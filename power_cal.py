"""从 xlsx 首列读取 EEG 原始信号，带通滤波后计算各节律绝对/相对功率。"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, welch
from scipy.fft import rfft, rfftfreq

_ALGO_DIR = Path(__file__).resolve().parent / "Algorithm"
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from MovementArtifact import (  # noqa: E402
    BandSuspiciousInfo,
    EegQualityInfo,
    EEG_ADAPTIVE_MAD_MULT,
    EEG_RAW_MAX_VALID,
    EEG_RAW_MIN_VALID,
    EEG_REJECT_SEGMENT_SEC,
    EEG_SEGMENT_MAX_DEVIATION,
    EEG_SEGMENT_MAX_PTP,
    EEG_SUSPICIOUS_ALPHA_RMS_FLOOR,
    EEG_SUSPICIOUS_ALPHA_RMS_MAD_MULT,
    EEG_SUSPICIOUS_ALPHA_RMS_RATIO,
    EEG_SUSPICIOUS_DELTA_RMS_MAD_MULT,
    EEG_SUSPICIOUS_DELTA_RMS_RATIO,
    EEG_SUSPICIOUS_MIN_DIFF,
    EEG_SUSPICIOUS_MIN_PTP,
    MODEL_ALPHA_SUSPICIOUS_DROP_RATIO,
    MODEL_ALPHA_SUSPICIOUS_WARN_RATIO,
    MODEL_REJECT_DROP_RATIO,
    MODEL_SUSPICIOUS_DROP_RATIO,
    MODEL_SUSPICIOUS_WARN_RATIO,
    MODEL_WINDOW_SEC,
    build_band_rms_suspicious,
    build_model_window_quality_table,
    build_threshold_rejection,
    clean_raw_signal,
    export_offline_raw_csvs,
    fit_bool_mask,
    merge_quality_with_band_suspicious,
    read_eeg_quality,
    rejected_spans as quality_mask_spans,
    slice_clean_raw_segment,
)

FILTER_ORDER = 4
KS1082_SAMPLE_RATE = 500.0
DEFAULT_SAMPLE_RATE = float(
    os.environ.get("KS1082_SAMPLE_RATE", str(KS1082_SAMPLE_RATE))
)

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0

# Experimental visualization path: remove alpha-suspicious samples from all
# bands, concatenate the remaining points, then lightly smooth the display.
ALPHA_REMOVAL_SMOOTH_SEC = 0.08

# 在 0.5–40 Hz 带通范围内的标准节律划分
EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

BAND_LABELS = {
    "delta": "δ",
    "theta": "θ",
    "alpha": "α",
    "beta": "β",
    "gamma": "gamma",
}

BAND_COLORS = {
    "delta": "#5B8FF9",
    "theta": "#5AD8A6",
    "alpha": "#F6BD16",
    "beta": "#E8684A",
    "gamma": "#9270CA",
}


def design_bandpass_sos(
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """设计 Butterworth 带通 SOS 系数（与 bandpass_filter 一致）。"""
    if low_hz <= 0:
        raise ValueError(f"低截止频率须 > 0，当前为 {low_hz}")
    if sample_rate <= 2 * high_hz:
        raise ValueError(
            f"采样率 {sample_rate} Hz 过低，无法设计 {high_hz} Hz 高通上限"
        )
    nyquist = sample_rate * 0.5
    return butter(
        order,
        (low_hz / nyquist, high_hz / nyquist),
        btype="bandpass",
        output="sos",
    )


class SampleRateEstimator:
    """统计实际接收样本速率，用于诊断是否丢样。"""

    def __init__(self, min_elapsed: float = 0.2, min_samples: int = 20) -> None:
        self._min_elapsed = min_elapsed
        self._min_samples = min_samples
        self._start: Optional[float] = None
        self._count = 0

    def reset(self) -> None:
        self._start = None
        self._count = 0

    def add(self, count: int = 1) -> None:
        if count <= 0:
            return
        now = time.monotonic()
        if self._start is None:
            self._start = now
        self._count += count

    def current_rate(self) -> Optional[float]:
        if self._start is None or self._count < self._min_samples:
            return None
        elapsed = time.monotonic() - self._start
        if elapsed < self._min_elapsed:
            return None
        rate = self._count / elapsed
        if 50.0 <= rate <= 2000.0:
            return rate
        return None


class RhythmStreamProcessor:
    """实时逐点节律滤波：先 0.5–40 Hz，再各节律窄带（与 extract_band_waveforms 一致）。"""

    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE) -> None:
        self._configured_rate = float(sample_rate)
        self._estimator = SampleRateEstimator()
        self._base_sos = design_bandpass_sos(
            self._configured_rate, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ
        )
        self._base_zi = sosfilt_zi(self._base_sos)
        self._band_sos: Dict[str, np.ndarray] = {}
        self._band_zi: Dict[str, np.ndarray] = {}
        for name, (low, high) in EEG_BANDS.items():
            self._band_sos[name] = design_bandpass_sos(
                self._configured_rate, low, high
            )
            self._band_zi[name] = sosfilt_zi(self._band_sos[name])
        self._initialized = False

    @property
    def sample_rate(self) -> float:
        return self._configured_rate

    @property
    def measured_sample_rate(self) -> Optional[float]:
        return self._estimator.current_rate()

    def reset(self) -> None:
        self._estimator.reset()
        self._base_zi = sosfilt_zi(self._base_sos)
        for name in EEG_BANDS:
            self._band_zi[name] = sosfilt_zi(self._band_sos[name])
        self._initialized = False

    def push(self, adc: int, band: Optional[str] = None) -> float:
        """band=None returns raw ADC; otherwise returns the selected rhythm band."""
        self._estimator.add(1)
        x = float(adc)
        if band is not None and band not in self._band_zi:
            raise ValueError(f"未知节律: {band!r}")

        if not self._initialized:
            self._base_zi = sosfilt_zi(self._base_sos) * x
        y_base, self._base_zi = sosfilt(self._base_sos, [x], zi=self._base_zi)
        base_value = float(y_base[0])
        if not self._initialized:
            for name in EEG_BANDS:
                self._band_zi[name] = sosfilt_zi(self._band_sos[name]) * base_value
            self._initialized = True

        selected_value = x
        for name in EEG_BANDS:
            y_band, self._band_zi[name] = sosfilt(
                self._band_sos[name], [base_value], zi=self._band_zi[name]
            )
            if name == band:
                selected_value = float(y_band[0])
        return selected_value


@dataclass(frozen=True)
class BandPowerResult:
    absolute: Dict[str, float]
    relative: Dict[str, float]
    total_power: float


@dataclass(frozen=True)
class PowerAnalysis:
    result: BandPowerResult
    freqs: np.ndarray
    psd: np.ndarray


@dataclass(frozen=True)
class SegmentBandComparison:
    """多段 EEG 各节律功率对比（2～4 段）；比值均相对第 1 段。"""

    ranges: Tuple[Tuple[float, float], ...]
    analyses: Tuple[PowerAnalysis, ...]
    absolute_ratios_vs_first: Tuple[Dict[str, float], ...]
    relative_ratios_vs_first: Tuple[Dict[str, float], ...]

    def __post_init__(self) -> None:
        if len(self.ranges) < 2:
            raise ValueError("段间对比至少需要 2 段")
        if len(self.ranges) != len(self.analyses):
            raise ValueError("ranges 与 analyses 数量须一致")

    @property
    def n_segments(self) -> int:
        return len(self.ranges)

    @property
    def range_a(self) -> Tuple[float, float]:
        return self.ranges[0]

    @property
    def range_b(self) -> Tuple[float, float]:
        return self.ranges[1]

    @property
    def analysis_a(self) -> PowerAnalysis:
        return self.analyses[0]

    @property
    def analysis_b(self) -> PowerAnalysis:
        return self.analyses[1]

    @property
    def absolute_ratio(self) -> Dict[str, float]:
        """兼容旧接口：第 2 段 / 第 1 段。"""
        return self.absolute_ratios_vs_first[1]

    @property
    def relative_ratio(self) -> Dict[str, float]:
        """兼容旧接口：第 2 段 / 第 1 段。"""
        return self.relative_ratios_vs_first[1]


SEGMENT_LABELS = ("A", "B", "C", "D")


def _ratios_vs_first(analyses: Sequence[PowerAnalysis]) -> Tuple[
    Tuple[Dict[str, float], ...],
    Tuple[Dict[str, float], ...],
]:
    first = analyses[0].result
    abs_list: List[Dict[str, float]] = []
    rel_list: List[Dict[str, float]] = []
    for analysis in analyses:
        abs_r: Dict[str, float] = {}
        rel_r: Dict[str, float] = {}
        for name in EEG_BANDS:
            a_abs = first.absolute[name]
            b_abs = analysis.result.absolute[name]
            a_rel = first.relative[name]
            b_rel = analysis.result.relative[name]
            abs_r[name] = b_abs / a_abs if a_abs > 0 else float("nan")
            rel_r[name] = b_rel / a_rel if a_rel > 0 else float("nan")
        abs_list.append(abs_r)
        rel_list.append(rel_r)
    return tuple(abs_list), tuple(rel_list)


def compare_multi_segment_band_powers(
    signal: np.ndarray,
    sample_rate: float,
    ranges: Sequence[Tuple[float, float]],
) -> SegmentBandComparison:
    """比较 2～4 段相同节律的绝对/相对功率；比值 = 各段 / 第 1 段。"""
    if not 2 <= len(ranges) <= 4:
        raise ValueError(f"段间对比支持 2～4 段，当前为 {len(ranges)} 段")
    analyses: List[PowerAnalysis] = []
    for start, end in ranges:
        seg = _slice_signal(signal, sample_rate, start, end)
        analyses.append(compute_band_powers(seg, sample_rate=sample_rate))
    abs_ratios, rel_ratios = _ratios_vs_first(analyses)
    return SegmentBandComparison(
        ranges=tuple((float(a), float(b)) for a, b in ranges),
        analyses=tuple(analyses),
        absolute_ratios_vs_first=abs_ratios,
        relative_ratios_vs_first=rel_ratios,
    )


def compare_segment_band_powers(
    signal: np.ndarray,
    sample_rate: float,
    range_a: Tuple[float, float],
    range_b: Tuple[float, float],
) -> SegmentBandComparison:
    """比较两段相同节律的绝对/相对功率，比值 = 段2 / 段1。"""
    return compare_multi_segment_band_powers(
        signal, sample_rate, (range_a, range_b)
    )


def compare_multi_segment_band_powers_cleaned(
    raw: np.ndarray,
    quality: EegQualityInfo,
    sample_rate: float,
    ranges: Sequence[Tuple[float, float]],
) -> SegmentBandComparison:
    """段间对比：各段先按原始时间切片，再去掉坏段/可疑段后算功率。"""
    if not 2 <= len(ranges) <= 4:
        raise ValueError(f"段间对比支持 2～4 段，当前为 {len(ranges)} 段")
    min_samples = max(1, int(sample_rate))
    analyses: List[PowerAnalysis] = []
    sizes: List[int] = []
    for start, end in ranges:
        seg = slice_clean_raw_segment(raw, quality, sample_rate, start, end)
        sizes.append(int(seg.size))
        if seg.size < min_samples:
            raise ValueError(
                f"删减后有效样本过少 {sizes}，请检查段间时间或坏段比例"
            )
        analyses.append(
            compute_band_powers(seg.astype(np.float64), sample_rate=sample_rate)
        )
    abs_ratios, rel_ratios = _ratios_vs_first(analyses)
    return SegmentBandComparison(
        ranges=tuple((float(a), float(b)) for a, b in ranges),
        analyses=tuple(analyses),
        absolute_ratios_vs_first=abs_ratios,
        relative_ratios_vs_first=rel_ratios,
    )


def compare_segment_band_powers_cleaned(
    raw: np.ndarray,
    quality: EegQualityInfo,
    sample_rate: float,
    range_a: Tuple[float, float],
    range_b: Tuple[float, float],
) -> SegmentBandComparison:
    """段间对比：各段先按原始时间切片，再去掉坏段/可疑段后算功率。"""
    return compare_multi_segment_band_powers_cleaned(
        raw, quality, sample_rate, (range_a, range_b)
    )


def format_segment_comparison(comp: SegmentBandComparison) -> str:
    labels = SEGMENT_LABELS[: comp.n_segments]
    range_bits = [
        f"{lab}: {rng[0]:g}–{rng[1]:g} s"
        for lab, rng in zip(labels, comp.ranges)
    ]
    lines = [
        "时间段  " + "  |  ".join(range_bits),
        "功率比 = 各段 / 段A（第 1 段）",
    ]
    header = f"{'节律':<8}"
    for lab in labels:
        header += f" {lab + '绝对':>12}"
    header += f" {'绝对比(相对A)':>14}"
    for lab in labels:
        header += f" {lab + '相对%':>10}"
    lines.append(header)
    lines.append("-" * max(78, len(header)))
    for name in EEG_BANDS:
        label = BAND_LABELS[name]
        row = f"{label:<8}"
        for analysis in comp.analyses:
            row += f" {analysis.result.absolute[name]:>12.4e}"
        # 用最后一段相对 A 的比值做摘要列；多段时各比值见图
        last_ratio = comp.absolute_ratios_vs_first[-1][name]
        row += f" {last_ratio:>14.3f}"
        for analysis in comp.analyses:
            row += f" {analysis.result.relative[name] * 100.0:>10.2f}"
        lines.append(row)
    if comp.n_segments > 2:
        lines.append("各段相对 A 的绝对功率比:")
        for i, lab in enumerate(labels[1:], start=1):
            bits = [
                f"{BAND_LABELS[n]}={comp.absolute_ratios_vs_first[i][n]:.3f}"
                for n in EEG_BANDS
            ]
            lines.append(f"  {lab}/A: " + ", ".join(bits))
    return "\n".join(lines)


def _slice_signal(
    signal: np.ndarray,
    sample_rate: float,
    start_seconds: float,
    end_seconds: float,
) -> np.ndarray:
    """按时间范围截取原始信号片段。"""
    n_total = len(signal)
    total_duration = n_total / sample_rate
    start = max(0.0, start_seconds)
    end = min(end_seconds, total_duration)
    if start >= end:
        raise ValueError(
            f"无效时间范围: {start_seconds:g}–{end_seconds:g} s "
            f"(数据总长 {total_duration:.2f} s)"
        )
    i_start = min(n_total - 1, int(start * sample_rate))
    i_end = max(i_start + 1, min(n_total, int(end * sample_rate)))
    return signal[i_start:i_end].copy()


def read_eeg_column(xlsx_path: str | Path, column: int = 0) -> np.ndarray:
    """读取 xlsx 指定列，丢弃非数值行。"""
    series = pd.read_excel(xlsx_path, usecols=[column], header=None).iloc[:, 0]
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size < int(DEFAULT_SAMPLE_RATE):
        raise ValueError(f"有效样本过少 ({values.size})，请检查文件首列是否为 EEG 数据")
    return values


def read_eeg_signal(data_path: str | Path, column: int = 0) -> np.ndarray:
    """读取 xlsx 或 csv 中的 EEG 原始序列（csv 优先 ch1_raw 列）。"""
    path = Path(data_path)
    if path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
        if "ch1_raw" in frame.columns:
            series = frame["ch1_raw"]
        else:
            series = frame.iloc[:, column]
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    else:
        values = read_eeg_column(path, column=column)
    if values.size < int(DEFAULT_SAMPLE_RATE):
        raise ValueError(
            f"有效样本过少 ({values.size})，请检查 {path.name} 是否包含 EEG 数据"
        )
    return values


def bandpass_filter(
    signal: np.ndarray,
    sample_rate: float,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    if low_hz <= 0:
        raise ValueError(f"低截止频率须 > 0，当前为 {low_hz}")
    if sample_rate <= 2 * high_hz:
        raise ValueError(
            f"采样率 {sample_rate} Hz 过低，无法设计 {high_hz} Hz 高通上限"
        )
    nyquist = sample_rate * 0.5
    sos = butter(
        order,
        (low_hz / nyquist, high_hz / nyquist),
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal, axis=0)


def _band_power_from_psd(
    freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float
) -> float:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def compute_band_powers(
    signal: np.ndarray,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    welch_seconds: float = 2.0,
) -> PowerAnalysis:
    """对滤波后信号做 Welch PSD，积分得到各频段绝对功率与相对功率。"""
    filtered = bandpass_filter(signal, sample_rate)
    nperseg = min(len(filtered), max(int(sample_rate * welch_seconds), 256))
    freqs, psd = welch(filtered, fs=sample_rate, nperseg=nperseg)

    absolute: Dict[str, float] = {}
    for name, (low, high) in EEG_BANDS.items():
        absolute[name] = _band_power_from_psd(freqs, psd, low, high)

    total = sum(absolute.values())
    if total <= 0.0:
        relative = {name: 0.0 for name in EEG_BANDS}
    else:
        relative = {name: absolute[name] / total for name in EEG_BANDS}

    result = BandPowerResult(absolute=absolute, relative=relative, total_power=total)
    return PowerAnalysis(result=result, freqs=freqs, psd=psd)


def compute_fft_spectrum(
    signal: np.ndarray,
    sample_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """对带通滤波后信号做单边 FFT，返回频率轴与幅度谱 (2|X(f)|/N，DC/Nyquist 为 |X|/N)。"""
    filtered = bandpass_filter(signal, sample_rate)
    n = len(filtered)
    spectrum = rfft(filtered)
    freqs = rfftfreq(n, d=1.0 / sample_rate)
    amplitude = (2.0 / n) * np.abs(spectrum)
    amplitude[0] /= 2.0
    if n % 2 == 0:
        amplitude[-1] /= 2.0
    return freqs, amplitude


def _smooth_spectrum_for_plot(
    freqs: np.ndarray,
    values: np.ndarray,
    *,
    max_points: int = 800,
) -> Tuple[np.ndarray, np.ndarray]:
    """将过密的 FFT 频点合并为更少的显示点，避免折线连成竖向“梳齿”。"""
    n = values.size
    if n <= max_points:
        return freqs, values
    n_bins = max_points
    edges = np.linspace(freqs[0], freqs[-1], n_bins + 1)
    idx = np.clip(np.digitize(freqs, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    f_out = np.bincount(idx, weights=freqs, minlength=n_bins) / np.maximum(counts, 1)
    v_out = np.bincount(idx, weights=values, minlength=n_bins) / np.maximum(counts, 1)
    valid = counts > 0
    return f_out[valid], v_out[valid]


def extract_band_waveforms(
    signal: np.ndarray,
    sample_rate: float,
) -> Dict[str, np.ndarray]:
    """先 0.5–40 Hz 带通，再对各节律窄带滤波，得到随时间变化的成分波形。"""
    base = bandpass_filter(signal, sample_rate)
    return {
        name: bandpass_filter(base, sample_rate, low_hz=low, high_hz=high)
        for name, (low, high) in EEG_BANDS.items()
    }


def _slice_waveforms(
    waveforms: Dict[str, np.ndarray],
    sample_rate: float,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """按时间范围截取波形。end_seconds 为 None 时取到数据末尾。"""
    n_total = len(next(iter(waveforms.values())))
    total_duration = n_total / sample_rate

    start = max(0.0, start_seconds)
    end = total_duration if end_seconds is None else min(end_seconds, total_duration)
    if start >= end:
        raise ValueError(
            f"无效波形时间范围: {start_seconds:g}–{end_seconds} s "
            f"(数据总长 {total_duration:.2f} s)"
        )

    i_start = min(n_total - 1, int(start * sample_rate))
    i_end = max(i_start + 1, min(n_total, int(end * sample_rate)))
    sliced = {name: wf[i_start:i_end] for name, wf in waveforms.items()}
    time_axis = np.arange(i_end - i_start, dtype=np.float64) / sample_rate + start
    return sliced, time_axis


def export_offline_waveform_csvs(
    data_path: str | Path,
    raw: np.ndarray,
    waveforms: Dict[str, np.ndarray],
    quality: EegQualityInfo,
    alpha_info: BandSuspiciousInfo | None,
    sample_rate: float,
) -> tuple[Path, Path, int]:
    """Save full and cleaned offline raw tables (no rhythm waveform columns)."""
    del waveforms  ## 节律波形仅用于绘图，离线 CSV 不再保存
    return export_offline_raw_csvs(
        data_path,
        raw,
        quality,
        sample_rate,
        alpha_info=alpha_info,
    )


def _moving_average(signal: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1 or signal.size < 3:
        return signal.copy()
    window_samples = min(window_samples, signal.size)
    if window_samples % 2 == 0:
        window_samples += 1
    kernel = np.ones(window_samples, dtype=np.float64) / window_samples
    pad = window_samples // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _smooth_only_near_gaps(
    original_signal: np.ndarray,
    keep_mask: np.ndarray,
    *,
    smooth_samples: int,
) -> np.ndarray:
    """Concatenate kept samples and smooth only around artificial join points."""
    kept_indices = np.flatnonzero(keep_mask)
    y = original_signal[kept_indices].astype(np.float64, copy=True)
    if y.size < 3 or smooth_samples <= 1:
        return y

    seam_positions = np.flatnonzero(np.diff(kept_indices) > 1) + 1
    if seam_positions.size == 0:
        return y

    smoothed = _moving_average(y, smooth_samples)
    half = max(1, smooth_samples // 2)
    for seam in seam_positions:
        start = max(0, seam - half)
        end = min(y.size, seam + half)
        y[start:end] = smoothed[start:end]
    return y


def remove_waveforms_keep_time_axis(
    waveforms: Dict[str, np.ndarray],
    remove_mask: np.ndarray,
    sample_rate: float,
    *,
    smooth_sec: float = ALPHA_REMOVAL_SMOOTH_SEC,
) -> Dict[str, np.ndarray]:
    """Blank removed samples with NaN, preserving the original time axis."""
    if not waveforms:
        return {}
    n_total = len(next(iter(waveforms.values())))
    remove_mask = fit_bool_mask(remove_mask, n_total)
    if not np.any(remove_mask):
        return {name: wf.copy() for name, wf in waveforms.items()}

    cleaned: Dict[str, np.ndarray] = {}
    for name, wf in waveforms.items():
        y = wf[:n_total].astype(np.float64, copy=True)
        y[remove_mask] = np.nan
        cleaned[name] = y
    return cleaned


def remove_waveforms_compress_time_axis(
    waveforms: Dict[str, np.ndarray],
    remove_mask: np.ndarray,
    sample_rate: float,
    *,
    smooth_sec: float = ALPHA_REMOVAL_SMOOTH_SEC,
) -> Dict[str, np.ndarray]:
    """Remove samples, concatenate the rest, and smooth only join seams."""
    if not waveforms:
        return {}
    n_total = len(next(iter(waveforms.values())))
    remove_mask = fit_bool_mask(remove_mask, n_total)
    if not np.any(remove_mask):
        return {name: wf.copy() for name, wf in waveforms.items()}

    keep_mask = ~remove_mask
    if np.count_nonzero(keep_mask) < max(3, int(sample_rate * 0.5)):
        return {name: wf.copy() for name, wf in waveforms.items()}

    smooth_samples = max(1, int(round(sample_rate * smooth_sec)))
    cleaned: Dict[str, np.ndarray] = {}
    for name, wf in waveforms.items():
        cleaned[name] = _smooth_only_near_gaps(
            wf[:n_total],
            keep_mask,
            smooth_samples=smooth_samples,
        )
    return cleaned


def waveform_y_limits(waveforms: Dict[str, np.ndarray]) -> Dict[str, tuple[float, float]]:
    limits: Dict[str, tuple[float, float]] = {}
    for name, wf in waveforms.items():
        finite = np.asarray(wf, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            limits[name] = (-1.0, 1.0)
            continue
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        if ymin == ymax:
            pad = max(1.0, abs(ymin) * 0.1)
        else:
            pad = (ymax - ymin) * 0.08
        limits[name] = (ymin - pad, ymax + pad)
    return limits


def format_results(result: BandPowerResult) -> str:
    lines = [
        f"{'节律':<8} {'频段(Hz)':<14} {'绝对功率':>14} {'相对功率(%)':>14}",
        "-" * 54,
    ]
    for name, (low, high) in EEG_BANDS.items():
        label = BAND_LABELS[name]
        abs_p = result.absolute[name]
        rel_p = result.relative[name] * 100.0
        lines.append(
            f"{label:<8} {low:>5.1f}-{high:<5.1f} {abs_p:>14.6e} {rel_p:>14.2f}"
        )
    lines.append("-" * 54)
    lines.append(f"{'合计':<8} {'0.5-40.0':<14} {result.total_power:>14.6e} {'100.00':>14}")
    return "\n".join(lines)


def analyze_xlsx(
    xlsx_path: str | Path,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> PowerAnalysis:
    raw = read_eeg_column(xlsx_path)
    return compute_band_powers(raw, sample_rate=sample_rate)


def _setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_band_powers(
    analysis: PowerAnalysis,
    title: str = "EEG 各节律功率",
    quality: EegQualityInfo | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制功率谱（分节律着色）与各节律绝对/相对功率柱状图。"""
    import matplotlib.pyplot as plt

    _setup_matplotlib()

    result = analysis.result
    names = list(EEG_BANDS.keys())
    labels = [
        f"{BAND_LABELS[n]}\n({EEG_BANDS[n][0]:g}-{EEG_BANDS[n][1]:g} Hz)" for n in names
    ]
    colors = [BAND_COLORS[n] for n in names]
    rel_pct = [result.relative[n] * 100.0 for n in names]
    abs_pow = [result.absolute[n] for n in names]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    quality_text = ""
    if quality is not None and quality.has_tag:
        quality_text = (
            f" | 阈值拒绝率 {quality.reject_rate:.1%}"
            f" | 可疑率 {quality.suspicious_rate:.1%}"
        )
    fig.suptitle(f"{title}{quality_text}", fontsize=14, fontweight="bold")

    ax_psd = axes[0]
    ax_psd.plot(analysis.freqs, analysis.psd, color="#333333", lw=0.8, label="PSD")
    ymax = float(np.max(analysis.psd)) if analysis.psd.size else 1.0
    for name, (low, high) in EEG_BANDS.items():
        mask = (analysis.freqs >= low) & (analysis.freqs <= high)
        ax_psd.fill_between(
            analysis.freqs,
            0,
            analysis.psd,
            where=mask,
            color=BAND_COLORS[name],
            alpha=0.45,
            label=BAND_LABELS[name],
        )
    ax_psd.set_xlim(BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)
    ax_psd.set_ylim(0, ymax * 1.08)
    ax_psd.set_xlabel("频率 (Hz)")
    ax_psd.set_ylabel("功率谱密度")
    ax_psd.set_title("Welch 功率谱（按节律分色）")
    ax_psd.legend(loc="upper right", ncol=3, fontsize=9)
    ax_psd.grid(True, alpha=0.3)

    ax_rel = axes[1]
    bars_rel = ax_rel.bar(labels, rel_pct, color=colors, edgecolor="white", linewidth=0.8)
    ax_rel.set_ylabel("相对功率 (%)")
    ax_rel.set_title("各节律相对功率")
    ax_rel.set_ylim(0, max(rel_pct) * 1.15 if rel_pct else 1.0)
    ax_rel.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars_rel, rel_pct):
        ax_rel.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax_abs = axes[2]
    bars_abs = ax_abs.bar(labels, abs_pow, color=colors, edgecolor="white", linewidth=0.8)
    ax_abs.set_ylabel("绝对功率")
    ax_abs.set_title("各节律绝对功率")
    ax_abs.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars_abs, abs_pow):
        ax_abs.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {out.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fft(
    signal: np.ndarray,
    sample_rate: float,
    *,
    time_range: tuple[float, float] | None = None,
    title: str = "EEG FFT 幅度谱",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制单边 FFT 幅度谱（0.5–40 Hz 带通后），按节律分色。

    time_range: (起始秒, 结束秒)；None 表示使用整段数据。
    """
    import matplotlib.pyplot as plt

    _setup_matplotlib()

    if time_range is not None:
        segment = _slice_signal(signal, sample_rate, time_range[0], time_range[1])
        t_label = f"{time_range[0]:g}–{time_range[1]:g} s"
    else:
        segment = signal
        duration = len(signal) / sample_rate
        t_label = f"0–{duration:.1f} s"

    freqs, amplitude = compute_fft_spectrum(segment, sample_rate)
    mask_plot = (freqs >= BANDPASS_LOW_HZ) & (freqs <= BANDPASS_HIGH_HZ)
    f_plot, a_plot = _smooth_spectrum_for_plot(
        freqs[mask_plot], amplitude[mask_plot]
    )

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    fig.suptitle(f"{title}（{t_label}）", fontsize=14, fontweight="bold")

    ymax = float(np.max(a_plot)) if a_plot.size else 1.0
    for name, (low, high) in EEG_BANDS.items():
        band_mask = (f_plot >= low) & (f_plot <= high)
        ax.fill_between(
            f_plot,
            0,
            a_plot,
            where=band_mask,
            color=BAND_COLORS[name],
            alpha=0.55,
            linewidth=0,
            label=BAND_LABELS[name],
        )
    ax.set_xlim(BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ)
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度 |X(f)|")
    ax.set_title("FFT 幅度谱（按节律分色，显示用频点合并）")
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"FFT 图已保存: {out.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_segment_power_comparison(
    comparison: SegmentBandComparison,
    *,
    title: str = "多段 EEG 节律功率对比",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制 2～4 段各节律相对/绝对功率及相对第 1 段比值柱状图。"""
    import matplotlib.pyplot as plt

    _setup_matplotlib()

    n_seg = comparison.n_segments
    seg_labels = SEGMENT_LABELS[:n_seg]
    names = list(EEG_BANDS.keys())
    band_labels = [
        f"{BAND_LABELS[n]}\n({EEG_BANDS[n][0]:g}-{EEG_BANDS[n][1]:g} Hz)" for n in names
    ]
    colors = [BAND_COLORS[n] for n in names]
    range_bits = "  |  ".join(
        f"{lab}: {rng[0]:g}–{rng[1]:g} s"
        for lab, rng in zip(seg_labels, comparison.ranges)
    )

    fig, axes = plt.subplots(3, 1, figsize=(max(10, 2.2 * n_seg + 6), 11), constrained_layout=True)
    fig.suptitle(
        f"{title}\n{range_bits}\n比值 = 各段 / 段A",
        fontsize=12,
        fontweight="bold",
    )

    x = np.arange(len(names))
    width = min(0.8 / n_seg, 0.28)
    offsets = (np.arange(n_seg) - (n_seg - 1) / 2.0) * width

    ax_rel = axes[0]
    for i, (lab, analysis, offset) in enumerate(
        zip(seg_labels, comparison.analyses, offsets)
    ):
        vals = [analysis.result.relative[n] * 100.0 for n in names]
        alpha = 0.45 + 0.55 * i / max(n_seg - 1, 1)
        ax_rel.bar(
            x + offset,
            vals,
            width,
            label=f"{lab} ({comparison.ranges[i][0]:g}–{comparison.ranges[i][1]:g} s)",
            color=colors,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.6,
        )
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(band_labels)
    ax_rel.set_ylabel("相对功率 (%)")
    ax_rel.set_title(f"各节律相对功率（{n_seg} 段）")
    ax_rel.legend(loc="upper right", fontsize=8)
    ax_rel.grid(True, axis="y", alpha=0.3)

    ax_abs = axes[1]
    for i, (lab, analysis, offset) in enumerate(
        zip(seg_labels, comparison.analyses, offsets)
    ):
        vals = [analysis.result.absolute[n] for n in names]
        alpha = 0.45 + 0.55 * i / max(n_seg - 1, 1)
        ax_abs.bar(
            x + offset,
            vals,
            width,
            label=lab,
            color=colors,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.6,
        )
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(band_labels)
    ax_abs.set_ylabel("绝对功率")
    ax_abs.set_title("各节律绝对功率")
    ax_abs.legend(loc="upper right", fontsize=8)
    ax_abs.grid(True, axis="y", alpha=0.3)

    ax_ratio = axes[2]
    # 对比段（B/C/D）相对 A 的绝对功率比；同节律分组
    n_ratio = n_seg - 1
    ratio_width = min(0.8 / max(n_ratio, 1), 0.28)
    ratio_offsets = (np.arange(n_ratio) - (n_ratio - 1) / 2.0) * ratio_width
    for j, (lab, offset) in enumerate(zip(seg_labels[1:], ratio_offsets)):
        vals = [
            comparison.absolute_ratios_vs_first[j + 1][n]
            if np.isfinite(comparison.absolute_ratios_vs_first[j + 1][n])
            else 0.0
            for n in names
        ]
        alpha = 0.55 + 0.45 * j / max(n_ratio - 1, 1)
        bars = ax_ratio.bar(
            x + offset,
            vals,
            ratio_width,
            label=f"{lab}/A",
            color=colors,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, name in zip(bars, names):
            ratio = comparison.absolute_ratios_vs_first[j + 1][name]
            if np.isfinite(ratio):
                ax_ratio.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{ratio:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax_ratio.axhline(1.0, color="#666666", ls="--", lw=1, label="比值 = 1")
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(band_labels)
    ax_ratio.set_ylabel("绝对功率比 (相对段A)")
    ax_ratio.set_title("各节律绝对功率比")
    ax_ratio.legend(loc="upper right", fontsize=8)
    ax_ratio.grid(True, axis="y", alpha=0.3)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"段间对比图已保存: {out.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_band_waveforms(
    waveforms: Dict[str, np.ndarray],
    sample_rate: float,
    *,
    title: str = "EEG 各节律时域波形",
    time_range: tuple[float, float] | None = None,
    max_seconds: float | None = 10.0,
    quality: EegQualityInfo | None = None,
    y_limits: Dict[str, tuple[float, float]] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制 δ/θ/α/β/γ 各节律窄带滤波后的波形（随时间变化）。

    time_range: (起始秒, 结束秒)，如 (10, 20) 显示 10–20 s；设置后优先于 max_seconds。
    max_seconds: 从 0 秒起显示时长；None 表示显示到数据末尾（仅当 time_range 未设置时生效）。
    """
    import matplotlib.pyplot as plt

    _setup_matplotlib()

    if time_range is not None:
        t_start, t_end = time_range
    elif max_seconds is None:
        t_start, t_end = 0.0, None
    else:
        t_start, t_end = 0.0, max_seconds

    sliced, time_axis = _slice_waveforms(
        waveforms, sample_rate, start_seconds=t_start, end_seconds=t_end
    )
    names = list(EEG_BANDS.keys())
    t0 = float(time_axis[0]) if time_axis.size else t_start
    t1 = float(time_axis[-1]) if time_axis.size else t_start
    rejected_spans: list[tuple[float, float]] = []
    if quality is not None and quality.has_rejection:
        rejected_spans = quality_mask_spans(
            quality.reject_mask,
            sample_rate,
            start_seconds=t0,
            end_seconds=t1,
        )
    suspicious_spans: list[tuple[float, float]] = []
    if (
        quality is not None
        and quality.has_suspicious
        and quality.suspicious_mask is not None
    ):
        suspicious_spans = quality_mask_spans(
            quality.suspicious_mask,
            sample_rate,
            start_seconds=t0,
            end_seconds=t1,
        )

    fig, axes = plt.subplots(len(names), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    if len(names) == 1:
        axes = [axes]
    quality_text = ""
    if quality is not None and quality.has_tag:
        quality_text = (
            f" | 阈值拒绝率 {quality.reject_rate:.1%}"
            f" | 可疑率 {quality.suspicious_rate:.1%}"
        )
    fig.suptitle(
        f"{title}（{t0:.1f}–{t1:.1f} s）{quality_text}",
        fontsize=14,
        fontweight="bold",
    )

    for ax, name in zip(axes, names):
        low, high = EEG_BANDS[name]
        y = sliced[name]
        ax.plot(time_axis, y, color=BAND_COLORS[name], lw=0.6)
        for span_start, span_end in suspicious_spans:
            ax.axvspan(span_start, span_end, color="#FBC02D", alpha=0.16, lw=0)
        for span_start, span_end in rejected_spans:
            ax.axvspan(span_start, span_end, color="#D32F2F", alpha=0.12, lw=0)
        ax.set_ylabel(f"{BAND_LABELS[name]}\n({low:g}-{high:g} Hz)")
        if y_limits is not None and name in y_limits:
            ax.set_ylim(*y_limits[name])
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    axes[-1].set_xlabel("时间 (s)")
    labels = []
    if rejected_spans:
        labels.append("红色背景=坏段")
    if suspicious_spans:
        labels.append("黄色背景=可疑段")
    title_suffix = f"；{'，'.join(labels)}" if labels else ""
    axes[0].set_title(f"各节律窄带滤波波形（自上而下：δ → γ）{title_suffix}")

    initial_xlim = axes[0].get_xlim()
    initial_ylims = [ax.get_ylim() for ax in axes]
    min_window = max(1.0 / sample_rate, 0.02)

    def _scroll_zoom(event) -> None:
        if event.inaxes not in axes:
            return
        scale = 0.8 if event.button == "up" else 1.25
        key = (event.key or "").lower()
        if "control" in key or "ctrl" in key:
            ax = event.inaxes
            y0, y1 = ax.get_ylim()
            center = event.ydata if event.ydata is not None else (y0 + y1) * 0.5
            half = max((y1 - y0) * scale * 0.5, 1e-12)
            ax.set_ylim(center - half, center + half)
        else:
            x0, x1 = axes[0].get_xlim()
            center = event.xdata if event.xdata is not None else (x0 + x1) * 0.5
            new_width = max((x1 - x0) * scale, min_window)
            data_min, data_max = initial_xlim
            if new_width >= data_max - data_min:
                new_x0, new_x1 = data_min, data_max
            else:
                new_x0 = center - new_width * 0.5
                new_x1 = center + new_width * 0.5
                if new_x0 < data_min:
                    new_x1 += data_min - new_x0
                    new_x0 = data_min
                if new_x1 > data_max:
                    new_x0 -= new_x1 - data_max
                    new_x1 = data_max
            for ax in axes:
                ax.set_xlim(new_x0, new_x1)
        fig.canvas.draw_idle()

    def _key_reset(event) -> None:
        if (event.key or "").lower() != "r":
            return
        for ax, ylim in zip(axes, initial_ylims):
            ax.set_xlim(*initial_xlim)
            ax.set_ylim(*ylim)
        fig.canvas.draw_idle()

    if show:
        fig.canvas.mpl_connect("scroll_event", _scroll_zoom)
        fig.canvas.mpl_connect("key_press_event", _key_reset)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"波形图已保存: {out.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _tagged_output_path(path: Path, suffix: str, extension: str) -> Path:
    if suffix:
        return path.parent / f"{path.stem}_{suffix}{extension}"
    return path.parent / f"{path.stem}{extension}"


def _run_power_analysis_pass(
    *,
    raw: np.ndarray,
    path: Path,
    sample_rate: float,
    pass_label: str,
    output_tag: str,
    quality: EegQualityInfo,
    enable_alpha_suspicious: bool,
    compare_segments: Sequence[Tuple[float, float]] | None,
    compare_segments_cleaned: bool,
    reference_raw: np.ndarray | None,
    reference_quality: EegQualityInfo | None,
    show_plot: bool,
    save_plot: bool,
    show_waveform: bool,
    save_waveform: bool,
    waveform_seconds: float | None,
    waveform_time_range: tuple[float, float] | None,
    show_fft: bool,
    save_fft: bool,
    fft_time_range: tuple[float, float] | None,
    show_segment_compare: bool,
    save_segment_compare: bool,
    save_model_window_table: bool,
    save_offline_waveform_data: bool,
    remove_alpha_artifact_segments: bool,
    alpha_artifact_removal_view: str,
) -> BandPowerResult:
    alpha_info: BandSuspiciousInfo | None = None
    quality_for_plot = quality
    if enable_alpha_suspicious:
        alpha_info = build_band_rms_suspicious(
            raw.astype(np.float64),
            sample_rate,
            band_name="alpha",
        )
        quality_for_plot = merge_quality_with_band_suspicious(quality, alpha_info)

    analysis = compute_band_powers(raw.astype(np.float64), sample_rate=sample_rate)
    result = analysis.result
    duration = len(raw) / sample_rate

    print()
    print("=" * 72)
    print(f"功率分析 · {pass_label}")
    print("=" * 72)
    print(f"文件: {path.resolve()}")
    print(f"样本数: {len(raw)}  |  采样率: {sample_rate:g} Hz  |  时长: {duration:.2f} s")
    if quality.has_tag:
        rejected = int(np.count_nonzero(quality.reject_mask))
        suspicious = (
            int(np.count_nonzero(quality.suspicious_mask))
            if quality.suspicious_mask is not None
            else 0
        )
        print(
            f"阈值拒绝: {quality.reject_rate:.1%} ({rejected}/{quality.reject_mask.size} 点)"
        )
        print(f"可疑片段: {quality.suspicious_rate:.1%} ({suspicious}/{quality.reject_mask.size} 点)")
    print(f"带通滤波: {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz (零相位 sosfiltfilt)")
    if alpha_info is not None:
        alpha_points = int(np.count_nonzero(alpha_info.mask))
        print(
            "Alpha suspicious: "
            f"{alpha_info.rate:.1%} ({alpha_points}/{len(raw)} samples), "
            f"RMS threshold={alpha_info.rms_threshold:.4g}, "
            f"median={alpha_info.rms_median:.4g}, MAD={alpha_info.rms_mad:.4g}"
        )
    print()
    print(format_results(result))

    if save_model_window_table:
        window_table = build_model_window_quality_table(
            quality_for_plot,
            alpha_info,
            sample_rate,
        )
        if not window_table.empty:
            window_path = _tagged_output_path(path, output_tag, "_model_windows.csv")
            window_table.to_csv(window_path, index=False, encoding="utf-8-sig")
            print(f"Model window quality table saved: {window_path.resolve()}")

    waveforms: Dict[str, np.ndarray] | None = None
    if save_offline_waveform_data or show_waveform or save_waveform:
        waveforms = extract_band_waveforms(raw.astype(np.float64), sample_rate)

    if save_offline_waveform_data and waveforms is not None:
        full_path, clean_path, removed_points = export_offline_waveform_csvs(
            _tagged_output_path(path, output_tag, ".csv"),
            raw,
            waveforms,
            quality_for_plot,
            alpha_info,
            sample_rate,
        )
        print(f"Offline full raw CSV saved: {full_path.resolve()}")
        print(
            f"Offline cleaned raw CSV saved: {clean_path.resolve()} "
            f"(removed {removed_points} samples)"
        )

    if show_plot or save_plot:
        plot_path = (
            _tagged_output_path(path, output_tag, "_band_power.png")
            if save_plot
            else None
        )
        plot_band_powers(
            analysis,
            title=f"EEG 节律功率 · {pass_label} · {path.stem}",
            quality=quality_for_plot,
            save_path=plot_path,
            show=show_plot,
        )

    if show_waveform or save_waveform:
        if waveforms is None:
            waveforms = extract_band_waveforms(raw.astype(np.float64), sample_rate)
        wf_path = (
            _tagged_output_path(path, output_tag, "_band_waveform.png")
            if save_waveform
            else None
        )
        plot_band_waveforms(
            waveforms,
            sample_rate,
            title=f"EEG 节律波形 · {pass_label} · {path.stem}",
            time_range=waveform_time_range,
            max_seconds=waveform_seconds,
            quality=quality_for_plot,
            save_path=wf_path,
            show=show_waveform,
        )
        if remove_alpha_artifact_segments and alpha_info is not None:
            removal_view = alpha_artifact_removal_view.lower().strip()
            if removal_view not in {"gap", "compressed", "both"}:
                removal_view = "both"
            gap_waveforms = remove_waveforms_keep_time_axis(
                waveforms,
                alpha_info.mask,
                sample_rate,
            )
            removed_points = int(np.count_nonzero(alpha_info.mask))
            removed_quality = EegQualityInfo(
                reject_mask=fit_bool_mask(alpha_info.mask, len(raw)),
                reject_rate=alpha_info.rate,
                has_tag=True,
                source="alpha_suspicious_removed",
                suspicious_mask=np.zeros(len(raw), dtype=bool),
                suspicious_rate=0.0,
            )
            original_limits = waveform_y_limits(waveforms)
            gap_path = (
                _tagged_output_path(path, f"{output_tag}_alpha_removed_gap", "_band_waveform.png")
                if save_waveform and removal_view in {"gap", "both"}
                else None
            )
            plot_band_waveforms(
                gap_waveforms,
                sample_rate,
                title=(
                    f"EEG alpha removed gap · {pass_label} · {path.stem} "
                    f"(removed {removed_points} samples)"
                ),
                time_range=waveform_time_range,
                max_seconds=waveform_seconds,
                quality=removed_quality,
                y_limits=original_limits,
                save_path=gap_path,
                show=show_waveform and removal_view in {"gap", "both"},
            )
            compressed_waveforms = remove_waveforms_compress_time_axis(
                waveforms,
                alpha_info.mask,
                sample_rate,
            )
            compressed_path = (
                _tagged_output_path(
                    path, f"{output_tag}_alpha_removed_compressed", "_band_waveform.png"
                )
                if save_waveform and removal_view in {"compressed", "both"}
                else None
            )
            plot_band_waveforms(
                compressed_waveforms,
                sample_rate,
                title=(
                    f"EEG alpha removed compressed · {pass_label} · {path.stem} "
                    f"(removed {removed_points} samples)"
                ),
                time_range=None,
                max_seconds=None,
                quality=None,
                y_limits=original_limits,
                save_path=compressed_path,
                show=show_waveform and removal_view in {"compressed", "both"},
            )

    if show_fft or save_fft:
        fft_path = _tagged_output_path(path, output_tag, "_fft.png") if save_fft else None
        plot_fft(
            raw.astype(np.float64),
            sample_rate,
            time_range=fft_time_range,
            title=f"EEG FFT · {pass_label} · {path.stem}",
            save_path=fft_path,
            show=show_fft,
        )

    if compare_segments is not None:
        ranges = tuple((float(a), float(b)) for a, b in compare_segments)
        try:
            if compare_segments_cleaned:
                if reference_raw is None or reference_quality is None:
                    raise ValueError("删减后段间对比缺少原始序列参考")
                comparison = compare_multi_segment_band_powers_cleaned(
                    reference_raw.astype(np.float64),
                    reference_quality,
                    sample_rate,
                    ranges,
                )
            else:
                comparison = compare_multi_segment_band_powers(
                    raw.astype(np.float64),
                    sample_rate,
                    ranges,
                )
        except ValueError as exc:
            print(f"跳过段间对比 ({pass_label}): {exc}")
        else:
            print()
            print(format_segment_comparison(comparison))
            if show_segment_compare or save_segment_compare:
                cmp_path = (
                    _tagged_output_path(path, output_tag, "_segment_compare.png")
                    if save_segment_compare
                    else None
                )
                plot_segment_power_comparison(
                    comparison,
                    title=f"段间节律功率 · {pass_label} · {path.stem}",
                    save_path=cmp_path,
                    show=show_segment_compare,
                )

    return result


def run_analysis(
    xlsx_path: str | Path,
    sample_rate: float,
    *,
    show_plot: bool = True,
    save_plot: bool = False,
    show_waveform: bool = True,
    save_waveform: bool = False,
    waveform_seconds: float | None = 10.0,
    waveform_time_range: tuple[float, float] | None = None,
    show_fft: bool = True,
    save_fft: bool = False,
    fft_time_range: tuple[float, float] | None = None,
    compare_segments: Sequence[Tuple[float, float]] | None = None,
    show_segment_compare: bool = True,
    save_segment_compare: bool = False,
    enable_alpha_suspicious: bool = True,
    save_model_window_table: bool = True,
    save_offline_waveform_data: bool = True,
    remove_alpha_artifact_segments: bool = True,
    alpha_artifact_removal_view: str = "both",
    dual_power_analysis: bool = False,
) -> BandPowerResult:
    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件: {path.resolve()}")

    raw_full = read_eeg_signal(path)
    quality_full = build_threshold_rejection(
        raw_full.astype(np.float64),
        sample_rate,
    )

    if dual_power_analysis:
        raw_cleaned, _, removed_points = clean_raw_signal(
            raw_full.astype(np.int64),
            quality_full,
        )
        quality_cleaned = build_threshold_rejection(
            raw_cleaned.astype(np.float64),
            sample_rate,
        )
        print(f"文件: {path.resolve()}")
        print(
            f"双份功率分析: 删减前 {raw_full.size} 点，删减后 {raw_cleaned.size} 点 "
            f"(剔除 {removed_points} 点)"
        )
        common_kwargs = dict(
            path=path,
            sample_rate=sample_rate,
            enable_alpha_suspicious=enable_alpha_suspicious,
            compare_segments=compare_segments,
            show_plot=show_plot,
            save_plot=save_plot,
            show_waveform=show_waveform,
            save_waveform=save_waveform,
            waveform_seconds=waveform_seconds,
            waveform_time_range=waveform_time_range,
            show_fft=show_fft,
            save_fft=save_fft,
            fft_time_range=fft_time_range,
            show_segment_compare=show_segment_compare,
            save_segment_compare=save_segment_compare,
            remove_alpha_artifact_segments=remove_alpha_artifact_segments,
            alpha_artifact_removal_view=alpha_artifact_removal_view,
        )
        _run_power_analysis_pass(
            raw=raw_full,
            pass_label="删减前",
            output_tag="before_removal",
            quality=quality_full,
            compare_segments_cleaned=False,
            reference_raw=None,
            reference_quality=None,
            save_model_window_table=save_model_window_table,
            save_offline_waveform_data=save_offline_waveform_data,
            **common_kwargs,
        )
        return _run_power_analysis_pass(
            raw=raw_cleaned.astype(np.float64),
            pass_label="删减后",
            output_tag="after_removal",
            quality=quality_cleaned,
            compare_segments_cleaned=True,
            reference_raw=raw_full,
            reference_quality=quality_full,
            save_model_window_table=False,
            save_offline_waveform_data=False,
            **common_kwargs,
        )

    quality = read_eeg_quality(
        path,
        expected_len=len(raw_full),
        raw_values=raw_full,
        sample_rate=sample_rate,
    )
    return _run_power_analysis_pass(
        raw=raw_full,
        path=path,
        sample_rate=sample_rate,
        pass_label=path.stem,
        output_tag="",
        quality=quality,
        enable_alpha_suspicious=enable_alpha_suspicious,
        compare_segments=compare_segments,
        compare_segments_cleaned=False,
        reference_raw=None,
        reference_quality=None,
        show_plot=show_plot,
        save_plot=save_plot,
        show_waveform=show_waveform,
        save_waveform=save_waveform,
        waveform_seconds=waveform_seconds,
        waveform_time_range=waveform_time_range,
        show_fft=show_fft,
        save_fft=save_fft,
        fft_time_range=fft_time_range,
        show_segment_compare=show_segment_compare,
        save_segment_compare=save_segment_compare,
        save_model_window_table=save_model_window_table,
        save_offline_waveform_data=save_offline_waveform_data,
        remove_alpha_artifact_segments=remove_alpha_artifact_segments,
        alpha_artifact_removal_view=alpha_artifact_removal_view,
    )


def main() -> None:
    # ========== 在此修改输入文件 ==========
    xlsx_file = r"Result/20260710_195055_前60秒双耳节拍200和210/eeg_raw_full.csv"
    sample_rate = DEFAULT_SAMPLE_RATE  # EEG 采样率 (Hz)，一般为 500
    show_plot = True        # 是否弹出功率图窗口
    save_plot = True       # 是否保存功率图 (*_band_power.png)
    show_waveform = True    # 是否弹出各节律波形图窗口
    save_waveform = True   # 是否保存波形图 (*_band_waveform.png)
    waveform_seconds = 0  # 从 0 秒起显示 N 秒；None 则显示整段（time_range 未设置时）
    waveform_time_range = (0, 360)  # 指定时间段 (起始秒, 结束秒)，如 (10, 20)
    show_fft = True
    save_fft = True
    fft_time_range = None  # None=整段 FFT；(10, 20) 仅对 10–20 s 画 FFT
    # 多段对比：2～4 段均可；比值=各段/第1段；None 关闭
    compare_segments = ((10, 50), (70, 110))  # 例如再加 (120,160), (180,220)
    show_segment_compare = True
    save_segment_compare = True
    enable_alpha_suspicious = True
    save_model_window_table = True
    save_offline_waveform_data = True
    # Checkbox-ready switch: True saves gap-view and compressed-view removal plots.
    remove_alpha_artifact_segments = True
    alpha_artifact_removal_view = "both"  # "gap" | "compressed" | "both"
    # =====================================

    run_analysis(
        xlsx_file,
        sample_rate,
        show_plot=show_plot,
        save_plot=save_plot,
        show_waveform=show_waveform,
        save_waveform=save_waveform,
        waveform_seconds=waveform_seconds,
        waveform_time_range=waveform_time_range,
        show_fft=show_fft,
        save_fft=save_fft,
        fft_time_range=fft_time_range,
        compare_segments=compare_segments,
        show_segment_compare=show_segment_compare,
        save_segment_compare=save_segment_compare,
        enable_alpha_suspicious=enable_alpha_suspicious,
        save_model_window_table=save_model_window_table,
        save_offline_waveform_data=save_offline_waveform_data,
        remove_alpha_artifact_segments=remove_alpha_artifact_segments,
        alpha_artifact_removal_view=alpha_artifact_removal_view,
    )


if __name__ == "__main__":
    main()
