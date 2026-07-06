"""从 xlsx 首列读取 EEG 原始信号，带通滤波后计算各节律绝对/相对功率。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, welch
from scipy.fft import rfft, rfftfreq

FILTER_ORDER = 4
KS1082_SAMPLE_RATE = 500.0
DEFAULT_SAMPLE_RATE = float(
    os.environ.get("KS1082_SAMPLE_RATE", str(KS1082_SAMPLE_RATE))
)

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0

# ===== EEG 阈值拒绝/可疑段参数 =====
# 检测按固定时长切片，目前每 1 秒判断一次；调小会更精细，但也更容易被瞬时噪声影响。
EEG_REJECT_SEGMENT_SEC = 1

# ADC 贴边饱和坏段阈值：原始值 <= 50 或 >= 4045 时标红。
# 适合 12 bit ADC (0-4095)。如果硬件有效范围变窄/变宽，再同步调整。
EEG_RAW_MIN_VALID = 50
EEG_RAW_MAX_VALID = 4045

# 原始 1 秒片段峰峰值坏段阈值：max(segment)-min(segment) 超过该值标红。
# 主要抓大幅运动伪迹、严重电极扰动、接触不良；调小会更严格。
EEG_SEGMENT_MAX_PTP = 1200.0

# 原始 1 秒片段偏离中位数坏段阈值：单点离该秒中位数太远时标红。
# 主要抓特别大的尖峰/跳变；调小会更严格。
EEG_SEGMENT_MAX_DEVIATION = 800.0

# 自适应可疑段倍率：阈值 = 全部 1 秒片段的 median + N*MAD。
# 同时作用于 EEG_SUSPICIOUS_MIN_PTP 和 EEG_SUSPICIOUS_MIN_DIFF 对应规则。
# 调小更敏感、黄段更多；调大更保守、黄段更少。
EEG_ADAPTIVE_MAD_MULT = 5.0

# 原始 1 秒峰峰值可疑段最低阈值。
# 实际阈值取 max(该值, median + EEG_ADAPTIVE_MAD_MULT*MAD)。
# 主要抓比全局背景明显更“抖”的片段；调小会更容易标黄。
EEG_SUSPICIOUS_MIN_PTP = 200.0

# 原始 1 秒内最大相邻跳变可疑段最低阈值。
# 实际阈值取 max(该值, median + EEG_ADAPTIVE_MAD_MULT*MAD)。
# 主要抓短促尖峰、突然跳点；调小会更容易标黄。
EEG_SUSPICIOUS_MIN_DIFF = 60.0

# δ(0.5-4 Hz)滤波后 RMS 可疑段 MAD 倍率。
# 阈值候选 = delta_rms_median + N*delta_rms_MAD。
# 主要抓低频慢漂/电极扰动/运动造成的 δ 行突刺；只标黄，不标红。
# 调小更容易抓低频异常；调大更保守。
EEG_SUSPICIOUS_DELTA_RMS_MAD_MULT = 6.0

# δ RMS 相对背景的最低倍数门槛。
# 最终 δ RMS 阈值 = max(median*该倍数, median + MAD_MULT*MAD)。
# 防止 MAD 很小时阈值过低；调小更敏感，调大更保守。
EEG_SUSPICIOUS_DELTA_RMS_RATIO = 2.0

# Alpha(8-13 Hz) RMS adaptive suspicious segment detection.
# This is intentionally a yellow/tag-only rule: strong alpha may be a real
# relaxed/eyes-closed state, so it should not directly reject data.
EEG_SUSPICIOUS_ALPHA_RMS_MAD_MULT = 6.0
EEG_SUSPICIOUS_ALPHA_RMS_RATIO = 2.0
EEG_SUSPICIOUS_ALPHA_RMS_FLOOR = 1.0

# Window-level model quality policy. A model can use this table to keep,
# downweight, or skip windows without pretending repaired EEG is real EEG.
MODEL_WINDOW_SEC = 5.0
MODEL_ALPHA_SUSPICIOUS_WARN_RATIO = 0.10
MODEL_ALPHA_SUSPICIOUS_DROP_RATIO = 0.30
MODEL_REJECT_DROP_RATIO = 0.20
MODEL_SUSPICIOUS_WARN_RATIO = 0.20
MODEL_SUSPICIOUS_DROP_RATIO = 0.50

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


@dataclass
class EegQualityInfo:
    reject_mask: np.ndarray
    reject_rate: float
    has_tag: bool = False
    source: str = "none"
    suspicious_mask: np.ndarray | None = None
    suspicious_rate: float = 0.0
    reject_reasons: list[str] | None = None
    suspicious_reasons: list[str] | None = None

    @property
    def has_rejection(self) -> bool:
        return bool(self.has_tag and self.reject_mask.size and np.any(self.reject_mask))

    @property
    def has_suspicious(self) -> bool:
        return bool(
            self.has_tag
            and self.suspicious_mask is not None
            and self.suspicious_mask.size
            and np.any(self.suspicious_mask)
        )


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
    """两段 EEG 各节律功率对比；ratio = 段2 / 段1。"""

    range_a: Tuple[float, float]
    range_b: Tuple[float, float]
    analysis_a: PowerAnalysis
    analysis_b: PowerAnalysis
    absolute_ratio: Dict[str, float]
    relative_ratio: Dict[str, float]


@dataclass(frozen=True)
class BandSuspiciousInfo:
    band_name: str
    mask: np.ndarray
    rate: float
    rms_threshold: float
    rms_median: float
    rms_mad: float
    segment_sec: float
    reasons: list[str]


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


def build_threshold_rejection(
    values: np.ndarray,
    sample_rate: float,
) -> EegQualityInfo:
    """按 1 s 片段打质量标签：严重坏段 rejected，可疑段 suspicious。"""
    count = values.size
    if count == 0:
        empty = np.zeros(0, dtype=bool)
        return EegQualityInfo(empty, 0.0, has_tag=False, suspicious_mask=empty)

    reject_mask = np.zeros(count, dtype=bool)
    suspicious_mask = np.zeros(count, dtype=bool)
    reject_reasons = [""] * count
    suspicious_reasons = [""] * count
    segment_size = max(1, int(round(sample_rate * EEG_REJECT_SEGMENT_SEC)))
    segments: list[tuple[int, int, np.ndarray]] = []
    ptp_values: list[float] = []
    diff_values: list[float] = []
    delta_rms_values: list[float] = []

    for start in range(0, count, segment_size):
        end = min(count, start + segment_size)
        segment = values[start:end].astype(np.float64, copy=False)
        segments.append((start, end, segment))
        ptp_values.append(float(np.ptp(segment)))
        diffs = np.diff(segment)
        diff_values.append(float(np.max(np.abs(diffs))) if diffs.size else 0.0)

    try:
        delta_signal = bandpass_filter(
            values.astype(np.float64, copy=False),
            sample_rate,
            0.5,
            4.0,
        )
    except ValueError:
        delta_signal = None

    for start, end, _segment in segments:
        if delta_signal is None:
            delta_rms_values.append(0.0)
            continue
        delta_segment = delta_signal[start:end]
        delta_rms_values.append(float(np.sqrt(np.mean(delta_segment * delta_segment))))

    def _adaptive_threshold(metrics: list[float], floor: float) -> float:
        if not metrics:
            return floor
        arr = np.asarray(metrics, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        return max(floor, median + EEG_ADAPTIVE_MAD_MULT * mad)

    ptp_suspicious_threshold = _adaptive_threshold(
        ptp_values,
        EEG_SUSPICIOUS_MIN_PTP,
    )
    diff_suspicious_threshold = _adaptive_threshold(
        diff_values,
        EEG_SUSPICIOUS_MIN_DIFF,
    )
    if delta_rms_values:
        delta_rms_arr = np.asarray(delta_rms_values, dtype=np.float64)
        delta_rms_median = float(np.median(delta_rms_arr))
        delta_rms_mad = float(np.median(np.abs(delta_rms_arr - delta_rms_median)))
        delta_rms_suspicious_threshold = max(
            delta_rms_median * EEG_SUSPICIOUS_DELTA_RMS_RATIO,
            delta_rms_median + EEG_SUSPICIOUS_DELTA_RMS_MAD_MULT * delta_rms_mad,
        )
    else:
        delta_rms_suspicious_threshold = float("inf")

    for idx, (start, end, segment) in enumerate(segments):
        reject_reason_parts: list[str] = []
        suspicious_reason_parts: list[str] = []
        if np.any(segment <= EEG_RAW_MIN_VALID) or np.any(segment >= EEG_RAW_MAX_VALID):
            reject_reason_parts.append("rail_saturation")
        ptp = ptp_values[idx]
        if ptp > EEG_SEGMENT_MAX_PTP:
            reject_reason_parts.append(f"ptp>{EEG_SEGMENT_MAX_PTP:g}")
        median = float(np.median(segment))
        max_dev = float(np.max(np.abs(segment - median)))
        if max_dev > EEG_SEGMENT_MAX_DEVIATION:
            reject_reason_parts.append(f"deviation>{EEG_SEGMENT_MAX_DEVIATION:g}")

        max_diff = diff_values[idx]
        if ptp > ptp_suspicious_threshold:
            suspicious_reason_parts.append(
                f"adaptive_ptp>{ptp_suspicious_threshold:g}"
            )
        if max_diff > diff_suspicious_threshold:
            suspicious_reason_parts.append(
                f"adaptive_diff>{diff_suspicious_threshold:g}"
            )
        delta_rms = delta_rms_values[idx]
        if delta_rms > delta_rms_suspicious_threshold:
            suspicious_reason_parts.append(
                f"delta_rms>{delta_rms_suspicious_threshold:g}"
            )

        if reject_reason_parts:
            reject_mask[start:end] = True
            reason = ";".join(reject_reason_parts)
            for point_idx in range(start, end):
                reject_reasons[point_idx] = reason
        elif suspicious_reason_parts:
            suspicious_mask[start:end] = True
            reason = ";".join(suspicious_reason_parts)
            for point_idx in range(start, end):
                suspicious_reasons[point_idx] = reason

    reject_rate = float(np.mean(reject_mask)) if reject_mask.size else 0.0
    suspicious_rate = (
        float(np.mean(suspicious_mask)) if suspicious_mask.size else 0.0
    )
    return EegQualityInfo(
        reject_mask,
        reject_rate,
        has_tag=True,
        source="computed",
        suspicious_mask=suspicious_mask,
        suspicious_rate=suspicious_rate,
        reject_reasons=reject_reasons,
        suspicious_reasons=suspicious_reasons,
    )


def _mad_threshold(
    metrics: np.ndarray,
    *,
    ratio: float,
    mad_mult: float,
    floor: float = 0.0,
) -> tuple[float, float, float]:
    if metrics.size == 0:
        return floor, 0.0, 0.0
    median = float(np.median(metrics))
    mad = float(np.median(np.abs(metrics - median)))
    threshold = max(floor, median * ratio, median + mad_mult * mad)
    return threshold, median, mad


def build_band_rms_suspicious(
    values: np.ndarray,
    sample_rate: float,
    *,
    band_name: str = "alpha",
    mad_mult: float = EEG_SUSPICIOUS_ALPHA_RMS_MAD_MULT,
    ratio: float = EEG_SUSPICIOUS_ALPHA_RMS_RATIO,
    floor: float = EEG_SUSPICIOUS_ALPHA_RMS_FLOOR,
) -> BandSuspiciousInfo:
    """Mark high-RMS band segments as suspicious without rejecting them."""
    count = values.size
    empty = np.zeros(count, dtype=bool)
    empty_reasons = [""] * count
    if count == 0:
        return BandSuspiciousInfo(
            band_name,
            empty,
            0.0,
            floor,
            0.0,
            0.0,
            EEG_REJECT_SEGMENT_SEC,
            empty_reasons,
        )
    if band_name not in EEG_BANDS:
        raise ValueError(f"未知节律: {band_name!r}")

    low, high = EEG_BANDS[band_name]
    try:
        band_signal = bandpass_filter(
            values.astype(np.float64, copy=False),
            sample_rate,
            low,
            high,
        )
    except ValueError:
        return BandSuspiciousInfo(
            band_name,
            empty,
            0.0,
            float("inf"),
            0.0,
            0.0,
            EEG_REJECT_SEGMENT_SEC,
            empty_reasons,
        )

    segment_size = max(1, int(round(sample_rate * EEG_REJECT_SEGMENT_SEC)))
    segments: list[tuple[int, int]] = []
    rms_values: list[float] = []
    for start in range(0, count, segment_size):
        end = min(count, start + segment_size)
        segment = band_signal[start:end]
        segments.append((start, end))
        rms_values.append(float(np.sqrt(np.mean(segment * segment))))

    rms_arr = np.asarray(rms_values, dtype=np.float64)
    threshold, median, mad = _mad_threshold(
        rms_arr,
        ratio=ratio,
        mad_mult=mad_mult,
        floor=floor,
    )
    mask = np.zeros(count, dtype=bool)
    reasons = [""] * count
    for idx, (start, end) in enumerate(segments):
        rms = rms_values[idx]
        if rms > threshold:
            mask[start:end] = True
            reason = f"{band_name}_rms>{threshold:g}"
            for point_idx in range(start, end):
                reasons[point_idx] = reason

    rate = float(np.mean(mask)) if mask.size else 0.0
    return BandSuspiciousInfo(
        band_name,
        mask,
        rate,
        threshold,
        median,
        mad,
        EEG_REJECT_SEGMENT_SEC,
        reasons,
    )


def merge_quality_with_band_suspicious(
    quality: EegQualityInfo,
    band_info: BandSuspiciousInfo,
) -> EegQualityInfo:
    """Add band-level suspicious tags to the general yellow mask."""
    count = quality.reject_mask.size
    band_mask = band_info.mask
    if band_mask.size < count:
        band_mask = np.pad(band_mask, (0, count - band_mask.size), constant_values=False)
    elif band_mask.size > count:
        band_mask = band_mask[:count]

    base_suspicious = (
        quality.suspicious_mask.copy()
        if quality.suspicious_mask is not None
        else np.zeros(count, dtype=bool)
    )
    band_mask = band_mask & ~quality.reject_mask
    suspicious_mask = base_suspicious | band_mask
    suspicious_rate = float(np.mean(suspicious_mask)) if suspicious_mask.size else 0.0

    suspicious_reasons = (
        list(quality.suspicious_reasons)
        if quality.suspicious_reasons is not None
        else [""] * count
    )
    if len(suspicious_reasons) < count:
        suspicious_reasons.extend([""] * (count - len(suspicious_reasons)))
    elif len(suspicious_reasons) > count:
        suspicious_reasons = suspicious_reasons[:count]

    for idx, is_band_suspicious in enumerate(band_mask):
        if not is_band_suspicious:
            continue
        reason = band_info.reasons[idx] if idx < len(band_info.reasons) else ""
        if not reason:
            reason = f"{band_info.band_name}_rms"
        if suspicious_reasons[idx]:
            suspicious_reasons[idx] = f"{suspicious_reasons[idx]};{reason}"
        else:
            suspicious_reasons[idx] = reason

    return EegQualityInfo(
        quality.reject_mask,
        quality.reject_rate,
        has_tag=quality.has_tag or band_info.mask.size > 0,
        source=quality.source,
        suspicious_mask=suspicious_mask,
        suspicious_rate=suspicious_rate,
        reject_reasons=quality.reject_reasons,
        suspicious_reasons=suspicious_reasons,
    )


def read_eeg_quality(
    data_path: str | Path,
    expected_len: int,
    raw_values: np.ndarray | None = None,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> EegQualityInfo:
    """读取 CSV 中的阈值拒绝 tag；旧文件没有 tag 时返回 0 拒绝率。"""
    path = Path(data_path)
    empty_mask = np.zeros(expected_len, dtype=bool)
    empty = EegQualityInfo(
        empty_mask,
        0.0,
        has_tag=False,
        suspicious_mask=empty_mask.copy(),
    )
    if path.suffix.lower() not in {".csv", ".txt"}:
        return empty

    frame = pd.read_csv(path)
    if "is_rejected" not in frame.columns:
        if raw_values is not None:
            return build_threshold_rejection(raw_values, sample_rate)
        return empty

    computed = (
        build_threshold_rejection(raw_values, sample_rate)
        if raw_values is not None
        else None
    )
    if computed is not None:
        return computed

    def _fit_mask(mask: np.ndarray) -> np.ndarray:
        if mask.size < expected_len:
            return np.pad(mask, (0, expected_len - mask.size), constant_values=False)
        if mask.size > expected_len:
            return mask[:expected_len]
        return mask

    flags = pd.to_numeric(frame["is_rejected"], errors="coerce").fillna(0)
    mask = _fit_mask(flags.to_numpy(dtype=np.float64) > 0)

    reject_rate = float(np.mean(mask)) if mask.size else 0.0
    if "reject_rate" in frame.columns:
        rates = pd.to_numeric(frame["reject_rate"], errors="coerce").dropna()
        if not rates.empty:
            reject_rate = float(rates.iloc[0])

    if "is_suspicious" in frame.columns:
        suspicious_flags = pd.to_numeric(
            frame["is_suspicious"], errors="coerce"
        ).fillna(0)
        suspicious_mask = _fit_mask(
            suspicious_flags.to_numpy(dtype=np.float64) > 0
        )
        suspicious_mask = suspicious_mask & ~mask
        suspicious_rate = (
            float(np.mean(suspicious_mask)) if suspicious_mask.size else 0.0
        )
        if "suspicious_rate" in frame.columns:
            rates = pd.to_numeric(
                frame["suspicious_rate"], errors="coerce"
            ).dropna()
            if not rates.empty:
                suspicious_rate = float(rates.iloc[0])
        suspicious_reasons = (
            frame["suspicious_reason"].fillna("").astype(str).to_list()
            if "suspicious_reason" in frame.columns
            else None
        )
    elif computed is not None and computed.suspicious_mask is not None:
        suspicious_mask = computed.suspicious_mask & ~mask
        suspicious_rate = (
            float(np.mean(suspicious_mask)) if suspicious_mask.size else 0.0
        )
        suspicious_reasons = computed.suspicious_reasons
    else:
        suspicious_mask = np.zeros(expected_len, dtype=bool)
        suspicious_rate = 0.0
        suspicious_reasons = None

    reject_reasons = (
        frame["reject_reason"].fillna("").astype(str).to_list()
        if "reject_reason" in frame.columns
        else None
    )
    return EegQualityInfo(
        mask,
        reject_rate,
        has_tag=True,
        source="csv",
        suspicious_mask=suspicious_mask,
        suspicious_rate=suspicious_rate,
        reject_reasons=reject_reasons,
        suspicious_reasons=suspicious_reasons,
    )


def _rejected_spans(
    reject_mask: np.ndarray,
    sample_rate: float,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """把逐点 reject mask 合并成时间段，供波形图背景标注。"""
    if reject_mask.size == 0 or not np.any(reject_mask):
        return []

    total_duration = reject_mask.size / sample_rate
    view_start = max(0.0, start_seconds)
    view_end = total_duration if end_seconds is None else min(end_seconds, total_duration)
    if view_start >= view_end:
        return []

    i_start = max(0, int(view_start * sample_rate))
    i_end = min(reject_mask.size, int(np.ceil(view_end * sample_rate)))
    local = reject_mask[i_start:i_end]
    spans: list[tuple[float, float]] = []
    in_span = False
    span_start = 0
    for offset, rejected in enumerate(local):
        if rejected and not in_span:
            in_span = True
            span_start = offset
        elif not rejected and in_span:
            abs_start = i_start + span_start
            abs_end = i_start + offset
            spans.append((abs_start / sample_rate, abs_end / sample_rate))
            in_span = False
    if in_span:
        abs_start = i_start + span_start
        abs_end = i_start + local.size
        spans.append((abs_start / sample_rate, abs_end / sample_rate))
    return spans


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


def compare_segment_band_powers(
    signal: np.ndarray,
    sample_rate: float,
    range_a: Tuple[float, float],
    range_b: Tuple[float, float],
) -> SegmentBandComparison:
    """比较两段相同节律的绝对/相对功率，比值 = 段2 / 段1。"""
    seg_a = _slice_signal(signal, sample_rate, range_a[0], range_a[1])
    seg_b = _slice_signal(signal, sample_rate, range_b[0], range_b[1])
    analysis_a = compute_band_powers(seg_a, sample_rate=sample_rate)
    analysis_b = compute_band_powers(seg_b, sample_rate=sample_rate)

    abs_ratio: Dict[str, float] = {}
    rel_ratio: Dict[str, float] = {}
    for name in EEG_BANDS:
        a_abs = analysis_a.result.absolute[name]
        b_abs = analysis_b.result.absolute[name]
        a_rel = analysis_a.result.relative[name]
        b_rel = analysis_b.result.relative[name]
        abs_ratio[name] = b_abs / a_abs if a_abs > 0 else float("nan")
        rel_ratio[name] = b_rel / a_rel if a_rel > 0 else float("nan")

    return SegmentBandComparison(
        range_a=range_a,
        range_b=range_b,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        absolute_ratio=abs_ratio,
        relative_ratio=rel_ratio,
    )


def format_segment_comparison(comp: SegmentBandComparison) -> str:
    ra, rb = comp.range_a, comp.range_b
    lines = [
        f"时间段 A: {ra[0]:g}–{ra[1]:g} s  |  时间段 B: {rb[0]:g}–{rb[1]:g} s",
        f"功率比 = B / A（段2 / 段1）",
        f"{'节律':<8} {'A绝对':>12} {'B绝对':>12} {'绝对比':>10} "
        f"{'A相对%':>10} {'B相对%':>10} {'相对比':>10}",
        "-" * 78,
    ]
    for name in EEG_BANDS:
        label = BAND_LABELS[name]
        a_abs = comp.analysis_a.result.absolute[name]
        b_abs = comp.analysis_b.result.absolute[name]
        a_rel = comp.analysis_a.result.relative[name] * 100.0
        b_rel = comp.analysis_b.result.relative[name] * 100.0
        ar = comp.absolute_ratio[name]
        rr = comp.relative_ratio[name]
        lines.append(
            f"{label:<8} {a_abs:>12.4e} {b_abs:>12.4e} {ar:>10.3f} "
            f"{a_rel:>10.2f} {b_rel:>10.2f} {rr:>10.3f}"
        )
    return "\n".join(lines)


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


def _fit_bool_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size < target_len:
        return np.pad(mask, (0, target_len - mask.size), constant_values=False)
    if mask.size > target_len:
        return mask[:target_len]
    return mask


def build_model_window_quality_table(
    quality: EegQualityInfo,
    alpha_info: BandSuspiciousInfo | None,
    sample_rate: float,
    *,
    window_sec: float = MODEL_WINDOW_SEC,
) -> pd.DataFrame:
    """Summarize quality tags per model input window."""
    count = quality.reject_mask.size
    if count == 0:
        return pd.DataFrame()

    window_size = max(1, int(round(sample_rate * window_sec)))
    reject_mask = _fit_bool_mask(quality.reject_mask, count)
    suspicious_mask = (
        _fit_bool_mask(quality.suspicious_mask, count)
        if quality.suspicious_mask is not None
        else np.zeros(count, dtype=bool)
    )
    alpha_mask = (
        _fit_bool_mask(alpha_info.mask, count)
        if alpha_info is not None
        else np.zeros(count, dtype=bool)
    )

    rows: list[dict[str, object]] = []
    for window_id, start in enumerate(range(0, count, window_size)):
        end = min(count, start + window_size)
        n = end - start
        reject_ratio = float(np.mean(reject_mask[start:end]))
        suspicious_ratio = float(np.mean(suspicious_mask[start:end]))
        alpha_ratio = float(np.mean(alpha_mask[start:end]))

        action = "keep"
        confidence_weight = 1.0
        if (
            reject_ratio >= MODEL_REJECT_DROP_RATIO
            or suspicious_ratio >= MODEL_SUSPICIOUS_DROP_RATIO
            or alpha_ratio >= MODEL_ALPHA_SUSPICIOUS_DROP_RATIO
        ):
            action = "drop"
            confidence_weight = 0.0
        elif (
            reject_ratio > 0.0
            or suspicious_ratio >= MODEL_SUSPICIOUS_WARN_RATIO
            or alpha_ratio >= MODEL_ALPHA_SUSPICIOUS_WARN_RATIO
        ):
            action = "low_confidence"
            confidence_weight = 0.5
        elif alpha_ratio > 0.0 or suspicious_ratio > 0.0:
            action = "downweight"
            confidence_weight = 0.75

        rows.append(
            {
                "window_id": window_id,
                "start_s": start / sample_rate,
                "end_s": end / sample_rate,
                "n_samples": n,
                "reject_ratio": reject_ratio,
                "suspicious_ratio": suspicious_ratio,
                "alpha_suspicious_ratio": alpha_ratio,
                "action": action,
                "confidence_weight": confidence_weight,
            }
        )

    return pd.DataFrame(rows)


def export_offline_waveform_csvs(
    data_path: str | Path,
    raw: np.ndarray,
    waveforms: Dict[str, np.ndarray],
    quality: EegQualityInfo,
    alpha_info: BandSuspiciousInfo | None,
    sample_rate: float,
) -> tuple[Path, Path, int]:
    """Save full and cleaned offline waveform tables for later model input."""
    path = Path(data_path)
    count = len(raw)
    index = np.arange(count, dtype=np.int64)
    reject_mask = _fit_bool_mask(quality.reject_mask, count)
    suspicious_mask = (
        _fit_bool_mask(quality.suspicious_mask, count)
        if quality.suspicious_mask is not None
        else np.zeros(count, dtype=bool)
    )
    alpha_mask = (
        _fit_bool_mask(alpha_info.mask, count)
        if alpha_info is not None
        else np.zeros(count, dtype=bool)
    )
    remove_mask = reject_mask | suspicious_mask

    full_data: dict[str, object] = {
        "index": index,
        "time_s": index / sample_rate,
        "ch1_raw": raw[:count],
    }
    for name in EEG_BANDS:
        full_data[f"{name}_waveform"] = waveforms[name][:count]
    full_data.update(
        {
            "is_rejected": reject_mask.astype(np.int8),
            "is_suspicious": suspicious_mask.astype(np.int8),
            "is_alpha_suspicious": alpha_mask.astype(np.int8),
            "is_removed_in_cleaned": remove_mask.astype(np.int8),
        }
    )

    full_path = path.parent / f"{path.stem}_offline_full_waveforms.csv"
    pd.DataFrame(full_data).to_csv(full_path, index=False, encoding="utf-8-sig")

    keep_mask = ~remove_mask
    kept_index = np.flatnonzero(keep_mask)
    clean_data: dict[str, object] = {
        "clean_index": np.arange(kept_index.size, dtype=np.int64),
        "clean_time_s": np.arange(kept_index.size, dtype=np.float64) / sample_rate,
        "original_index": kept_index,
        "original_time_s": kept_index / sample_rate,
        "ch1_raw": raw[kept_index],
    }
    for name in EEG_BANDS:
        clean_data[f"{name}_waveform"] = waveforms[name][kept_index]

    clean_path = path.parent / f"{path.stem}_offline_cleaned_waveforms.csv"
    pd.DataFrame(clean_data).to_csv(clean_path, index=False, encoding="utf-8-sig")
    return full_path, clean_path, int(np.count_nonzero(remove_mask))


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
    remove_mask = _fit_bool_mask(remove_mask, n_total)
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
    remove_mask = _fit_bool_mask(remove_mask, n_total)
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
    title: str = "两段 EEG 节律功率对比",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """绘制两段各节律相对/绝对功率及 B/A 比值柱状图。"""
    import matplotlib.pyplot as plt

    _setup_matplotlib()

    ra, rb = comparison.range_a, comparison.range_b
    names = list(EEG_BANDS.keys())
    labels = [
        f"{BAND_LABELS[n]}\n({EEG_BANDS[n][0]:g}-{EEG_BANDS[n][1]:g} Hz)" for n in names
    ]
    colors = [BAND_COLORS[n] for n in names]

    rel_a = [comparison.analysis_a.result.relative[n] * 100.0 for n in names]
    rel_b = [comparison.analysis_b.result.relative[n] * 100.0 for n in names]
    abs_a = [comparison.analysis_a.result.absolute[n] for n in names]
    abs_b = [comparison.analysis_b.result.absolute[n] for n in names]
    rel_ratios = [comparison.relative_ratio[n] for n in names]
    abs_ratios = [comparison.absolute_ratio[n] for n in names]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    fig.suptitle(
        f"{title}\nA: {ra[0]:g}–{ra[1]:g} s  vs  B: {rb[0]:g}–{rb[1]:g} s  |  比值 = B / A",
        fontsize=13,
        fontweight="bold",
    )

    x = np.arange(len(names))
    width = 0.36

    ax_rel = axes[0]
    ax_rel.bar(x - width / 2, rel_a, width, label=f"A ({ra[0]:g}–{ra[1]:g} s)", color=colors, alpha=0.55, edgecolor="white")
    ax_rel.bar(x + width / 2, rel_b, width, label=f"B ({rb[0]:g}–{rb[1]:g} s)", color=colors, edgecolor="white", linewidth=0.8)
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(labels)
    ax_rel.set_ylabel("相对功率 (%)")
    ax_rel.set_title("各节律相对功率")
    ax_rel.legend(loc="upper right")
    ax_rel.grid(True, axis="y", alpha=0.3)

    ax_abs = axes[1]
    ax_abs.bar(x - width / 2, abs_a, width, label=f"A", color=colors, alpha=0.55, edgecolor="white")
    ax_abs.bar(x + width / 2, abs_b, width, label=f"B", color=colors, edgecolor="white", linewidth=0.8)
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(labels)
    ax_abs.set_ylabel("绝对功率")
    ax_abs.set_title("各节律绝对功率")
    ax_abs.legend(loc="upper right")
    ax_abs.grid(True, axis="y", alpha=0.3)

    ax_ratio = axes[2]
    valid_abs = [r if np.isfinite(r) else 0.0 for r in abs_ratios]
    bars = ax_ratio.bar(labels, valid_abs, color=colors, edgecolor="white", linewidth=0.8)
    ax_ratio.axhline(1.0, color="#666666", ls="--", lw=1, label="比值 = 1")
    ax_ratio.set_ylabel("绝对功率比 (B / A)")
    ax_ratio.set_title("各节律绝对功率比")
    ax_ratio.legend(loc="upper right")
    ax_ratio.grid(True, axis="y", alpha=0.3)
    for bar, abs_r, rel_r in zip(bars, abs_ratios, rel_ratios):
        if np.isfinite(abs_r):
            ax_ratio.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{abs_r:.2f}\n(相对 {rel_r:.2f})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

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
        rejected_spans = _rejected_spans(
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
        suspicious_spans = _rejected_spans(
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
    compare_segments: tuple[tuple[float, float], tuple[float, float]] | None = None,
    show_segment_compare: bool = True,
    save_segment_compare: bool = False,
    enable_alpha_suspicious: bool = True,
    save_model_window_table: bool = True,
    save_offline_waveform_data: bool = True,
    remove_alpha_artifact_segments: bool = True,
    alpha_artifact_removal_view: str = "both",
) -> BandPowerResult:
    path = Path(xlsx_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件: {path.resolve()}")

    raw = read_eeg_signal(path)
    quality = read_eeg_quality(
        path,
        expected_len=len(raw),
        raw_values=raw,
        sample_rate=sample_rate,
    )
    alpha_info: BandSuspiciousInfo | None = None
    quality_for_plot = quality
    if enable_alpha_suspicious:
        alpha_info = build_band_rms_suspicious(
            raw,
            sample_rate,
            band_name="alpha",
        )
        quality_for_plot = merge_quality_with_band_suspicious(quality, alpha_info)
    analysis = compute_band_powers(raw, sample_rate=sample_rate)
    result = analysis.result
    duration = len(raw) / sample_rate

    print(f"文件: {path.resolve()}")
    print(f"样本数: {len(raw)}  |  采样率: {sample_rate:g} Hz  |  时长: {duration:.2f} s")
    if quality.has_tag:
        rejected = int(np.count_nonzero(quality.reject_mask))
        suspicious = (
            int(np.count_nonzero(quality.suspicious_mask))
            if quality.suspicious_mask is not None
            else 0
        )
        source_label = "CSV tag" if quality.source == "csv" else "现场计算"
        print(
            f"阈值拒绝: {quality.reject_rate:.1%} ({rejected}/{len(raw)} 点, {source_label})"
        )
        print(f"可疑片段: {quality.suspicious_rate:.1%} ({suspicious}/{len(raw)} 点)")
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
            window_path = path.parent / f"{path.stem}_model_windows.csv"
            window_table.to_csv(window_path, index=False, encoding="utf-8-sig")
            print(f"Model window quality table saved: {window_path.resolve()}")

    waveforms: Dict[str, np.ndarray] | None = None
    if save_offline_waveform_data or show_waveform or save_waveform:
        waveforms = extract_band_waveforms(raw, sample_rate)

    if save_offline_waveform_data and waveforms is not None:
        full_path, clean_path, removed_points = export_offline_waveform_csvs(
            path,
            raw,
            waveforms,
            quality_for_plot,
            alpha_info,
            sample_rate,
        )
        print(f"Offline full waveform CSV saved: {full_path.resolve()}")
        print(
            f"Offline cleaned waveform CSV saved: {clean_path.resolve()} "
            f"(removed {removed_points} samples)"
        )

    if show_plot or save_plot:
        plot_path = path.parent / f"{path.stem}_band_power.png" if save_plot else None
        plot_band_powers(
            analysis,
            title=f"EEG 节律功率 · {path.stem}",
            quality=quality_for_plot,
            save_path=plot_path,
            show=show_plot,
        )

    if show_waveform or save_waveform:
        if waveforms is None:
            waveforms = extract_band_waveforms(raw, sample_rate)
        wf_path = path.parent / f"{path.stem}_band_waveform.png" if save_waveform else None
        plot_band_waveforms(
            waveforms,
            sample_rate,
            title=f"EEG 节律波形 · {path.stem}",
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
                reject_mask=_fit_bool_mask(alpha_info.mask, len(raw)),
                reject_rate=alpha_info.rate,
                has_tag=True,
                source="alpha_suspicious_removed",
                suspicious_mask=np.zeros(len(raw), dtype=bool),
                suspicious_rate=0.0,
            )
            original_limits = waveform_y_limits(waveforms)
            gap_path = (
                path.parent / f"{path.stem}_alpha_removed_gap_band_waveform.png"
                if save_waveform and removal_view in {"gap", "both"}
                else None
            )
            plot_band_waveforms(
                gap_waveforms,
                sample_rate,
                title=(
                    f"EEG alpha removed + smoothed 路 {path.stem} "
                    f"(removed {removed_points} samples, original time axis)"
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
                path.parent / f"{path.stem}_alpha_removed_compressed_band_waveform.png"
                if save_waveform and removal_view in {"compressed", "both"}
                else None
            )
            plot_band_waveforms(
                compressed_waveforms,
                sample_rate,
                title=(
                    f"EEG alpha removed compressed view - {path.stem} "
                    f"(removed {removed_points} samples, compressed time axis)"
                ),
                time_range=None,
                max_seconds=None,
                quality=None,
                y_limits=original_limits,
                save_path=compressed_path,
                show=show_waveform and removal_view in {"compressed", "both"},
            )

    if show_fft or save_fft:
        fft_path = path.parent / f"{path.stem}_fft.png" if save_fft else None
        plot_fft(
            raw,
            sample_rate,
            time_range=fft_time_range,
            title=f"EEG FFT · {path.stem}",
            save_path=fft_path,
            show=show_fft,
        )

    if compare_segments is not None:
        range_a, range_b = compare_segments
        try:
            comparison = compare_segment_band_powers(raw, sample_rate, range_a, range_b)
        except ValueError as exc:
            print(f"跳过段间对比: {exc}")
        else:
            print()
            print(format_segment_comparison(comparison))
            if show_segment_compare or save_segment_compare:
                cmp_path = path.parent / f"{path.stem}_segment_compare.png" if save_segment_compare else None
                plot_segment_power_comparison(
                    comparison,
                    title=f"段间节律功率 · {path.stem}",
                    save_path=cmp_path,
                    show=show_segment_compare,
                )

    return result


def main() -> None:
    # ========== 在此修改输入文件 ==========
    xlsx_file = r"Sample/close60open60_1/close60open60_1_converted.csv"
    sample_rate = DEFAULT_SAMPLE_RATE  # EEG 采样率 (Hz)，一般为 500
    show_plot = True        # 是否弹出功率图窗口
    save_plot = True       # 是否保存功率图 (*_band_power.png)
    show_waveform = True    # 是否弹出各节律波形图窗口
    save_waveform = True   # 是否保存波形图 (*_band_waveform.png)
    waveform_seconds = 0  # 从 0 秒起显示 N 秒；None 则显示整段（time_range 未设置时）
    waveform_time_range = (0, 120)  # 指定时间段 (起始秒, 结束秒)，如 (10, 20)
    show_fft = True
    save_fft = True
    fft_time_range = None  # None=整段 FFT；(10, 20) 仅对 10–20 s 画 FFT
    # 两段对比：计算相同节律功率比 B/A，并出对比图
    compare_segments = ((20, 30), (40, 50))  # (段A起止秒), (段B起止秒)；None 关闭
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
