"""α 频率变异（AFV）与变化率。

按 Schwabedal et al. (JSR 2016) / Elemind 入睡试验用法：
在 α 阳性段内用相邻正峰间隔估瞬时频率 f_n，AFV = std(f_n)。
另给出滑动窗 AFV 时间序列与早/晚段相对变化率。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, iirnotch

from erp import ERP_NOTCH_HZ, ERP_NOTCH_Q

ALPHA_BANDPASS_HZ = (7.0, 13.0)
AFV_MIN_SEGMENT_SEC = 0.30
AFV_PAPER_WINDOW_SEC = 300.0  ## Elemind：开段前 5 min
AFV_SLIDE_WIN_SEC = 30.0
AFV_SLIDE_HOP_SEC = 5.0
AFV_MIN_INTERVALS = 20


@dataclass(frozen=True)
class AfvAnalysis:
    sample_rate: float
    duration_s: float
    alpha_band_hz: Tuple[float, float]
    amp_threshold: float
    n_intervals: int
    median_freq_hz: float
    afv_hz: float
    ## 开段前 5 min（论文窗口；记录不足 5 min 则用全长）
    paper_window_s: float
    paper_n_intervals: int
    paper_median_freq_hz: float
    paper_afv_hz: float
    ## 早半段 / 晚半段与相对变化率
    early_afv_hz: float
    late_afv_hz: float
    change_rate: float  ## (late-early)/early，无量纲
    ## 滑动窗
    slide_times_s: np.ndarray
    slide_afv_hz: np.ndarray
    slide_median_freq_hz: np.ndarray
    slide_n: np.ndarray
    ## 瞬时频率序列（全段阳性峰）
    inst_times_s: np.ndarray
    inst_freq_hz: np.ndarray
    note: str = ""


def _notch(eeg: np.ndarray, sample_rate: float) -> np.ndarray:
    y = np.asarray(eeg, dtype=np.float64)
    fs = float(sample_rate)
    if fs <= 0 or y.size < 24:
        return y
    nyq = 0.5 * fs
    for f0 in ERP_NOTCH_HZ:
        if f0 <= 0 or f0 >= nyq * 0.95:
            continue
        b, a = iirnotch(f0 / nyq, ERP_NOTCH_Q)
        y = filtfilt(b, a, y)
    return y


def _alpha_bandpass(
    eeg: np.ndarray,
    sample_rate: float,
    band_hz: Tuple[float, float] = ALPHA_BANDPASS_HZ,
) -> np.ndarray:
    fs = float(sample_rate)
    lo, hi = float(band_hz[0]), float(band_hz[1])
    nyq = 0.5 * fs
    if hi >= nyq * 0.95:
        hi = max(lo + 1.0, nyq * 0.45)
    if lo <= 0 or hi <= lo or eeg.size < 48:
        return np.asarray(eeg, dtype=np.float64)
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.asarray(eeg, dtype=np.float64))


def load_eeg_csv_for_afv(path: Path) -> Tuple[np.ndarray, float]:
    """读取 EEG CSV（需 ch1_raw / ch1）；不要求 pink_seq。"""
    path = Path(path)
    frame = pd.read_csv(path)
    columns_lower = {str(col).strip().lower(): col for col in frame.columns}
    eeg_col = None
    for name in ("ch1_raw", "ch1"):
        if name in columns_lower:
            eeg_col = columns_lower[name]
            break
    if eeg_col is None:
        raw_cols = [col for col in frame.columns if str(col).lower().endswith("_raw")]
        if raw_cols:
            eeg_col = raw_cols[0]
        else:
            skip = {"index", "time_s", "time", "timestamp", "pink_seq", "iaf_chz"}
            data_cols = [col for col in frame.columns if str(col).lower() not in skip]
            if not data_cols:
                raise ValueError(f"{path.name} 找不到 EEG 数据列")
            eeg_col = data_cols[0]

    eeg = pd.to_numeric(frame[eeg_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(eeg)
    eeg = eeg[mask]
    if eeg.size < 48:
        raise ValueError(f"有效样本过少 ({eeg.size}): {path.name}")

    sample_rate = 500.0
    if "time_s" in columns_lower:
        t = pd.to_numeric(frame[columns_lower["time_s"]], errors="coerce").to_numpy(
            dtype=np.float64
        )
        t = t[mask]
        if t.size >= 2:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                sample_rate = float(1.0 / np.median(dt))
    return eeg, sample_rate


def _instantaneous_freqs(
    alpha: np.ndarray,
    sample_rate: float,
    *,
    amp_threshold: Optional[float] = None,
    min_segment_sec: float = AFV_MIN_SEGMENT_SEC,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """返回 (peak_times_s, inst_freq_hz, used_amp_threshold)。"""
    fs = float(sample_rate)
    analytic = hilbert(np.asarray(alpha, dtype=np.float64))
    radius = np.abs(analytic)
    phase = np.angle(analytic)
    ## 峰-峰值 ≈ 2·radius；阈值用相对幅度（适配 ADC 原始单位）
    if amp_threshold is None or not np.isfinite(amp_threshold) or amp_threshold <= 0:
        med = float(np.median(radius))
        amp_threshold = max(med * 0.5, 1e-9)
    thr = float(amp_threshold)
    positive = radius >= thr
    min_len = max(3, int(round(min_segment_sec * fs)))

    ## 标记长度 ≥ min_len 的阳性连续段
    padded = np.concatenate([[False], positive, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    keep = np.zeros_like(positive, dtype=bool)
    for a, b in zip(starts, ends):
        if (b - a) >= min_len:
            keep[a:b] = True

    ## Hilbert 相位穿越 +π/2 + 2πk（正峰），与 Schwabedal 一致
    ph = np.unwrap(phase)
    level = np.floor((ph - (np.pi / 2.0)) / (2.0 * np.pi))
    cross = np.diff(level) > 0
    peak_idx = np.where(cross & keep[1:] & keep[:-1])[0] + 1
    if peak_idx.size < 2:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            thr,
        )

    t = peak_idx.astype(np.float64) / fs
    dt = np.diff(t)
    ## 只保留 α 合理周期（约 7–13 Hz → 0.077–0.143 s，放宽到 0.05–0.20）
    ok = (dt >= 0.050) & (dt <= 0.200)
    if not np.any(ok):
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            thr,
        )
    times = t[1:][ok]
    freqs = 1.0 / dt[ok]
    return times, freqs.astype(np.float64), thr


def _stats(freqs: np.ndarray) -> Tuple[int, float, float]:
    v = np.asarray(freqs, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0, float("nan"), float("nan")
    return int(v.size), float(np.median(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0


def analyze_afv(
    eeg: np.ndarray,
    sample_rate: float,
    *,
    band_hz: Tuple[float, float] = ALPHA_BANDPASS_HZ,
    amp_threshold: Optional[float] = None,
    paper_window_sec: float = AFV_PAPER_WINDOW_SEC,
    slide_win_sec: float = AFV_SLIDE_WIN_SEC,
    slide_hop_sec: float = AFV_SLIDE_HOP_SEC,
    min_intervals: int = AFV_MIN_INTERVALS,
) -> AfvAnalysis:
    fs = float(sample_rate)
    if fs <= 0:
        raise ValueError(f"无效采样率: {sample_rate}")
    y = _alpha_bandpass(_notch(eeg - np.median(eeg), fs), fs, band_hz)
    times, freqs, thr = _instantaneous_freqs(y, fs, amp_threshold=amp_threshold)
    n, med_f, afv = _stats(freqs)
    if n < min_intervals:
        raise ValueError(
            f"有效 α 峰间隔过少 ({n} < {min_intervals})。"
            "请确认记录足够长且存在明显 α 活动。"
        )

    dur = float(eeg.size / fs)
    ## 论文窗：开段前 5 min
    paper_s = min(float(paper_window_sec), dur)
    paper_mask = times <= paper_s
    pn, pmed, pafv = _stats(freqs[paper_mask])

    ## 早/晚半段变化率
    mid = dur * 0.5
    early = freqs[times <= mid]
    late = freqs[times > mid]
    en, _, eafv = _stats(early)
    ln, _, lafv = _stats(late)
    if en >= 8 and ln >= 8 and np.isfinite(eafv) and eafv > 1e-12:
        change_rate = float((lafv - eafv) / eafv)
    else:
        change_rate = float("nan")

    ## 滑动窗 AFV
    win = max(5.0, float(slide_win_sec))
    hop = max(1.0, float(slide_hop_sec))
    slide_t, slide_afv, slide_med, slide_n = [], [], [], []
    t0 = 0.0
    while t0 + win * 0.5 <= dur:
        t1 = t0 + win
        m = (times >= t0) & (times < t1)
        nn, mm, aa = _stats(freqs[m])
        if nn >= max(8, min_intervals // 4):
            slide_t.append(t0 + win * 0.5)
            slide_afv.append(aa)
            slide_med.append(mm)
            slide_n.append(nn)
        t0 += hop

    note = (
        f"方法：α 带通 {band_hz[0]:.0f}–{band_hz[1]:.0f} Hz → Hilbert；"
        f"包络≥{thr:.4g} 且段长≥{AFV_MIN_SEGMENT_SEC*1000:.0f} ms 为阳性；"
        f"相位穿越 +90° 得正峰；f=1/Δt；AFV=std(f)。"
        f"变化率=(晚半段AFV−早半段AFV)/早半段AFV。"
    )

    return AfvAnalysis(
        sample_rate=fs,
        duration_s=dur,
        alpha_band_hz=(float(band_hz[0]), float(band_hz[1])),
        amp_threshold=thr,
        n_intervals=n,
        median_freq_hz=med_f,
        afv_hz=afv,
        paper_window_s=paper_s,
        paper_n_intervals=pn,
        paper_median_freq_hz=pmed,
        paper_afv_hz=pafv,
        early_afv_hz=eafv,
        late_afv_hz=lafv,
        change_rate=change_rate,
        slide_times_s=np.asarray(slide_t, dtype=np.float64),
        slide_afv_hz=np.asarray(slide_afv, dtype=np.float64),
        slide_median_freq_hz=np.asarray(slide_med, dtype=np.float64),
        slide_n=np.asarray(slide_n, dtype=np.int64),
        inst_times_s=times,
        inst_freq_hz=freqs,
        note=note,
    )


def analyze_afv_csv(path: Path, **kwargs) -> AfvAnalysis:
    eeg, fs = load_eeg_csv_for_afv(path)
    return analyze_afv(eeg, fs, **kwargs)


def format_afv_report(result: AfvAnalysis) -> str:
    lo, hi = result.alpha_band_hz
    cr = result.change_rate
    cr_txt = f"{cr:+.1%}" if np.isfinite(cr) else "N/A"
    lines = [
        f"AFV: {result.duration_s:.1f} s @ {result.sample_rate:.1f} Hz，"
        f"α {lo:.0f}–{hi:.0f} Hz，间隔 n={result.n_intervals}",
        f"全段：中位频率 {result.median_freq_hz:.3f} Hz，AFV={result.afv_hz:.4f} Hz",
        f"开段前 {result.paper_window_s:.0f} s（论文窗）："
        f"AFV={result.paper_afv_hz:.4f} Hz（n={result.paper_n_intervals}）",
        f"早半段 AFV={result.early_afv_hz:.4f} Hz，"
        f"晚半段 AFV={result.late_afv_hz:.4f} Hz，"
        f"变化率={cr_txt}",
        result.note,
    ]
    return "\n".join(lines)


def export_afv_summary_csv(path: Path, result: AfvAnalysis) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["sample_rate_hz", f"{result.sample_rate:.6g}"])
        writer.writerow(["duration_s", f"{result.duration_s:.6g}"])
        writer.writerow(["amp_threshold", f"{result.amp_threshold:.6g}"])
        writer.writerow(["n_intervals", result.n_intervals])
        writer.writerow(["median_freq_hz", f"{result.median_freq_hz:.6g}"])
        writer.writerow(["afv_hz", f"{result.afv_hz:.6g}"])
        writer.writerow(["paper_window_s", f"{result.paper_window_s:.6g}"])
        writer.writerow(["paper_afv_hz", f"{result.paper_afv_hz:.6g}"])
        writer.writerow(["early_afv_hz", f"{result.early_afv_hz:.6g}"])
        writer.writerow(["late_afv_hz", f"{result.late_afv_hz:.6g}"])
        writer.writerow(["change_rate", f"{result.change_rate:.6g}"])
        writer.writerow(["note", result.note])
        writer.writerow([])
        writer.writerow(["slide_time_s", "slide_afv_hz", "slide_median_freq_hz", "slide_n"])
        for t, a, m, n in zip(
            result.slide_times_s,
            result.slide_afv_hz,
            result.slide_median_freq_hz,
            result.slide_n,
        ):
            writer.writerow(
                [f"{float(t):.6f}", f"{float(a):.6g}", f"{float(m):.6g}", int(n)]
            )
        writer.writerow([])
        writer.writerow(["inst_time_s", "inst_freq_hz"])
        for t, f in zip(result.inst_times_s, result.inst_freq_hz):
            writer.writerow([f"{float(t):.6f}", f"{float(f):.6g}"])
    return path
