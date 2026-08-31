"""听觉 ERP 离线分析：按 pink_seq 跳变切 epoch，计算峰潜伏期与起始潜伏期。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

ERP_BANDPASS_HZ = (1.0, 30.0)
## 工频及其二次谐波；窄带零相位陷波，几乎不碰 N1/P2（~1–20 Hz）能量
ERP_NOTCH_HZ = (50.0, 100.0)
ERP_NOTCH_Q = 30.0
ERP_EPOCH_PRE_SEC = 0.100
ERP_EPOCH_POST_SEC = 0.500
ERP_BASELINE_PRE_SEC = 0.100
## 听觉皮层长潜伏期成分常用窗（相对刺激 0 ms）
ERP_P1_WINDOW_MS = (40.0, 80.0)  ## P1/P50：正向峰
ERP_N1_WINDOW_MS = (70.0, 150.0)
ERP_P2_WINDOW_MS = (150.0, 280.0)
ERP_MAIN_PEAK_WINDOW_MS = (50.0, 300.0)
ERP_ONSET_MIN_MS = 20.0
ERP_ONSET_SD_MULT = 3.0
ERP_ONSET_HOLD_MS = 10.0
ERP_MIN_EVENTS = 5


@dataclass(frozen=True)
class ErpPeakInfo:
    name: str
    latency_ms: float
    amplitude: float


@dataclass(frozen=True)
class ErpAnalysis:
    sample_rate: float
    n_events: int
    n_epochs: int
    times_ms: np.ndarray
    average: np.ndarray
    epochs: np.ndarray  ## shape (n_epochs, n_times)，基线校正后的单次叠加段
    sem: np.ndarray  ## 各时间点标准误
    p1: Optional[ErpPeakInfo]
    n1: Optional[ErpPeakInfo]
    p2: Optional[ErpPeakInfo]
    main_peak: Optional[ErpPeakInfo]
    onset_latency_ms: Optional[float]
    onset_threshold: float
    note: str = ""


def load_eeg_pink_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """读取 CSV，返回 (eeg, pink_seq, sample_rate)。缺 pink_seq 列则报错。"""
    path = Path(path)
    frame = pd.read_csv(path)
    columns_lower = {str(col).strip().lower(): col for col in frame.columns}
    if "pink_seq" not in columns_lower:
        raise ValueError(f"{path.name} 缺少 pink_seq 列，无法做 ERP 分析（请用含粉噪序号的录制文件）")

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
            skip = {"index", "time_s", "time", "timestamp", "pink_seq"}
            data_cols = [col for col in frame.columns if str(col).lower() not in skip]
            if not data_cols:
                raise ValueError(f"{path.name} 找不到 EEG 数据列")
            eeg_col = data_cols[0]

    eeg = pd.to_numeric(frame[eeg_col], errors="coerce").to_numpy(dtype=np.float64)
    pink = pd.to_numeric(frame[columns_lower["pink_seq"]], errors="coerce").to_numpy(
        dtype=np.float64
    )
    mask = np.isfinite(eeg) & np.isfinite(pink)
    eeg = eeg[mask]
    pink = pink[mask].astype(np.int64)
    if eeg.size < 8:
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
    return eeg, pink, sample_rate


def detect_pink_seq_events(pink_seq: np.ndarray) -> np.ndarray:
    """相邻采样 pink_seq 变化且两侧均 >= 0 的样本索引（变化后的点）。"""
    seq = np.asarray(pink_seq, dtype=np.int64)
    if seq.size < 2:
        return np.zeros(0, dtype=np.int64)
    prev = seq[:-1]
    curr = seq[1:]
    changed = (curr != prev) & (prev >= 0) & (curr >= 0)
    return np.flatnonzero(changed) + 1


def _notch_line_noise(eeg: np.ndarray, sample_rate: float) -> np.ndarray:
    """50/100 Hz 窄带零相位陷波（工频滤波）。"""
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


def _bandpass(eeg: np.ndarray, sample_rate: float) -> np.ndarray:
    fs = float(sample_rate)
    lo, hi = ERP_BANDPASS_HZ
    nyq = 0.5 * fs
    if hi >= nyq * 0.95:
        hi = max(lo + 1.0, nyq * 0.45)
    if lo <= 0 or hi <= lo or eeg.size < 24:
        return np.asarray(eeg, dtype=np.float64)
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.asarray(eeg, dtype=np.float64))


def _preprocess_for_erp(eeg: np.ndarray, sample_rate: float) -> np.ndarray:
    """工频陷波 → 1–30 Hz 带通；全程零相位，不人为推迟潜伏期。"""
    return _bandpass(_notch_line_noise(eeg, sample_rate), sample_rate)


def _peak_in_window(
    times_ms: np.ndarray,
    average: np.ndarray,
    window_ms: Tuple[float, float],
    *,
    mode: str,
    name: str,
) -> Optional[ErpPeakInfo]:
    lo, hi = window_ms
    mask = (times_ms >= lo) & (times_ms <= hi)
    if not np.any(mask):
        return None
    seg_t = times_ms[mask]
    seg_y = average[mask]
    if mode == "min":
        idx = int(np.argmin(seg_y))
    elif mode == "max":
        idx = int(np.argmax(seg_y))
    else:
        idx = int(np.argmax(np.abs(seg_y)))
    return ErpPeakInfo(name=name, latency_ms=float(seg_t[idx]), amplitude=float(seg_y[idx]))


def _onset_latency_ms(
    times_ms: np.ndarray,
    average: np.ndarray,
    *,
    baseline_sd: float,
    sample_rate: float,
) -> Tuple[Optional[float], float]:
    threshold = float(ERP_ONSET_SD_MULT * max(baseline_sd, 1e-12))
    hold_n = max(1, int(round(ERP_ONSET_HOLD_MS * 1e-3 * sample_rate)))
    post = (times_ms >= ERP_ONSET_MIN_MS) & (times_ms <= ERP_EPOCH_POST_SEC * 1000.0)
    idxs = np.flatnonzero(post)
    if idxs.size < hold_n:
        return None, threshold
    abs_y = np.abs(average)
    for start in idxs:
        end = start + hold_n
        if end > average.size:
            break
        if np.all(abs_y[start:end] >= threshold):
            return float(times_ms[start]), threshold
    return None, threshold


def analyze_auditory_erp(
    eeg: np.ndarray,
    pink_seq: np.ndarray,
    sample_rate: float,
    *,
    min_events: int = ERP_MIN_EVENTS,
) -> ErpAnalysis:
    """对已对齐的 EEG 与 pink_seq 做听觉 ERP 分析。"""
    fs = float(sample_rate)
    if fs <= 0:
        raise ValueError(f"采样率无效: {sample_rate}")
    eeg = np.asarray(eeg, dtype=np.float64)
    pink = np.asarray(pink_seq, dtype=np.int64)
    if eeg.size != pink.size:
        raise ValueError(f"EEG 与 pink_seq 长度不一致: {eeg.size} vs {pink.size}")

    filtered = _preprocess_for_erp(eeg, fs)
    events = detect_pink_seq_events(pink)
    n_events = int(events.size)
    if n_events < min_events:
        raise ValueError(f"有效粉噪事件过少 ({n_events} < {min_events})，无法可靠估计 ERP")

    pre_n = int(round(ERP_EPOCH_PRE_SEC * fs))
    post_n = int(round(ERP_EPOCH_POST_SEC * fs))
    epoch_len = pre_n + post_n + 1
    times_ms = (np.arange(epoch_len, dtype=np.float64) - pre_n) * (1000.0 / fs)

    epochs: List[np.ndarray] = []
    for onset in events:
        start = int(onset) - pre_n
        end = int(onset) + post_n + 1
        if start < 0 or end > filtered.size:
            continue
        epoch = filtered[start:end].copy()
        baseline = epoch[: max(1, pre_n)]
        epoch -= float(np.mean(baseline))
        epochs.append(epoch)

    n_epochs = len(epochs)
    if n_epochs < min_events:
        raise ValueError(
            f"完整 epoch 过少 ({n_epochs} < {min_events}；事件 {n_events}，"
            f"多数靠近文件首尾被丢弃）"
        )

    stack = np.vstack(epochs)
    average = np.mean(stack, axis=0)
    # 标准误：用于阴影带（mean ± SEM）
    if stack.shape[0] > 1:
        sem = np.std(stack, axis=0, ddof=1) / np.sqrt(float(stack.shape[0]))
    else:
        sem = np.zeros_like(average)
    baseline_mask = times_ms < 0
    baseline_sd = float(np.std(average[baseline_mask])) if np.any(baseline_mask) else 0.0

    p1 = _peak_in_window(times_ms, average, ERP_P1_WINDOW_MS, mode="max", name="P1")
    n1 = _peak_in_window(times_ms, average, ERP_N1_WINDOW_MS, mode="min", name="N1")
    p2 = _peak_in_window(times_ms, average, ERP_P2_WINDOW_MS, mode="max", name="P2")
    main_peak = _peak_in_window(
        times_ms, average, ERP_MAIN_PEAK_WINDOW_MS, mode="abs", name="主峰"
    )
    onset_ms, threshold = _onset_latency_ms(
        times_ms, average, baseline_sd=baseline_sd, sample_rate=fs
    )

    note = ""
    if n_epochs < n_events:
        note = f"丢弃 {n_events - n_epochs} 个越界 epoch"
    return ErpAnalysis(
        sample_rate=fs,
        n_events=n_events,
        n_epochs=n_epochs,
        times_ms=times_ms,
        average=average,
        epochs=stack,
        sem=sem,
        p1=p1,
        n1=n1,
        p2=p2,
        main_peak=main_peak,
        onset_latency_ms=onset_ms,
        onset_threshold=threshold,
        note=note,
    )


def analyze_erp_csv(path: Path, *, min_events: int = ERP_MIN_EVENTS) -> ErpAnalysis:
    eeg, pink, fs = load_eeg_pink_csv(path)
    return analyze_auditory_erp(eeg, pink, fs, min_events=min_events)


def format_erp_report(result: ErpAnalysis) -> str:
    lines = [
        f"ERP 分析: {result.n_epochs}/{result.n_events} epoch @ {result.sample_rate:.1f} Hz",
        "预处理: 50/100 Hz 工频陷波 + 1–30 Hz 带通（零相位）",
    ]
    if result.note:
        lines.append(result.note)

    def _peak_line(peak: Optional[ErpPeakInfo]) -> str:
        if peak is None:
            return f"{'—'}: 未检出"
        return f"{peak.name}: 潜伏期 {peak.latency_ms:.1f} ms，幅值 {peak.amplitude:.3f}"

    lines.append(_peak_line(result.p1))
    lines.append(_peak_line(result.n1))
    lines.append(_peak_line(result.p2))
    lines.append(_peak_line(result.main_peak))
    if result.onset_latency_ms is None:
        lines.append(f"起始潜伏期: 未检出（阈值 {result.onset_threshold:.3f}）")
    else:
        lines.append(
            f"起始潜伏期: {result.onset_latency_ms:.1f} ms"
            f"（阈值 {result.onset_threshold:.3f}）"
        )
    return "\n".join(lines)


def export_erp_summary_csv(path: Path, result: ErpAnalysis) -> Path:
    """写出指标摘要与平均波形到 path（同名 *_erp_summary.csv 由调用方决定）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["sample_rate_hz", f"{result.sample_rate:.6g}"])
        writer.writerow(["n_events", result.n_events])
        writer.writerow(["n_epochs", result.n_epochs])
        writer.writerow(
            [
                "p1_latency_ms",
                "" if result.p1 is None else f"{result.p1.latency_ms:.3f}",
            ]
        )
        writer.writerow(
            [
                "p1_amplitude",
                "" if result.p1 is None else f"{result.p1.amplitude:.6g}",
            ]
        )
        writer.writerow(
            [
                "n1_latency_ms",
                "" if result.n1 is None else f"{result.n1.latency_ms:.3f}",
            ]
        )
        writer.writerow(
            [
                "n1_amplitude",
                "" if result.n1 is None else f"{result.n1.amplitude:.6g}",
            ]
        )
        writer.writerow(
            [
                "p2_latency_ms",
                "" if result.p2 is None else f"{result.p2.latency_ms:.3f}",
            ]
        )
        writer.writerow(
            [
                "p2_amplitude",
                "" if result.p2 is None else f"{result.p2.amplitude:.6g}",
            ]
        )
        writer.writerow(
            [
                "main_peak_latency_ms",
                "" if result.main_peak is None else f"{result.main_peak.latency_ms:.3f}",
            ]
        )
        writer.writerow(
            [
                "main_peak_amplitude",
                ""
                if result.main_peak is None
                else f"{result.main_peak.amplitude:.6g}",
            ]
        )
        writer.writerow(
            [
                "onset_latency_ms",
                ""
                if result.onset_latency_ms is None
                else f"{result.onset_latency_ms:.3f}",
            ]
        )
        writer.writerow(["onset_threshold", f"{result.onset_threshold:.6g}"])
        writer.writerow(["note", result.note])
        writer.writerow([])
        writer.writerow(["time_ms", "average_erp"])
        for t, y in zip(result.times_ms, result.average):
            writer.writerow([f"{float(t):.6f}", f"{float(y):.6g}"])
    return path
