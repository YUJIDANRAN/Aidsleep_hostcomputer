"""α 闭环离线验证：粉噪刺激时刻相对 α 相位（onset / P1）的锁相统计。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, iirnotch

from erp import (
    ERP_NOTCH_HZ,
    ERP_NOTCH_Q,
    detect_pink_seq_events,
    load_eeg_pink_csv,
)

ALPHA_BANDPASS_HZ = (7.5, 12.5)
## 标定 P1 潜伏期（ms）；锁相验证严格按此取样，不从同文件 ERP 自动改写
ALPHA_CALIBRATED_P1_MS = 45.0
ALPHA_MIN_EVENTS = 8
## 相位约定：0° = α 正峰，±180° = 波谷（Hilbert 角）
ALPHA_TROUGH_DEG = 180.0
ALPHA_HIT_HALF_WIDTH_DEG = 45.0
ALPHA_EPOCH_PRE_SEC = 0.200
ALPHA_EPOCH_POST_SEC = 0.500
ALPHA_HIST_BINS = 18


@dataclass(frozen=True)
class PhaseLockStats:
    n: int
    mean_deg: float
    plv: float
    hit_rate: float
    angular_std_deg: float
    phases_deg: np.ndarray


@dataclass(frozen=True)
class AlphaPhaseAnalysis:
    sample_rate: float
    n_events: int
    alpha_band_hz: Tuple[float, float]
    p1_latency_ms: float
    p1_source: str
    target_deg: float
    hit_half_width_deg: float
    onset: PhaseLockStats
    at_p1: PhaseLockStats
    times_ms: np.ndarray
    alpha_average: np.ndarray
    alpha_sem: np.ndarray
    hist_edges_deg: np.ndarray
    hist_onset: np.ndarray
    hist_p1: np.ndarray
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


def _wrap_deg(deg: np.ndarray) -> np.ndarray:
    """映射到 (-180, 180]。"""
    x = np.asarray(deg, dtype=np.float64)
    return (x + 180.0) % 360.0 - 180.0


def _circular_mean_deg(phases_deg: np.ndarray) -> float:
    rad = np.deg2rad(np.asarray(phases_deg, dtype=np.float64))
    if rad.size == 0:
        return float("nan")
    mean = np.angle(np.mean(np.exp(1j * rad)))
    return float(_wrap_deg(np.rad2deg(mean)))


def _plv(phases_deg: np.ndarray) -> float:
    rad = np.deg2rad(np.asarray(phases_deg, dtype=np.float64))
    if rad.size == 0:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * rad))))


def _angular_std_deg(phases_deg: np.ndarray) -> float:
    """圆周标准差（度），由平均合成长度 R 推导。"""
    r = _plv(phases_deg)
    if not np.isfinite(r) or r <= 0:
        return float("nan")
    r = min(1.0, max(1e-12, r))
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(r))))


def _hit_rate(
    phases_deg: np.ndarray,
    target_deg: float,
    half_width_deg: float,
) -> float:
    if phases_deg.size == 0:
        return float("nan")
    err = np.abs(_wrap_deg(np.asarray(phases_deg, dtype=np.float64) - target_deg))
    return float(np.mean(err <= half_width_deg))


def _phase_stats(
    phases_deg: np.ndarray,
    target_deg: float,
    half_width_deg: float,
) -> PhaseLockStats:
    ph = _wrap_deg(np.asarray(phases_deg, dtype=np.float64))
    return PhaseLockStats(
        n=int(ph.size),
        mean_deg=_circular_mean_deg(ph),
        plv=_plv(ph),
        hit_rate=_hit_rate(ph, target_deg, half_width_deg),
        angular_std_deg=_angular_std_deg(ph),
        phases_deg=ph,
    )


def _resolve_p1_ms(p1_latency_ms: Optional[float]) -> Tuple[float, str]:
    """严格使用标定 45 ms；仅当调用方显式传入时才覆盖。"""
    if p1_latency_ms is not None and np.isfinite(p1_latency_ms) and p1_latency_ms >= 0:
        return float(p1_latency_ms), "调用方指定"
    return float(ALPHA_CALIBRATED_P1_MS), f"标定 {ALPHA_CALIBRATED_P1_MS:.0f} ms"


def analyze_alpha_phase_lock(
    eeg: np.ndarray,
    pink_seq: np.ndarray,
    sample_rate: float,
    *,
    band_hz: Tuple[float, float] = ALPHA_BANDPASS_HZ,
    p1_latency_ms: Optional[float] = None,
    target_deg: float = ALPHA_TROUGH_DEG,
    hit_half_width_deg: float = ALPHA_HIT_HALF_WIDTH_DEG,
    min_events: int = ALPHA_MIN_EVENTS,
) -> AlphaPhaseAnalysis:
    """对已对齐的 EEG / pink_seq 做 α 锁相验证。"""
    fs = float(sample_rate)
    if fs <= 0:
        raise ValueError(f"无效采样率: {sample_rate}")

    events = detect_pink_seq_events(pink_seq)
    n_events = int(events.size)
    if n_events < min_events:
        raise ValueError(f"有效粉噪事件过少 ({n_events} < {min_events})，无法可靠估计锁相")

    p1_ms, p1_source = _resolve_p1_ms(p1_latency_ms)
    p1_shift = int(round(p1_ms * 1e-3 * fs))

    alpha = _alpha_bandpass(_notch(eeg, fs), fs, band_hz)
    phase = np.angle(hilbert(alpha))  ## rad, 0=正峰

    onset_idx = []
    p1_idx = []
    for ev in events:
        i0 = int(ev)
        i1 = i0 + p1_shift
        if i0 < 0 or i0 >= phase.size:
            continue
        if i1 < 0 or i1 >= phase.size:
            continue
        onset_idx.append(i0)
        p1_idx.append(i1)

    if len(onset_idx) < min_events:
        raise ValueError(
            f"落在有效范围内的事件过少 ({len(onset_idx)} < {min_events})"
        )

    onset_deg = _wrap_deg(np.rad2deg(phase[np.asarray(onset_idx, dtype=np.int64)]))
    p1_deg = _wrap_deg(np.rad2deg(phase[np.asarray(p1_idx, dtype=np.int64)]))
    target = float(_wrap_deg(np.asarray([target_deg]))[0])
    half = float(hit_half_width_deg)

    onset_stats = _phase_stats(onset_deg, target, half)
    p1_stats = _phase_stats(p1_deg, target, half)

    edges = np.linspace(-180.0, 180.0, ALPHA_HIST_BINS + 1)
    hist_onset, _ = np.histogram(onset_stats.phases_deg, bins=edges)
    hist_p1, _ = np.histogram(p1_stats.phases_deg, bins=edges)

    pre_n = int(round(ALPHA_EPOCH_PRE_SEC * fs))
    post_n = int(round(ALPHA_EPOCH_POST_SEC * fs))
    epochs = []
    for ev in events:
        i0 = int(ev)
        a = i0 - pre_n
        b = i0 + post_n + 1
        if a < 0 or b > alpha.size:
            continue
        epochs.append(alpha[a:b])
    if not epochs:
        raise ValueError("无法切出有效 α epoch（记录过短或事件靠边）")
    epochs_arr = np.asarray(epochs, dtype=np.float64)
    times_ms = (np.arange(-pre_n, post_n + 1, dtype=np.float64) / fs) * 1000.0
    avg = np.mean(epochs_arr, axis=0)
    sem = np.std(epochs_arr, axis=0, ddof=1) / np.sqrt(epochs_arr.shape[0])

    note = (
        f"相位约定: 0°=α正峰, ±180°=波谷；目标={target:.0f}°±{half:.0f}°；"
        f"P1 取样点={p1_ms:.1f} ms（{p1_source}）"
    )
    return AlphaPhaseAnalysis(
        sample_rate=fs,
        n_events=n_events,
        alpha_band_hz=(float(band_hz[0]), float(band_hz[1])),
        p1_latency_ms=p1_ms,
        p1_source=p1_source,
        target_deg=target,
        hit_half_width_deg=half,
        onset=onset_stats,
        at_p1=p1_stats,
        times_ms=times_ms,
        alpha_average=avg,
        alpha_sem=sem,
        hist_edges_deg=edges,
        hist_onset=hist_onset.astype(np.float64),
        hist_p1=hist_p1.astype(np.float64),
        note=note,
    )


def analyze_alpha_phase_csv(
    path: Path,
    *,
    band_hz: Tuple[float, float] = ALPHA_BANDPASS_HZ,
    p1_latency_ms: Optional[float] = None,
    target_deg: float = ALPHA_TROUGH_DEG,
    hit_half_width_deg: float = ALPHA_HIT_HALF_WIDTH_DEG,
    min_events: int = ALPHA_MIN_EVENTS,
) -> AlphaPhaseAnalysis:
    eeg, pink, fs = load_eeg_pink_csv(path)
    return analyze_alpha_phase_lock(
        eeg,
        pink,
        fs,
        band_hz=band_hz,
        p1_latency_ms=p1_latency_ms,
        target_deg=target_deg,
        hit_half_width_deg=hit_half_width_deg,
        min_events=min_events,
    )


def format_alpha_phase_report(result: AlphaPhaseAnalysis) -> str:
    lo, hi = result.alpha_band_hz

    def _block(label: str, s: PhaseLockStats) -> list[str]:
        return [
            f"[{label}] n={s.n}",
            f"  平均相位 {s.mean_deg:.1f}°  PLV={s.plv:.3f}  "
            f"角标准差 {s.angular_std_deg:.1f}°",
            f"  落在目标窗比例 {s.hit_rate:.1%} "
            f"（目标 {result.target_deg:.0f}°±{result.hit_half_width_deg:.0f}°）",
        ]

    lines = [
        f"α 锁相验证: {result.n_events} 事件 @ {result.sample_rate:.1f} Hz",
        f"α 带通: {lo:.1f}–{hi:.1f} Hz（零相位）+ 50/100 Hz 陷波",
        f"P1 取样: {result.p1_latency_ms:.1f} ms（{result.p1_source}）",
        result.note,
        *_block("刺激 onset", result.onset),
        *_block("onset + P1", result.at_p1),
    ]
    return "\n".join(lines)


def export_alpha_phase_summary_csv(path: Path, result: AlphaPhaseAnalysis) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["sample_rate_hz", f"{result.sample_rate:.6g}"])
        writer.writerow(["n_events", result.n_events])
        writer.writerow(["alpha_lo_hz", f"{result.alpha_band_hz[0]:.6g}"])
        writer.writerow(["alpha_hi_hz", f"{result.alpha_band_hz[1]:.6g}"])
        writer.writerow(["p1_latency_ms", f"{result.p1_latency_ms:.3f}"])
        writer.writerow(["p1_source", result.p1_source])
        writer.writerow(["target_deg", f"{result.target_deg:.3f}"])
        writer.writerow(["hit_half_width_deg", f"{result.hit_half_width_deg:.3f}"])
        writer.writerow(["onset_n", result.onset.n])
        writer.writerow(["onset_mean_deg", f"{result.onset.mean_deg:.6g}"])
        writer.writerow(["onset_plv", f"{result.onset.plv:.6g}"])
        writer.writerow(["onset_hit_rate", f"{result.onset.hit_rate:.6g}"])
        writer.writerow(["onset_angular_std_deg", f"{result.onset.angular_std_deg:.6g}"])
        writer.writerow(["p1_n", result.at_p1.n])
        writer.writerow(["p1_mean_deg", f"{result.at_p1.mean_deg:.6g}"])
        writer.writerow(["p1_plv", f"{result.at_p1.plv:.6g}"])
        writer.writerow(["p1_hit_rate", f"{result.at_p1.hit_rate:.6g}"])
        writer.writerow(["p1_angular_std_deg", f"{result.at_p1.angular_std_deg:.6g}"])
        writer.writerow(["note", result.note])
        writer.writerow([])
        writer.writerow(["event_index", "onset_phase_deg", "p1_phase_deg"])
        n = min(result.onset.phases_deg.size, result.at_p1.phases_deg.size)
        for i in range(n):
            writer.writerow(
                [
                    i,
                    f"{float(result.onset.phases_deg[i]):.6f}",
                    f"{float(result.at_p1.phases_deg[i]):.6f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["time_ms", "alpha_average", "alpha_sem"])
        for t, y, s in zip(result.times_ms, result.alpha_average, result.alpha_sem):
            writer.writerow([f"{float(t):.6f}", f"{float(y):.6g}", f"{float(s):.6g}"])
    return path
