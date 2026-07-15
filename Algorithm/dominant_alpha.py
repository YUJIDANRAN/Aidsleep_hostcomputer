"""
[已弃用] Alpha 主导子频 Goertzel 相位跟踪。

助眠闭环已改为 IAF + ecHT，见 Algorithm/iaf_echt.py。
本文件仅保留作对照/回退，在线与标定不再导入。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Sequence

import numpy as np

from power_cal import bandpass_filter

# 与 View/handle_event.py、Controller 一致
HILBERT_WINDOW_SEC = 0.5
MIN_SAMPLES_RATIO = 0.5
ALPHA_FREQ_MIN_HZ = 7.0
ALPHA_FREQ_MAX_HZ = 14.0
FREQ_STEP_HZ = 0.5
NARROW_BANDWIDTH_HZ = 2.0
TROUGH_PHASE_RAD = math.pi
POWER_STICK_RATIO = 0.85  ## 主导频切换滞后，避免 9/10 Hz 来回跳


@dataclass(frozen=True)
class DominantAlphaSnapshot:
  ready: bool
  phase_rad: float = 0.0
  inst_freq_hz: float = 10.0
  seconds_to_trough: float = 0.0
  dominant_amplitude: float = 0.0


def alpha_candidate_frequencies(
    *,
    f_min: float = ALPHA_FREQ_MIN_HZ,
    f_max: float = ALPHA_FREQ_MAX_HZ,
    step: float = FREQ_STEP_HZ,
) -> np.ndarray:
    count = int(round((f_max - f_min) / step)) + 1
    freqs = f_min + step * np.arange(count, dtype=np.float64)
    return freqs[(freqs >= f_min) & (freqs <= f_max)]


def goertzel_complex(
    signal: np.ndarray, sample_rate: float, frequency_hz: float
) -> complex:
    """单频 Goertzel，返回窗口末端该频率的复振幅（幅值+相位）。"""
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0.0 + 0.0j
    omega = 2.0 * math.pi * frequency_hz / sample_rate
    coeff = 2.0 * math.cos(omega)
    s1 = 0.0
    s2 = 0.0
    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0
    real = s1 - s2 * math.cos(omega)
    imag = s2 * math.sin(omega)
    return complex(real, imag)


def scan_dominant_frequency(
    signal: np.ndarray,
    sample_rate: float,
    *,
    frequencies: np.ndarray | None = None,
    last_freq_hz: float | None = None,
) -> tuple[float, float, float]:
    """
    在 Alpha 候选频率中找幅值最大者。

    返回 (dominant_freq_hz, amplitude, phase_rad)。
    """
    freqs = (
        np.asarray(frequencies, dtype=np.float64)
        if frequencies is not None
        else alpha_candidate_frequencies()
    )
    if freqs.size == 0:
        return 10.0, 0.0, 0.0

    best_freq = float(freqs[0])
    best_amp = 0.0
    best_phase = 0.0
    last_amp = 0.0

    for freq in freqs:
        z = goertzel_complex(signal, sample_rate, float(freq))
        amp = abs(z)
        if last_freq_hz is not None and abs(float(freq) - last_freq_hz) < 1e-6:
            last_amp = amp
        if amp > best_amp:
            best_amp = amp
            best_freq = float(freq)
            best_phase = float(np.angle(z) % (2.0 * math.pi))

    if (
        last_freq_hz is not None
        and last_amp >= best_amp * POWER_STICK_RATIO
        and last_amp > 0.0
    ):
        z_last = goertzel_complex(signal, sample_rate, float(last_freq_hz))
        return float(last_freq_hz), float(abs(z_last)), float(
            np.angle(z_last) % (2.0 * math.pi)
        )

    return best_freq, float(best_amp), best_phase


def compute_seconds_to_trough(phase_rad: float, freq_hz: float) -> float:
    freq_hz = float(np.clip(freq_hz, ALPHA_FREQ_MIN_HZ, ALPHA_FREQ_MAX_HZ))
    delta = (TROUGH_PHASE_RAD - phase_rad) % (2.0 * math.pi)
    return float(delta / (2.0 * math.pi * freq_hz))


class DominantAlphaPhaseTracker:
    """Alpha 带通序列 → 主导子频 Goertzel 相位与下一波谷时间。"""

    def __init__(
        self,
        sample_rate: float,
        window: int | None = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        if window is None:
            window = max(64, int(round(sample_rate * HILBERT_WINDOW_SEC)))
        self.window = int(window)
        self._min_samples = max(64, int(self.window * MIN_SAMPLES_RATIO))
        self._buf: Deque[float] = deque(maxlen=self.window)
        self._last_freq_hz: float | None = None
        self._frequencies = alpha_candidate_frequencies()

    def reset(self) -> None:
        self._buf.clear()
        self._last_freq_hz = None

    def push(self, alpha: float) -> DominantAlphaSnapshot:
        self._buf.append(float(alpha))
        return self.snapshot()

    def snapshot(self) -> DominantAlphaSnapshot:
        if len(self._buf) < self._min_samples:
            return DominantAlphaSnapshot(ready=False)

        arr = np.asarray(self._buf, dtype=np.float64)
        freq_hz, amplitude, phase_rad = scan_dominant_frequency(
            arr,
            self.sample_rate,
            frequencies=self._frequencies,
            last_freq_hz=self._last_freq_hz,
        )
        self._last_freq_hz = freq_hz
        seconds_to_trough = compute_seconds_to_trough(phase_rad, freq_hz)
        return DominantAlphaSnapshot(
            ready=True,
            phase_rad=phase_rad,
            inst_freq_hz=freq_hz,
            seconds_to_trough=seconds_to_trough,
            dominant_amplitude=amplitude,
        )


def estimate_dominant_freq_hz(
    alpha: np.ndarray,
    sample_rate: float,
    *,
    bursts: Sequence[object] | None = None,
) -> float:
    """从 burst 记录或整段信号估计标定用的主导 Alpha 频率。"""
    if bursts:
        freqs = [
            float(getattr(b, "inst_freq_hz"))
            for b in bursts
            if getattr(b, "inst_freq_hz", 0.0) > 0.0
        ]
        if freqs:
            return float(np.median(freqs))

    tracker = DominantAlphaPhaseTracker(sample_rate)
    collected: list[float] = []
    for value in alpha:
        snap = tracker.push(float(value))
        if snap.ready:
            collected.append(snap.inst_freq_hz)
    if collected:
        return float(np.median(collected))
    return 10.0


def narrowband_alpha(
    alpha: np.ndarray,
    sample_rate: float,
    center_hz: float,
    *,
    bandwidth_hz: float = NARROW_BANDWIDTH_HZ,
) -> np.ndarray:
    """以主导频率为中心的窄带 Alpha 成分。"""
    half = bandwidth_hz / 2.0
    low = max(0.5, center_hz - half)
    high = min(sample_rate / 2.0 - 1.0, center_hz + half)
    if low >= high:
        return alpha.astype(np.float64, copy=True)
    return bandpass_filter(alpha.astype(np.float64), sample_rate, low, high)


def run_tracker_series(
    alpha: Iterable[float], sample_rate: float
) -> list[DominantAlphaSnapshot]:
    """离线：逐点复现在线主导频相位跟踪。"""
    tracker = DominantAlphaPhaseTracker(sample_rate)
    return [tracker.push(float(v)) for v in alpha]


def find_goertzel_trough_indices(
    snapshots: Sequence[DominantAlphaSnapshot],
    sample_rate: float,
    *,
    min_distance: int | None = None,
) -> np.ndarray:
    """
    与在线 DominantAlphaPhaseTracker 同一定义找波谷：
    seconds_to_trough 从接近 0 突然跳回接近一个周期 → 刚越过相位 π。
    """
    if min_distance is None:
        min_distance = max(1, int(round(0.08 * sample_rate)))
    troughs: list[int] = []
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]
        if not prev.ready or not curr.ready:
            continue
        period = 1.0 / max(float(curr.inst_freq_hz), 1e-6)
        # 越过波谷：距下一谷的时间突然增大，且越过前已接近谷底
        if (
            curr.seconds_to_trough > prev.seconds_to_trough + 0.35 * period
            and prev.seconds_to_trough < 0.30 * period
        ):
            idx = i - 1
            if not troughs or idx - troughs[-1] >= min_distance:
                troughs.append(idx)
    return np.asarray(troughs, dtype=np.int64)
