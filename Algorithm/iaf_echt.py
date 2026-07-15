"""
IAF（个体 Alpha 峰频）+ ecHT（端点校正 Hilbert）相位跟踪。

对齐 Elemind / Schreglmann 闭环思路：
1. 从近期 EEG 估计 IAF（Welch PSD + 1/f 去趋势峰）；
2. 在滑动窗上对解析谱做因果带通（中心≈IAF），取窗末样本相位；
3. 波谷 = 相位 π，用 IAF 换算 seconds_to_trough。

在线与离线标定共用本模块，保证口径一致。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Sequence, Tuple

import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift, next_fast_len
from scipy.signal import butter, freqz, welch

# 与既有助眠管线窗口量级一致
ECHT_WINDOW_SEC = 0.5
IAF_BUFFER_SEC = 4.0
IAF_UPDATE_SEC = 0.5
MIN_SAMPLES_RATIO = 0.5
ALPHA_FREQ_MIN_HZ = 7.0
ALPHA_FREQ_MAX_HZ = 14.0
DEFAULT_IAF_HZ = 10.0
ECHT_FILT_ORDER = 2
TROUGH_PHASE_RAD = math.pi
IAF_STICK_HZ = 0.25  ## IAF 抖动抑制


@dataclass(frozen=True)
class EcHTSnapshot:
    ready: bool
    phase_rad: float = 0.0
    inst_freq_hz: float = DEFAULT_IAF_HZ
    seconds_to_trough: float = 0.0
    dominant_amplitude: float = 0.0
    iaf_hz: float = DEFAULT_IAF_HZ


def clamp_iaf_hz(freq_hz: float) -> float:
    return float(np.clip(freq_hz, ALPHA_FREQ_MIN_HZ, ALPHA_FREQ_MAX_HZ))


def echt_band_edges(iaf_hz: float) -> Tuple[float, float]:
    """
    因果带通半宽取 IAF/4（总带宽 IAF/2），与 meegkit/Schreglmann 经验一致。
    """
    f0 = clamp_iaf_hz(iaf_hz)
    half_bw = max(1.0, 0.25 * f0)
    low = max(0.5, f0 - half_bw)
    high = f0 + half_bw
    return float(low), float(high)


def estimate_iaf_hz(
    signal: np.ndarray,
    sample_rate: float,
    *,
    f_min: float = ALPHA_FREQ_MIN_HZ,
    f_max: float = ALPHA_FREQ_MAX_HZ,
) -> float:
    """Welch PSD + log(f)–log(P) 线性去趋势后，在 Alpha 带取残差峰为 IAF。"""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size < 64 or sample_rate <= 0:
        return DEFAULT_IAF_HZ

    nperseg = int(min(arr.size, max(64, round(sample_rate * 2.0))))
    freqs, psd = welch(arr, fs=float(sample_rate), nperseg=nperseg)
    mask = (freqs >= f_min) & (freqs <= f_max) & (psd > 0) & (freqs > 0)
    if not np.any(mask):
        return DEFAULT_IAF_HZ

    f = freqs[mask]
    p = psd[mask]
    if f.size < 3:
        return clamp_iaf_hz(float(f[int(np.argmax(p))]))

    lf = np.log(f)
    lp = np.log(p)
    slope, intercept = np.polyfit(lf, lp, 1)
    residual = lp - (intercept + slope * lf)
    return clamp_iaf_hz(float(f[int(np.argmax(residual))]))


def _build_echt_kernels(
    n_fft: int,
    sample_rate: float,
    l_freq: float,
    h_freq: float,
    filt_order: int = ECHT_FILT_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """解析谱乘子 h_ 与因果 Butterworth 频响 coef_（对齐 meegkit.phase.ECHT）。"""
    h = np.zeros(n_fft, dtype=np.float64)
    h[0] = 1.0
    h[1 : (n_fft // 2) + 1] = 2.0
    if n_fft % 2 == 0:
        h[n_fft // 2] = 1.0

    nyq = 0.5 * float(sample_rate)
    low = max(1e-6, min(l_freq, nyq * 0.99))
    high = max(low * 1.01, min(h_freq, nyq * 0.999))
    b, a = butter(int(filt_order), [low / nyq, high / nyq], btype="band")
    T = n_fft / float(sample_rate)
    filt_freq = np.ceil(np.arange(-n_fft / 2.0, n_fft / 2.0) / T)
    coef = freqz(b, a, worN=filt_freq, fs=float(sample_rate))[1]
    return h, np.asarray(coef, dtype=np.complex128)


def echt_endpoint_analytic(
    signal: np.ndarray,
    sample_rate: float,
    l_freq: float,
    h_freq: float,
    *,
    filt_order: int = ECHT_FILT_ORDER,
    n_fft: int | None = None,
    h: np.ndarray | None = None,
    coef: np.ndarray | None = None,
) -> complex:
    """
    对滑动窗做 ecHT，返回窗末（最近样本）的解析值。
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = int(x.size)
    if n < 8:
        return complex(x[-1] if n else 0.0, 0.0)

    if n_fft is None:
        n_fft = int(next_fast_len(n))
    if h is None or coef is None:
        h, coef = _build_echt_kernels(n_fft, sample_rate, l_freq, h_freq, filt_order)

    Xf = fft(x, n_fft)
    Xf = Xf * h
    Xf = fftshift(Xf)
    Xf = Xf * coef
    Xf = ifftshift(Xf)
    z = ifft(Xf)
    return complex(z[n - 1])


def compute_seconds_to_trough(phase_rad: float, freq_hz: float) -> float:
    freq_hz = clamp_iaf_hz(freq_hz)
    delta = (TROUGH_PHASE_RAD - phase_rad) % (2.0 * math.pi)
    return float(delta / (2.0 * math.pi * freq_hz))


class IAFEcHTPhaseTracker:
    """因果 Alpha（或更宽）序列 → IAF + ecHT 相位与下一波谷时间。"""

    def __init__(
        self,
        sample_rate: float,
        window: int | None = None,
        *,
        iaf_buffer: int | None = None,
        filt_order: int = ECHT_FILT_ORDER,
    ) -> None:
        self.sample_rate = float(sample_rate)
        if window is None:
            window = max(64, int(round(self.sample_rate * ECHT_WINDOW_SEC)))
        self.window = int(window)
        if iaf_buffer is None:
            iaf_buffer = max(self.window, int(round(self.sample_rate * IAF_BUFFER_SEC)))
        self.iaf_buffer = int(iaf_buffer)
        self.filt_order = int(filt_order)

        self._min_samples = max(64, int(self.window * MIN_SAMPLES_RATIO))
        self._min_iaf_samples = max(self._min_samples, int(round(self.sample_rate * 1.0)))
        self._iaf_update_every = max(1, int(round(self.sample_rate * IAF_UPDATE_SEC)))

        self._echt_buf: Deque[float] = deque(maxlen=self.window)
        self._iaf_buf: Deque[float] = deque(maxlen=self.iaf_buffer)
        self._iaf_hz = DEFAULT_IAF_HZ
        self._iaf_ready = False
        self._since_iaf_update = 0

        self._n_fft = int(next_fast_len(self.window))
        self._cache_key: Tuple[float, float, int] | None = None
        self._h: np.ndarray | None = None
        self._coef: np.ndarray | None = None

    def reset(self) -> None:
        self._echt_buf.clear()
        self._iaf_buf.clear()
        self._iaf_hz = DEFAULT_IAF_HZ
        self._iaf_ready = False
        self._since_iaf_update = 0
        self._cache_key = None
        self._h = None
        self._coef = None

    def _ensure_kernels(self, l_freq: float, h_freq: float) -> None:
        key = (round(l_freq, 3), round(h_freq, 3), self._n_fft)
        if key == self._cache_key and self._h is not None and self._coef is not None:
            return
        self._h, self._coef = _build_echt_kernels(
            self._n_fft,
            self.sample_rate,
            l_freq,
            h_freq,
            self.filt_order,
        )
        self._cache_key = key

    def _maybe_update_iaf(self) -> None:
        if len(self._iaf_buf) < self._min_iaf_samples:
            return
        self._since_iaf_update += 1
        need = (not self._iaf_ready) or (self._since_iaf_update >= self._iaf_update_every)
        if not need:
            return
        new_iaf = estimate_iaf_hz(np.asarray(self._iaf_buf, dtype=np.float64), self.sample_rate)
        if self._iaf_ready and abs(new_iaf - self._iaf_hz) < IAF_STICK_HZ:
            new_iaf = self._iaf_hz
        self._iaf_hz = new_iaf
        self._iaf_ready = True
        self._since_iaf_update = 0

    def push(self, sample: float) -> EcHTSnapshot:
        v = float(sample)
        self._echt_buf.append(v)
        self._iaf_buf.append(v)
        self._maybe_update_iaf()
        return self.snapshot()

    def snapshot(self) -> EcHTSnapshot:
        if len(self._echt_buf) < self._min_samples or not self._iaf_ready:
            return EcHTSnapshot(ready=False, iaf_hz=self._iaf_hz, inst_freq_hz=self._iaf_hz)

        arr = np.asarray(self._echt_buf, dtype=np.float64)
        l_freq, h_freq = echt_band_edges(self._iaf_hz)
        self._ensure_kernels(l_freq, h_freq)
        z_end = echt_endpoint_analytic(
            arr,
            self.sample_rate,
            l_freq,
            h_freq,
            filt_order=self.filt_order,
            n_fft=self._n_fft,
            h=self._h,
            coef=self._coef,
        )
        phase_rad = float(np.angle(z_end) % (2.0 * math.pi))
        amplitude = float(abs(z_end))
        freq_hz = self._iaf_hz
        return EcHTSnapshot(
            ready=True,
            phase_rad=phase_rad,
            inst_freq_hz=freq_hz,
            seconds_to_trough=compute_seconds_to_trough(phase_rad, freq_hz),
            dominant_amplitude=amplitude,
            iaf_hz=freq_hz,
        )


def estimate_iaf_from_series(
    alpha: np.ndarray,
    sample_rate: float,
    *,
    bursts: Sequence[object] | None = None,
) -> float:
    """标定报告用：优先 burst 记录中的 IAF/频率中位数，否则整段估计。"""
    if bursts:
        freqs = [
            float(getattr(b, "inst_freq_hz"))
            for b in bursts
            if getattr(b, "inst_freq_hz", 0.0) > 0.0
        ]
        if freqs:
            return clamp_iaf_hz(float(np.median(freqs)))

    if alpha.size >= 64:
        return estimate_iaf_hz(alpha, sample_rate)
    return DEFAULT_IAF_HZ


def run_tracker_series(
    alpha: Iterable[float], sample_rate: float
) -> list[EcHTSnapshot]:
    """离线逐点复现在线 IAF+ecHT 跟踪。"""
    tracker = IAFEcHTPhaseTracker(sample_rate)
    return [tracker.push(float(v)) for v in alpha]


def find_echt_trough_indices(
    snapshots: Sequence[EcHTSnapshot],
    sample_rate: float,
    *,
    min_distance: int | None = None,
) -> np.ndarray:
    """
    与在线同一定义找波谷：
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
        if (
            curr.seconds_to_trough > prev.seconds_to_trough + 0.35 * period
            and prev.seconds_to_trough < 0.30 * period
        ):
            idx = i - 1
            if not troughs or idx - troughs[-1] >= min_distance:
                troughs.append(idx)
    return np.asarray(troughs, dtype=np.int64)
