"""
三轴加速度振动分析（oscillator_serial → osc_data）。

采样：1 ms/点 (1000 Hz)，每 100 点一批上传。

处理流程：
  1. 原始计数 → m/s²（1000 计数 ≈ 9.80665 m/s²，即 1g）
  2. 三轴 0.15 Hz 高通去重力 → 合成加速度幅值
  3. 0.5–40 Hz 总带通 → 各节律窄带 → 界面显示加速度 (m/s²)
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, detrend, hilbert, sosfilt, sosfilt_zi, sosfiltfilt, welch

FILTER_ORDER = 4
OSC_SAMPLE_RATE = 1000.0  ## 1 ms 采样 → 1000 Hz
MCU_BATCH_SIZE = 100  ## 与固件 BUFFER_SIZE 一致
DEFAULT_SAMPLE_RATE = float(os.environ.get("OSC_SAMPLE_RATE", str(OSC_SAMPLE_RATE)))

## 传感器标定：静止时约 1000 计数对应 1g = 9.80665 m/s²
GRAVITY_M_S2 = float(os.environ.get("OSC_GRAVITY_M_S2", "9.80665"))
SENSOR_UNITS_PER_G = float(os.environ.get("OSC_SENSOR_UNITS_PER_G", "1000"))
M_S2_PER_SENSOR_UNIT = GRAVITY_M_S2 / SENSOR_UNITS_PER_G
ACCEL_DISPLAY_UNIT = "m/s²"
DISP_DISPLAY_UNIT = "mm"
M_TO_MM = 1000.0  ## 离线位移换算用

BASE_LOW_HZ = 0.5
BASE_HIGH_HZ = 40.0
GRAVITY_HP_HZ = 0.15  ## 去重力高通（低于 0.5 Hz，避免把慢速大位移当漂移滤掉）
PEAK_FILTER_HALF_BW_HZ = 1.5
DEFAULT_PEAK_CENTER_HZ = 10.0
MIN_SCALE_FREQ_HZ = 0.1  ## 积分缩放最低频率，防止除零

## 与 EEG 相同的节律划分（0.5–40 Hz 内）；机械振动可视情况调整
OSC_BANDS: Dict[str, Tuple[float, float]] = {
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


def sensor_units_to_m_s2(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """MCU 原始计数 → m/s²。"""
    s = M_S2_PER_SENSOR_UNIT
    return x * s, y * s, z * s


def sensor_array_to_m_s2(arr: np.ndarray) -> np.ndarray:
    """N 或 N×3 数组 → m/s²。"""
    return np.asarray(arr, dtype=np.float64) * M_S2_PER_SENSOR_UNIT


def displacement_m_to_display(displacement_m: float) -> float:
    """位移 m → 界面显示 mm。"""
    return displacement_m * M_TO_MM


def band_center_hz(band: str) -> float:
    low, high = OSC_BANDS[band]
    return (low + high) * 0.5


def accel_to_displacement_at_freq(
    accel: float | np.ndarray,
    freq_hz: float,
) -> float | np.ndarray:
    """
    窄带加速度 → 位移（频域二次积分）。
    对正弦振动：位移幅值 = 加速度幅值 / (2πf)²，比泄漏时域积分幅值更准确。
    """
    f = max(float(freq_hz), MIN_SCALE_FREQ_HZ)
    w = 2.0 * np.pi * f
    return -np.asarray(accel, dtype=np.float64) / (w * w)


def scale_freq_for_band(band: str, dominant_freq_hz: float) -> float:
    """主振频率落在该带内则用实测频率，否则用频段中心。"""
    low, high = OSC_BANDS[band]
    if low <= dominant_freq_hz < high:
        return dominant_freq_hz
    return band_center_hz(band)


def design_highpass_sos(
    sample_rate: float,
    cutoff_hz: float = BASE_LOW_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """设计 Butterworth 高通 SOS，用于各轴去重力/去直流。"""
    if cutoff_hz <= 0:
        raise ValueError(f"截止频率须 > 0，当前为 {cutoff_hz}")
    nyquist = sample_rate * 0.5
    if cutoff_hz >= nyquist:
        raise ValueError(f"截止频率 {cutoff_hz} Hz 须低于奈奎斯特 {nyquist} Hz")
    return butter(order, cutoff_hz / nyquist, btype="high", output="sos")


def design_bandpass_sos(
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """设计 Butterworth 带通 SOS 系数。"""
    if low_hz <= 0:
        raise ValueError(f"低截止频率须 > 0，当前为 {low_hz}")
    if sample_rate <= 2 * high_hz:
        raise ValueError(
            f"采样率 {sample_rate} Hz 过低，无法设计 {high_hz} Hz 通带上限"
        )
    nyquist = sample_rate * 0.5
    return butter(
        order,
        (low_hz / nyquist, high_hz / nyquist),
        btype="bandpass",
        output="sos",
    )


def _bandpass_offline(
    signal: np.ndarray,
    sample_rate: float,
    low_hz: float = BASE_LOW_HZ,
    high_hz: float = BASE_HIGH_HZ,
) -> np.ndarray:
    """离线零相位带通（批处理用）。"""
    sos = design_bandpass_sos(sample_rate, low_hz, high_hz)
    return sosfiltfilt(sos, signal, axis=0)


def _highpass_offline(
    signal: np.ndarray,
    sample_rate: float,
    cutoff_hz: float = BASE_LOW_HZ,
) -> np.ndarray:
    """离线零相位高通（批处理用）。"""
    sos = design_highpass_sos(sample_rate, cutoff_hz)
    return sosfiltfilt(sos, signal, axis=0)


def _band_for_frequency(freq_hz: float) -> str:
    """将频率映射到所在节律带名称。"""
    for name, (low, high) in OSC_BANDS.items():
        if low <= freq_hz < high:
            return name
    if freq_hz >= OSC_BANDS["gamma"][0]:
        return "gamma"
    return "delta"


@dataclass(frozen=True)
class OscBatchResult:
    """一批加速度数据的分析结果。"""

    vibration: np.ndarray  ## 合成加速度 (N,)
    band_waveforms: Dict[str, np.ndarray]  ## 各节律 **位移** 波形
    band_envelopes: Dict[str, np.ndarray]  ## 位移 Hilbert 包络
    dominant_freq_hz: np.ndarray
    dominant_band: np.ndarray
    dominant_waveform: np.ndarray  ## 主振频带 **位移** 波形
    sample_rate: float = DEFAULT_SAMPLE_RATE


def accel_to_displacement(
    accel: np.ndarray,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    freq_hz: Optional[float] = None,
) -> np.ndarray:
    """离线：带通加速度 → 位移。freq_hz 给定则用频域缩放，否则时域积分+去趋势。"""
    if accel.size == 0:
        return accel
    if freq_hz is not None and freq_hz > 0:
        return accel_to_displacement_at_freq(accel, freq_hz)
    dt = 1.0 / sample_rate
    vel = np.cumsum(accel, dtype=np.float64) * dt
    vel = detrend(vel, type="linear")
    disp = np.cumsum(vel, dtype=np.float64) * dt
    return detrend(disp, type="linear")


class OscAxisHighpass:
    """单轴实时高通，用于去除重力/直流后再合成振动量。"""

    def __init__(
        self,
        sample_rate: float,
        cutoff_hz: float = GRAVITY_HP_HZ,
    ) -> None:
        self._sos = design_highpass_sos(sample_rate, cutoff_hz)
        self._zi = sosfilt_zi(self._sos)

    def reset(self) -> None:
        self._zi = sosfilt_zi(self._sos)

    def process(self, value: float) -> float:
        y, self._zi = sosfilt(self._sos, [value], zi=self._zi)
        return float(y[0])


class OscStreamProcessor:
    """
    实时逐点：三轴加速度 → 带通 → 各频段 / M_Fre 窄带加速度 (m/s²)。
    """

    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE) -> None:
        self._configured_rate = float(sample_rate)
        self._hp_x = OscAxisHighpass(self._configured_rate)
        self._hp_y = OscAxisHighpass(self._configured_rate)
        self._hp_z = OscAxisHighpass(self._configured_rate)
        self._base_sos = design_bandpass_sos(
            self._configured_rate, BASE_LOW_HZ, BASE_HIGH_HZ
        )
        self._base_zi = sosfilt_zi(self._base_sos)
        self._band_sos: Dict[str, np.ndarray] = {}
        self._band_zi: Dict[str, np.ndarray] = {}
        self._last_band_accel: Dict[str, float] = {name: 0.0 for name in OSC_BANDS}
        self._last_band_outputs: Dict[str, float] = {name: 0.0 for name in OSC_BANDS}
        self._last_axis_accel: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        for name, (low, high) in OSC_BANDS.items():
            self._band_sos[name] = design_bandpass_sos(
                self._configured_rate, low, high
            )
            self._band_zi[name] = sosfilt_zi(self._band_sos[name])
        self._vibration_buf: Deque[float] = deque(maxlen=int(sample_rate * 4))
        self._base_buf: Deque[float] = deque(maxlen=int(sample_rate * 4))
        self._last_base = 0.0
        self._last_peak_accel = 0.0
        self._last_m_freq_accel = 0.0
        self._dominant_band = "alpha"
        self._dominant_freq_hz = DEFAULT_PEAK_CENTER_HZ
        self._peak_center_hz = DEFAULT_PEAK_CENTER_HZ
        self._peak_sos = design_bandpass_sos(
            self._configured_rate,
            max(BASE_LOW_HZ, DEFAULT_PEAK_CENTER_HZ - PEAK_FILTER_HALF_BW_HZ),
            min(BASE_HIGH_HZ, DEFAULT_PEAK_CENTER_HZ + PEAK_FILTER_HALF_BW_HZ),
        )
        self._peak_zi = sosfilt_zi(self._peak_sos)
        self._dom_update_counter = 0
        self._dom_update_interval = max(20, int(sample_rate * 0.05))

    @property
    def sample_rate(self) -> float:
        return self._configured_rate

    @property
    def dominant_band(self) -> str:
        return self._dominant_band

    @property
    def dominant_freq_hz(self) -> float:
        return self._dominant_freq_hz

    @property
    def axis_accel(self) -> Dict[str, float]:
        """三轴高通后加速度 (m/s²)，键为 x/y/z。"""
        return dict(self._last_axis_accel)

    def reset(self) -> None:
        self._hp_x.reset()
        self._hp_y.reset()
        self._hp_z.reset()
        self._base_zi = sosfilt_zi(self._base_sos)
        for name in OSC_BANDS:
            self._band_zi[name] = sosfilt_zi(self._band_sos[name])
            self._last_band_accel[name] = 0.0
            self._last_band_outputs[name] = 0.0
        self._last_axis_accel = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._vibration_buf.clear()
        self._base_buf.clear()
        self._last_base = 0.0
        self._last_peak_accel = 0.0
        self._last_m_freq_accel = 0.0
        self._dominant_band = "alpha"
        self._dominant_freq_hz = DEFAULT_PEAK_CENTER_HZ
        self._peak_center_hz = DEFAULT_PEAK_CENTER_HZ
        self._peak_sos = design_bandpass_sos(
            self._configured_rate,
            max(BASE_LOW_HZ, DEFAULT_PEAK_CENTER_HZ - PEAK_FILTER_HALF_BW_HZ),
            min(BASE_HIGH_HZ, DEFAULT_PEAK_CENTER_HZ + PEAK_FILTER_HALF_BW_HZ),
        )
        self._peak_zi = sosfilt_zi(self._peak_sos)
        self._dom_update_counter = 0

    def _rebuild_peak_filter(self, center_hz: float) -> None:
        """按主振频率重建 M_Fre 窄带滤波器。"""
        center_hz = float(max(BASE_LOW_HZ + 0.1, min(BASE_HIGH_HZ - 0.1, center_hz)))
        if abs(center_hz - self._peak_center_hz) < 0.8:
            return
        self._peak_center_hz = center_hz
        low = max(BASE_LOW_HZ, center_hz - PEAK_FILTER_HALF_BW_HZ)
        high = min(BASE_HIGH_HZ, center_hz + PEAK_FILTER_HALF_BW_HZ)
        if high - low < 0.5:
            high = low + 0.5
        self._peak_sos = design_bandpass_sos(self._configured_rate, low, high)
        self._peak_zi = sosfilt_zi(self._peak_sos)

    @staticmethod
    def _vibration_scalar(hx: float, hy: float, hz: float) -> float:
        return float(np.sqrt(hx * hx + hy * hy + hz * hz))

    def _process_vibration(self, x: float, y: float, z: float) -> float:
        """原始计数 → m/s² → 高通 → 带通 → 保存各频段加速度。"""
        x, y, z = sensor_units_to_m_s2(x, y, z)
        hx = self._hp_x.process(x)
        hy = self._hp_y.process(y)
        hz = self._hp_z.process(z)
        self._last_axis_accel["x"] = hx
        self._last_axis_accel["y"] = hy
        self._last_axis_accel["z"] = hz
        vib = self._vibration_scalar(hx, hy, hz)
        self._vibration_buf.append(vib)
        y_base, self._base_zi = sosfilt(self._base_sos, [vib], zi=self._base_zi)
        self._last_base = float(y_base[0])
        self._base_buf.append(self._last_base)
        for name in OSC_BANDS:
            y_band, self._band_zi[name] = sosfilt(
                self._band_sos[name], y_base, zi=self._band_zi[name]
            )
            accel = float(y_band[0])
            self._last_band_accel[name] = accel
        y_peak, self._peak_zi = sosfilt(
            self._peak_sos, [self._last_base], zi=self._peak_zi
        )
        self._last_peak_accel = float(y_peak[0])
        return vib

    def _maybe_update_dominant_band(self) -> None:
        """对 0.5–40 Hz 带通后的信号做 Welch，估计主振频率并更新 M_Fre 窄带滤波器。"""
        self._dom_update_counter += 1
        if self._dom_update_counter < self._dom_update_interval:
            return
        self._dom_update_counter = 0
        buf = np.asarray(self._base_buf, dtype=np.float64)
        min_len = max(64, int(self._configured_rate * 0.25))
        if len(buf) < min_len:
            return
        seg = buf[-min(len(buf), int(self._configured_rate * 0.5)) :]
        nperseg = min(len(seg), max(64, len(seg) // 2))
        freqs, psd = welch(seg, fs=self._configured_rate, nperseg=nperseg)
        mask = (freqs >= BASE_LOW_HZ) & (freqs <= BASE_HIGH_HZ)
        if not np.any(mask):
            return
        f_band = freqs[mask]
        p_band = psd[mask]
        peak_idx = int(np.argmax(p_band))
        self._dominant_freq_hz = float(f_band[peak_idx])
        self._dominant_band = _band_for_frequency(self._dominant_freq_hz)
        self._rebuild_peak_filter(self._dominant_freq_hz)

    def push(self, x: float, y: float, z: float, band: Optional[str] = None) -> float:
        """
        送入 MCU 原始计数 (x,y,z)。
        band=None 返回合成加速度 (m/s²)；否则返回对应频段窄带加速度 (m/s²)。
        """
        vib = self._process_vibration(x, y, z)
        self._maybe_update_dominant_band()
        if band is None:
            return vib
        if band not in self._last_band_accel:
            raise ValueError(f"未知节律: {band!r}")
        accel = self._last_band_accel[band]
        self._last_band_outputs[band] = accel
        return accel

    def push_axes(self, x: float, y: float, z: float) -> Dict[str, float]:
        """处理一点并返回三轴高通加速度 (m/s²)。"""
        self._process_vibration(x, y, z)
        return self.axis_accel

    def push_m_freq(self, x: float, y: float, z: float) -> float:
        """返回主振频率窄带加速度 (m/s²)。"""
        self._process_vibration(x, y, z)
        self._maybe_update_dominant_band()
        self._last_m_freq_accel = self._last_peak_accel
        return self._last_m_freq_accel

    def recent_vibration(self) -> np.ndarray:
        return np.asarray(self._vibration_buf, dtype=np.float64)


def acc_batch_to_arrays(
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x,y,z) 序列 → 三轴 numpy 数组。"""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points 须为 N×3 的 (x,y,z) 序列")
    return arr[:, 0], arr[:, 1], arr[:, 2]


def preprocess_vibration(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """
    原始计数 → m/s² → 高通 → 合成 → 0.5–40 Hz 带通。
    """
    x = sensor_array_to_m_s2(x)
    y = sensor_array_to_m_s2(y)
    z = sensor_array_to_m_s2(z)
    hx = _highpass_offline(x, sample_rate, GRAVITY_HP_HZ)
    hy = _highpass_offline(y, sample_rate, GRAVITY_HP_HZ)
    hz = _highpass_offline(z, sample_rate, GRAVITY_HP_HZ)
    vib = np.sqrt(hx * hx + hy * hy + hz * hz)
    return _bandpass_offline(vib, sample_rate, BASE_LOW_HZ, BASE_HIGH_HZ)


def extract_band_waveforms(
    vibration: np.ndarray,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> Dict[str, np.ndarray]:
    """各节律带通加速度 → 按主振频率二次积分 → 位移波形 (mm)。"""
    accel_bands = {
        name: _bandpass_offline(vibration, sample_rate, low, high)
        for name, (low, high) in OSC_BANDS.items()
    }
    _, dom_freqs = sliding_dominant_frequency(vibration, sample_rate)
    dom_freq = (
        float(np.median(dom_freqs))
        if len(dom_freqs) > 0
        else band_center_hz("delta")
    )
    return {
        name: displacement_m_to_display(
            accel_to_displacement(
                accel,
                sample_rate,
                freq_hz=scale_freq_for_band(name, dom_freq),
            )
        )
        for name, accel in accel_bands.items()
    }


def extract_peak_displacement_waveform(
    vibration: np.ndarray,
    sample_rate: float,
    center_hz: float,
) -> np.ndarray:
    """主振频率窄带加速度 → 位移。"""
    low = max(BASE_LOW_HZ, center_hz - PEAK_FILTER_HALF_BW_HZ)
    high = min(BASE_HIGH_HZ, center_hz + PEAK_FILTER_HALF_BW_HZ)
    if high - low < 0.5:
        high = low + 0.5
    accel = _bandpass_offline(vibration, sample_rate, low, high)
    return displacement_m_to_display(
        accel_to_displacement(accel, sample_rate, freq_hz=center_hz)
    )


def hilbert_envelopes(waveforms: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """对各节律波形做 Hilbert 变换，取包络 |analytic| → 该频段振动强度随时间。"""
    return {
        name: np.abs(hilbert(wave))
        for name, wave in waveforms.items()
    }


def _band_power_from_psd(
    freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float
) -> float:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def sliding_dominant_frequency(
    vibration: np.ndarray,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    window_seconds: float = 0.5,
    hop_seconds: Optional[float] = None,
    fmin: float = BASE_LOW_HZ,
    fmax: float = BASE_HIGH_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    滑窗 Welch 估计主振频率 f_dom(t)。
    返回 (窗中心时间秒, 主振频率 Hz)。
    比 Hilbert 瞬时频率更适合多频段叠加的振动信号。
    """
    n = len(vibration)
    if n < 8:
        return np.array([]), np.array([])

    win = max(8, int(window_seconds * sample_rate))
    hop = max(1, int((hop_seconds or window_seconds * 0.25) * sample_rate))
    times: List[float] = []
    freqs_out: List[float] = []

    for start in range(0, n - win + 1, hop):
        seg = vibration[start : start + win]
        nperseg = min(len(seg), max(win // 2, 64))
        freqs, psd = welch(seg, fs=sample_rate, nperseg=nperseg)
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            continue
        f_band = freqs[mask]
        p_band = psd[mask]
        peak_idx = int(np.argmax(p_band))
        center_t = (start + win * 0.5) / sample_rate
        times.append(center_t)
        freqs_out.append(float(f_band[peak_idx]))

    return np.asarray(times), np.asarray(freqs_out)


def build_dominant_waveform(
    band_waveforms: Dict[str, np.ndarray],
    sample_rate: float,
    window_seconds: float = 0.5,
    hop_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    按滑窗功率最大的节律，拼接「主振频带时序波形」。
    返回 (与 vibration 等长的波形, 每采样点所属主振节律名列表的稀疏表示用 band_names_per_hop)。
    实现：每个 hop 段内，选 PSD 功率最大的节律带，取该带 band_waveforms 对应片段。
    """
    vib = next(iter(band_waveforms.values()))
    n = len(vib)
    if n == 0:
        return np.array([]), np.array([]), []

    win = max(8, int(window_seconds * sample_rate))
    hop = max(1, int((hop_seconds or window_seconds * 0.25) * sample_rate))
    out = np.zeros(n, dtype=np.float64)
    band_names: List[str] = []

    for start in range(0, n - win + 1, hop):
        end = min(start + win, n)
        seg_len = end - start
        best_name = "alpha"
        best_power = -1.0
        for name, (low, high) in OSC_BANDS.items():
            seg = band_waveforms[name][start:end]
            if len(seg) < 8:
                continue
            nperseg = min(len(seg), max(seg_len // 2, 32))
            freqs, psd = welch(seg, fs=sample_rate, nperseg=nperseg)
            power = _band_power_from_psd(freqs, psd, low, high)
            if power > best_power:
                best_power = power
                best_name = name
        out[start:end] = band_waveforms[best_name][start:end]
        band_names.append(best_name)

    times = np.arange(n, dtype=np.float64) / sample_rate
    return out, times, band_names


def process_acc_batch(
    points: Sequence[Tuple[float, float, float]],
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    window_seconds: float = 0.5,
) -> OscBatchResult:
    """
    处理 oscillator_serial 一批 100 点 (x,y,z)：
      振动标量 → 节律波形 → Hilbert 包络 → 主振频率/主振波形
    """
    x, y, z = acc_batch_to_arrays(points)
    vibration = preprocess_vibration(x, y, z, sample_rate)
    band_waves = extract_band_waveforms(vibration, sample_rate)
    envelopes = hilbert_envelopes(band_waves)
    dom_times, dom_freqs = sliding_dominant_frequency(
        vibration, sample_rate, window_seconds=window_seconds
    )
    dom_bands = np.array([_band_for_frequency(f) for f in dom_freqs], dtype=object)
    dom_wave, _, _ = build_dominant_waveform(
        band_waves, sample_rate, window_seconds=window_seconds
    )
    return OscBatchResult(
        vibration=vibration,
        band_waveforms=band_waves,
        band_envelopes=envelopes,
        dominant_freq_hz=dom_freqs,
        dominant_band=dom_bands,
        dominant_waveform=dom_wave,
        sample_rate=sample_rate,
    )


def process_acc_stream_batches(
    batches: Sequence[Sequence[Tuple[float, float, float]]],
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> OscBatchResult:
    """多批串联为一段连续信号后再分析（适合缓冲若干批后画图）。"""
    all_points: List[Tuple[float, float, float]] = []
    for batch in batches:
        all_points.extend(batch)
    return process_acc_batch(all_points, sample_rate=sample_rate)
