"""
立体声正弦波输出控制：左/右声道频率、相位与播放时长；
助眠闭环：Alpha 波谷相位锁定短促粉噪 burst 刺激。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from audiio import (
    DEFAULT_AMPLITUDE,
    DEFAULT_SAMPLE_RATE,
    StereoSineAudioOutput,
    default_output_device,
)

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]

SLEEP_AID_WARMUP_SEC = 15.0
SLEEP_AID_MIN_INTERVAL_SEC = 0.5
SLEEP_AID_ERP_LATENCY_SEC = 0.062
SLEEP_AID_AUDIO_LATENCY_SEC = 0.0  ## 音频线直连振子：模拟传输可忽略（非蓝牙）
SLEEP_AID_FILTER_DELAY_SEC = 0.030
SLEEP_AID_PLAY_LATENCY = "low"  ## sounddevice 低延迟输出（声卡缓冲仍须实测标定）
SLEEP_AID_TRIGGER_TOLERANCE_SEC = 0.004
SLEEP_AID_BURST_DURATION_MS = 20.0
SLEEP_AID_BURST_AMPLITUDE = 0.3


@dataclass(frozen=True)
class StereoAudioParams:
    """一次播放所需的参数。"""

    left_frequency_hz: float
    right_frequency_hz: float
    left_phase_deg: float
    right_phase_deg: float
    duration_sec: float
    left_amplitude: float = DEFAULT_AMPLITUDE
    right_amplitude: float = DEFAULT_AMPLITUDE


class StereoAudioController:
    """封装左右声道正弦波的开始/停止与参数应用。"""

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        device: Optional[int | str] = None,
        on_stopped: Optional[Callable[[], None]] = None,
    ) -> None:
        if device is None:
            device_idx, _ = default_output_device()
            device = device_idx
        self._audio = StereoSineAudioOutput(
            sample_rate=sample_rate,
            device=device,
        )
        self._on_stopped = on_stopped
        self._lock = threading.Lock()
        self._playing = False
        self._params: Optional[StereoAudioParams] = None
        self._stop_timer: Optional[threading.Timer] = None

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def current_params(self) -> Optional[StereoAudioParams]:
        with self._lock:
            return self._params

    @staticmethod
    def parse_params(
        *,
        left_frequency_hz: str,
        right_frequency_hz: str,
        left_phase_deg: str,
        right_phase_deg: str,
        duration_sec: str,
        left_amplitude: float = DEFAULT_AMPLITUDE,
        right_amplitude: float = DEFAULT_AMPLITUDE,
    ) -> StereoAudioParams:
        """从 UI 文本解析并校验参数。"""
        try:
            left_f = float(left_frequency_hz.strip())
            right_f = float(right_frequency_hz.strip())
            left_p = float(left_phase_deg.strip())
            right_p = float(right_phase_deg.strip())
            duration = float(duration_sec.strip())
        except ValueError as exc:
            raise ValueError("频率、相位、时长须为数字") from exc

        if left_f < 0 or right_f < 0:
            raise ValueError("频率不能为负数")
        if duration < 0:
            raise ValueError("时长不能为负数")

        return StereoAudioParams(
            left_frequency_hz=left_f,
            right_frequency_hz=right_f,
            left_phase_deg=left_p,
            right_phase_deg=right_p,
            duration_sec=duration,
            left_amplitude=left_amplitude,
            right_amplitude=right_amplitude,
        )

    def start(self, params: StereoAudioParams) -> None:
        """按参数开始输出；duration_sec>0 时在后台定时自动停止。"""
        with self._lock:
            self._cancel_timer_locked()
            self._apply_params_locked(params)
            if not self._audio.is_playing:
                self._audio.start()
            self._playing = True
            self._params = params
            if params.duration_sec > 0:
                self._stop_timer = threading.Timer(
                    params.duration_sec,
                    self._auto_stop,
                )
                self._stop_timer.daemon = True
                self._stop_timer.start()

    def stop(self) -> None:
        """立即停止输出。"""
        with self._lock:
            self._stop_locked(notify=False)

    def toggle(self, params: StereoAudioParams) -> bool:
        """
        切换播放状态。
        若正在播放则停止并返回 False；否则按 params 开始并返回 True。
        """
        with self._lock:
            if self._playing:
                self._stop_locked(notify=False)
                return False
        self.start(params)
        return True

    def _apply_params_locked(self, params: StereoAudioParams) -> None:
        self._audio.set_stereo(
            left_frequency_hz=params.left_frequency_hz,
            right_frequency_hz=params.right_frequency_hz,
            left_phase_deg=params.left_phase_deg,
            right_phase_deg=params.right_phase_deg,
            left_amplitude=params.left_amplitude,
            right_amplitude=params.right_amplitude,
        )

    def _cancel_timer_locked(self) -> None:
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

    def _stop_locked(self, *, notify: bool) -> None:
        self._cancel_timer_locked()
        if self._audio.is_playing:
            self._audio.stop()
        self._playing = False
        self._params = None
        if notify and self._on_stopped is not None:
            self._on_stopped()

    def _auto_stop(self) -> None:
        with self._lock:
            self._stop_locked(notify=True)

    def shutdown(self) -> None:
        """程序退出前释放音频资源。"""
        with self._lock:
            self._stop_locked(notify=False)


@dataclass(frozen=True)
class AlphaPhaseSnapshot:
    """Alpha Hilbert 瞬时相位与下一波谷预测。"""

    ready: bool
    phase_rad: float = 0.0
    inst_freq_hz: float = 10.0
    seconds_to_trough: float = 0.0


@dataclass(frozen=True)
class SleepAidParams:
    """助眠闭环 burst 参数。"""

    warmup_sec: float = SLEEP_AID_WARMUP_SEC
    min_interval_sec: float = SLEEP_AID_MIN_INTERVAL_SEC
    erp_latency_sec: float = SLEEP_AID_ERP_LATENCY_SEC
    audio_latency_sec: float = SLEEP_AID_AUDIO_LATENCY_SEC
    filter_delay_sec: float = SLEEP_AID_FILTER_DELAY_SEC
    trigger_tolerance_sec: float = SLEEP_AID_TRIGGER_TOLERANCE_SEC
    burst_duration_ms: float = SLEEP_AID_BURST_DURATION_MS
    burst_amplitude: float = SLEEP_AID_BURST_AMPLITUDE


def make_pink_noise_burst_stereo(
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    duration_ms: float = SLEEP_AID_BURST_DURATION_MS,
    amplitude: float = SLEEP_AID_BURST_AMPLITUDE,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """生成 Hanning 包络立体声粉噪 burst，形状 (frames, 2)。"""
    frames = max(1, int(sample_rate * duration_ms / 1000.0))
    gen = rng if rng is not None else np.random.default_rng()
    white = gen.standard_normal(frames)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(frames, d=1.0 / sample_rate)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spec *= 1.0 / np.sqrt(freqs)
    mono = np.fft.irfft(spec, n=frames)
    peak = float(np.max(np.abs(mono)))
    if peak > 0.0:
        mono /= peak
    envelope = np.hanning(frames)
    mono = amplitude * envelope * mono
    stereo = np.column_stack([mono, mono]).astype(np.float32)
    return stereo


class SleepAidStimulusController:
    """
    助眠音效：按 Alpha 波谷预测发出短 burst。
    暖机期内不触发；相邻触发最小间隔 500 ms。
    """

    def __init__(
        self,
        params: Optional[SleepAidParams] = None,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        device: Optional[int | str] = None,
        on_triggered: Optional[Callable[[float], None]] = None,
    ) -> None:
        if sd is None:
            raise ImportError(
                "sounddevice is required for sleep-aid burst: pip install sounddevice"
            )
        self._params = params or SleepAidParams()
        self._sample_rate = float(sample_rate)
        if device is None:
            device_idx, _ = default_output_device()
            device = device_idx
        self._device = device
        self._on_triggered = on_triggered
        self._rng = np.random.default_rng()
        self._lock = threading.Lock()
        self._active = False
        self._started_at = 0.0
        self._last_trigger_at = 0.0
        self._trigger_count = 0

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def params(self) -> SleepAidParams:
        return self._params

    @property
    def trigger_count(self) -> int:
        with self._lock:
            return self._trigger_count

    @property
    def total_latency_sec(self) -> float:
        p = self._params
        return p.erp_latency_sec + p.audio_latency_sec + p.filter_delay_sec

    def warmup_remaining(self, now: Optional[float] = None) -> float:
        """距暖机结束剩余秒数；未激活时返回 0。"""
        with self._lock:
            if not self._active:
                return 0.0
            t = time.monotonic() if now is None else now
            return max(0.0, self._params.warmup_sec - (t - self._started_at))

    def start(self) -> None:
        with self._lock:
            self._active = True
            self._started_at = time.monotonic()
            self._last_trigger_at = 0.0
            self._trigger_count = 0

    def stop(self) -> None:
        with self._lock:
            self._active = False

    def shutdown(self) -> None:
        self.stop()

    def process_snapshot(
        self,
        snapshot: AlphaPhaseSnapshot,
        now: Optional[float] = None,
    ) -> bool:
        """
        根据相位快照判断是否触发 burst。
        当 seconds_to_trough ≈ total_latency 时立即播放。
        """
        if not snapshot.ready:
            return False
        t = time.monotonic() if now is None else now
        with self._lock:
            if not self._active:
                return False
            if t - self._started_at < self._params.warmup_sec:
                return False
            if (
                self._last_trigger_at > 0.0
                and t - self._last_trigger_at < self._params.min_interval_sec
            ):
                return False

            target = self.total_latency_sec
            if abs(snapshot.seconds_to_trough - target) > self._params.trigger_tolerance_sec:
                return False

            self._last_trigger_at = t
            self._trigger_count += 1
            count = self._trigger_count

        self._play_burst()
        if self._on_triggered is not None:
            self._on_triggered(float(count))
        return True

    def _play_burst(self) -> None:
        burst = make_pink_noise_burst_stereo(
            sample_rate=self._sample_rate,
            duration_ms=self._params.burst_duration_ms,
            amplitude=self._params.burst_amplitude,
            rng=self._rng,
        )
        sd.play(
            burst,
            samplerate=self._sample_rate,
            device=self._device,
            blocking=False,
            latency=SLEEP_AID_PLAY_LATENCY,
        )
