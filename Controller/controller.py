"""
立体声正弦波输出控制：左/右声道频率、相位与播放时长。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from audiio import (
    DEFAULT_AMPLITUDE,
    DEFAULT_SAMPLE_RATE,
    StereoSineAudioOutput,
    default_output_device,
)


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
