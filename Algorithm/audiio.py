"""
立体声音频线输出：左/右声道独立正弦波，频率与相位可配置。

典型用法::

    from audiio import StereoSineAudioOutput, phase_from_degrees

    audio = StereoSineAudioOutput(sample_rate=44100)
    audio.set_left(frequency_hz=10.0, phase_rad=0.0, amplitude=0.3)
    audio.set_right(frequency_hz=10.0, phase_rad=phase_from_degrees(90), amplitude=0.3)
    audio.start()
    ...
    audio.stop()

依赖: sounddevice, numpy
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

DEFAULT_SAMPLE_RATE = 44100.0
DEFAULT_BLOCKSIZE = 1024
DEFAULT_AMPLITUDE = 0.3
MAX_AMPLITUDE = 1.0


def _require_sounddevice() -> None:
    if sd is None:
        raise ImportError(
            "sounddevice is required: pip install sounddevice"
        ) from _SOUNDDEVICE_IMPORT_ERROR


def phase_from_degrees(degrees: float) -> float:
    """角度 → 弧度，便于设置相位。"""
    return float(np.deg2rad(degrees))


def phase_from_radians(radians: float) -> float:
    """规范化相位到 [0, 2π)。"""
    twopi = 2.0 * np.pi
    return float(radians % twopi)


@dataclass
class SineChannelState:
    """单声道正弦波状态。"""

    frequency_hz: float = 440.0
    phase_rad: float = 0.0  ## 当前相位累加器 (rad)
    amplitude: float = DEFAULT_AMPLITUDE
    enabled: bool = True


class StereoSineAudioOutput:
    """
    通过音频接口输出立体声正弦波。
    左声道 → 左耳/左通道；右声道 → 右耳/右通道。
    """

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        device: Optional[int | str] = None,
    ) -> None:
        _require_sounddevice()
        if sample_rate <= 0:
            raise ValueError(f"sample_rate 须 > 0，当前为 {sample_rate}")
        if blocksize <= 0:
            raise ValueError(f"blocksize 须 > 0，当前为 {blocksize}")

        self._sample_rate = float(sample_rate)
        self._blocksize = int(blocksize)
        self._device = device
        self._lock = threading.Lock()
        self._left = SineChannelState()
        self._right = SineChannelState()
        self._stream: Optional[sd.OutputStream] = None

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    def set_left(
        self,
        *,
        frequency_hz: Optional[float] = None,
        phase_rad: Optional[float] = None,
        phase_deg: Optional[float] = None,
        amplitude: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """配置左声道。phase_deg 与 phase_rad 二选一；给定相位会重置该声道相位。"""
        self._apply_channel(
            self._left,
            frequency_hz=frequency_hz,
            phase_rad=phase_rad,
            phase_deg=phase_deg,
            amplitude=amplitude,
            enabled=enabled,
        )

    def set_right(
        self,
        *,
        frequency_hz: Optional[float] = None,
        phase_rad: Optional[float] = None,
        phase_deg: Optional[float] = None,
        amplitude: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """配置右声道。"""
        self._apply_channel(
            self._right,
            frequency_hz=frequency_hz,
            phase_rad=phase_rad,
            phase_deg=phase_deg,
            amplitude=amplitude,
            enabled=enabled,
        )

    def set_stereo(
        self,
        *,
        left_frequency_hz: Optional[float] = None,
        right_frequency_hz: Optional[float] = None,
        left_phase_rad: Optional[float] = None,
        right_phase_rad: Optional[float] = None,
        left_phase_deg: Optional[float] = None,
        right_phase_deg: Optional[float] = None,
        left_amplitude: Optional[float] = None,
        right_amplitude: Optional[float] = None,
        left_enabled: Optional[bool] = None,
        right_enabled: Optional[bool] = None,
    ) -> None:
        """同时配置左右声道。"""
        self.set_left(
            frequency_hz=left_frequency_hz,
            phase_rad=left_phase_rad,
            phase_deg=left_phase_deg,
            amplitude=left_amplitude,
            enabled=left_enabled,
        )
        self.set_right(
            frequency_hz=right_frequency_hz,
            phase_rad=right_phase_rad,
            phase_deg=right_phase_deg,
            amplitude=right_amplitude,
            enabled=right_enabled,
        )

    def get_left(self) -> SineChannelState:
        with self._lock:
            return SineChannelState(
                frequency_hz=self._left.frequency_hz,
                phase_rad=self._left.phase_rad,
                amplitude=self._left.amplitude,
                enabled=self._left.enabled,
            )

    def get_right(self) -> SineChannelState:
        with self._lock:
            return SineChannelState(
                frequency_hz=self._right.frequency_hz,
                phase_rad=self._right.phase_rad,
                amplitude=self._right.amplitude,
                enabled=self._right.enabled,
            )

    def start(self) -> None:
        """开始输出。若已在播放则忽略。"""
        if self.is_playing:
            return
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self._blocksize,
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """停止输出并释放音频流。"""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def __enter__(self) -> StereoSineAudioOutput:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def _apply_channel(
        self,
        state: SineChannelState,
        *,
        frequency_hz: Optional[float],
        phase_rad: Optional[float],
        phase_deg: Optional[float],
        amplitude: Optional[float],
        enabled: Optional[bool],
    ) -> None:
        if phase_rad is not None and phase_deg is not None:
            raise ValueError("phase_rad 与 phase_deg 不能同时指定")
        with self._lock:
            if frequency_hz is not None:
                if frequency_hz < 0:
                    raise ValueError(f"frequency_hz 须 >= 0，当前为 {frequency_hz}")
                state.frequency_hz = float(frequency_hz)
            if phase_rad is not None:
                state.phase_rad = phase_from_radians(phase_rad)
            elif phase_deg is not None:
                state.phase_rad = phase_from_radians(phase_from_degrees(phase_deg))
            if amplitude is not None:
                state.amplitude = float(
                    max(0.0, min(MAX_AMPLITUDE, amplitude))
                )
            if enabled is not None:
                state.enabled = bool(enabled)

    def _render_channel(
        self,
        state: SineChannelState,
        frames: int,
    ) -> np.ndarray:
        if not state.enabled or state.amplitude <= 0.0 or state.frequency_hz <= 0.0:
            state.phase_rad = phase_from_radians(state.phase_rad)
            return np.zeros(frames, dtype=np.float32)

        omega = 2.0 * np.pi * state.frequency_hz / self._sample_rate
        phases = state.phase_rad + omega * np.arange(frames, dtype=np.float64)
        samples = (state.amplitude * np.sin(phases)).astype(np.float32)
        state.phase_rad = phase_from_radians(float(phases[-1] + omega))
        return samples

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[audiio] {status}")  ## 欠载/过载提示

        with self._lock:
            left = self._render_channel(self._left, frames)
            right = self._render_channel(self._right, frames)

        outdata[:, 0] = left
        outdata[:, 1] = right


class PinkNoiseBurstOutput:
    """
    持久立体声 OutputStream + burst 队列。
    主线程仅 enqueue 预生成缓冲，PortAudio 回调线程负责播放，避免每次 sd.play 开流卡顿。
    """

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        blocksize: int = 256,
        device: Optional[int | str] = None,
        max_queued_bursts: int = 2,
    ) -> None:
        _require_sounddevice()
        self._sample_rate = float(sample_rate)
        self._blocksize = max(64, int(blocksize))
        self._device = device
        self._max_queued = max(1, int(max_queued_bursts))
        self._lock = threading.Lock()
        self._queue: Deque[np.ndarray] = deque(maxlen=self._max_queued)
        self._current: Optional[np.ndarray] = None
        self._pos = 0
        self._stream: Optional[sd.OutputStream] = None
        self._dropped = 0

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    def start(self) -> None:
        if self.is_running:
            return
        stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=2,
            dtype="float32",
            blocksize=self._blocksize,
            device=self._device,
            callback=self._audio_callback,
        )
        stream.start()
        with self._lock:
            self._stream = stream

    def stop(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._queue.clear()
            self._current = None
            self._pos = 0
        if stream is not None:
            stream.stop()
            stream.close()

    def queue_burst(self, stereo: np.ndarray) -> bool:
        """入队预生成 burst；队列满时丢弃并返回 False。"""
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError("burst 须为 (frames, 2) float32 数组")
        with self._lock:
            if len(self._queue) >= self._max_queued:
                self._dropped += 1
                return False
            self._queue.append(stereo)
            return True

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[audiio burst] {status}")

        outdata.fill(0.0)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._current is None:
                    if not self._queue:
                        break
                    self._current = self._queue.popleft()
                    self._pos = 0
                remaining = len(self._current) - self._pos
                if remaining <= 0:
                    self._current = None
                    continue
                n = min(frames - filled, remaining)
                outdata[filled : filled + n] = self._current[self._pos : self._pos + n]
                self._pos += n
                filled += n
                if self._pos >= len(self._current):
                    self._current = None


def make_alert_chime(
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.28,
) -> np.ndarray:
    """生成短促双音提示（叮—咚），形状 (frames, 2)。"""
    tone_ms = (180.0, 220.0)
    gap_ms = 60.0
    freqs_hz = (880.0, 1174.7)
    chunks: List[np.ndarray] = []
    for i, (freq, dur_ms) in enumerate(zip(freqs_hz, tone_ms)):
        frames = max(1, int(sample_rate * dur_ms / 1000.0))
        t = np.arange(frames, dtype=np.float64) / sample_rate
        env = np.hanning(frames)
        mono = (amplitude * env * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        chunks.append(np.column_stack([mono, mono]))
        if i + 1 < len(freqs_hz):
            gap_frames = max(1, int(sample_rate * gap_ms / 1000.0))
            chunks.append(np.zeros((gap_frames, 2), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def play_alert_chime(
    *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.28,
    device: Optional[int | str] = None,
    blocking: bool = False,
) -> bool:
    """
    播放测试结束提示音。
    默认非阻塞（后台线程），避免卡住 UI；失败时返回 False。
    """
    if sd is None:
        return False

    waveform = make_alert_chime(sample_rate=sample_rate, amplitude=amplitude)

    def _play() -> None:
        import time as _time

        # 给前一个 OutputStream（助眠 burst/连续音频）释放声卡留出一点时间
        _time.sleep(0.15)
        try:
            sd.play(waveform, samplerate=sample_rate, device=device, blocking=True)
            sd.wait()
        except Exception as exc:  # pragma: no cover
            print(f"[audiio alert] 播放失败: {exc}")

    if blocking:
        _play()
        return True
    threading.Thread(target=_play, name="alert-chime", daemon=True).start()
    return True


def list_audio_devices() -> List[str]:
    """列出可用音频输出设备，便于选择 device 参数。"""
    _require_sounddevice()
    lines: List[str] = []
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] >= 2:
            name = dev["name"]
            host = sd.query_hostapis(dev["hostapi"])["name"]
            lines.append(f"[{idx}] {name} ({host}, out={dev['max_output_channels']})")
    return lines


def default_output_device() -> Tuple[Optional[int], str]:
    """返回 (device_index, device_name)。"""
    _require_sounddevice()
    dev = sd.query_devices(kind="output")
    return dev.get("index"), str(dev.get("name", ""))


def generate_stereo_sine_block(
    *,
    sample_rate: float,
    frames: int,
    left_frequency_hz: float = 440.0,
    right_frequency_hz: float = 440.0,
    left_phase_rad: float = 0.0,
    right_phase_rad: float = 0.0,
    left_amplitude: float = DEFAULT_AMPLITUDE,
    right_amplitude: float = DEFAULT_AMPLITUDE,
    start_sample: int = 0,
) -> np.ndarray:
    """
    离线生成一段立体声正弦波 (frames, 2)，用于测试或保存 WAV。
    start_sample 为全局起始采样序号，用于拼接连续相位。
    """
    t = (start_sample + np.arange(frames, dtype=np.float64)) / sample_rate
    left = left_amplitude * np.sin(
        2.0 * np.pi * left_frequency_hz * t + left_phase_rad
    )
    right = right_amplitude * np.sin(
        2.0 * np.pi * right_frequency_hz * t + right_phase_rad
    )
    return np.column_stack([left, right]).astype(np.float32)


if __name__ == "__main__":
    import time as _time

    print("可用立体声输出设备:")
    for line in list_audio_devices():
        print(" ", line)
    idx, name = default_output_device()
    print(f"默认输出: [{idx}] {name}")

    player = StereoSineAudioOutput(sample_rate=44100, device=idx)
    player.set_stereo(
        left_frequency_hz=100.0,
        right_frequency_hz=100.0,
        left_phase_deg=0.0,
        right_phase_deg=90.0,
        left_amplitude=0.25,
        right_amplitude=0.25,
    )
    print("播放 5 秒: 左 10 Hz 0°, 右 10 Hz 90°")
    with player:
        _time.sleep(5.0)
    print("结束")
