"""
KS1082 串口协议（与 STM32 固件 send_data_cnt 逻辑一致）。

每次 DMA 完成发送一个采样时刻的数据：
  send_data_cnt == 0 时先发校位 0xFF，再发 CH1_L、CH1_H
  其余时刻只发 CH1_L、CH1_H
  send_data_cnt += 2，到 32 归零 → 每 16 个采样重复一次校位

CH1 数据：
  数字滤波开：Send_Filter_Data = (u16)(Filter_Data * 10000)
  数字滤波关：ADC_Buffer[0]（12 位 ADC，0~4095）
先发低字节 L，再高字节 H；若 L==0xFF 则线上发 0xFE，解码时还原。

若固件仍发送 CH2（每点再发 2 字节），将 CH2_BYTES_ON_WIRE 设为 2。
当前默认仅 CH1（CH2_BYTES_ON_WIRE=0）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

try:
    import serial
    from serial import Serial
except ImportError as exc:  # pragma: no cover
    Serial = None  # type: ignore[misc, assignment]
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None

SYNC_BYTE = 0xFF
SYNC_CNT_STEP = 2  ## 固件 send_data_cnt 每次 +2
SYNC_CNT_WRAP = 32  ## 固件 send_data_cnt 到 32 归零
SAMPLES_PER_SYNC_GROUP = SYNC_CNT_WRAP // SYNC_CNT_STEP  ## 16 点/组
CH1_BYTES_ON_WIRE = 2
CH2_BYTES_ON_WIRE = 0  ## 仅 CH1=0；若固件仍发 CH2 则改为 2
WIRE_BYTES_PER_TICK = CH1_BYTES_ON_WIRE + CH2_BYTES_ON_WIRE
DUAL_CHANNEL_COUNT = 2
MULTI_CHANNEL_COUNT = 6
MULTI_BYTES_PER_TICK = MULTI_CHANNEL_COUNT * 2
MCU_SAMPLE_RATE = 500
BYTES_PER_SECOND = int(
    MCU_SAMPLE_RATE
    * (
        1 + WIRE_BYTES_PER_TICK
        + (SAMPLES_PER_SYNC_GROUP - 1) * WIRE_BYTES_PER_TICK
    )
    / SAMPLES_PER_SYNC_GROUP
)
POLL_READ_MAX_BYTES = 8192
MAX_PARSER_BUF = 8192
LOW_BYTE_ESCAPE = 0xFE
UINT16_MASK = 0xFFFF  ## 下位机按 uint16_t 传输，可能是 ADC count，也可能是滤波/缩放后的协议值
RAW_TYPICAL_MID = 2000.0  ## raw 显示默认中心（0~4096 量程中部）
RAW_TYPICAL_AMP = 2200.0  ## raw 显示默认半幅


@dataclass(frozen=True)
class AdcSample:
    """One ADC sample tick; channels has CH1 in single mode, CH1~CH6 in multi mode."""

    channel1: int
    channel2: int = 0
    channels: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.channels:
            object.__setattr__(self, "channels", (self.channel1,))

    @staticmethod
    def decode_u16_le(data_low: int, data_high: int) -> int:
        """Decode little-endian word; 0xFE on low byte is restored to 0xFF."""
        low = data_low & 0xFF
        if low == LOW_BYTE_ESCAPE:
            low = 0xFF
        high = data_high & 0xFF
        return ((high << 8) | low) & UINT16_MASK

    @staticmethod
    def decode_ch1_le(data_low: int, data_high: int) -> int:
        return AdcSample.decode_u16_le(data_low, data_high)

    @classmethod
    def from_ch1_bytes(cls, data_low: int, data_high: int) -> "AdcSample":
        value = cls.decode_ch1_le(data_low, data_high)
        return cls(channel1=value, channels=(value,))

    @classmethod
    def from_channel_bytes(cls, payload: bytes, channel_count: int) -> "AdcSample":
        values = tuple(
            cls.decode_u16_le(payload[index * 2], payload[index * 2 + 1])
            for index in range(channel_count)
        )
        return cls(channel1=values[0] if values else 0, channels=values)


class AdcStreamParser:
    """
    按固件 send_data_cnt 解析：
      _ticks_in_group == 0 → 消费 0xFF 校位，再读 CH1（及可选 CH2 丢弃）
      _ticks_in_group == 1..15 → 直接读 CH1
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()
        self._ticks_in_group = 0  ## 0~15，对应 send_data_cnt 0,2,...,30
        self._channel_count = getattr(self, "_channel_count", 1)

    def set_channel_count(self, channel_count: int) -> None:
        """Switch between single-channel, two-channel and six-channel wire protocols."""
        if channel_count >= MULTI_CHANNEL_COUNT:
            self._channel_count = MULTI_CHANNEL_COUNT
        elif channel_count >= DUAL_CHANNEL_COUNT:
            self._channel_count = DUAL_CHANNEL_COUNT
        else:
            self._channel_count = 1
        self._buf.clear()
        self._ticks_in_group = 0

    @property
    def channel_count(self) -> int:
        return self._channel_count

    def _trim_buffer_if_needed(self) -> None:
        if len(self._buf) <= MAX_PARSER_BUF:
            return
        sync_at = self._buf.rfind(SYNC_BYTE)
        if sync_at >= 0:
            self._buf = self._buf[sync_at:]
        else:
            self._buf.clear()
        self._ticks_in_group = 0

    def feed(self, byte: int) -> Optional[AdcSample]:
        self._buf.append(byte & 0xFF)
        samples = self._parse_all()
        return samples[0] if samples else None

    def feed_many(self, data: bytes) -> List[AdcSample]:
        if not data:
            return []
        self._buf.extend(data)
        self._trim_buffer_if_needed()
        return self._parse_all()

    def _consume_ch2_if_present(self) -> None:
        if CH2_BYTES_ON_WIRE > 0 and len(self._buf) >= CH2_BYTES_ON_WIRE:
            del self._buf[:CH2_BYTES_ON_WIRE]

    def _parse_all(self) -> List[AdcSample]:
        out: List[AdcSample] = []
        while self._buf:
            bytes_per_tick = (
                self._channel_count * 2
                if self._channel_count > 1
                else CH1_BYTES_ON_WIRE + CH2_BYTES_ON_WIRE
            )
            if self._ticks_in_group == 0:
                if self._buf[0] != SYNC_BYTE:
                    del self._buf[0]
                    continue
                if len(self._buf) < 1 + bytes_per_tick:
                    break
                del self._buf[0]
            if self._channel_count > 1:
                if len(self._buf) < bytes_per_tick:
                    break
                payload = bytes(self._buf[:bytes_per_tick])
                del self._buf[:bytes_per_tick]
                out.append(AdcSample.from_channel_bytes(payload, self._channel_count))
            else:
                if len(self._buf) < CH1_BYTES_ON_WIRE:
                    break
                data_low, data_high = self._buf[0], self._buf[1]
                del self._buf[:CH1_BYTES_ON_WIRE]
                self._consume_ch2_if_present()
                out.append(AdcSample.from_ch1_bytes(data_low, data_high))
            self._ticks_in_group = (self._ticks_in_group + 1) % SAMPLES_PER_SYNC_GROUP
        return out



class Ks1082Serial:
    """串口封装：打开端口、轮询读取、解析为 AdcSample。"""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.1,
        bytesize: int = 8,
        parity: str = "N",
        stopbits: float = 1,
    ) -> None:
        if Serial is None:
            raise ImportError(
                "pyserial is required: pip install pyserial"
            ) from _SERIAL_IMPORT_ERROR
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self._ser: Optional[Serial] = None
        self._parser = AdcStreamParser()
        self._io_lock = threading.Lock()
        self._io_enabled = False
        self.on_sample: Optional[Callable[[AdcSample], None]] = None
        self.bytes_received = 0
        self._recent_raw = bytearray()

    def open(self) -> None:
        if self._ser and self._ser.is_open:
            self._io_enabled = True
            return
        self._ser = Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
        )
        self._ser.dtr = False
        self._ser.rts = False
        self._ser.reset_input_buffer()
        self._parser.reset()
        self.bytes_received = 0
        self._recent_raw.clear()
        self._io_enabled = True

    def close(self) -> None:
        # Disable I/O first so any in-flight drain exits quickly, then close under lock.
        self._io_enabled = False
        with self._io_lock:
            ser = self._ser
            self._ser = None
            if ser is not None:
                try:
                    cancel_read = getattr(ser, "cancel_read", None)
                    if callable(cancel_read):
                        cancel_read()
                except Exception:
                    pass
                try:
                    if ser.is_open:
                        ser.reset_input_buffer()
                except Exception:
                    pass
                try:
                    if ser.is_open:
                        ser.close()
                except Exception:
                    pass
            self._parser.reset()

    def ensure_io_enabled(self) -> None:
        """Re-enable reads while the port remains open (e.g. after stop→start)."""
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._io_enabled = True
            except Exception:
                pass

    def __enter__(self) -> Ks1082Serial:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        ser = self._ser
        if ser is None:
            return False
        try:
            return bool(ser.is_open)
        except Exception:
            return False

    def _require_open(self) -> Serial:
        if not self.is_open or self._ser is None:
            raise RuntimeError("serial port is not open")
        return self._ser

    def flush_input_buffer(self) -> int:
        """丢弃串口接收缓冲中的待发字节，避免暂停期间积压。"""
        with self._io_lock:
            ser = self._ser
            if ser is None or not ser.is_open or not self._io_enabled:
                return 0
            try:
                waiting = ser.in_waiting
                if waiting:
                    ser.read(waiting)
            except Exception:
                waiting = 0
            self._parser.reset()
            return waiting

    def discard_pending_input(self) -> int:
        """暂停采集时调用：读空硬件缓冲并重置解析状态。"""
        return self.flush_input_buffer()

    def set_channel_count(self, channel_count: int) -> None:
        with self._io_lock:
            self._parser.set_channel_count(channel_count)

    @property
    def channel_count(self) -> int:
        return self._parser.channel_count

    def feed_bytes(self, data: bytes) -> List[AdcSample]:
        self.bytes_received += len(data)
        self._recent_raw.extend(data)
        if len(self._recent_raw) > 64:
            del self._recent_raw[:-64]
        samples = self._parser.feed_many(data)
        if self.on_sample:
            for sample in samples:
                self.on_sample(sample)
        return samples

    def drain_available(self) -> List[AdcSample]:
        """Read a bounded amount of waiting bytes. Safe to call from a worker thread."""
        if not self._io_enabled:
            return []
        max_bytes = 65536
        with self._io_lock:
            if not self._io_enabled or self._ser is None or not self._ser.is_open:
                return []
            ser = self._ser
            chunks: List[bytes] = []
            total = 0
            try:
                while total < max_bytes:
                    waiting = ser.in_waiting
                    if not waiting:
                        break
                    to_read = min(waiting, max_bytes - total)
                    chunk = ser.read(to_read)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
            except Exception:
                return []
            if not chunks:
                return []
            # Parse under the same lock so flush/close cannot race the parser.
            return self.feed_bytes(b"".join(chunks))

    def poll_once(self, max_reads: int = 4) -> List[AdcSample]:
        return self.drain_available()

    def peek_recent_raw(self, nbytes: int = 16) -> str:
        data = bytes(self._recent_raw[-nbytes:])
        if not data:
            return "(空)"
        return data.hex(" ")

    def read_sample(
        self,
        timeout: Optional[float] = None,
    ) -> AdcSample:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for sample in self.poll_once():
                return sample
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("no ADC sample received within timeout")
            if self._ser and self._ser.timeout is not None:
                chunk = self._ser.read(1)
                if chunk:
                    self.feed_bytes(chunk)
            else:
                time.sleep(0.001)

    def iter_samples(self, timeout: Optional[float] = None) -> Iterator[AdcSample]:
        while True:
            yield self.read_sample(timeout=timeout)


def list_ports() -> List[str]:
    if Serial is None:
        raise ImportError("pyserial is required: pip install pyserial") from _SERIAL_IMPORT_ERROR
    from serial.tools import list_ports as lp

    lines: List[str] = []
    for p in lp.comports():
        desc = p.description or "未知设备"
        lines.append(f"{p.device} ({desc})")
    return lines
