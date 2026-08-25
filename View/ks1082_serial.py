"""
KS1082 / 上位机串口协议。

单通道（HostUart_SendEeg，每采样 1 帧 9 字节）：
  AA 55 | 01 | 04 | seq_L seq_H | eeg_L eeg_H | xor
  xor = 01 ^ 04 ^ payload[0..3]
  seq / eeg 均为小端 uint16。

多通道（≥2）仍使用旧线格式：
  send_data_cnt == 0 时先发校位 0xFF，再发各通道 L/H；
  其余时刻只发通道数据；若 L==0xFF 则线上发 0xFE，解码时还原。
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

# 单通道成帧协议（与 HostUart_SendEeg 一致）
HOST_UART_SYNC0 = 0xAA
HOST_UART_SYNC1 = 0x55
HOST_UART_TYPE_EEG = 0x01
HOST_UART_LEN_EEG = 0x04
HOST_UART_FRAME_SIZE = 2 + 1 + 1 + HOST_UART_LEN_EEG + 1  ## 9

# 多通道旧协议
SYNC_BYTE = 0xFF
SYNC_CNT_STEP = 2
SYNC_CNT_WRAP = 32
SAMPLES_PER_SYNC_GROUP = SYNC_CNT_WRAP // SYNC_CNT_STEP
CH1_BYTES_ON_WIRE = 2
CH2_BYTES_ON_WIRE = 0
WIRE_BYTES_PER_TICK = CH1_BYTES_ON_WIRE + CH2_BYTES_ON_WIRE
DUAL_CHANNEL_COUNT = 2
MULTI_CHANNEL_COUNT = 6
MULTI_BYTES_PER_TICK = MULTI_CHANNEL_COUNT * 2
MCU_SAMPLE_RATE = 500
BYTES_PER_SECOND = MCU_SAMPLE_RATE * HOST_UART_FRAME_SIZE
POLL_READ_MAX_BYTES = 8192
MAX_PARSER_BUF = 8192
LOW_BYTE_ESCAPE = 0xFE
UINT16_MASK = 0xFFFF
RAW_TYPICAL_MID = 2000.0
RAW_TYPICAL_AMP = 2200.0


@dataclass(frozen=True)
class AdcSample:
    """One ADC sample tick; channels has CH1 in single mode, CH1~CH6 in multi mode."""

    channel1: int
    channel2: int = 0
    channels: Tuple[int, ...] = ()
    sequence: Optional[int] = None  ## 单通道成帧协议中的 pink_seq；多通道为 None

    def __post_init__(self) -> None:
        if not self.channels:
            object.__setattr__(self, "channels", (self.channel1,))

    @staticmethod
    def decode_u16_le(data_low: int, data_high: int, *, escape: bool = True) -> int:
        """Decode little-endian word; optional 0xFE→0xFF restore for old multi protocol."""
        low = data_low & 0xFF
        if escape and low == LOW_BYTE_ESCAPE:
            low = 0xFF
        high = data_high & 0xFF
        return ((high << 8) | low) & UINT16_MASK

    @staticmethod
    def decode_ch1_le(data_low: int, data_high: int) -> int:
        return AdcSample.decode_u16_le(data_low, data_high, escape=True)

    @classmethod
    def from_ch1_bytes(cls, data_low: int, data_high: int) -> "AdcSample":
        value = cls.decode_ch1_le(data_low, data_high)
        return cls(channel1=value, channels=(value,))

    @classmethod
    def from_framed_eeg(cls, seq: int, eeg: int) -> "AdcSample":
        value = int(eeg) & UINT16_MASK
        return cls(channel1=value, channels=(value,), sequence=int(seq) & UINT16_MASK)

    @classmethod
    def from_channel_bytes(cls, payload: bytes, channel_count: int) -> "AdcSample":
        values = tuple(
            cls.decode_u16_le(payload[index * 2], payload[index * 2 + 1], escape=True)
            for index in range(channel_count)
        )
        return cls(channel1=values[0] if values else 0, channels=values)


class AdcStreamParser:
    """
    单通道：解析 AA 55 | 01 | 04 | seq | eeg | xor。
    多通道：旧 send_data_cnt + 0xFF 校位协议。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()
        self._ticks_in_group = 0
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
        if self._channel_count == 1:
            sync_at = -1
            for index in range(len(self._buf) - 1, 0, -1):
                if self._buf[index - 1] == HOST_UART_SYNC0 and self._buf[index] == HOST_UART_SYNC1:
                    sync_at = index - 1
                    break
            if sync_at >= 0:
                self._buf = self._buf[sync_at:]
            else:
                self._buf.clear()
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

    @staticmethod
    def _framed_xor(payload: bytes) -> int:
        xorv = HOST_UART_TYPE_EEG ^ HOST_UART_LEN_EEG
        for value in payload:
            xorv ^= value
        return xorv & 0xFF

    def _parse_framed_single(self) -> List[AdcSample]:
        out: List[AdcSample] = []
        while True:
            if len(self._buf) < HOST_UART_FRAME_SIZE:
                break
            # 找帧头 AA 55
            sync_at = -1
            for index in range(len(self._buf) - 1):
                if (
                    self._buf[index] == HOST_UART_SYNC0
                    and self._buf[index + 1] == HOST_UART_SYNC1
                ):
                    sync_at = index
                    break
            if sync_at < 0:
                # 可能只剩半个同步字
                if self._buf and self._buf[-1] == HOST_UART_SYNC0:
                    self._buf = bytearray([HOST_UART_SYNC0])
                else:
                    self._buf.clear()
                break
            if sync_at > 0:
                del self._buf[:sync_at]
            if len(self._buf) < HOST_UART_FRAME_SIZE:
                break
            frame_type = self._buf[2]
            length = self._buf[3]
            payload = bytes(self._buf[4:8])
            checksum = self._buf[8]
            if (
                frame_type != HOST_UART_TYPE_EEG
                or length != HOST_UART_LEN_EEG
                or checksum != self._framed_xor(payload)
            ):
                # 坏帧：丢掉当前 AA，从下一个字节继续找头
                del self._buf[0]
                continue
            seq = AdcSample.decode_u16_le(payload[0], payload[1], escape=False)
            eeg = AdcSample.decode_u16_le(payload[2], payload[3], escape=False)
            out.append(AdcSample.from_framed_eeg(seq, eeg))
            del self._buf[:HOST_UART_FRAME_SIZE]
        return out

    def _parse_legacy_multi(self) -> List[AdcSample]:
        out: List[AdcSample] = []
        while self._buf:
            bytes_per_tick = self._channel_count * 2
            if self._ticks_in_group == 0:
                if self._buf[0] != SYNC_BYTE:
                    del self._buf[0]
                    continue
                if len(self._buf) < 1 + bytes_per_tick:
                    break
                del self._buf[0]
            if len(self._buf) < bytes_per_tick:
                break
            payload = bytes(self._buf[:bytes_per_tick])
            del self._buf[:bytes_per_tick]
            out.append(AdcSample.from_channel_bytes(payload, self._channel_count))
            self._ticks_in_group = (self._ticks_in_group + 1) % SAMPLES_PER_SYNC_GROUP
        return out

    def _parse_all(self) -> List[AdcSample]:
        if self._channel_count == 1:
            return self._parse_framed_single()
        return self._parse_legacy_multi()


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
