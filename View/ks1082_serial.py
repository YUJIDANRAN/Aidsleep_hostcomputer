"""
KS1082 ADC UART stream protocol (STM32 -> host).

Each sample sends CH1 only, low byte first:
  USART: CH1_L, CH1_H

If CH1_L would be 0xFF on the MCU, firmware sends 0xFE instead (wire never has 0xFF
in the low byte). Host decodes: value = (CH1_H << 8) | CH1_L.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

try:
    import serial
    from serial import Serial
except ImportError as exc:  # pragma: no cover
    Serial = None  # type: ignore[misc, assignment]
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None

BYTES_PER_SAMPLE = 2
POLL_READ_MAX_BYTES = 512
ADC_12BIT_MAX = 4095


@dataclass(frozen=True)
class AdcSample:
    channel1: int
    channel2: int = 0

    @staticmethod
    def _decode(low: int, high: int) -> int:
        return ((high & 0xFF) << 8) | (low & 0xFF)

    @classmethod
    def from_bytes(cls, ch1_l: int, ch1_h: int) -> AdcSample:
        return cls(channel1=cls._decode(ch1_l, ch1_h))


class AdcStreamParser:
    """Consume a continuous CH1_L, CH1_H byte stream (2 bytes per sample)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()

    def feed(self, byte: int) -> Optional[AdcSample]:
        self._buf.append(byte & 0xFF)
        return self._try_emit_one()

    def feed_many(self, data: bytes) -> List[AdcSample]:
        if not data:
            return []
        self._buf.extend(data)
        if len(self._buf) > 512:
            over = len(self._buf) - 256
            keep_odd = over % BYTES_PER_SAMPLE
            del self._buf[: over - keep_odd]
        usable = len(self._buf) - (len(self._buf) % BYTES_PER_SAMPLE)
        if usable < BYTES_PER_SAMPLE:
            return []
        raw = bytes(self._buf[:usable])
        del self._buf[:usable]
        return [
            AdcSample.from_bytes(raw[i], raw[i + 1])
            for i in range(0, usable, BYTES_PER_SAMPLE)
        ]

    def _try_emit_one(self) -> Optional[AdcSample]:
        if len(self._buf) < BYTES_PER_SAMPLE:
            return None
        sample = AdcSample.from_bytes(self._buf[0], self._buf[1])
        del self._buf[:BYTES_PER_SAMPLE]
        return sample


class Ks1082Serial:
    """Serial wrapper: read CH1 ADC samples from KS1082 stream."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.1,
    ) -> None:
        if Serial is None:
            raise ImportError(
                "pyserial is required: pip install pyserial"
            ) from _SERIAL_IMPORT_ERROR
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[Serial] = None
        self._parser = AdcStreamParser()
        self.on_sample: Optional[Callable[[AdcSample], None]] = None
        self.bytes_received = 0
        self._recent_raw = bytearray()

    def open(self) -> None:
        if self._ser and self._ser.is_open:
            return
        self._ser = Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            rtscts=False,
            dsrdtr=False,
            xonxoff=False,
        )
        # 避免 DTR/RTS 影响部分 CH340/STM32 板卡
        self._ser.dtr = False
        self._ser.rts = False
        self._ser.reset_input_buffer()
        self._parser.reset()
        self.bytes_received = 0
        self._recent_raw.clear()

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def __enter__(self) -> Ks1082Serial:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _require_open(self) -> Serial:
        if not self.is_open or self._ser is None:
            raise RuntimeError("serial port is not open")
        return self._ser

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

    def poll_once(self, max_reads: int = 4) -> List[AdcSample]:
        ser = self._require_open()
        chunks: List[bytes] = []
        for _ in range(max_reads):
            waiting = ser.in_waiting
            if waiting:
                chunk = ser.read(min(waiting, POLL_READ_MAX_BYTES))
            elif not chunks:
                chunk = ser.read(1)
            else:
                break
            if not chunk:
                break
            chunks.append(chunk)
        if not chunks:
            return []
        return self.feed_bytes(b"".join(chunks))

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
