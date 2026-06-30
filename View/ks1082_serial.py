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

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

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
MCU_SAMPLE_RATE = 500
BYTES_PER_SECOND = int(
    MCU_SAMPLE_RATE
    * (
        1 + WIRE_BYTES_PER_TICK
        + (SAMPLES_PER_SYNC_GROUP - 1) * WIRE_BYTES_PER_TICK
    )
    / SAMPLES_PER_SYNC_GROUP
)
POLL_READ_MAX_BYTES = 512
MAX_PARSER_BUF = 8192
LOW_BYTE_ESCAPE = 0xFE
ADC_12BIT_MASK = 0x0FFF  ## 数值有效范围 0~4095
RAW_TYPICAL_MID = 2000.0  ## raw 显示默认中心（0~4096 量程中部）
RAW_TYPICAL_AMP = 2200.0  ## raw 显示默认半幅


@dataclass(frozen=True)
class AdcSample:
    """单点 CH1 采样。"""

    channel1: int
    channel2: int = 0

    @staticmethod
    def decode_ch1_le(data_low: int, data_high: int) -> int:
        """
        解码 CH1：先发 L 后发 H；0xFE 还原为低字节 0xFF。
        结果限制在 12 位 ADC / Send_Filter_Data 常见范围 0~4095。
        """
        low = data_low & 0xFF
        if low == LOW_BYTE_ESCAPE:
            low = 0xFF
        high = data_high & 0xFF
        return ((high << 8) | low) & ADC_12BIT_MASK

    @classmethod
    def from_ch1_bytes(cls, data_low: int, data_high: int) -> AdcSample:
        return cls(channel1=cls.decode_ch1_le(data_low, data_high))


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
            if self._ticks_in_group == 0:
                if self._buf[0] != SYNC_BYTE:
                    del self._buf[0]
                    continue
                del self._buf[0]
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

    def flush_input_buffer(self) -> int:
        """丢弃串口接收缓冲中的待发字节，避免暂停期间积压。"""
        if not self.is_open or self._ser is None:
            return 0
        waiting = self._ser.in_waiting
        if waiting:
            self._ser.read(waiting)
        self._parser.reset()
        return waiting

    def discard_pending_input(self) -> int:
        """暂停采集时调用：读空硬件缓冲并重置解析状态。"""
        return self.flush_input_buffer()

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


def _parse_eeg_raw_csv(path: Path) -> List[int]:
    """读取 eeg_raw.csv，返回 ch1 序列（回放固定按 500 Hz）。"""
    values: List[int] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV 文件为空")

    header = [cell.strip().lower() for cell in rows[0]]
    data_rows = rows[1:]
    ch1_key: Optional[str] = None
    for key in ("ch1_raw", "channel1", "ch1", "raw"):
        if key in header:
            ch1_key = key
            break

    if ch1_key is not None:
        ch1_idx = header.index(ch1_key)
        for row in data_rows:
            if len(row) <= ch1_idx:
                continue
            try:
                values.append(int(float(row[ch1_idx].strip())))
            except ValueError:
                continue
    else:
        for row in data_rows:
            if not row:
                continue
            try:
                values.append(int(float(row[-1].strip())))
            except ValueError:
                continue

    if not values:
        raise ValueError("CSV 中未找到有效的 CH1 数据")

    return values


class EegCsvReplay:
    """用已保存的 eeg_raw.csv 按采样率模拟串口输入。"""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = str(Path(csv_path).resolve())
        self._values: List[int] = []
        self._sample_rate = float(MCU_SAMPLE_RATE)
        self._index = 0
        self._open = False
        self._playback_origin: Optional[float] = None
        self.bytes_received = 0
        self._finished_logged = False

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def sample_count(self) -> int:
        return len(self._values)

    @property
    def finished(self) -> bool:
        return self._index >= len(self._values)

    def open(self) -> None:
        path = Path(self.csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"EEG CSV 不存在: {path}")
        self._values = _parse_eeg_raw_csv(path)
        self._sample_rate = float(MCU_SAMPLE_RATE)
        self._index = 0
        self._playback_origin = None
        self.bytes_received = 0
        self._finished_logged = False
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def reset_playback(self) -> None:
        self._index = 0
        self._playback_origin = None
        self.bytes_received = 0
        self._finished_logged = False

    def flush_input_buffer(self) -> int:
        """重置到文件开头（兼容串口 flush 接口）。"""
        flushed_samples = self._index
        self.reset_playback()
        return flushed_samples * CH1_BYTES_ON_WIRE

    def discard_pending_input(self) -> int:
        """暂停时冻结播放进度。"""
        self._playback_origin = None
        return 0

    def _ensure_playback_origin(self, now: float) -> None:
        if self._playback_origin is None:
            self._playback_origin = now - (self._index / self._sample_rate)

    def poll_once(self, max_reads: int = 4) -> List[AdcSample]:
        del max_reads  ## 与 Ks1082Serial 签名一致，CSV 回放忽略
        if not self.is_open or self.finished:
            return []
        now = time.monotonic()
        self._ensure_playback_origin(now)
        elapsed = now - self._playback_origin
        target_index = min(len(self._values), int(elapsed * self._sample_rate))
        out: List[AdcSample] = []
        while self._index < target_index:
            out.append(AdcSample(channel1=self._values[self._index]))
            self._index += 1
            self.bytes_received += CH1_BYTES_ON_WIRE
        return out


def list_ports() -> List[str]:
    if Serial is None:
        raise ImportError("pyserial is required: pip install pyserial") from _SERIAL_IMPORT_ERROR
    from serial.tools import list_ports as lp

    lines: List[str] = []
    for p in lp.comports():
        desc = p.description or "未知设备"
        lines.append(f"{p.device} ({desc})")
    return lines
