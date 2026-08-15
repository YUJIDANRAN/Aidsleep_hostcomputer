"""
三轴加速度传感器串口协议（与 IGT6 STM32 固件双缓冲发送格式一致）。

上电后 MCU 先发一行握手：
  "IGT6 UART OK\\r\\n"

之后每次 DMA 发送一整块缓冲区（BUFFER_SIZE=100 点）：
  每行: "%ld,%ld,%ld\\r\\n"  → x,y,z（int32，单位 mg）
  块末: "---\\r\\n"

对应固件变量：
  AccDataPoint_t buffer_A/B[BUFFER_SIZE]
  read_buffer → (int32_t) 取整 → snprintf 逐行发送 → HAL_UART_Transmit_DMA
"""

from __future__ import annotations  ## 前向引用类型

import threading
import time  ## 阻塞读超时
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

BUFFER_SIZE = 100  ## 与固件 BUFFER_SIZE 一致，每批 100 点
BATCH_SEPARATOR = "---"  ## 固件块末分隔符（不含 \\r\\n）
UART_READY_MARKER = "IGT6 UART OK"  ## 固件上电握手行
DEFAULT_BAUDRATE = 115200  ## USART1 默认波特率
POLL_READ_MAX_BYTES = 4096  ## 单次 poll 最多读取字节数
MAX_PARSER_BUF = 16384  ## 文本解析缓冲上限，防止异常数据撑爆内存


@dataclass(frozen=True)
class AccDataPoint:
    """单点三轴加速度 (mg)，对应 AccDataPoint_t { x, y, z } 经 (int32_t) 取整后发送。"""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class AccDataBatch:
    """一帧完整缓冲区数据，对应 read_buffer 中的一次 DMA 发送。"""

    points: tuple[AccDataPoint, ...]

    @property
    def size(self) -> int:
        return len(self.points)


class AccStreamParser:
    """
    解析 MCU 文本流：
      IGT6 UART OK
      12,34,1005
      ...
      (共 100 行，整数 mg)
      ---
    支持跨包拆行：半行留在 _buf，下次 feed_many 继续拼接。
    """

    def __init__(self, expected_size: int = BUFFER_SIZE) -> None:
        self._expected_size = expected_size  ## 期望每批点数，默认 100
        self.reset()

    def reset(self) -> None:
        self._buf = ""  ## 未完成的行片段
        self._pending: List[AccDataPoint] = []  ## 当前批次已解析的点

    def _trim_buffer_if_needed(self) -> None:
        """缓冲过长时丢弃旧数据，尽量从最近的分隔符处重新同步。"""
        if len(self._buf) <= MAX_PARSER_BUF:
            return
        sep = self._buf.rfind(BATCH_SEPARATOR)
        if sep >= 0:
            self._buf = self._buf[sep:]  ## 保留最后一个 --- 之后的内容
        else:
            self._buf = self._buf[-MAX_PARSER_BUF // 2 :]  ## 找不到分隔符则截断
        self._pending.clear()  ## 丢弃未完成的批次，避免脏数据

    @staticmethod
    def _parse_point(line: str) -> Optional[AccDataPoint]:
        """解析一行 'x,y,z'（整数 mg，与固件 %ld 输出一致），格式非法则返回 None。"""
        parts = line.strip().split(",")
        if len(parts) != 3:
            return None
        try:
            x, y, z = (int(p.strip(), 10) for p in parts)
        except ValueError:
            return None
        return AccDataPoint(x=float(x), y=float(y), z=float(z))

    def feed_many(self, data: bytes | str) -> List[AccDataBatch]:
        """喂入串口原始数据，返回本次解析出的完整批次列表。"""
        if isinstance(data, bytes):
            text = data.decode("ascii", errors="ignore")  ## MCU 发送 ASCII 文本
        else:
            text = data
        if not text:
            return []

        self._buf += text.replace("\r\n", "\n").replace("\r", "\n")  ## 统一换行符
        self._trim_buffer_if_needed()

        batches: List[AccDataBatch] = []
        while True:
            nl = self._buf.find("\n")
            if nl < 0:
                break  ## 没有完整行，等待下次数据

            line = self._buf[:nl]
            self._buf = self._buf[nl + 1 :]

            stripped = line.strip()
            if not stripped:
                continue  ## 跳过空行
            if stripped == UART_READY_MARKER:
                continue  ## 固件上电握手，非采样数据
            if stripped == BATCH_SEPARATOR:
                if self._pending:
                    batches.append(AccDataBatch(points=tuple(self._pending)))
                    self._pending.clear()
                continue  ## 块结束，与固件 ---\\r\\n 对应

            point = self._parse_point(stripped)
            if point is not None:
                self._pending.append(point)
                if len(self._pending) >= self._expected_size:
                    batches.append(AccDataBatch(points=tuple(self._pending)))
                    self._pending.clear()  ## 满 100 点也组批，防止分隔符丢失

        return batches


class OscillatorSerial:
    """串口封装：打开端口、轮询读取、解析为三轴加速度批次。"""

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = 0.1,
        batch_size: int = BUFFER_SIZE,
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
        self.batch_size = batch_size
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self._ser: Optional[Serial] = None
        self._parser = AccStreamParser(expected_size=batch_size)
        self._io_lock = threading.Lock()
        self.on_batch: Optional[Callable[[AccDataBatch], None]] = None  ## 收到整批时的回调
        self.bytes_received = 0  ## 累计接收字节数（调试用）
        self._recent_raw = bytearray()  ## 最近原始字节，peek 调试用

    def open(self) -> None:
        if self._ser and self._ser.is_open:
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
        self._ser.dtr = False  ## 避免部分 USB 转串口误触发复位
        self._ser.rts = False
        self._ser.reset_input_buffer()
        self._parser.reset()
        self.bytes_received = 0
        self._recent_raw.clear()

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def __enter__(self) -> OscillatorSerial:
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
        with self._io_lock:
            waiting = self._ser.in_waiting
            if waiting:
                self._ser.read(waiting)
            self._parser.reset()
            return waiting

    def discard_pending_input(self) -> int:
        """暂停采集时调用：读空硬件缓冲并重置解析状态。"""
        return self.flush_input_buffer()

    def feed_bytes(self, data: bytes) -> List[AccDataBatch]:
        """将原始字节送入解析器，可选触发 on_batch 回调。"""
        self.bytes_received += len(data)
        self._recent_raw.extend(data)
        if len(self._recent_raw) > 128:
            del self._recent_raw[:-128]  ## 只保留最近 128 字节供调试
        batches = self._parser.feed_many(data)
        if self.on_batch:
            for batch in batches:
                self.on_batch(batch)
        return batches

    def drain_available(self) -> List[AccDataBatch]:
        """Read every waiting byte without blocking. Safe to call from a worker thread."""
        ser = self._require_open()
        with self._io_lock:
            chunks: List[bytes] = []
            while True:
                waiting = ser.in_waiting
                if not waiting:
                    break
                chunk = ser.read(waiting)
                if not chunk:
                    break
                chunks.append(chunk)
            if not chunks:
                return []
            return self.feed_bytes(b"".join(chunks))

    def poll_once(self, max_reads: int = 4) -> List[AccDataBatch]:
        return self.drain_available()

    def peek_recent_raw(self, nbytes: int = 64) -> str:
        """查看最近收到的原始 ASCII，串口调试时用。"""
        data = bytes(self._recent_raw[-nbytes:])
        if not data:
            return "(空)"
        try:
            return data.decode("ascii", errors="replace")
        except Exception:
            return data.hex(" ")

    def read_batch(self, timeout: Optional[float] = None) -> AccDataBatch:
        """阻塞直到收到下一批数据，超时抛 TimeoutError。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for batch in self.poll_once():
                return batch
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("no accelerometer batch received within timeout")
            if self._ser and self._ser.timeout is not None:
                chunk = self._ser.read(1)
                if chunk:
                    batches = self.feed_bytes(chunk)
                    if batches:
                        return batches[0]
            else:
                time.sleep(0.001)

    def iter_batches(self, timeout: Optional[float] = None) -> Iterator[AccDataBatch]:
        """无限迭代，逐批 yield。"""
        while True:
            yield self.read_batch(timeout=timeout)


def list_ports() -> List[str]:
    """列出本机可用串口，供 UI 下拉选择。"""
    if Serial is None:
        raise ImportError("pyserial is required: pip install pyserial") from _SERIAL_IMPORT_ERROR
    from serial.tools import list_ports as lp

    lines: List[str] = []
    for p in lp.comports():
        desc = p.description or "未知设备"
        lines.append(f"{p.device} ({desc})")
    return lines
