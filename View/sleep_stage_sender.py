"""EDF 单通道按 30 秒 epoch 发送到 STM32 睡眠分期网络。"""

from __future__ import annotations

import csv
import math
import re
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
from scipy.signal import resample_poly

from analysis_plot_view import (
    _normalize_edf_signal_to_uv,
    load_eeg_file_info,
)


TARGET_FS = 100.0
EPOCH_SECONDS = 30
EPOCH_SAMPLES = 3000
DEFAULT_DATA_DIR = Path(r"D:\eeglab2026.0.0\opensource_psg\EPCTL01-2025\EPCTL01")


def parse_sleep_labels(path: Optional[Path]) -> Dict[int, str]:
    """读取“阶段 起始秒 持续秒”标签，返回 epoch 编号到阶段的映射。"""
    result: Dict[int, str] = {}
    if path is None:
        return result
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,;]+", line)
            if len(parts) < 3:
                continue
            try:
                stage = parts[0].strip()
                start_s = float(parts[1])
                duration_s = float(parts[2])
            except ValueError:
                continue
            if stage and start_s >= 0.0 and duration_s > 0.0:
                result[int(round(start_s / EPOCH_SECONDS))] = stage
    return result


def resample_to_model_rate(raw: np.ndarray, source_fs: float) -> np.ndarray:
    """整段重采样到 100 Hz，避免逐帧重采样造成边界不连续。"""
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("所选通道没有数据")
    if not math.isfinite(source_fs) or source_fs <= 0:
        raise ValueError(f"采样率无效: {source_fs}")
    if not np.all(np.isfinite(values)):
        count = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"所选通道含 {count} 个 NaN/Inf，请先处理坏段")
    if abs(source_fs - TARGET_FS) < 1.0e-6:
        return values
    ratio = Fraction(TARGET_FS / source_fs).limit_denominator(10000)
    return np.asarray(
        resample_poly(values, ratio.numerator, ratio.denominator),
        dtype=np.float64,
    )


def load_edf_epoch_range(
    path: Path, channel_index: int, first_epoch: int, last_epoch: int
) -> tuple[np.ndarray, float, str, str, int]:
    """只读取所选 epoch 范围，返回严格对齐的 100 Hz 连续信号。

    范围两端额外读取 1 秒，再在重采样后裁掉。这样既避免加载整夜通道，
    也减少 resample_poly 在所选范围边缘补零造成的瞬态。
    """
    import pyedflib

    info = load_eeg_file_info(path)
    if not info.channel_labels:
        raise ValueError("EDF 中没有通道")
    index = max(0, min(int(channel_index), len(info.channel_labels) - 1))
    source_fs = float(info.channel_rates[index])
    sample_count = int(info.channel_samples[index])
    total_epochs = int(math.floor(sample_count / source_fs / EPOCH_SECONDS))
    first = max(0, int(first_epoch))
    last = min(int(last_epoch), total_epochs - 1)
    if total_epochs <= 0 or first > last:
        raise ValueError(f"epoch 范围无效；可用范围为 0~{total_epochs - 1}")

    requested_start_s = first * EPOCH_SECONDS
    requested_end_s = (last + 1) * EPOCH_SECONDS
    file_end_s = sample_count / source_fs
    padded_start_s = max(0.0, requested_start_s - 1.0)
    padded_end_s = min(file_end_s, requested_end_s + 1.0)
    source_start = int(round(padded_start_s * source_fs))
    source_length = int(round((padded_end_s - padded_start_s) * source_fs))

    reader = pyedflib.EdfReader(str(path))
    try:
        raw = reader.readSignal(index, source_start, source_length)
        try:
            physical_unit = reader.getPhysicalDimension(index)
        except Exception:
            physical_unit = info.channel_units[index]
    finally:
        reader.close()

    raw, unit = _normalize_edf_signal_to_uv(raw, physical_unit)
    padded_100_hz = resample_to_model_rate(raw, source_fs)
    crop_start = int(round((requested_start_s - padded_start_s) * TARGET_FS))
    required = (last - first + 1) * EPOCH_SAMPLES
    signal = padded_100_hz[crop_start:crop_start + required]
    if signal.size != required:
        raise RuntimeError(
            f"重采样后点数异常：得到 {signal.size}，期望 {required}"
        )
    return signal, source_fs, info.channel_labels[index], unit, total_epochs


def parse_mcu_reply(lines: List[str]) -> Dict[str, str]:
    """把一帧多行回复整理成表格字段。"""
    result = {"ack": "", "cnn": "", "sequence": "", "stage": "", "status": ""}
    for line in lines:
        if line.startswith("ACK "):
            result["ack"] = line
        elif line.startswith("CNN "):
            result["cnn"] = line
        elif line.startswith("WAIT_SEQUENCE "):
            result["sequence"] = line
        elif line.startswith("STAGE "):
            result["stage"] = line
        elif line.startswith("ERR"):
            result["status"] = line
        elif line == "READY" and not result["status"]:
            result["status"] = "READY"
    return result


class SleepStageSendWorker(QtCore.QObject):
    """后台执行 EDF 读取、重采样和串口握手，防止界面卡顿。"""

    log = QtCore.pyqtSignal(str)
    prepared = QtCore.pyqtSignal(int, float, str, str)
    epoch_result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, *, edf_path: Path, label_path: Optional[Path],
                 channel_index: int, port: str, first_epoch: int,
                 last_epoch: int) -> None:
        super().__init__()
        self.edf_path = edf_path
        self.label_path = label_path
        self.channel_index = channel_index
        self.port = port
        self.first_epoch = first_epoch
        self.last_epoch = last_epoch
        self.stop_requested = False

    def _write_all(self, ser, payload: bytes) -> None:
        """按 512 字节分块发送，以便及时响应停止操作。"""
        offset = 0
        while offset < len(payload):
            if self.stop_requested:
                raise InterruptedError("用户停止发送")
            end = min(offset + 512, len(payload))
            written = ser.write(payload[offset:end])
            if not written:
                raise RuntimeError("串口写入了 0 字节")
            offset += int(written)
        ser.flush()

    def _read_until_ready(self, ser) -> List[str]:
        """READY 是帧间握手；收到它之前绝不发送下一帧。"""
        deadline = time.monotonic() + 45.0
        lines: List[str] = []
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise InterruptedError("用户停止发送")
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            self.log.emit(f"MCU > {line}")
            if line == "READY":
                return lines
        raise TimeoutError("等待 MCU 返回 READY 超时（45 秒）")

    @QtCore.pyqtSlot()
    def run(self) -> None:
        ser = None
        try:
            from serial import Serial

            self.log.emit(
                f"读取 EDF 通道 {self.channel_index}，epoch "
                f"{self.first_epoch}~{self.last_epoch}: {self.edf_path.name}"
            )
            signal, source_fs, channel, unit, total_epochs = load_edf_epoch_range(
                self.edf_path,
                self.channel_index,
                self.first_epoch,
                self.last_epoch,
            )
            first = max(0, self.first_epoch)
            last = min(self.last_epoch, total_epochs - 1)
            labels = parse_sleep_labels(self.label_path)
            self.prepared.emit(total_epochs, float(source_fs), channel, unit)

            ser = Serial(
                self.port, 115200, bytesize=8, parity="N", stopbits=1,
                timeout=0.2, write_timeout=2.0,
                rtscts=False, dsrdtr=False, xonxoff=False,
            )
            ser.dtr = False
            ser.rts = False
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            self.log.emit(f"串口已打开: {self.port} @ 115200 8N1")

            total = last - first + 1
            for sent, epoch_index in enumerate(range(first, last + 1), 1):
                if self.stop_requested:
                    raise InterruptedError("用户停止发送")
                begin = (epoch_index - first) * EPOCH_SAMPLES
                epoch = signal[begin:begin + EPOCH_SAMPLES]
                if epoch.size != EPOCH_SAMPLES:
                    raise RuntimeError(f"epoch {epoch_index} 不是 3000 点")
                if float(np.var(epoch)) < 1.0e-12:
                    raise ValueError(f"epoch {epoch_index} 是平坦信号")

                payload_text = "EPOCH_BEGIN\n"
                payload_text += "\n".join(f"{value:.9g}" for value in epoch)
                payload_text += "\nEPOCH_END\n"
                payload = payload_text.encode("ascii")
                label = labels.get(epoch_index, "")
                self.log.emit(
                    f"PC  > epoch={epoch_index} time={epoch_index * 30}s "
                    f"label={label or '-'} samples=3000 bytes={len(payload)}"
                )
                self._write_all(ser, payload)
                lines = self._read_until_ready(ser)
                parsed = parse_mcu_reply(lines)
                parsed.update({
                    "epoch": epoch_index,
                    "time_s": epoch_index * EPOCH_SECONDS,
                    "label": label,
                    "channel": channel,
                    "source_fs": float(source_fs),
                    "reply": "\n".join(lines),
                })
                self.epoch_result.emit(parsed)
                self.progress.emit(sent, total)
                if str(parsed["status"]).startswith("ERR"):
                    raise RuntimeError(str(parsed["status"]))

            self.finished.emit(True, f"发送完成：{total} 个 epoch")
        except InterruptedError as exc:
            self.finished.emit(False, str(exc))
        except Exception as exc:
            self.finished.emit(False, f"发送失败：{exc}")
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass


class SleepStageSenderWidget(QtWidgets.QWidget):
    """动态加入主窗口 tabWidget_wave_display 的功能页。"""

    serial_claim_requested = QtCore.pyqtSignal(str)

    def __init__(self, default_port: str = "COM6", parent=None) -> None:
        super().__init__(parent)
        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[SleepStageSendWorker] = None
        self.results: List[Dict[str, object]] = []
        self._build_ui(default_port)
        self._wire_ui()
        self.refresh_ports()
        self._set_default_paths()

    def _build_ui(self, default_port: str) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        files = QtWidgets.QGridLayout()
        self.edf_edit = QtWidgets.QLineEdit(self)
        self.edf_button = QtWidgets.QPushButton("选择 EDF...", self)
        self.label_edit = QtWidgets.QLineEdit(self)
        self.label_button = QtWidgets.QPushButton("选择标签...", self)
        files.addWidget(QtWidgets.QLabel("EDF/BDF", self), 0, 0)
        files.addWidget(self.edf_edit, 0, 1)
        files.addWidget(self.edf_button, 0, 2)
        files.addWidget(QtWidgets.QLabel("标签", self), 1, 0)
        files.addWidget(self.label_edit, 1, 1)
        files.addWidget(self.label_button, 1, 2)
        root.addLayout(files)

        options = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("读取通道", self)
        self.channel_combo = QtWidgets.QComboBox(self)
        self.channel_combo.setMinimumWidth(210)
        self.port_combo = QtWidgets.QComboBox(self)
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(120)
        self.port_combo.setEditText(default_port)
        self.port_button = QtWidgets.QPushButton("刷新串口", self)
        options.addWidget(self.load_button)
        options.addWidget(QtWidgets.QLabel("通道", self))
        options.addWidget(self.channel_combo, 1)
        options.addWidget(QtWidgets.QLabel("串口", self))
        options.addWidget(self.port_combo)
        options.addWidget(self.port_button)
        root.addLayout(options)

        send_row = QtWidgets.QHBoxLayout()
        self.first_spin = QtWidgets.QSpinBox(self)
        self.last_spin = QtWidgets.QSpinBox(self)
        for spin in (self.first_spin, self.last_spin):
            spin.setRange(0, 999999)
        self.last_spin.setValue(11)
        self.one_button = QtWidgets.QPushButton("发送当前帧", self)
        self.range_button = QtWidgets.QPushButton("连续发送范围", self)
        self.stop_button = QtWidgets.QPushButton("停止", self)
        self.stop_button.setEnabled(False)
        send_row.addWidget(QtWidgets.QLabel("起始 epoch", self))
        send_row.addWidget(self.first_spin)
        send_row.addWidget(QtWidgets.QLabel("结束 epoch", self))
        send_row.addWidget(self.last_spin)
        send_row.addWidget(self.one_button)
        send_row.addWidget(self.range_button)
        send_row.addWidget(self.stop_button)
        root.addLayout(send_row)

        self.info = QtWidgets.QLabel(
            "epoch 从 0 开始；每帧 30 秒/3000 点；收到 READY 后才发下一帧。", self
        )
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.table = QtWidgets.QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(
            ["epoch", "时间(s)", "真实标签", "CNN", "LSTM/STAGE", "状态"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(105)
        root.addWidget(self.table)

        bottom = QtWidgets.QHBoxLayout()
        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMinimumHeight(75)
        self.save_button = QtWidgets.QPushButton("保存结果 CSV...", self)
        self.clear_button = QtWidgets.QPushButton("清空结果", self)
        buttons = QtWidgets.QVBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        bottom.addWidget(self.log, 1)
        bottom.addLayout(buttons)
        root.addLayout(bottom)

    def _wire_ui(self) -> None:
        self.edf_button.clicked.connect(self.browse_edf)
        self.label_button.clicked.connect(self.browse_label)
        self.load_button.clicked.connect(self.load_channels)
        self.port_button.clicked.connect(self.refresh_ports)
        self.one_button.clicked.connect(self.send_one)
        self.range_button.clicked.connect(self.send_range)
        self.stop_button.clicked.connect(self.stop)
        self.save_button.clicked.connect(self.save_results)
        self.clear_button.clicked.connect(self.clear_results)

    def _set_default_paths(self) -> None:
        edf = DEFAULT_DATA_DIR / "EPCTL01 - fixed.edf"
        label = DEFAULT_DATA_DIR / "test1.txt"
        if edf.exists():
            self.edf_edit.setText(str(edf))
        if label.exists():
            self.label_edit.setText(str(label))

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def current_port(self) -> str:
        text = self.port_combo.currentText().strip()
        data = self.port_combo.currentData()
        if data and (text == str(data) or text.startswith(f"{data} ")):
            return str(data)
        return text.split(" ", 1)[0]

    def refresh_ports(self) -> None:
        previous = self.current_port() or "COM6"
        self.port_combo.clear()
        try:
            from serial.tools import list_ports
            for port in list_ports.comports():
                self.port_combo.addItem(
                    f"{port.device} ({port.description or '未知设备'})", port.device
                )
        except Exception as exc:
            self.append_log(f"刷新串口失败: {exc}")
        self.port_combo.setEditText(previous)

    def browse_edf(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 EDF/BDF", self.edf_edit.text() or str(DEFAULT_DATA_DIR),
            "EDF/BDF (*.edf *.bdf);;所有文件 (*)"
        )
        if path:
            self.edf_edit.setText(path)
            self.load_channels()

    def browse_label(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择睡眠标签", self.label_edit.text() or str(DEFAULT_DATA_DIR),
            "标签 (*.txt *.csv);;所有文件 (*)"
        )
        if path:
            self.label_edit.setText(path)

    def optional_label_path(self) -> Optional[Path]:
        text = self.label_edit.text().strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_file():
            raise FileNotFoundError(f"标签文件不存在: {path}")
        return path

    def load_channels(self) -> None:
        try:
            path = Path(self.edf_edit.text().strip())
            if not path.is_file():
                raise FileNotFoundError(f"EDF 文件不存在: {path}")
            info = load_eeg_file_info(path)
            self.channel_combo.clear()
            for i, name in enumerate(info.channel_labels):
                self.channel_combo.addItem(
                    f"{i}: {name}  {info.channel_rates[i]:g}Hz  "
                    f"{info.channel_units[i]}  {info.channel_samples[i]}点", i
                )
            if self.channel_combo.count() == 0:
                raise ValueError("EDF 中没有可用通道")
            available = max(
                int(round(n / fs * TARGET_FS)) // EPOCH_SAMPLES
                for n, fs in zip(info.channel_samples, info.channel_rates) if fs > 0
            )
            maximum = max(0, available - 1)
            self.first_spin.setMaximum(maximum)
            self.last_spin.setMaximum(maximum)
            self.last_spin.setValue(min(11, maximum))
            label_count = len(parse_sleep_labels(self.optional_label_path()))
            self.info.setText(
                f"共 {len(info.channel_labels)} 个通道，约 {available} 个 epoch，"
                f"标签 {label_count} 个；发送前仅重采样到 100 Hz。"
            )
            self.append_log(f"已读取: {path.name}")
        except Exception as exc:
            self.channel_combo.clear()
            QtWidgets.QMessageBox.warning(self, "读取失败", str(exc))

    def send_one(self) -> None:
        self._start(self.first_spin.value(), self.first_spin.value())

    def send_range(self) -> None:
        self._start(self.first_spin.value(), self.last_spin.value())

    def _start(self, first: int, last: int) -> None:
        if self.thread is not None:
            return
        try:
            edf = Path(self.edf_edit.text().strip())
            if not edf.is_file():
                raise FileNotFoundError(f"EDF 文件不存在: {edf}")
            if self.channel_combo.count() == 0:
                self.load_channels()
            if self.channel_combo.count() == 0:
                return
            channel = int(self.channel_combo.currentData())
            port = self.current_port()
            if not port:
                raise ValueError("请选择串口")
            if first > last:
                raise ValueError("起始 epoch 不能大于结束 epoch")
            label = self.optional_label_path()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "无法发送", str(exc))
            return

        self.serial_claim_requested.emit(port)
        self._set_running(True)
        self.progress.setRange(0, last - first + 1)
        self.progress.setValue(0)
        thread = QtCore.QThread(self)
        worker = SleepStageSendWorker(
            edf_path=edf, label_path=label, channel_index=channel,
            port=port, first_epoch=first, last_epoch=last,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self.append_log)
        worker.prepared.connect(self.on_prepared)
        worker.epoch_result.connect(self.on_result)
        worker.progress.connect(self.on_progress)
        worker.finished.connect(self.on_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.thread, self.worker = thread, worker
        thread.start()

    def _set_running(self, running: bool) -> None:
        for widget in (
            self.edf_button, self.label_button, self.load_button,
            self.channel_combo, self.port_combo, self.port_button,
            self.first_spin, self.last_spin, self.one_button, self.range_button,
        ):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def on_prepared(self, total: int, source_fs: float, channel: str, unit: str) -> None:
        self.info.setText(
            f"{channel} ({unit})，原采样率 {source_fs:g} Hz，共 {total} 个完整 epoch；"
            "发送采样率 100 Hz。"
        )

    def on_result(self, result: Dict[str, object]) -> None:
        self.results.append(dict(result))
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            result.get("epoch", ""), result.get("time_s", ""), result.get("label", ""),
            result.get("cnn", ""), result.get("stage", "") or result.get("sequence", ""),
            result.get("status", ""),
        ]
        for column, value in enumerate(values):
            self.table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.table.scrollToBottom()

    def on_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.progress.setFormat(f"{current}/{total}")

    def on_finished(self, success: bool, message: str) -> None:
        self.append_log(message)
        if not success:
            self.info.setText(message)

    def on_thread_finished(self) -> None:
        self.thread = None
        self.worker = None
        self._set_running(False)

    def stop(self) -> None:
        if self.worker is not None:
            # 工作线程正忙时 queued slot 不会执行，所以直接设置简单停止标志。
            self.worker.stop_requested = True
            self.stop_button.setEnabled(False)
            self.append_log("正在停止...")

    def clear_results(self) -> None:
        if self.thread is None:
            self.results.clear()
            self.table.setRowCount(0)
            self.log.clear()

    def save_results(self) -> None:
        if not self.results:
            QtWidgets.QMessageBox.information(self, "没有结果", "当前没有可保存结果")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存结果", "sleep_stage_mcu_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        fields = ["epoch", "time_s", "label", "channel", "source_fs",
                  "ack", "cnn", "sequence", "stage", "status", "reply"]
        with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.results)
        self.append_log(f"已保存: {path}")

    def shutdown(self) -> None:
        self.stop()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
