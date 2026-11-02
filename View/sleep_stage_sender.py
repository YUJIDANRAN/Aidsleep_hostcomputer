"""EDF/BDF 或 OpenBCI 单通道按 30 秒 epoch 发送到 STM32 睡眠分期网络。"""

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
from scipy.signal import butter, lfilter, resample, resample_poly

from analysis_plot_view import (
    _brainflow_columns_and_labels,
    _looks_like_brainflow_raw,
    _read_sibling_openbci_sample_rate,
    _normalize_edf_signal_to_uv,
    load_eeg_file_channel_window,
    load_eeg_file_info,
)


TARGET_FS = 100.0
DOD_TRAINING_FS = 250.0
DOD_CONTEXT_SECONDS = 60.0
EPOCH_SECONDS = 30
EPOCH_SAMPLES = 3000
DEFAULT_DATA_DIR = Path(r"D:\eeglab2026.0.0\opensource_psg\EPCTL01-2025\EPCTL01")
STAGE_LABELS = ("W", "N1", "N2", "N3", "R")
OPENBCI_CHANNELS = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2")


def prediction_label(reply: object) -> str:
    """仅提取显示类别，原始 MCU 回复仍完整保存在结果中。"""
    match = re.match(r"^(?:CNN|STAGE) class=(\d+)(?:\s|$)", str(reply))
    if match and 0 <= int(match[1]) < len(STAGE_LABELS):
        return STAGE_LABELS[int(match[1])]
    return ""


def stage_matches_label(result: Dict[str, object]) -> str:
    """无最终分期或有效真值时不判错；CSV 留空，UI 显示破折号。"""
    predicted = prediction_label(result.get("stage", ""))
    label = str(result.get("label", "")).strip().upper()
    if label == "REM":
        label = "R"
    if not predicted or label not in STAGE_LABELS:
        return ""
    return "true" if predicted == label else "false"


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


def count_brainflow_rows(path: Path) -> int:
    """Count rows exactly without loading a potentially multi-gigabyte CSV."""
    line_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if path.stat().st_size and last_byte != b"\n":
        line_count += 1
    return line_count


def load_openbci_epoch_range(
    path: Path, channel_index: int, first_epoch: int, last_epoch: int
) -> tuple[np.ndarray, float, str, str, int]:
    """Load one BrainFlow EXG channel with DOD context and preprocessing."""
    if not _looks_like_brainflow_raw(path):
        raise ValueError("请选择 OpenBCI BrainFlow-RAW_*.csv")
    columns, _labels, _n_columns = _brainflow_columns_and_labels(path)
    if not 0 <= int(channel_index) < min(len(columns), len(OPENBCI_CHANNELS)):
        raise ValueError(f"EXG 通道序号无效: {channel_index}")
    source_fs = float(_read_sibling_openbci_sample_rate(path) or 500.0)
    sample_count = count_brainflow_rows(path)
    total_epochs = int(math.floor(sample_count / source_fs / EPOCH_SECONDS))
    first = max(0, int(first_epoch))
    last = min(int(last_epoch), total_epochs - 1)
    if total_epochs <= 0 or first > last:
        raise ValueError(f"epoch 范围无效；可用范围为 0~{total_epochs - 1}")

    requested_start_s = first * EPOCH_SECONDS
    requested_end_s = (last + 1) * EPOCH_SECONDS
    file_end_s = sample_count / source_fs
    padded_start_s = max(0.0, requested_start_s - DOD_CONTEXT_SECONDS)
    padded_end_s = min(file_end_s, requested_end_s + DOD_CONTEXT_SECONDS)
    raw, loaded_fs, _label, unit, actual_start_s = load_eeg_file_channel_window(
        path,
        int(channel_index),
        start_s=padded_start_s,
        duration_s=padded_end_s - padded_start_s,
    )
    expected_source_samples = int(round((padded_end_s - padded_start_s) * source_fs))
    if raw.size != expected_source_samples:
        raise ValueError(
            f"所选 BrainFlow 通道样本数异常: 得到 {raw.size}, 期望 {expected_source_samples}；"
            "请检查文件是否有空值或行缺失"
        )
    if unit.strip().lower() not in {"uv", "µv", "μv"}:
        raise ValueError(f"BrainFlow 单位不是 uV: {unit}")
    if abs(float(loaded_fs) - source_fs) > 1.0e-6:
        raise ValueError("BrainFlow 文件采样率读取不一致")

    processed = preprocess_dod_training_signal(raw, source_fs)
    crop_start = int(round((requested_start_s - actual_start_s) * TARGET_FS))
    required = (last - first + 1) * EPOCH_SAMPLES
    signal = processed[crop_start:crop_start + required]
    if signal.size != required:
        raise RuntimeError(
            f"DOD 训练链处理后点数异常：得到 {signal.size}，期望 {required}"
        )
    return signal, source_fs, f"{OPENBCI_CHANNELS[channel_index]} (EXG {channel_index})", unit, total_epochs


def preprocess_dod_training_signal(raw: np.ndarray, source_fs: float) -> np.ndarray:
    """复现 DOD 训练脚本的连续信号预处理顺序。

    源信号先抗混叠重采样到 250 Hz，再执行五阶因果 0.5~40 Hz
    Butterworth 带通滤波，最后按训练脚本使用 FFT 重采样到 100 Hz。
    每个 epoch 的标准化和量化仍由 MCU 端已有代码完成。
    """
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("所选通道没有数据")
    if not math.isfinite(source_fs) or source_fs <= 0:
        raise ValueError(f"采样率无效: {source_fs}")
    if not np.all(np.isfinite(values)):
        count = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"所选通道含 {count} 个 NaN/Inf，请先处理坏段")

    if abs(source_fs - DOD_TRAINING_FS) < 1.0e-6:
        at_training_rate = values
    else:
        ratio = Fraction(DOD_TRAINING_FS / source_fs).limit_denominator(10000)
        at_training_rate = np.asarray(
            resample_poly(values, ratio.numerator, ratio.denominator),
            dtype=np.float64,
        )

    nyquist = 0.5 * DOD_TRAINING_FS
    b, a = butter(5, [0.5 / nyquist, 40.0 / nyquist], btype="band")
    filtered = lfilter(b, a, at_training_rate)
    # DOD trainer.py 使用 int(len(x) * 100 / 250)，这里保持相同的截断规则。
    target_count = int(filtered.size * TARGET_FS / DOD_TRAINING_FS)
    return np.asarray(resample(filtered, target_count), dtype=np.float64)


def resample_to_model_rate(raw: np.ndarray, source_fs: float) -> np.ndarray:
    """兼容旧调用名；现在返回 DOD 训练预处理后的 100 Hz 信号。"""
    return preprocess_dod_training_signal(raw, source_fs)


def load_edf_epoch_range(
    path: Path, channel_index: int, first_epoch: int, last_epoch: int
) -> tuple[np.ndarray, float, str, str, int]:
    """只读取所选 epoch 范围，返回按 DOD 训练链处理的 100 Hz 信号。

    范围两端额外读取 60 秒，再在完整预处理后裁掉，以减小因果 IIR
    初始状态和 FFT 重采样边界对目标 epoch 的影响。
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
    padded_start_s = max(0.0, requested_start_s - DOD_CONTEXT_SECONDS)
    padded_end_s = min(file_end_s, requested_end_s + DOD_CONTEXT_SECONDS)
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
    padded_100_hz = preprocess_dod_training_signal(raw, source_fs)
    crop_start = int(round((requested_start_s - padded_start_s) * TARGET_FS))
    required = (last - first + 1) * EPOCH_SAMPLES
    signal = padded_100_hz[crop_start:crop_start + required]
    if signal.size != required:
        raise RuntimeError(
            f"DOD 训练链处理后点数异常：得到 {signal.size}，期望 {required}"
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
    """后台执行文件读取、DOD 预处理和串口握手，防止界面卡顿。"""

    log = QtCore.pyqtSignal(str)
    prepared = QtCore.pyqtSignal(int, float, str, str)
    epoch_result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(bool, str)

    def __init__(self, *, source_path: Path, source_kind: str,
                 label_path: Optional[Path],
                 channel_index: int, port: str, first_epoch: int,
                 last_epoch: int) -> None:
        super().__init__()
        self.source_path = source_path
        self.source_kind = source_kind
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
                f"读取 {self.source_kind} 通道 {self.channel_index}，epoch "
                f"{self.first_epoch}~{self.last_epoch}: {self.source_path.name}；"
                "执行 DOD 训练预处理"
            )
            loader = load_openbci_epoch_range if self.source_kind == "OpenBCI" else load_edf_epoch_range
            signal, source_fs, channel, unit, total_epochs = loader(
                self.source_path, self.channel_index, self.first_epoch, self.last_epoch,
            )
            first = max(0, self.first_epoch)
            last = min(self.last_epoch, total_epochs - 1)
            labels = parse_sleep_labels(self.label_path) if self.source_kind != "OpenBCI" else {}
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
                roundtrip_started = time.perf_counter()
                self._write_all(ser, payload)
                upload_finished = time.perf_counter()
                lines = self._read_until_ready(ser)
                response_finished = time.perf_counter()
                upload_ms = (upload_finished - roundtrip_started) * 1000.0
                response_ms = (response_finished - upload_finished) * 1000.0
                roundtrip_ms = (response_finished - roundtrip_started) * 1000.0
                parsed = parse_mcu_reply(lines)
                parsed.update({
                    "epoch": epoch_index,
                    "sequence_position": (sent - 1) % 12 + 1,
                    "time_s": epoch_index * EPOCH_SECONDS,
                    "label": label,
                    "channel": channel,
                    "source_file": str(self.source_path),
                    "source_type": self.source_kind,
                    "source_fs": float(source_fs),
                    "upload_ms": upload_ms,
                    "response_ms": response_ms,
                    "roundtrip_ms": roundtrip_ms,
                    "reply": "\n".join(lines),
                })
                self.log.emit(
                    f"TIME epoch={epoch_index} upload={upload_ms:.1f} ms "
                    f"mcu_response={response_ms:.1f} ms total={roundtrip_ms:.1f} ms"
                )
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
        self.edf_mode_check = QtWidgets.QCheckBox("EDF/BDF", self)
        self.openbci_mode_check = QtWidgets.QCheckBox("OpenBCI BrainFlow CSV", self)
        self.source_mode_group = QtWidgets.QButtonGroup(self)
        self.source_mode_group.setExclusive(True)
        self.source_mode_group.addButton(self.edf_mode_check, 0)
        self.source_mode_group.addButton(self.openbci_mode_check, 1)
        self.edf_mode_check.setChecked(True)
        files.addWidget(QtWidgets.QLabel("输入格式", self), 0, 0)
        files.addWidget(self.edf_mode_check, 0, 1)
        files.addWidget(self.openbci_mode_check, 0, 2)

        self.edf_path_label = QtWidgets.QLabel("EDF/BDF", self)
        self.edf_edit = QtWidgets.QLineEdit(self)
        self.edf_button = QtWidgets.QPushButton("选择 EDF...", self)
        self.openbci_path_label = QtWidgets.QLabel("BrainFlow CSV", self)
        self.openbci_edit = QtWidgets.QLineEdit(self)
        self.openbci_button = QtWidgets.QPushButton("选择 CSV...", self)
        self.label_edit = QtWidgets.QLineEdit(self)
        self.label_button = QtWidgets.QPushButton("选择标签...", self)
        self.reference_path_label = QtWidgets.QLabel("真实标签", self)
        files.addWidget(self.edf_path_label, 1, 0)
        files.addWidget(self.edf_edit, 1, 1)
        files.addWidget(self.edf_button, 1, 2)
        files.addWidget(self.openbci_path_label, 2, 0)
        files.addWidget(self.openbci_edit, 2, 1)
        files.addWidget(self.openbci_button, 2, 2)
        files.addWidget(self.reference_path_label, 3, 0)
        files.addWidget(self.label_edit, 3, 1)
        files.addWidget(self.label_button, 3, 2)
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
            "epoch 从 0 开始；发送前执行 DOD 训练预处理；"
            "每帧 30 秒/3000 点；收到 READY 后才发下一帧。", self
        )
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.table = QtWidgets.QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels(
            ["epoch", "真实标签", "CNN", "LSTM", "一致", "MCU响应(ms)", "状态"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeaderItem(3).setToolTip(
            "本次发送每 12 帧的序号,最终类别；LSTM 使用 12 帧滑动窗口，预热后每帧输出。"
        )
        self.table.horizontalHeaderItem(4).setToolTip(
            "LSTM 最终类别与真实标签是否一致；无结果或无有效标签时显示 —。"
        )
        header = self.table.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignCenter)
        header.setStretchLastSection(False)

        # 类别使用短标签，剩余宽度留给 LSTM 的序号和类别。
        for column in (0, 1, 2, 4, 5, 6):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.Fixed)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setMinimumSectionSize(40)
        self.table.setColumnWidth(0, 54)   # epoch
        self.table.setColumnWidth(1, 62)   # 真实标签
        self.table.setColumnWidth(2, 54)   # CNN 类别
        self.table.setColumnWidth(
            4, max(64, self.table.fontMetrics().horizontalAdvance("false") + 16)
        )
        self.table.setColumnWidth(5, 92)   # 发送完成到 READY，近似 MCU 处理耗时
        self.table.setColumnWidth(6, 58)   # 状态
        self.table.setTextElideMode(QtCore.Qt.ElideRight)
        self.table.setWordWrap(False)
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
        self.edf_mode_check.toggled.connect(self._update_source_mode)
        self.openbci_mode_check.toggled.connect(self._update_source_mode)
        self.edf_button.clicked.connect(self.browse_edf)
        self.openbci_button.clicked.connect(self.browse_openbci)
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
        self._update_source_mode()

    def is_openbci_mode(self) -> bool:
        return self.openbci_mode_check.isChecked()

    def _update_source_mode(self, *_args) -> None:
        openbci = self.is_openbci_mode()
        can_change_mode = self.thread is None and not self.results
        can_edit_source = self.thread is None and not self.results
        self.edf_mode_check.setEnabled(can_change_mode)
        self.openbci_mode_check.setEnabled(can_change_mode)
        for widget in (self.edf_path_label, self.edf_edit, self.edf_button):
            widget.setEnabled(can_edit_source and not openbci)
        for widget in (self.openbci_path_label, self.openbci_edit, self.openbci_button):
            widget.setEnabled(can_edit_source and openbci)
        for widget in (self.label_edit, self.label_button):
            widget.setEnabled(can_edit_source and not openbci)
        for widget in (self.load_button, self.channel_combo):
            widget.setEnabled(can_edit_source)
        self._update_source_hint()

    def _update_source_hint(self) -> None:
        if self.is_openbci_mode():
            self.info.setText(
                "OpenBCI BrainFlow CSV：EXG0–7 映射为 Fp1/Fp2/F3/F4/C3/C4/O1/O2；"
                "按 µV 和文件采样率读取，执行与 EDF 相同的 DOD 预处理，30 秒/3000 点后串口发送；"
                "结果单独导出，之后可与上位机 YASA CSV 按 epoch 对照。"
            )
        else:
            self.info.setText(
                "epoch 从 0 开始；发送前执行 DOD 训练预处理；"
                "每帧 30 秒/3000 点；收到 READY 后才发下一帧。"
            )

    def browse_openbci(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 OpenBCI BrainFlow CSV",
            self.openbci_edit.text() or str(Path(r"E:\OpenBCI\OpenBCI_GUI\Recordings")),
            "BrainFlow CSV (BrainFlow-RAW_*.csv);;CSV (*.csv);;所有文件 (*)"
        )
        if path:
            self.openbci_edit.setText(path)
            self.load_channels()

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
            self.channel_combo.clear()
            if self.is_openbci_mode():
                path = Path(self.openbci_edit.text().strip())
                if not path.is_file():
                    raise FileNotFoundError(f"BrainFlow CSV 不存在: {path}")
                if not _looks_like_brainflow_raw(path):
                    raise ValueError("请选择 OpenBCI BrainFlow-RAW_*.csv")
                columns, _labels, _column_count = _brainflow_columns_and_labels(path)
                sample_rate = float(_read_sibling_openbci_sample_rate(path) or 500.0)
                brainflow_info = load_eeg_file_info(path)
                sample_count = int(brainflow_info.channel_samples[0])
                channel_count = min(len(columns), len(OPENBCI_CHANNELS))
                for i, _column in enumerate(columns[:channel_count]):
                    self.channel_combo.addItem(
                        f"{i}: {OPENBCI_CHANNELS[i]} (EXG {i})  {sample_rate:g}Hz  uV  {sample_count}点", i
                    )
                if self.channel_combo.count() > 3:
                    self.channel_combo.setCurrentIndex(3)
            else:
                path = Path(self.edf_edit.text().strip())
                if not path.is_file():
                    raise FileNotFoundError(f"EDF 文件不存在: {path}")
                info = load_eeg_file_info(path)
                channel_count = len(info.channel_labels)
                for i, name in enumerate(info.channel_labels):
                    self.channel_combo.addItem(
                        f"{i}: {name}  {info.channel_rates[i]:g}Hz  "
                        f"{info.channel_units[i]}  {info.channel_samples[i]}点", i
                    )
            if self.channel_combo.count() == 0:
                raise ValueError("输入文件中没有可用通道")
            if self.is_openbci_mode():
                available = int(math.floor(sample_count / sample_rate / EPOCH_SECONDS))
            else:
                available = max(
                    int(round(n / fs * TARGET_FS)) // EPOCH_SAMPLES
                    for n, fs in zip(info.channel_samples, info.channel_rates) if fs > 0
                )
            maximum = max(0, available - 1)
            if self.is_openbci_mode():
                # File-size row estimates can be low for variable-width BrainFlow rows.
                # The worker counts CSV records exactly and validates the requested end epoch.
                maximum = 999999
            self.first_spin.setMaximum(maximum)
            self.last_spin.setMaximum(maximum)
            self.last_spin.setValue(min(11, maximum))
            if self.is_openbci_mode():
                labels = {}
                ref_name = "参照标签"
            else:
                labels = parse_sleep_labels(self.optional_label_path())
                ref_name = "标签"
            self.info.setText(
                f"共 {channel_count} 个通道，约 {available} 个 epoch，"
                f"{ref_name} {len(labels)} 个；发送前：源采样率→250 Hz、0.5~40 Hz "
                "五阶因果带通、FFT→100 Hz。"
            )
            if self.is_openbci_mode():
                self.info.setText(
                    f"BrainFlow CSV: 约 {sample_count} 点 @ {sample_rate:g} Hz，{available} 个完整 epoch；"
                    "通道映射 EXG0–7→Fp1/Fp2/F3/F4/C3/C4/O1/O2；"
                    "DOD 预处理：→250 Hz、0.5–40 Hz 五阶因果带通、FFT→100 Hz；"
                    "IGT6 结果单独导出，之后再和 YASA CSV 对照。输入范围会按实际 CSV 行数校验。"
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
            openbci = self.is_openbci_mode()
            source = Path((self.openbci_edit if openbci else self.edf_edit).text().strip())
            if not source.is_file():
                raise FileNotFoundError(
                    f"{'BrainFlow CSV' if openbci else 'EDF/BDF'} 文件不存在: {source}"
                )
            if openbci and not _looks_like_brainflow_raw(source):
                raise ValueError("请选择 OpenBCI BrainFlow-RAW_*.csv")
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
            label = None if openbci else self.optional_label_path()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "无法发送", str(exc))
            return

        self.serial_claim_requested.emit(port)
        self._set_running(True)
        self.progress.setRange(0, last - first + 1)
        self.progress.setValue(0)
        thread = QtCore.QThread(self)
        worker = SleepStageSendWorker(
            source_path=source, source_kind="OpenBCI" if openbci else "EDF/BDF",
            label_path=label, channel_index=channel,
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
            self.port_combo, self.port_button,
            self.first_spin, self.last_spin, self.one_button, self.range_button,
        ):
            widget.setEnabled(not running)
        self._update_source_mode()
        self.stop_button.setEnabled(running)

    def on_prepared(self, total: int, source_fs: float, channel: str, unit: str) -> None:
        self.info.setText(
            f"{channel} ({unit})，原采样率 {source_fs:g} Hz，共 {total} 个完整 epoch；"
            "DOD 训练预处理后发送 100 Hz 数据。"
        )

    def on_result(self, result: Dict[str, object]) -> None:
        result = dict(result)
        result["stage_matches_label"] = stage_matches_label(result)
        self.results.append(result)
        row = self.table.rowCount()
        self.table.insertRow(row)
        cnn = prediction_label(result.get("cnn", ""))
        stage = prediction_label(result.get("stage", ""))
        position = result.get("sequence_position", row % 12 + 1)
        stage_display = stage or ("等待" if result.get("sequence") else "—")
        values = [
            result.get("epoch", ""), result.get("label", ""), cnn or "—",
            f"{position}/12,{stage_display}", result["stage_matches_label"] or "—",
            f"{float(result.get('response_ms', 0.0)):.1f}",
            result.get("status", ""),
        ]
        for column, value in enumerate(values):
            text = str(value)
            item = QtWidgets.QTableWidgetItem(text)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            # 精简显示，悬停仍可查看原始回复和分数。
            tooltip = text
            if column == 2:
                tooltip = str(result.get("cnn", ""))
            elif column == 3:
                tooltip = str(result.get("stage", "") or result.get("sequence", ""))
            item.setToolTip(tooltip)
            self.table.setItem(row, column, item)
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
            self._update_source_mode()

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
                  "source_file", "source_type",
                  "upload_ms", "response_ms", "roundtrip_ms",
                  "ack", "cnn", "sequence", "stage", "status", "reply",
                  "stage_matches_label"]
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
