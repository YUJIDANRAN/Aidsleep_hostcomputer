"""Embedded matplotlib views for offline EEG waveform and power plots."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from power_cal import (  # noqa: E402
    BAND_COLORS,
    BAND_LABELS,
    DEFAULT_SAMPLE_RATE,
    EEG_BANDS,
    extract_band_waveforms,
)

CHANNEL_ORDER: List[str] = ["raw", "delta", "theta", "alpha", "beta", "gamma"]
CHANNEL_LABELS = {
    "raw": "raw",
    **{name: BAND_LABELS[name] for name in EEG_BANDS},
}
CHANNEL_COLORS = {
    "raw": "#212121",
    **BAND_COLORS,
}
## 粉噪落在 α 相位分区的标注色
PINK_PHASE_COLORS = {
    "peak": "#1565C0",
    "falling": "#EF6C00",
    "trough": "#C62828",
    "rising": "#2E7D32",
}
PINK_PHASE_LABELS = {
    "peak": "峰",
    "falling": "下降沿",
    "trough": "谷",
    "rising": "上升沿",
}
MAX_PLOT_POINTS = 25000
MAX_PINK_MARKERS = 800

class OfflineEegFileInfo(NamedTuple):
    channel_labels: List[str]
    channel_rates: List[float]
    channel_samples: List[int]
    channel_units: List[str]


class _TabularEegFile(NamedTuple):
    frame: pd.DataFrame
    channel_columns: List[object]
    channel_labels: List[str]
    sample_rate: float
    unit: str


# Higher cap for short windows so local EDF browsing keeps detail.
MAX_PLOT_POINTS_SHORT = 200000


def _read_openbci_sample_rate_from_comments(path: Path) -> Optional[float]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(32):
                line = handle.readline()
                if not line:
                    break
                text = line.strip()
                if not text.startswith("%"):
                    break
                match = re.search(r"Sample\s+Rate\s*=\s*([0-9.]+)\s*Hz", text, re.I)
                if match:
                    rate = float(match.group(1))
                    if rate > 0:
                        return rate
    except OSError:
        return None
    return None


def _read_sibling_openbci_sample_rate(path: Path) -> Optional[float]:
    for sibling in sorted(path.parent.glob("OpenBCI-RAW-*.txt")):
        rate = _read_openbci_sample_rate_from_comments(sibling)
        if rate is not None:
            return rate
    return None


def _looks_like_openbci_raw(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(8):
                line = handle.readline()
                if not line:
                    break
                text = line.strip()
                if text.startswith("%OpenBCI") or text.startswith("%Sample Rate"):
                    return True
                if text and not text.startswith("%"):
                    return "EXG Channel" in text and "Sample Index" in text
    except OSError:
        return False
    return False


def _looks_like_brainflow_raw(path: Path) -> bool:
    if path.name.lower().startswith("brainflow-raw"):
        return True
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            line = handle.readline()
    except OSError:
        return False
    return "\t" in line and "," not in line


def _looks_like_openbci_console_log(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            head = "".join(handle.readline() for _ in range(12))
    except OSError:
        return False
    lowered = head.lower()
    return (
        "[board_logger]" in lowered
        or "[brainflow_logger]" in lowered
        or "incoming json:" in lowered
        or "failed to establish connection" in lowered
    )


def _raise_openbci_console_log_error(path: Path) -> None:
    raise ValueError(
        f"{path.name} 是 OpenBCI/BrainFlow 控制台日志，不是 EEG 数据文件；"
        "本上位机的 OpenBCI 离线分析请改选 "
        "Recordings/OpenBCISession_*/BrainFlow-RAW_*.csv。"
    )


def is_openbci_brainflow_eeg_file(path: Path) -> bool:
    """Return True for BrainFlow RAW matrices used as OpenBCI EEG input."""
    return _looks_like_brainflow_raw(path)


def _raise_openbci_text_unsupported(path: Path) -> None:
    raise ValueError(
        f"{path.name} 是 OpenBCI GUI 文本导出，不作为本上位机离线分析输入；"
        "请改选同一会话目录下的 BrainFlow-RAW_*.csv。"
    )


def _timestamp_sample_rate(frame: pd.DataFrame, column: object) -> Optional[float]:
    if column not in frame.columns or len(frame) < 2:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size < 2:
        return None
    dt = np.diff(values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not dt.size:
        return None
    median_dt = float(np.median(dt))
    if median_dt > 1000.0:
        median_dt /= 1000.0
    elif median_dt > 10.0:
        median_dt /= 1000.0
    if median_dt <= 0:
        return None
    rate = 1.0 / median_dt
    if 1.0 <= rate <= 5000.0:
        return float(rate)
    return None


def _load_openbci_text_table(path: Path) -> _TabularEegFile:
    sample_rate = _read_openbci_sample_rate_from_comments(path) or float(DEFAULT_SAMPLE_RATE)
    frame = pd.read_csv(path, comment="%", skipinitialspace=True)
    exg_columns = [col for col in frame.columns if str(col).strip().startswith("EXG Channel")]
    if not exg_columns:
        raise ValueError(f"OpenBCI 文件未找到 EXG Channel 列: {path.name}")
    labels = []
    for index, col in enumerate(exg_columns):
        match = re.search(r"EXG Channel\s+(\d+)", str(col), re.I)
        labels.append(f"EXG {int(match.group(1))}" if match else f"EXG {index}")
    return _TabularEegFile(frame, exg_columns, labels, float(sample_rate), "uV")


def _openbci_columns_and_labels(path: Path) -> tuple[List[str], List[str]]:
    frame = pd.read_csv(path, comment="%", skipinitialspace=True, nrows=0)
    exg_columns = [str(col) for col in frame.columns if str(col).strip().startswith("EXG Channel")]
    labels = []
    for index, col in enumerate(exg_columns):
        match = re.search(r"EXG Channel\s+(\d+)", str(col), re.I)
        labels.append(f"EXG {int(match.group(1))}" if match else f"EXG {index}")
    return exg_columns, labels


def _openbci_header_line_index(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            if text and not text.startswith("%"):
                return index
    raise ValueError(f"OpenBCI 文件未找到表头: {path.name}")


def _estimate_openbci_data_rows(path: Path) -> int:
    header_index = _openbci_header_line_index(path)
    with path.open("rb") as handle:
        for _ in range(header_index + 1):
            handle.readline()
        line = handle.readline()
        while line and not line.strip():
            line = handle.readline()
    line_len = max(1, len(line))
    data_bytes = max(0, path.stat().st_size - len(line) * 0 - 1)
    return max(0, int(data_bytes / line_len))


def _load_brainflow_raw_table(path: Path) -> _TabularEegFile:
    frame = pd.read_csv(path, sep="\t", header=None)
    if frame.shape[1] < 9:
        raise ValueError(f"BrainFlow RAW 列数不足，无法解析 EXG 通道: {path.name}")
    exg_columns = list(range(1, min(9, int(frame.shape[1]))))
    labels = [f"EXG {index}" for index in range(len(exg_columns))]
    sample_rate = (
        _read_sibling_openbci_sample_rate(path)
        or _timestamp_sample_rate(frame, 22)
        or float(DEFAULT_SAMPLE_RATE)
    )
    return _TabularEegFile(frame, exg_columns, labels, float(sample_rate), "uV")


def _brainflow_columns_and_labels(path: Path) -> tuple[List[int], List[str], int]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()
    n_columns = len(first.split()) if first else 0
    if n_columns < 9:
        raise ValueError(f"BrainFlow RAW 列数不足，无法解析 EXG 通道: {path.name}")
    columns = list(range(1, min(9, n_columns)))
    labels = [f"EXG {index}" for index in range(len(columns))]
    return columns, labels, n_columns


def _count_nonempty_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line = handle.readline()
    line_len = max(1, len(line))
    return max(0, int(path.stat().st_size / line_len))


def _load_generic_eeg_table(path: Path) -> _TabularEegFile:
    frame = pd.read_csv(path)
    lower_columns = {str(col).strip().lower() for col in frame.columns}
    feature_required = {"epoch", "start_s", "end_s", "duration_s"}
    if feature_required.issubset(lower_columns) and (
        "stage_yasa" in lower_columns
        or "yasa_confidence" in lower_columns
        or "spindle_count" in lower_columns
    ):
        raise ValueError(
            f"{path.name} 是睡眠 epoch 特征/分期结果表，不是原始 EEG 波形；"
            "纺锤波检测请加载同一会话目录下的 BrainFlow-RAW_*.csv，并选择对应 EXG 通道。"
        )
    if "ch1_raw" in frame.columns:
        columns = ["ch1_raw"]
        labels = ["CH1"]
    elif "ch1" in frame.columns:
        columns = ["ch1"]
        labels = ["CH1"]
    else:
        raw_columns = [
            col for col in frame.columns if str(col).lower().endswith("_raw")
        ]
        if raw_columns:
            columns = raw_columns
            labels = [
                (str(col)[: -len("_raw")] or f"CH{index + 1}").upper()
                for index, col in enumerate(raw_columns)
            ]
        else:
            skip_names = {"index", "time_s", "time", "timestamp", "pink_seq"}
            data_columns = [
                col for col in frame.columns if str(col).lower() not in skip_names
            ]
            columns = [data_columns[0] if data_columns else frame.columns[0]]
            labels = ["CH1"]
    sample_rate = float(DEFAULT_SAMPLE_RATE)
    if "time_s" in frame.columns:
        rate = _timestamp_sample_rate(frame, "time_s")
        if rate is not None:
            sample_rate = rate
    return _TabularEegFile(frame, list(columns), labels, sample_rate, "raw")


def _load_tabular_eeg_file(path: Path) -> _TabularEegFile:
    if _looks_like_openbci_console_log(path):
        _raise_openbci_console_log_error(path)
    if _looks_like_openbci_raw(path):
        _raise_openbci_text_unsupported(path)
    if _looks_like_brainflow_raw(path):
        return _load_brainflow_raw_table(path)
    return _load_generic_eeg_table(path)


def _adaptive_max_plot_points(n_samples: int, sample_rate: float) -> int:
    """Choose a plot-point budget based on visible duration."""
    n = max(0, int(n_samples))
    fs = float(sample_rate) if sample_rate and sample_rate > 0 else float(DEFAULT_SAMPLE_RATE)
    duration_s = n / fs if fs > 0 else 0.0
    if duration_s <= 0:
        return MAX_PLOT_POINTS
    if duration_s <= 60:
        return max(MAX_PLOT_POINTS_SHORT, n)
    if duration_s <= 180:
        return MAX_PLOT_POINTS_SHORT
    if duration_s <= 600:
        return 80000
    return MAX_PLOT_POINTS



def _time_tick_step_seconds(span_s: float) -> float:
    """Choose a readable x-axis major tick spacing."""
    span = abs(float(span_s))
    if span <= 0:
        return 60.0
    if span > 120:
        return 60.0
    if span > 40:
        return 10.0
    if span > 15:
        return 5.0
    if span > 5:
        return 1.0
    return 0.5


def _load_eeg_csv_with_rate(path: Path) -> tuple[np.ndarray, float]:
    """Read EEG CSV/TXT and return (raw, sample_rate)."""
    if _looks_like_openbci_console_log(path):
        _raise_openbci_console_log_error(path)
    if _looks_like_openbci_raw(path):
        _raise_openbci_text_unsupported(path)
    if _looks_like_brainflow_raw(path):
        raw, sample_rate, _label, _unit = load_eeg_file_channel(path, 0)
        return raw, sample_rate
    table = _load_tabular_eeg_file(path)
    series = table.frame[table.channel_columns[0]]
    raw = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if raw.size < 8:
        raise ValueError(f"有效样本过少 ({raw.size}): {path.name}")
    return raw, float(table.sample_rate)


def _csv_channel_label(path: Path) -> str:
    """Return the first EEG-like CSV column label, e.g. ch2_raw -> CH2."""
    try:
        table = _load_tabular_eeg_file(path)
    except Exception:
        return "CH1"
    return table.channel_labels[0] if table.channel_labels else "CH1"


def _load_eeg_edf_with_rate(path: Path) -> tuple[np.ndarray, float]:
    """Read the first signal channel from EDF/BDF and return (raw, sample_rate)."""
    try:
        import pyedflib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "缂哄皯 pyedflib锛屾棤娉曡鍙?EDF/BDF銆傝鍏堟墽琛? pip install pyedflib"
        ) from exc

    reader = pyedflib.EdfReader(str(path))
    try:
        n_signals = int(reader.signals_in_file)
        if n_signals <= 0:
            raise ValueError(f"EDF/BDF 涓病鏈変俊鍙烽€氶亾: {path.name}")
        raw = reader.readSignal(0).astype(np.float64, copy=False)
        try:
            sample_rate = float(reader.getSampleFrequency(0))
        except AttributeError:
            sample_rate = float(reader.samplefrequency(0))
    finally:
        close = getattr(reader, "close", None) or getattr(reader, "_close", None)
        if close is not None:
            close()

    if raw.size < 8:
        raise ValueError(f"鏈夋晥鏍锋湰杩囧皯 ({raw.size}): {path.name}")
    if sample_rate <= 0:
        raise ValueError(f"EDF/BDF 閲囨牱鐜囨棤鏁?({sample_rate:g}): {path.name}")
    return raw, sample_rate


def _open_pyedflib_reader(path: Path):
    try:
        import pyedflib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "缂哄皯 pyedflib锛屾棤娉曡鍙?EDF/BDF銆傝鍏堟墽琛? pip install pyedflib"
        ) from exc
    return pyedflib.EdfReader(str(path))


def _close_edf_reader(reader) -> None:
    close = getattr(reader, "close", None) or getattr(reader, "_close", None)
    if close is not None:
        close()


def _edf_physical_dimension(reader, channel_index: int) -> str:
    try:
        unit = str(reader.getPhysicalDimension(channel_index)).strip()
    except Exception:
        unit = ""
    if unit in {"uV", "uv", "UV", "microV"}:
        return "uV"
    return unit or "uV"


def load_eeg_file_info(path: Path) -> OfflineEegFileInfo:
    """Return channel labels/rates/sample counts for CSV/TXT/EDF/BDF."""
    suffix = path.suffix.lower()
    if suffix not in {".edf", ".bdf"}:
        if _looks_like_openbci_console_log(path):
            _raise_openbci_console_log_error(path)
        if _looks_like_openbci_raw(path):
            _raise_openbci_text_unsupported(path)
        if _looks_like_brainflow_raw(path):
            _columns, labels, _n_columns = _brainflow_columns_and_labels(path)
            n_rows = _count_nonempty_rows(path)
            rate = _read_sibling_openbci_sample_rate(path) or float(DEFAULT_SAMPLE_RATE)
            return OfflineEegFileInfo(
                labels,
                [float(rate) for _ in labels],
                [int(n_rows) for _ in labels],
                ["uV" for _ in labels],
            )
        table = _load_tabular_eeg_file(path)
        samples: List[int] = []
        labels: List[str] = []
        for index, column in enumerate(table.channel_columns):
            raw = pd.to_numeric(table.frame[column], errors="coerce").dropna()
            samples.append(int(raw.size))
            labels.append(
                table.channel_labels[index]
                if index < len(table.channel_labels)
                else f"CH{index + 1}"
            )
        rates = [float(table.sample_rate) for _ in labels]
        units = [str(table.unit or "raw") for _ in labels]
        return OfflineEegFileInfo(labels, rates, samples, units)

    reader = _open_pyedflib_reader(path)
    try:
        n_signals = int(reader.signals_in_file)
        if n_signals <= 0:
            raise ValueError(f"EDF/BDF 涓病鏈変俊鍙烽€氶亾: {path.name}")
        try:
            labels = list(reader.getSignalLabels())
        except Exception:
            labels = []
        try:
            samples = [int(v) for v in reader.getNSamples()]
        except Exception:
            samples = [0 for _ in range(n_signals)]
        rates: List[float] = []
        clean_labels: List[str] = []
        units: List[str] = []
        for index in range(n_signals):
            try:
                rate = float(reader.getSampleFrequency(index))
            except AttributeError:
                rate = float(reader.samplefrequency(index))
            rates.append(rate)
            label = labels[index].strip() if index < len(labels) else ""
            clean_labels.append(label or f"CH{index + 1}")
            units.append(_edf_physical_dimension(reader, index))
    finally:
        _close_edf_reader(reader)

    return OfflineEegFileInfo(clean_labels, rates, samples, units)


def load_eeg_file_channel(path: Path, channel_index: int = 0) -> tuple[np.ndarray, float, str, str]:
    """Read one EEG channel and return (values, sample_rate, label, y_unit)."""
    suffix = path.suffix.lower()
    if suffix not in {".edf", ".bdf"}:
        if _looks_like_openbci_console_log(path):
            _raise_openbci_console_log_error(path)
        if _looks_like_openbci_raw(path):
            _raise_openbci_text_unsupported(path)
        if _looks_like_brainflow_raw(path):
            columns, labels, _n_columns = _brainflow_columns_and_labels(path)
            index = max(0, min(int(channel_index), len(columns) - 1))
            frame = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                usecols=[columns[index]],
                engine="python",
            )
            raw = pd.to_numeric(frame[columns[index]], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if raw.size < 8:
                raise ValueError(f"有效样本过少 ({raw.size}): {path.name}")
            sample_rate = _read_sibling_openbci_sample_rate(path) or float(DEFAULT_SAMPLE_RATE)
            return raw, float(sample_rate), labels[index], "uV"
        table = _load_tabular_eeg_file(path)
        n_channels = len(table.channel_columns)
        if n_channels <= 0:
            raise ValueError(f"文件未找到可用 EEG 通道: {path.name}")
        index = max(0, min(int(channel_index), n_channels - 1))
        column = table.channel_columns[index]
        raw = pd.to_numeric(table.frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if raw.size < 8:
            raise ValueError(f"有效样本过少 ({raw.size}): {path.name}")
        label = table.channel_labels[index] if index < len(table.channel_labels) else f"CH{index + 1}"
        return raw, float(table.sample_rate), label, str(table.unit or "raw")

    reader = _open_pyedflib_reader(path)
    try:
        n_signals = int(reader.signals_in_file)
        if n_signals <= 0:
            raise ValueError(f"EDF/BDF 涓病鏈変俊鍙烽€氶亾: {path.name}")
        index = max(0, min(int(channel_index), n_signals - 1))
        try:
            labels = list(reader.getSignalLabels())
        except Exception:
            labels = []
        label = labels[index].strip() if index < len(labels) else ""
        label = label or f"CH{index + 1}"
        unit = _edf_physical_dimension(reader, index)
        raw = reader.readSignal(index).astype(np.float64, copy=False)
        try:
            sample_rate = float(reader.getSampleFrequency(index))
        except AttributeError:
            sample_rate = float(reader.samplefrequency(index))
    finally:
        _close_edf_reader(reader)

    if raw.size < 8:
        raise ValueError(f"鏈夋晥鏍锋湰杩囧皯 ({raw.size}): {path.name}")
    if sample_rate <= 0:
        raise ValueError(f"EDF/BDF 閲囨牱鐜囨棤鏁?({sample_rate:g}): {path.name}")
    return raw, sample_rate, label, unit


def load_eeg_file_channel_window(
    path: Path,
    channel_index: int = 0,
    *,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
) -> tuple[np.ndarray, float, str, str, float]:
    """Read a time window from a large text EEG file when possible."""
    if _looks_like_openbci_console_log(path):
        _raise_openbci_console_log_error(path)
    start_s = max(0.0, float(start_s))
    if duration_s is not None:
        duration_s = max(0.0, float(duration_s))

    if _looks_like_openbci_raw(path):
        _raise_openbci_text_unsupported(path)

    if _looks_like_brainflow_raw(path):
        columns, labels, _n_columns = _brainflow_columns_and_labels(path)
        index = max(0, min(int(channel_index), len(columns) - 1))
        sample_rate = _read_sibling_openbci_sample_rate(path) or float(DEFAULT_SAMPLE_RATE)
        start_row = max(0, int(round(start_s * sample_rate)))
        n_rows = None if duration_s is None else max(8, int(round(duration_s * sample_rate)))
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[columns[index]],
            skiprows=start_row,
            nrows=n_rows,
        )
        raw = pd.to_numeric(frame[columns[index]], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if raw.size < 8:
            raise ValueError(f"有效样本过少 ({raw.size}): {path.name}")
        return raw, float(sample_rate), labels[index], "uV", start_row / float(sample_rate)

    raw, sample_rate, label, unit = load_eeg_file_channel(path, channel_index)
    if duration_s is None:
        return raw, sample_rate, label, unit, 0.0
    i0 = max(0, int(round(start_s * sample_rate)))
    i1 = min(int(raw.size), i0 + max(8, int(round(duration_s * sample_rate))))
    sliced = raw[i0:i1]
    if sliced.size < 8:
        raise ValueError(f"有效样本过少 ({sliced.size}): {path.name}")
    return sliced, sample_rate, label, unit, i0 / float(sample_rate)


def load_eeg_file_channels_window(
    path: Path,
    channel_indices: Sequence[int],
    *,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
) -> List[tuple[np.ndarray, float, str, str, float]]:
    """Read several channels from the same time window with one BrainFlow CSV scan."""
    if _looks_like_openbci_console_log(path):
        _raise_openbci_console_log_error(path)
    if _looks_like_openbci_raw(path):
        _raise_openbci_text_unsupported(path)
    indices = [int(index) for index in channel_indices]
    if not indices:
        return []
    if not _looks_like_brainflow_raw(path):
        return [
            load_eeg_file_channel_window(
                path,
                index,
                start_s=start_s,
                duration_s=duration_s,
            )
            for index in indices
        ]

    columns, labels, _n_columns = _brainflow_columns_and_labels(path)
    clamped = [max(0, min(index, len(columns) - 1)) for index in indices]
    sample_rate = _read_sibling_openbci_sample_rate(path) or float(DEFAULT_SAMPLE_RATE)
    start_s = max(0.0, float(start_s))
    duration = None if duration_s is None else max(0.0, float(duration_s))
    start_row = max(0, int(round(start_s * sample_rate)))
    n_rows = None if duration is None else max(8, int(round(duration * sample_rate)))
    read_columns = sorted({columns[index] for index in clamped})
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=read_columns,
        skiprows=start_row,
        nrows=n_rows,
    )
    result: List[tuple[np.ndarray, float, str, str, float]] = []
    for index in clamped:
        column = columns[index]
        raw = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if raw.size < 8:
            raise ValueError(f"有效样本过少 ({raw.size}): {path.name}")
        result.append((raw, float(sample_rate), labels[index], "uV", start_row / float(sample_rate)))
    return result


def load_eeg_csv_with_rate(path: Path) -> tuple[np.ndarray, float]:
    """Read an offline EEG file (CSV/TXT/EDF/BDF) and return (raw, sample_rate)."""
    suffix = path.suffix.lower()
    if suffix in {".edf", ".bdf"}:
        raw, sample_rate, _label, _unit = load_eeg_file_channel(path, 0)
        return raw, sample_rate
    return _load_eeg_csv_with_rate(path)


def _downsample_pair(
    time_s: np.ndarray, values: np.ndarray, max_points: int = MAX_PLOT_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample long signals while preserving min/max envelope."""
    n = int(values.size)
    if n <= max_points:
        return time_s, values
    n_bins = max(1, max_points // 2)
    bin_size = int(np.ceil(n / n_bins))
    out_t: List[float] = []
    out_y: List[float] = []
    for start in range(0, n, bin_size):
        end = min(n, start + bin_size)
        segment = values[start:end]
        t_seg = time_s[start:end]
        finite_idx = np.flatnonzero(np.isfinite(segment))
        if finite_idx.size == 0:
            continue
        finite_segment = segment[finite_idx]
        i_min = int(finite_idx[int(np.argmin(finite_segment))])
        i_max = int(finite_idx[int(np.argmax(finite_segment))])
        if i_min <= i_max:
            out_t.extend((float(t_seg[i_min]), float(t_seg[i_max])))
            out_y.extend((float(segment[i_min]), float(segment[i_max])))
        else:
            out_t.extend((float(t_seg[i_max]), float(t_seg[i_min])))
            out_y.extend((float(segment[i_max]), float(segment[i_min])))
    return np.asarray(out_t, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


class _MatplotlibHostView(QtWidgets.QWidget):
    """Shared matplotlib toolbar and canvas host."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        empty_message: str = "",
    ) -> None:
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas, stretch=1)
        if empty_message:
            self._draw_empty(empty_message)

        self.scroll_panel = QtWidgets.QWidget(self)
        scroll_layout = QtWidgets.QHBoxLayout(self.scroll_panel)
        scroll_layout.setContentsMargins(6, 2, 6, 2)
        scroll_layout.setSpacing(6)
        self.prev_button = QtWidgets.QPushButton("<", self.scroll_panel)
        self.next_button = QtWidgets.QPushButton(">", self.scroll_panel)
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.scroll_panel)
        self.time_slider.setRange(0, 0)
        self.time_slider.setEnabled(False)
        self.time_status = QtWidgets.QLabel("0.0-0.0 s", self.scroll_panel)
        self.time_status.setMinimumWidth(170)
        self.time_status.setAlignment(QtCore.Qt.AlignCenter)
        scroll_layout.addWidget(QtWidgets.QLabel("时间", self.scroll_panel))
        scroll_layout.addWidget(self.prev_button)
        scroll_layout.addWidget(self.time_slider, stretch=1)
        scroll_layout.addWidget(self.next_button)
        scroll_layout.addWidget(self.time_status)
        self._layout.addWidget(self.scroll_panel)
        self.scroll_panel.hide()

    @property
    def figure(self) -> Figure:
        return self._figure

    def refresh(self) -> None:
        self._figure.tight_layout()
        self._canvas.draw_idle()
        self._enable_pan()

    def _enable_pan(self) -> None:
        try:
            if self._toolbar.mode != "pan/zoom":
                self._toolbar.pan()
        except Exception:
            pass

    def _draw_empty(self, message: str) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#616161")
        self._canvas.draw_idle()


class AnalysisPlotView(_MatplotlibHostView):
    """Single/multi-segment band-power plot view."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, empty_message="填写时间段后点击功率对比")

    def clear(self) -> None:
        self._draw_empty("填写时间段后点击功率对比")


class OfflineRhythmStackView(_MatplotlibHostView):
    """Stacked offline EEG waveform view."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._sample_rate = float(DEFAULT_SAMPLE_RATE)
        self._time_s = np.zeros(0, dtype=np.float64)
        self._channels: Dict[str, np.ndarray] = {}
        self._visible: List[str] = list(CHANNEL_ORDER)
        self._source_name = ""
        self._axes: List = []
        self._remove_mask: Optional[np.ndarray] = None
        self._y_limits: Optional[tuple[float, float]] = None
        self._raw_y_label = "raw"
        self._display_note = ""
        self._pink_event_times_s = np.zeros(0, dtype=np.float64)
        self._show_pink_on_alpha = False
        super().__init__(parent, empty_message="选择 CSV/EDF 后点击加载")

    @property
    def has_data(self) -> bool:
        return bool(self._channels) and self._time_s.size > 0

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def pink_event_count(self) -> int:
        return int(self._pink_event_times_s.size)

    def clear(self) -> None:
        self._time_s = np.zeros(0, dtype=np.float64)
        self._channels.clear()
        self._source_name = ""
        self._axes = []
        self._remove_mask = None
        self._y_limits = None
        self._raw_y_label = "raw"
        self._display_note = ""
        self._pink_event_times_s = np.zeros(0, dtype=np.float64)
        self._show_pink_on_alpha = False
        self._draw_empty("选择 CSV/EDF 后点击加载")

    def set_pink_events(self, event_times_s: Optional[np.ndarray]) -> None:
        """设置粉噪发射时刻（绝对时间轴，与 _time_s 一致）。"""
        if event_times_s is None:
            self._pink_event_times_s = np.zeros(0, dtype=np.float64)
        else:
            arr = np.asarray(event_times_s, dtype=np.float64).reshape(-1)
            arr = arr[np.isfinite(arr)]
            self._pink_event_times_s = np.sort(arr)
        if self.has_data:
            self.redraw()

    def set_show_pink_on_alpha(self, enabled: bool) -> None:
        flag = bool(enabled)
        if flag == self._show_pink_on_alpha:
            return
        self._show_pink_on_alpha = flag
        if self.has_data:
            self.redraw()

    def set_y_limits(self, y_min: Optional[float], y_max: Optional[float]) -> None:
        if y_min is None or y_max is None:
            self._y_limits = None
        elif y_max <= y_min:
            raise ValueError("Y轴上限必须大于下限")
        else:
            self._y_limits = (float(y_min), float(y_max))
        if self.has_data:
            self.redraw()

    def load_raw(
        self,
        raw: np.ndarray,
        sample_rate: float,
        *,
        source_name: str = "",
        time_offset_s: float = 0.0,
        remove_mask: Optional[np.ndarray] = None,
        y_label: str = "raw",
        display_raw: Optional[np.ndarray] = None,
        display_note: str = "",
        pink_event_times_s: Optional[np.ndarray] = None,
    ) -> tuple[int, float]:
        """Load a raw segment into the waveform view."""
        raw_arr = np.asarray(raw, dtype=np.float64)
        if raw_arr.size < 8:
            raise ValueError(f"鏈夋晥鏍锋湰杩囧皯 ({raw_arr.size})")
        display_arr = raw_arr
        if display_raw is not None:
            display_arr = np.asarray(display_raw, dtype=np.float64)
            if display_arr.shape != raw_arr.shape:
                raise ValueError("display_raw length does not match raw length")
        fs = float(sample_rate)
        bands = extract_band_waveforms(raw_arr, fs)
        n = int(raw_arr.size)
        time_s = float(time_offset_s) + np.arange(n, dtype=np.float64) / fs
        self._sample_rate = fs
        self._time_s = time_s
        self._channels = {"raw": display_arr, **bands}
        self._source_name = source_name or "offline"
        self._raw_y_label = y_label or "raw"
        self._display_note = str(display_note or "")
        self._visible = ["raw"]
        if pink_event_times_s is None:
            self._pink_event_times_s = np.zeros(0, dtype=np.float64)
        else:
            arr = np.asarray(pink_event_times_s, dtype=np.float64).reshape(-1)
            self._pink_event_times_s = np.sort(arr[np.isfinite(arr)])
        if remove_mask is None:
            self._remove_mask = None
        else:
            mask = np.asarray(remove_mask, dtype=bool).reshape(-1)
            if mask.size != n:
                raise ValueError(
                    f"remove_mask length {mask.size} does not match raw length {n}"
                )
            self._remove_mask = mask
        self.redraw()
        self._enable_pan()
        return n, self._sample_rate

    def load_file(self, path: Path) -> tuple[int, float]:
        raw, sample_rate = load_eeg_csv_with_rate(path)
        return self.load_raw(raw, sample_rate, source_name=path.name)

    def set_visible_channels(self, names: Sequence[str]) -> None:
        # Keep raw in front whenever it exists.
        ordered = ["raw"] if "raw" in self._channels else []
        for name in CHANNEL_ORDER:
            if name == "raw":
                continue
            if name in names and name in self._channels:
                ordered.append(name)
        if ordered == self._visible and self.has_data:
            return
        self._visible = ordered
        if self.has_data:
            self.redraw()
        else:
            self._draw_empty("璇峰厛鍔犺浇 CSV/EDF")

    @staticmethod
    def _mask_true_spans(mask: np.ndarray) -> List[tuple]:
        """Return contiguous True spans as [start, end) index pairs."""
        spans: List[tuple] = []
        n = int(mask.size)
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < n and mask[j]:
                j += 1
            spans.append((i, j))
            i = j
        return spans

    @staticmethod
    def _phase_region(deg: float) -> str:
        x = float((deg + 180.0) % 360.0 - 180.0)
        if abs(x) <= 45.0:
            return "peak"
        if abs(x) >= 135.0:
            return "trough"
        if x > 0:
            return "falling"
        return "rising"

    def _draw_pink_on_alpha(self, ax) -> None:
        """在 α 波形上标注粉噪发射时刻，颜色表示峰/谷/上升/下降沿。"""
        if not self._show_pink_on_alpha:
            return
        events = self._pink_event_times_s
        alpha = self._channels.get("alpha")
        if events is None or events.size == 0 or alpha is None or self._time_s.size < 8:
            return
        t0 = float(self._time_s[0])
        t1 = float(self._time_s[-1])
        in_win = events[(events >= t0) & (events <= t1)]
        if in_win.size == 0:
            return
        try:
            from scipy.signal import hilbert
        except Exception:
            hilbert = None
        phases = None
        if hilbert is not None and alpha.size == self._time_s.size:
            try:
                phases = np.angle(hilbert(np.asarray(alpha, dtype=np.float64)))
            except Exception:
                phases = None

        idx = np.searchsorted(self._time_s, in_win)
        idx = np.clip(idx, 0, self._time_s.size - 1)
        if idx.size > MAX_PINK_MARKERS:
            step = int(np.ceil(idx.size / MAX_PINK_MARKERS))
            sel = np.arange(0, idx.size, step)
            idx = idx[sel]
            in_win = in_win[sel]

        y_vals = np.asarray(alpha, dtype=np.float64)[idx]
        region_points: Dict[str, List[tuple]] = {
            "peak": [],
            "falling": [],
            "trough": [],
            "rising": [],
        }
        for k, i_ev in enumerate(idx):
            if phases is not None:
                deg = float(np.rad2deg(phases[int(i_ev)]))
                region = self._phase_region(deg)
            else:
                region = "trough"
            region_points[region].append((float(in_win[k]), float(y_vals[k])))

        for region, pts in region_points.items():
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(
                xs,
                ys,
                s=22,
                c=PINK_PHASE_COLORS[region],
                marker="o",
                zorder=6,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
                label=f"pink·{PINK_PHASE_LABELS[region]}",
            )
            for x in xs:
                ax.axvline(x, color=PINK_PHASE_COLORS[region], alpha=0.18, lw=0.7)

        ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

    def redraw(self) -> None:
        self._figure.clear()
        self._axes = []
        visible = [name for name in self._visible if name in self._channels]
        if "raw" in self._channels and "raw" not in visible:
            visible = ["raw"] + visible
        if (
            self._show_pink_on_alpha
            and "alpha" in self._channels
            and "alpha" not in visible
        ):
            visible.append("alpha")
        if not visible or self._time_s.size == 0:
            self._draw_empty("璇峰厛鍔犺浇 CSV/EDF")
            return

        n = len(visible)
        max_pts = _adaptive_max_plot_points(int(self._time_s.size), self._sample_rate)
        axes = self._figure.subplots(n, 1, sharex=True, squeeze=False)
        reject_spans = (
            self._mask_true_spans(self._remove_mask)
            if self._remove_mask is not None and self._remove_mask.size
            else []
        )
        dt = (
            float(self._time_s[1] - self._time_s[0])
            if self._time_s.size >= 2
            else (1.0 / max(self._sample_rate, 1.0))
        )
        for i, name in enumerate(visible):
            ax = axes[i, 0]
            t, y = _downsample_pair(
                self._time_s, self._channels[name], max_points=max_pts
            )
            color = CHANNEL_COLORS.get(name, "#1976D2")
            ax.plot(t, y, color=color, linewidth=0.8)
            if name == "raw" and reject_spans:
                t_full = self._time_s
                for i0, i1 in reject_spans:
                    t0 = float(t_full[i0])
                    t1 = float(t_full[i1 - 1]) + dt
                    ax.axvspan(t0, t1, color="#E53935", alpha=0.28, lw=0)
            if name == "alpha":
                self._draw_pink_on_alpha(ax)
            y_label = self._raw_y_label if name == "raw" else CHANNEL_LABELS.get(name, name)
            ax.set_ylabel(y_label, fontsize=9)
            if self._y_limits is not None:
                ax.set_ylim(*self._y_limits)
            elif name == "raw":
                finite_y = y[np.isfinite(y)]
                if finite_y.size:
                    y0 = float(np.min(finite_y))
                    y1 = float(np.max(finite_y))
                    if y1 > y0:
                        pad = max((y1 - y0) * 0.05, 1.0)
                        ax.set_ylim(y0 - pad, y1 + pad)
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=8)
            self._axes.append(ax)
        axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
        self._bind_time_axis_ticks(self._axes)
        title = self._source_name or "offline"
        mark_tip = " | bad segments marked" if reject_spans else ""
        display_tip = f" | {self._display_note}" if self._display_note else ""
        pink_tip = ""
        if self._show_pink_on_alpha:
            pink_tip = f" | pink on α ({self.pink_event_count} events)"
        self._figure.suptitle(
            f"{title} | {self._sample_rate:.0f} Hz | waveform always shown{mark_tip}{display_tip}{pink_tip}",
            fontsize=10,
        )
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _bind_time_axis_ticks(self, axes: Sequence) -> None:
        """Keep x-axis tick spacing readable after pan/zoom."""
        if not axes:
            return
        def _apply(_ax=None) -> None:
            ref = axes[-1]
            x0, x1 = ref.get_xlim()
            step = _time_tick_step_seconds(x1 - x0)
            locator = MultipleLocator(step)
            for ax in axes:
                ax.xaxis.set_major_locator(locator)
                ax.grid(True, which="major", alpha=0.25)

        _apply()
        # sharex锛氭寕鍦ㄤ换涓€杞村嵆鍙紝缂╂斁宸ュ叿鏍忎細瑙﹀彂 xlim_changed
        axes[-1].callbacks.connect("xlim_changed", _apply)

