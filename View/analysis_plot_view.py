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
    bandpass_filter,
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
MAX_PLOT_POINTS = 25000
MULTI_CHANNEL_COLORS = [
    "#D32F2F",
    "#1976D2",
    "#388E3C",
    "#F57C00",
    "#7B1FA2",
    "#0097A7",
    "#C2185B",
    "#5D4037",
    "#455A64",
    "#689F38",
    "#512DA8",
    "#E64A19",
]
COMPARE_WAVEFORM_MODES = [
    ("raw", "raw data"),
    ("delta", "Delta"),
    ("theta", "theta"),
    ("alpha", "Alpha"),
    ("beta", "Beta"),
    ("gamma", "gamma"),
    ("sigma", "Sigma"),
]
COMPARE_BAND_LABELS = {
    "raw": "raw data",
    **{name: BAND_LABELS.get(name, name) for name in EEG_BANDS},
    "sigma": "σ",
}
COMPARE_BAND_RANGES = {
    **EEG_BANDS,
    "sigma": (12.0, 16.0),
}

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
            skip_names = {"index", "time_s", "time", "timestamp"}
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


class OfflineMultiChannelCompareDialog(QtWidgets.QDialog):
    """Overlay several offline channels in one shared time/value axis."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("多通道波形比较")
        self.resize(980, 680)
        self._time_s = np.zeros(0, dtype=np.float64)
        self._channels: List[tuple[str, np.ndarray, np.ndarray, str]] = []
        self._sample_rate = float(DEFAULT_SAMPLE_RATE)
        self._source_name = ""
        self._y_label = "raw"
        self._y_limits: Optional[tuple[float, float]] = None
        self._checks: List[QtWidgets.QCheckBox] = []
        self._mode_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._band_cache: Dict[str, List[np.ndarray]] = {}
        self._current_index = 0
        self._source_path: Optional[Path] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        mode_box = QtWidgets.QGroupBox("比较波形")
        mode_layout = QtWidgets.QHBoxLayout(mode_box)
        mode_layout.setContentsMargins(10, 18, 10, 10)
        mode_layout.setSpacing(14)
        for mode, label in COMPARE_WAVEFORM_MODES:
            check = QtWidgets.QCheckBox(label, self)
            check.setChecked(mode == "raw")
            check.toggled.connect(self.redraw)
            self._mode_group.addButton(check)
            self._mode_checks[mode] = check
            mode_layout.addWidget(check)
        self._offset_display_check = QtWidgets.QCheckBox("错开显示", self)
        self._offset_display_check.setToolTip("将可见通道按幅值间隔上下错开，便于观察峰谷位置")
        self._offset_display_check.toggled.connect(self.redraw)
        mode_layout.addWidget(self._offset_display_check)
        self._phase_button = QtWidgets.QPushButton("同相分析...", self)
        self._phase_button.setToolTip("计算可见通道相对参考通道的相关系数和最佳滞后")
        self._phase_button.clicked.connect(self._show_phase_analysis)
        mode_layout.addWidget(self._phase_button)
        mode_layout.addStretch(1)
        layout.addWidget(mode_box, stretch=0)

        controls_box = QtWidgets.QGroupBox("通道曲线")
        controls_layout = QtWidgets.QGridLayout(controls_box)
        controls_layout.setContentsMargins(10, 18, 10, 10)
        controls_layout.setHorizontalSpacing(14)
        controls_layout.setVerticalSpacing(6)
        self._controls_layout = controls_layout
        layout.addWidget(controls_box, stretch=0)

        self._plot = _MatplotlibHostView(self, empty_message="点击多通道比较加载当前窗口")
        layout.addWidget(self._plot, stretch=1)

    def load_channels(
        self,
        channels: Sequence[tuple],
        sample_rate: float,
        *,
        start_s: float,
        source_name: str,
        y_label: str,
        y_limits: Optional[tuple[float, float]] = None,
        current_index: int = 0,
        source_path: Optional[Path] = None,
    ) -> None:
        self._sample_rate = float(sample_rate)
        self._channels = []
        self._band_cache = {}
        for item in channels:
            if len(item) == 4:
                label, raw_values, display_values, note = item
            else:
                label, display_values, note = item
                raw_values = display_values
            raw_arr = np.asarray(raw_values, dtype=np.float64)
            display_arr = np.asarray(display_values, dtype=np.float64)
            if raw_arr.size >= 8 and display_arr.size >= 8:
                self._channels.append(
                    (str(label), raw_arr, display_arr, str(note or ""))
                )
        self._source_name = str(source_name or "offline")
        self._y_label = str(y_label or "raw")
        self._y_limits = y_limits
        self._current_index = max(0, int(current_index))
        self._source_path = Path(source_path) if source_path is not None else None
        n = min(
            (
                min(raw_values.size, display_values.size)
                for _label, raw_values, display_values, _note in self._channels
            ),
            default=0,
        )
        if n <= 0:
            self._time_s = np.zeros(0, dtype=np.float64)
        else:
            self._channels = [
                (label, raw_values[:n], display_values[:n], note)
                for label, raw_values, display_values, note in self._channels
            ]
            self._time_s = float(start_s) + np.arange(n, dtype=np.float64) / max(
                self._sample_rate, 1.0
            )
        self._rebuild_checkboxes(current_index=current_index)
        self.redraw()

    def _rebuild_checkboxes(self, *, current_index: int) -> None:
        while self._controls_layout.count():
            item = self._controls_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._checks = []
        for index, (label, _raw_values, _display_values, _note) in enumerate(self._channels):
            check = QtWidgets.QCheckBox(label, self)
            check.setChecked(True)
            color = MULTI_CHANNEL_COLORS[index % len(MULTI_CHANNEL_COLORS)]
            weight = "font-weight: 600;" if index == int(current_index) else ""
            check.setStyleSheet(f"QCheckBox {{ color: {color}; {weight} }}")
            check.toggled.connect(self.redraw)
            row = index // 4
            col = index % 4
            self._controls_layout.addWidget(check, row, col, 1, 1)
            self._checks.append(check)

    def _current_mode(self) -> str:
        for mode, check in self._mode_checks.items():
            if check.isChecked():
                return mode
        return "raw"

    def _mode_values(self, mode: str) -> List[np.ndarray]:
        if mode == "raw":
            return [display_values for _label, _raw_values, display_values, _note in self._channels]
        cached = self._band_cache.get(mode)
        if cached is not None:
            return cached
        if mode not in COMPARE_BAND_RANGES:
            return [display_values for _label, _raw_values, display_values, _note in self._channels]
        low, high = COMPARE_BAND_RANGES[mode]
        values: List[np.ndarray] = []
        for _label, raw_values, _display_values, _note in self._channels:
            base = bandpass_filter(raw_values, self._sample_rate)
            values.append(
                bandpass_filter(base, self._sample_rate, low_hz=low, high_hz=high)
            )
        self._band_cache[mode] = values
        return values

    def _visible_channel_items(self) -> List[tuple[int, str]]:
        return [
            (index, item[0])
            for index, item in enumerate(self._channels)
            if index < len(self._checks) and self._checks[index].isChecked()
        ]

    @staticmethod
    def _prepare_corr_array(values: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size < 8:
            return None
        finite = np.isfinite(arr)
        if int(np.count_nonzero(finite)) < 8:
            return None
        mean = float(np.nanmean(arr[finite]))
        arr = arr.copy()
        arr[~finite] = mean
        arr -= float(np.mean(arr))
        std = float(np.std(arr))
        if not np.isfinite(std) or std <= 1e-12:
            return None
        return arr / std

    @staticmethod
    def _zero_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        mask = np.isfinite(a) & np.isfinite(b)
        if int(np.count_nonzero(mask)) < 8:
            return None
        aa = np.asarray(a[mask], dtype=np.float64)
        bb = np.asarray(b[mask], dtype=np.float64)
        aa -= float(np.mean(aa))
        bb -= float(np.mean(bb))
        denom = float(np.std(aa) * np.std(bb))
        if denom <= 1e-12:
            return None
        return float(np.mean(aa * bb) / denom)

    @staticmethod
    def _phase_comment(zero_corr: Optional[float], best_corr: Optional[float], lag_ms: float) -> str:
        z = -2.0 if zero_corr is None else float(zero_corr)
        b = -2.0 if best_corr is None else float(best_corr)
        lag = abs(float(lag_ms))
        if z >= 0.80 and lag <= 50.0:
            return "同步较好"
        if z >= 0.50 and lag <= 120.0:
            return "部分同步"
        if b >= 0.50:
            return "形状相似但有滞后"
        if z <= -0.30:
            return "可能反相/错位"
        return "同步性弱"

    def _phase_metrics(
        self,
        ref: np.ndarray,
        values: np.ndarray,
        max_lag: int,
    ) -> tuple[Optional[float], Optional[float], float]:
        n = min(int(ref.size), int(values.size))
        if n < 8:
            return None, None, 0.0
        ref = np.asarray(ref[:n], dtype=np.float64)
        values = np.asarray(values[:n], dtype=np.float64)
        zero = self._zero_corr(ref, values)
        ref_norm = self._prepare_corr_array(ref)
        arr_norm = self._prepare_corr_array(values)
        if ref_norm is None or arr_norm is None:
            return zero, None, 0.0
        lag_limit = min(int(max_lag), n - 2)
        corr = np.correlate(arr_norm, ref_norm, mode="full") / float(n)
        lags = np.arange(-n + 1, n)
        mask = np.abs(lags) <= lag_limit
        if not np.any(mask):
            return zero, None, 0.0
        masked_corr = corr[mask]
        masked_lags = lags[mask]
        best_pos = int(np.argmax(masked_corr))
        best = float(masked_corr[best_pos])
        lag_ms = float(masked_lags[best_pos] / max(self._sample_rate, 1.0) * 1000.0)
        return zero, best, lag_ms

    def _phase_rows_for_current_window(
        self,
        visible: Sequence[tuple[int, str]],
        mode_values: Sequence[np.ndarray],
        ref_index: int,
        max_n: int,
    ) -> List[List[str]]:
        ref = np.asarray(mode_values[ref_index][:max_n], dtype=np.float64)
        max_lag = min(int(round(max(self._sample_rate, 1.0) * 1.0)), max_n - 2)
        rows: List[List[str]] = []
        for index, label in visible:
            zero, best, lag_ms = self._phase_metrics(
                ref,
                np.asarray(mode_values[index][:max_n], dtype=np.float64),
                max_lag,
            )
            rows.append(
                [
                    label,
                    "-" if zero is None else f"{zero:.3f}",
                    "-" if best is None else f"{best:.3f}",
                    f"{lag_ms:.0f}",
                    "-",
                    self._phase_comment(zero, best, lag_ms),
                ]
            )
        return rows

    @staticmethod
    def _stage_key(text: object) -> str:
        value = str(text or "").strip().upper()
        if value in {"R", "REM"}:
            return "REM"
        return value

    def _stage_file_for_channel(self, index: int) -> Optional[Path]:
        if self._source_path is None:
            return None
        candidates = [
            self._source_path.parent / f"sleep_epoch_features_mne_EXG_{index}_Full.csv",
            self._source_path.parent / f"sleep_epoch_features_mne_EXG_{index}.csv",
        ]
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _common_stage_epochs(
        self,
        channel_indices: Sequence[int],
        stages: Sequence[str],
    ) -> List[tuple[int, float, float]]:
        selected = {self._stage_key(stage) for stage in stages}
        common: Optional[set[int]] = None
        epoch_times: Dict[int, tuple[float, float]] = {}
        for index in channel_indices:
            path = self._stage_file_for_channel(index)
            if path is None:
                raise ValueError(f"未找到 EXG {index} 的睡眠分期文件")
            frame = pd.read_csv(path)
            required = {"epoch", "start_s", "end_s", "stage_yasa"}
            if not required.issubset(set(frame.columns)):
                raise ValueError(f"{path.name} 缺少睡眠分期列")
            epochs: set[int] = set()
            for row in frame.itertuples(index=False):
                stage = self._stage_key(getattr(row, "stage_yasa"))
                if stage not in selected:
                    continue
                epoch = int(getattr(row, "epoch"))
                epochs.add(epoch)
                if epoch not in epoch_times:
                    epoch_times[epoch] = (
                        float(getattr(row, "start_s")),
                        float(getattr(row, "end_s")),
                    )
            common = epochs if common is None else common.intersection(epochs)
        return [
            (epoch, epoch_times[epoch][0], epoch_times[epoch][1])
            for epoch in sorted(common or set())
            if epoch in epoch_times and epoch_times[epoch][1] > epoch_times[epoch][0]
        ]

    def _values_for_loaded_window(
        self,
        loaded: Sequence[tuple[np.ndarray, float, str, str, float]],
        mode: str,
    ) -> List[np.ndarray]:
        raw_values: List[np.ndarray] = []
        display_values: List[np.ndarray] = []
        for raw, _fs, _label, unit, _actual_start in loaded:
            arr = np.asarray(raw, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            baseline = float(np.nanmedian(finite)) if finite.size else 0.0
            if self._source_path is not None and is_openbci_brainflow_eeg_file(self._source_path):
                raw_arr = arr - baseline
                display_arr = raw_arr
            else:
                raw_arr = arr
                display_arr = arr
            raw_values.append(raw_arr)
            display_values.append(display_arr)
        if mode == "raw":
            return display_values
        if mode not in COMPARE_BAND_RANGES:
            return display_values
        low, high = COMPARE_BAND_RANGES[mode]
        filtered: List[np.ndarray] = []
        for values in raw_values:
            base = bandpass_filter(values, self._sample_rate)
            filtered.append(
                bandpass_filter(base, self._sample_rate, low_hz=low, high_hz=high)
            )
        return filtered

    def _phase_rows_for_sleep_stages(
        self,
        visible: Sequence[tuple[int, str]],
        mode: str,
        ref_index: int,
        stages: Sequence[str],
    ) -> tuple[List[List[str]], int]:
        if self._source_path is None:
            raise ValueError("当前比较窗口没有关联原始文件路径")
        channel_indices = [index for index, _label in visible]
        epochs = self._common_stage_epochs(channel_indices, stages)
        if not epochs:
            raise ValueError("未找到可见通道共同属于所选睡眠阶段的 epoch")
        max_lag = int(round(max(self._sample_rate, 1.0) * 1.0))
        labels = {index: label for index, label in visible}
        metrics: Dict[int, List[tuple[Optional[float], Optional[float], float]]] = {
            index: [] for index, _label in visible
        }
        ref_pos = channel_indices.index(ref_index) if ref_index in channel_indices else 0
        if float(epochs[-1][2] - epochs[0][1]) <= 3600.0:
            blocks: List[List[tuple[int, float, float]]] = [epochs]
        else:
            blocks = []
            for epoch in epochs:
                if not blocks or epoch[1] > blocks[-1][-1][2] + 1e-6:
                    blocks.append([epoch])
                else:
                    blocks[-1].append(epoch)
        for block in blocks:
            block_start = float(block[0][1])
            block_end = float(block[-1][2])
            loaded = load_eeg_file_channels_window(
                self._source_path,
                channel_indices,
                start_s=block_start,
                duration_s=float(block_end - block_start),
            )
            block_values = self._values_for_loaded_window(loaded, mode)
            if ref_pos >= len(block_values):
                continue
            for _epoch, start_s, end_s in block:
                i0 = max(0, int(round((float(start_s) - block_start) * self._sample_rate)))
                i1 = max(i0 + 1, int(round((float(end_s) - block_start) * self._sample_rate)))
                ref = block_values[ref_pos][i0:i1]
                for pos, index in enumerate(channel_indices):
                    if pos >= len(block_values):
                        continue
                    values = block_values[pos][i0:i1]
                    metrics[index].append(self._phase_metrics(ref, values, max_lag))
                if QtWidgets.QApplication.instance() is not None:
                    QtWidgets.QApplication.processEvents()
            if QtWidgets.QApplication.instance() is not None:
                QtWidgets.QApplication.processEvents()
        rows: List[List[str]] = []
        for index, _label in visible:
            vals = metrics.get(index, [])
            zeros = [zero for zero, _best, _lag in vals if zero is not None]
            bests = [best for _zero, best, _lag in vals if best is not None]
            lags = [lag for _zero, best, lag in vals if best is not None]
            zero_med = float(np.median(zeros)) if zeros else None
            best_med = float(np.median(bests)) if bests else None
            lag_med = float(np.median(lags)) if lags else 0.0
            rows.append(
                [
                    labels[index],
                    "-" if zero_med is None else f"{zero_med:.3f}",
                    "-" if best_med is None else f"{best_med:.3f}",
                    f"{lag_med:.0f}",
                    str(len(vals)),
                    self._phase_comment(zero_med, best_med, lag_med),
                ]
            )
        return rows, len(epochs)

    @QtCore.pyqtSlot()
    def _show_phase_analysis(self) -> None:
        if not self._channels or self._time_s.size == 0:
            QtWidgets.QMessageBox.information(self, "同相分析", "当前没有可分析的数据")
            return
        visible = self._visible_channel_items()
        if len(visible) < 2:
            QtWidgets.QMessageBox.information(self, "同相分析", "请至少勾选两个通道")
            return
        mode = self._current_mode()
        mode_values = self._mode_values(mode)
        ref_index = self._current_index if any(i == self._current_index for i, _ in visible) else visible[0][0]
        if ref_index >= len(mode_values):
            QtWidgets.QMessageBox.information(self, "同相分析", "参考通道数据无效")
            return
        n_total = min(len(values) for values in mode_values if len(values) > 0)
        max_n = min(n_total, int(round(max(self._sample_rate, 1.0) * 20.0)), 10000)
        if max_n < 8:
            QtWidgets.QMessageBox.information(self, "同相分析", "有效样本过少")
            return
        ref_label = self._channels[ref_index][0]

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("同相分析")
        dialog.resize(780, 420)
        layout = QtWidgets.QVBoxLayout(dialog)
        mode_label = COMPARE_BAND_LABELS.get(mode, mode)
        duration_s = max_n / max(self._sample_rate, 1.0)
        clipped = "，长窗口仅分析开头片段" if max_n < n_total else ""
        hint = QtWidgets.QLabel(
            f"参考通道：{ref_label}；波形：{mode_label}；分析时长：{duration_s:.1f}s{clipped}"
        )
        layout.addWidget(hint)

        stage_box = QtWidgets.QGroupBox("分析范围")
        stage_layout = QtWidgets.QHBoxLayout(stage_box)
        stage_layout.setContentsMargins(10, 18, 10, 10)
        stage_layout.setSpacing(12)
        current_check = QtWidgets.QCheckBox("当前窗口", stage_box)
        current_check.setChecked(True)
        stage_layout.addWidget(current_check)
        stage_checks: Dict[str, QtWidgets.QCheckBox] = {}
        for stage in ("W", "N1", "N2", "N3", "REM"):
            check = QtWidgets.QCheckBox(stage, stage_box)
            check.toggled.connect(lambda checked, c=current_check: c.setChecked(False) if checked else None)
            stage_layout.addWidget(check)
            stage_checks[stage] = check
        current_check.toggled.connect(
            lambda checked: [check.setChecked(False) for check in stage_checks.values()] if checked else None
        )
        analyze_button = QtWidgets.QPushButton("重新分析", stage_box)
        stage_layout.addWidget(analyze_button)
        stage_layout.addStretch(1)
        layout.addWidget(stage_box)

        table = QtWidgets.QTableWidget(dialog)
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(
            ["通道", "零滞后相关", "最佳相关", "最佳滞后(ms)", "epoch数", "判断"]
        )
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, stretch=1)

        def _fill_table(rows: Sequence[Sequence[str]]) -> None:
            table.setRowCount(len(rows))
            for row, cells in enumerate(rows):
                for col, text in enumerate(cells):
                    item = QtWidgets.QTableWidgetItem(str(text))
                    if col:
                        item.setTextAlignment(QtCore.Qt.AlignCenter)
                    table.setItem(row, col, item)
            table.resizeColumnsToContents()

        def _run_analysis() -> None:
            try:
                selected_stages = [
                    stage for stage, check in stage_checks.items() if check.isChecked()
                ]
                if selected_stages:
                    current_check.blockSignals(True)
                    current_check.setChecked(False)
                    current_check.blockSignals(False)
                    rows, epoch_count = self._phase_rows_for_sleep_stages(
                        visible,
                        mode,
                        ref_index,
                        selected_stages,
                    )
                    hint.setText(
                        f"参考通道：{ref_label}；波形：{mode_label}；阶段：{','.join(selected_stages)}；"
                        f"共同 epoch：{epoch_count}；表内为各 epoch 指标中位数"
                    )
                    _fill_table(rows)
                    return
                current_check.setChecked(True)
                rows = self._phase_rows_for_current_window(
                    visible,
                    mode_values,
                    ref_index,
                    max_n,
                )
                hint.setText(
                    f"参考通道：{ref_label}；波形：{mode_label}；分析时长：{duration_s:.1f}s{clipped}"
                )
                _fill_table(rows)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dialog, "同相分析失败", str(exc))

        analyze_button.clicked.connect(_run_analysis)
        _run_analysis()
        close_button = QtWidgets.QPushButton("关闭", dialog)
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignRight)
        dialog.exec_()

    @QtCore.pyqtSlot()
    def redraw(self) -> None:
        self._plot.figure.clear()
        if not self._channels or self._time_s.size == 0:
            self._plot._draw_empty("当前窗口没有可比较的多通道数据")
            return
        visible = [(index, self._channels[index]) for index, _label in self._visible_channel_items()]
        if not visible:
            self._plot._draw_empty("请至少勾选一个通道")
            return
        ax = self._plot.figure.add_subplot(111)
        max_pts = _adaptive_max_plot_points(int(self._time_s.size), self._sample_rate)
        mode = self._current_mode()
        mode_values = self._mode_values(mode)
        plot_values: List[np.ndarray] = []
        for index, _item in visible:
            if index < len(mode_values):
                plot_values.append(np.asarray(mode_values[index], dtype=np.float64))
        offsets: Dict[int, float] = {}
        if self._offset_display_check.isChecked() and plot_values:
            spans = []
            for values in plot_values:
                finite = values[np.isfinite(values)]
                if finite.size:
                    spans.append(float(np.nanpercentile(finite, 95.0) - np.nanpercentile(finite, 5.0)))
            spacing = max(float(np.median(spans)) if spans else 1.0, 1.0) * 1.25
            for order, (index, _item) in enumerate(visible):
                offsets[index] = (len(visible) - 1 - order) * spacing
        for index, (label, _raw_values, _display_values, _note) in visible:
            if index >= len(mode_values):
                continue
            values = np.asarray(mode_values[index], dtype=np.float64)
            if offsets:
                values = values + offsets.get(index, 0.0)
            t, y = _downsample_pair(self._time_s, values, max_points=max_pts)
            color = MULTI_CHANNEL_COLORS[index % len(MULTI_CHANNEL_COLORS)]
            ax.plot(t, y, color=color, linewidth=0.85, label=label)
        ax.set_xlabel("Time (s)", fontsize=9)
        mode_label = COMPARE_BAND_LABELS.get(mode, mode)
        ylabel = self._y_label if mode == "raw" else f"{mode_label} ({self._y_label})"
        if offsets:
            ylabel = f"{ylabel} + offset"
        ax.set_ylabel(ylabel, fontsize=9)
        if self._y_limits is not None and mode == "raw" and not offsets:
            ax.set_ylim(*self._y_limits)
        ax.grid(True, which="major", alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=2)
        start = float(self._time_s[0])
        end = float(self._time_s[-1] + 1.0 / max(self._sample_rate, 1.0))
        self._plot.figure.suptitle(
            f"{self._source_name} | {mode_label} | {start:.1f}-{end:.1f}s | {self._sample_rate:.0f} Hz",
            fontsize=10,
        )
        self._plot.refresh()


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

    def clear(self) -> None:
        self._time_s = np.zeros(0, dtype=np.float64)
        self._channels.clear()
        self._source_name = ""
        self._axes = []
        self._remove_mask = None
        self._y_limits = None
        self._raw_y_label = "raw"
        self._display_note = ""
        self._draw_empty("选择 CSV/EDF 后点击加载")

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

    def redraw(self) -> None:
        self._figure.clear()
        self._axes = []
        visible = [name for name in self._visible if name in self._channels]
        if "raw" in self._channels and "raw" not in visible:
            visible = ["raw"] + visible
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
        self._figure.suptitle(
            f"{title} | {self._sample_rate:.0f} Hz | waveform always shown{mark_tip}{display_tip}",
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

