"""Embedded matplotlib views for offline EEG waveform and power plots."""

from __future__ import annotations

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
MAX_PLOT_POINTS = 25000

class OfflineEegFileInfo(NamedTuple):
    channel_labels: List[str]
    channel_rates: List[float]
    channel_samples: List[int]
    channel_units: List[str]


# Higher cap for short windows so local EDF browsing keeps detail.
MAX_PLOT_POINTS_SHORT = 200000


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
    frame = pd.read_csv(path)
    if "ch1_raw" in frame.columns:
        series = frame["ch1_raw"]
    elif "ch1" in frame.columns:
        series = frame["ch1"]
    else:
        series = frame.iloc[:, 0]
    raw = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if raw.size < 8:
        raise ValueError(f"有效样本过少 ({raw.size}): {path.name}")

    sample_rate = float(DEFAULT_SAMPLE_RATE)
    if "time_s" in frame.columns and len(frame) >= 2:
        dt = np.diff(pd.to_numeric(frame["time_s"], errors="coerce").dropna().to_numpy())
        dt = dt[dt > 0]
        if dt.size:
            sample_rate = float(1.0 / np.median(dt))
    return raw, sample_rate


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
        raw, sample_rate = _load_eeg_csv_with_rate(path)
        return OfflineEegFileInfo(["CH1"], [float(sample_rate)], [int(raw.size)], ["raw"])

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
        raw, sample_rate = _load_eeg_csv_with_rate(path)
        return raw, sample_rate, "CH1", "raw"

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
        i_min = int(np.argmin(segment))
        i_max = int(np.argmax(segment))
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
    ) -> tuple[int, float]:
        """Load a raw segment into the waveform view."""
        raw_arr = np.asarray(raw, dtype=np.float64)
        if raw_arr.size < 8:
            raise ValueError(f"鏈夋晥鏍锋湰杩囧皯 ({raw_arr.size})")
        fs = float(sample_rate)
        bands = extract_band_waveforms(raw_arr, fs)
        n = int(raw_arr.size)
        time_s = float(time_offset_s) + np.arange(n, dtype=np.float64) / fs
        self._sample_rate = fs
        self._time_s = time_s
        self._channels = {"raw": raw_arr, **bands}
        self._source_name = source_name or "offline"
        self._raw_y_label = y_label or "raw"
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
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=8)
            self._axes.append(ax)
        axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
        self._bind_time_axis_ticks(self._axes)
        title = self._source_name or "offline"
        mark_tip = " | bad segments marked" if reject_spans else ""
        self._figure.suptitle(
            f"{title} | {self._sample_rate:.0f} Hz | waveform always shown{mark_tip}",
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

