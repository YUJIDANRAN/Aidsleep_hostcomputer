"""主显示区嵌入的 matplotlib 视图：离线分层波形 + 功率分析图。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

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


def _time_tick_step_seconds(span_s: float) -> float:
    """按可见时间跨度选刻度间隔；宽窗用 60 s，放大后加密以保证仍有刻度。"""
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


def load_eeg_csv_with_rate(path: Path) -> tuple[np.ndarray, float]:
    """读取 EEG CSV，返回 (raw, sample_rate)。优先 ch1_raw，可用 time_s 估采样率。"""
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


def _downsample_pair(
    time_s: np.ndarray, values: np.ndarray, max_points: int = MAX_PLOT_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """过长时按桶保留 min/max，避免等间隔抽点造成假尖峰/形貌失真。"""
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
    """工具栏 + FigureCanvas 公共底座。"""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        empty_message: str = "",
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)
        if empty_message:
            self._draw_empty(empty_message)

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
    """单段 band_power / 多段功率对比图。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, empty_message="填写段时间后点击「功率对比」")

    def clear(self) -> None:
        self._draw_empty("填写段时间后点击「功率对比」")


class OfflineRhythmStackView(_MatplotlibHostView):
    """分层波形：勾选通道显示；工具栏可拖拽平移/缩放横纵轴。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._sample_rate = float(DEFAULT_SAMPLE_RATE)
        self._time_s = np.zeros(0, dtype=np.float64)
        self._channels: Dict[str, np.ndarray] = {}
        self._visible: List[str] = list(CHANNEL_ORDER)
        self._source_name = ""
        self._axes: List = []
        super().__init__(parent, empty_message="选择 CSV 并点击「加载」")

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
        self._draw_empty("选择 CSV 并点击「加载」")

    def load_raw(
        self,
        raw: np.ndarray,
        sample_rate: float,
        *,
        source_name: str = "",
        time_offset_s: float = 0.0,
    ) -> tuple[int, float]:
        """用已截取/拼接的 raw 填充视图；横轴 = time_offset_s + 局部时间。"""
        raw_arr = np.asarray(raw, dtype=np.float64)
        if raw_arr.size < 8:
            raise ValueError(f"有效样本过少 ({raw_arr.size})")
        fs = float(sample_rate)
        bands = extract_band_waveforms(raw_arr, fs)
        n = int(raw_arr.size)
        time_s = float(time_offset_s) + np.arange(n, dtype=np.float64) / fs
        self._sample_rate = fs
        self._time_s = time_s
        self._channels = {"raw": raw_arr, **bands}
        self._source_name = source_name or "offline"
        self._visible = ["raw"]
        self.redraw()
        self._enable_pan()
        return n, self._sample_rate

    def load_file(self, path: Path) -> tuple[int, float]:
        raw, sample_rate = load_eeg_csv_with_rate(path)
        return self.load_raw(raw, sample_rate, source_name=path.name)

    def set_visible_channels(self, names: Sequence[str]) -> None:
        # raw 始终保留在最前
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
            self._draw_empty("请先加载 CSV")

    def redraw(self) -> None:
        self._figure.clear()
        self._axes = []
        visible = [name for name in self._visible if name in self._channels]
        if "raw" in self._channels and "raw" not in visible:
            visible = ["raw"] + visible
        if not visible or self._time_s.size == 0:
            self._draw_empty("请先加载 CSV")
            return

        n = len(visible)
        axes = self._figure.subplots(n, 1, sharex=True, squeeze=False)
        for i, name in enumerate(visible):
            ax = axes[i, 0]
            t, y = _downsample_pair(self._time_s, self._channels[name])
            color = CHANNEL_COLORS.get(name, "#1976D2")
            ax.plot(t, y, color=color, linewidth=0.8)
            ax.set_ylabel(CHANNEL_LABELS.get(name, name), fontsize=9)
            ax.grid(True, which="major", alpha=0.25)
            ax.tick_params(labelsize=8)
            self._axes.append(ax)
        axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
        self._bind_time_axis_ticks(self._axes)
        title = self._source_name or "offline"
        self._figure.suptitle(
            f"{title}  |  {self._sample_rate:.0f} Hz  |  raw 必显，勾选叠加节律",
            fontsize=10,
        )
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _bind_time_axis_ticks(self, axes: Sequence) -> None:
        """缩放/平移后仍保持横轴刻度可见（宽窗 60 s，缩放过细则加密）。"""
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
        # sharex：挂在任一轴即可，缩放工具栏会触发 xlim_changed
        axes[-1].callbacks.connect("xlim_changed", _apply)