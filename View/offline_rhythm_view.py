"""离线 EEG 文件查看：分层显示 raw / 各节律，支持拖拽平移与缩放坐标轴。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import sys

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
    n = int(values.size)
    if n <= max_points:
        return time_s, values
    step = int(np.ceil(n / max_points))
    return time_s[::step], values[::step]


class OfflineRhythmStackView(QtWidgets.QWidget):
    """分层波形：勾选通道显示；工具栏可拖拽平移/缩放横纵轴。"""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._sample_rate = float(DEFAULT_SAMPLE_RATE)
        self._time_s = np.zeros(0, dtype=np.float64)
        self._channels: Dict[str, np.ndarray] = {}
        self._visible: List[str] = list(CHANNEL_ORDER)
        self._source_name = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

        self._axes: List = []
        self._draw_empty("选择 CSV 并点击「加载」")

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
        self._draw_empty("选择 CSV 并点击「加载」")

    def load_file(self, path: Path) -> tuple[int, float]:
        raw, sample_rate = load_eeg_csv_with_rate(path)
        bands = extract_band_waveforms(raw, sample_rate)
        n = int(raw.size)
        time_s = np.arange(n, dtype=np.float64) / float(sample_rate)
        self._sample_rate = float(sample_rate)
        self._time_s = time_s
        self._channels = {"raw": raw.astype(np.float64, copy=False), **bands}
        self._source_name = path.name
        self._visible = [name for name in CHANNEL_ORDER if name in self._channels]
        self.redraw()
        # 默认进入平移模式，便于拖动横/纵轴
        try:
            if self._toolbar.mode != "pan/zoom":
                self._toolbar.pan()
        except Exception:
            pass
        return n, self._sample_rate

    def set_visible_channels(self, names: Sequence[str]) -> None:
        ordered = [name for name in CHANNEL_ORDER if name in names and name in self._channels]
        if ordered == self._visible and self.has_data:
            return
        self._visible = ordered
        if self.has_data:
            self.redraw()
        elif not ordered:
            self._draw_empty("请勾选至少一种波形（raw / 节律）")

    def redraw(self) -> None:
        self._figure.clear()
        self._axes = []
        visible = [name for name in self._visible if name in self._channels]
        if not visible or self._time_s.size == 0:
            self._draw_empty("请勾选至少一种波形（raw / 节律）")
            return

        n = len(visible)
        axes = self._figure.subplots(n, 1, sharex=True, squeeze=False)
        for i, name in enumerate(visible):
            ax = axes[i, 0]
            t, y = _downsample_pair(self._time_s, self._channels[name])
            color = CHANNEL_COLORS.get(name, "#1976D2")
            ax.plot(t, y, color=color, linewidth=0.8)
            ax.set_ylabel(CHANNEL_LABELS.get(name, name), fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
            self._axes.append(ax)
        axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
        title = self._source_name or "offline"
        self._figure.suptitle(
            f"{title}  |  {self._sample_rate:.0f} Hz  |  工具栏：平移/缩放拖动坐标轴",
            fontsize=10,
        )
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _draw_empty(self, message: str) -> None:
        self._figure.clear()
        self._axes = []
        ax = self._figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#616161")
        self._canvas.draw_idle()
