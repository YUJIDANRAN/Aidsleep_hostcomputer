"""主窗口：串口采集、节律/原始波形显示、UI 事件绑定。"""

from __future__ import annotations  ## 前向引用类型

import csv  ## 定时测试 EEG raw 导出
import io  ## 捕获 power_cal 分析输出
import os  ## 环境变量读串口
import sys  ## 路径与退出
import time  ## 状态栏刷新节流
from contextlib import redirect_stdout
from collections import deque  ## 波形点缓冲
from datetime import datetime  ## CSV 文件名时间戳
from pathlib import Path  ## 项目根路径
from typing import Deque, Dict, Iterable, List, Optional, Tuple  ## 类型标注

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets  ## Qt 界面

_ROOT = Path(__file__).resolve().parent.parent  ## 项目根目录
_VIEW_DIR = Path(__file__).resolve().parent
_ALGO_DIR = _ROOT / "Algorithm"
_CTRL_DIR = _ROOT / "Controller"
for _path in (_ROOT, _ALGO_DIR, _VIEW_DIR, _CTRL_DIR):
    _ps = str(_path)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from power_cal import (  ## 节律滤波与频段定义
    BAND_LABELS,
    DEFAULT_SAMPLE_RATE,
    EEG_BANDS,
    RhythmStreamProcessor,
    compare_multi_segment_band_powers,
    compute_band_powers,
    plot_band_powers,
    plot_segment_power_comparison,
    run_analysis,
    slice_signal,
)
from MovementArtifact import (  ## 阈值拒绝 / 质量标记
    RealtimeAlphaThresholdRejector,
    RealtimeQualityGate,
    build_threshold_rejection,
    clean_raw_signal,
)
from iaf_echt import IAFEcHTPhaseTracker
from osc_data import (  ## 振子三轴加速度节律分析
    ACCEL_DISPLAY_UNIT,
    OSC_BANDS,
    OSC_SAMPLE_RATE,
    BAND_LABELS as OSC_BAND_LABELS,
    OscStreamProcessor,
    SENSOR_UNITS_PER_G,
    GRAVITY_M_S2,
)
from MainWindow import Ui_MainWindow  ## Designer 生成的 UI
from ks1082_serial import (
    Ks1082Serial,
    MCU_SAMPLE_RATE,
    RAW_TYPICAL_AMP,
    RAW_TYPICAL_MID,
    list_ports,
)
from analysis_plot_view import (
    CHANNEL_ORDER,
    AnalysisPlotView,
    OfflineRhythmStackView,
    load_eeg_csv_with_rate,
)
from oscillator_serial import OscillatorSerial, BUFFER_SIZE as OSC_BUFFER_SIZE
from controller import (
    AlphaPhaseSnapshot,
    SLEEP_AID_WARMUP_SEC,
    SleepAidStimulusController,
    StereoAudioController,
    StereoAudioParams,
)


DEFAULT_SERIAL_PORT = "COM6"  ## EEG 默认串口
DEFAULT_OSC_SERIAL_PORT = "COM7"  ## 振子默认串口
DEFAULT_SERIAL_BAUDRATE = 115200
DEFAULT_SERIAL_BYTESIZE = 8
DEFAULT_SERIAL_PARITY = "N"
DEFAULT_SERIAL_STOPBITS = 1
LOG_MAX_LINES = 20  ## 日志最大行数
OSC_DISPLAY_RATE = 100  ## 界面波形约 100 点/秒（由 1000 Hz 每 10 点取 1 点）
OSC_DECIM_FACTOR = max(1, int(OSC_SAMPLE_RATE / OSC_DISPLAY_RATE))  ## 1000→100 显示
OSC_PLOT_WINDOW_SECONDS = 60.0
OSC_PLOT_MAX_POINTS = int(OSC_DISPLAY_RATE * OSC_PLOT_WINDOW_SECONDS)
OSC_DEFAULT_Y_MID = 0.0
OSC_DEFAULT_Y_AMP = 5.0  ## 加速度默认半幅 (m/s²)，约 ±0.5g
OSC_MIN_Y_AMP = 0.01
OSC_MAX_Y_AMP = 50.0
OSC_Y_AXIS_LABEL = f"加速度 ({ACCEL_DISPLAY_UNIT})"
OSC_AXIS_COLORS = {
    "x": "#C62828",
    "y": "#2E7D32",
    "z": "#1565C0",
}
M_FREQ_PLOT_COLOR = "#6A1B9A"
AUDIO_DEFAULT_LEFT_FREQ_HZ = "10"
AUDIO_DEFAULT_RIGHT_FREQ_HZ = "10"
AUDIO_DEFAULT_DURATION_SEC = "5"
AUDIO_DEFAULT_LEFT_PHASE_DEG = "0"
AUDIO_DEFAULT_RIGHT_PHASE_DEG = "90"
DEFAULT_EEG_CSV_DIR = _ROOT / "Result"  ## timeEdit 定时测试默认 CSV 目录
TEST_WARMUP_SEC = 10.0  ## 有定时测试时，开始后前 10 s 不保存数据
LONG_RECORD_CHUNK_SEC = 300.0  ## 长时记录：每 5 分钟自动存一份
WAVEFORM_WAKE_SEC = 300.0  ## 长时记录：波形显示窗口 5 分钟
LONG_NORMAL_RAW_MIN = 900  ## 长时记录正常段：raw 下限
LONG_NORMAL_RAW_MAX = 1300  ## 长时记录正常段：raw 上限
LONG_NORMAL_MIN_DURATION_SEC = 120.0  ## 长时记录正常段：最短连续时长
MAX_COMPARE_SEGMENTS = 4
EEG_REJECT_RATE_WARN = 0.20  ## 拒绝率超过 20% 时提示本次采集不宜用于分析
TROUGH_CAL_SCRIPT = _ALGO_DIR / "TroughCalibrator.py"
RAW_DISPLAY_RATE = 100  ## 波形显示约 100 点/秒
RAW_DECIM_FACTOR = max(1, int(MCU_SAMPLE_RATE / RAW_DISPLAY_RATE))  ## 500→100 降采样比
RAW_PLOT_WINDOW_SECONDS = 60.0  ## 波形时间窗 (s)
RAW_PLOT_MAX_POINTS = int(RAW_DISPLAY_RATE * RAW_PLOT_WINDOW_SECONDS)  ## 缓冲约 200 点
SERIAL_POLL_INTERVAL_MS = max(2, int(1000 / MCU_SAMPLE_RATE))  ## 串口轮询 2 ms
MAX_SAMPLES_PER_POLL = 80  ## 单次 poll 最多处理的原始样本数，防止恢复后积压卡顿
EEG_MULTI_CHANNEL_COUNT = 6  ## six-channel EEG protocol: CH1~CH6
MIN_PLOT_WIDTH = 200  ## 绘图区最小宽度
MIN_PLOT_HEIGHT = 120  ## 绘图区最小高度
AXIS_MARGIN_LEFT = 72  ## 左留白给 Y 轴
AXIS_MARGIN_BOTTOM = 48  ## 下留白给 X 轴
AXIS_MARGIN_TOP = 36  ## 上留白
AXIS_MARGIN_RIGHT = 16  ## 右留白
Y_AXIS_WHEEL_BAND = 48  ## Y 轴滚轮热区：左留白 + 绘图区左缘条带 (px)
X_AXIS_WHEEL_BAND = 48  ## X 轴滚轮热区：下留白 + 绘图区下缘条带 (px)
WAVE_LINE_WIDTH = 1.0  ## 波形线宽（屏幕像素，cosmetic pen）
PLOT_REFRESH_INTERVAL_S = 0.04  ## 波形最高刷新约 25 fps
AXIS_REDRAW_INTERVAL_S = 0.5  ## 坐标轴最多约 2 fps 重绘
Y_AXIS_REDRAW_REL_EPS = 0.08  ## Y 轴量程变化超过 8% 才重绘刻度
RAW_DEFAULT_Y_MID = float(RAW_TYPICAL_MID)  ## raw 默认 Y 中心（0~4096）
RAW_DEFAULT_Y_AMP = float(RAW_TYPICAL_AMP)  ## raw 默认 Y 半幅
RHYTHM_DEFAULT_Y_MID =15.0  ## 节律波形典型中心（0~500）
RHYTHM_DEFAULT_Y_AMP = 50.0  ## 节律半幅，使 0~500 居中显示并留边距
RHYTHM_PLOT_COLORS = {  ## 节律波形线条（较 BAND_COLORS 更深）
    "delta": "#0D47A1",
    "theta": "#1B5E20",
    "alpha": "#E65100",
    "beta": "#B71C1C",
    "gamma": "#4A148C",
}
WHEEL_Y_ZOOM_STEP = 1.12  ## 滚轮每档 Y 轴缩放比例
WHEEL_X_ZOOM_STEP = 1.12  ## 滚轮每档 X 轴（时间）缩放比例
MIN_X_TIME_ZOOM = 0.2  ## X 最小缩放（时间窗最长）
MIN_X_VISIBLE_POINTS = 50  ## X 至少显示点数，过少会连成三角折线
MIN_Y_AMP = 20.0  ## Y 半幅下限
MAX_Y_AMP = 200000.0  ## Y 半幅上限
SLEEP_AID_BURST_LOG_EVERY = 10  ## burst 日志每 N 次写一条，减轻 QTextEdit 重绘


def _to_alpha_phase_snapshot(snap) -> AlphaPhaseSnapshot:
    """EcHTSnapshot → Controller 使用的 AlphaPhaseSnapshot。"""
    return AlphaPhaseSnapshot(
        ready=bool(snap.ready),
        phase_rad=float(getattr(snap, "phase_rad", 0.0)),
        inst_freq_hz=float(getattr(snap, "inst_freq_hz", 10.0)),
        seconds_to_trough=float(getattr(snap, "seconds_to_trough", 0.0)),
    )


def _make_wave_pen(color: str) -> QtGui.QPen:
    pen = QtGui.QPen(QtGui.QColor(color), WAVE_LINE_WIDTH)
    pen.setCosmetic(True)  ## 线宽不随缩放变粗
    return pen


def _format_tick(value: float) -> str:
    """坐标轴刻度文字格式化。"""
    av = abs(value)
    if av >= 1000:
        return f"{value:.0f}"
    if av >= 10:
        return f"{value:.1f}"
    if av >= 1:
        return f"{value:.2f}"
    if av >= 0.01:
        return f"{value:.3f}"
    if av >= 1e-6:
        return f"{value:.2e}"
    return f"{value:.1e}"


class AlphaWaveformView(QtWidgets.QGraphicsView):
    """实时波形视图（原始或各节律带通）。"""

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)  ## 调用父类构造
        self._sample_rate = sample_rate  ## 时间轴采样率
        self._scene = QtWidgets.QGraphicsScene(self)  ## 绘图场景
        self.setScene(self._scene)  ## 绑定场景
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)  ## 关闭抗锯齿，减轻卡顿
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.MinimalViewportUpdate)  ## 只重绘变化区域
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  ## 隐藏横滚条
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  ## 隐藏竖滚条
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#FAFAFA")))  ## 浅灰背景
        self.setFocusPolicy(QtCore.Qt.WheelFocus)  ## 鼠标在波形区时可接收滚轮
        self.setFrameShape(QtWidgets.QFrame.NoFrame)  ## 去掉边框，与 graphicsView 重合
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)  ## 左上角对齐，避免留白

        self._plot_left = AXIS_MARGIN_LEFT  ## 绘图区左边界
        self._plot_top = AXIS_MARGIN_TOP  ## 绘图区上边界
        self._legend_text = "RAW CH1"  ## 图例（configure_display 会覆盖）
        self._y_axis_label = "幅值"
        self._max_points = RAW_PLOT_MAX_POINTS  ## 数据缓冲最大点数
        self._plot_width = MIN_PLOT_WIDTH  ## 绘图区宽（随控件 resize 更新）
        self._plot_height = MIN_PLOT_HEIGHT  ## 绘图区高（随控件 resize 更新）

        self._path = QtGui.QPainterPath()  ## 波形路径
        self._path_item = self._scene.addPath(  ## 添加到场景
            self._path,
            pen=_make_wave_pen("#1976D2"),
        )
        self._path_item.setZValue(2)  ## 曲线在网格之上
        self._path_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)

        self._legend = self._scene.addText("")  ## 图例文字项
        self._legend.setDefaultTextColor(QtGui.QColor("#424242"))
        self._legend.setZValue(3)  ## 图例在最上层
        self._legend.setAcceptedMouseButtons(QtCore.Qt.NoButton)

        self._points: Deque[float] = deque(maxlen=self._max_points)  ## 波形数据环缓冲
        self._multi_mode = False  ## True：多曲线（如 X/Y/Z 三轴）
        self._series_buffers: Dict[str, Deque[float]] = {}
        self._series_path_items: Dict[str, QtWidgets.QGraphicsPathItem] = {}
        self._axis_items: list[QtWidgets.QGraphicsItem] = []  ## 坐标轴图形项
        self._reject_flags: Deque[int] = deque(maxlen=self._max_points)
        self._reject_items: list[QtWidgets.QGraphicsItem] = []
        self._y_mid = 0.0  ## Y 轴中心
        self._y_amp = 1.0  ## Y 轴半幅
        self._min_y_amp = MIN_Y_AMP  ## 滚轮缩放 Y 半幅下限
        self._max_y_amp = MAX_Y_AMP  ## 滚轮缩放 Y 半幅上限
        self._fixed_y_axis = False  ## True：固定刻度，滚轮缩放/平移
        self._use_full_plot_height = False  ## True：波形纵向尽量铺满绘图区
        self._x_time_zoom = 1.0  ## X 时间缩放，越大可见时间窗越短
        self._x_pan_ratio = 0.0  ## X 时间平移，0=最新在右，1=最旧在左
        self._path_dirty = False  ## 是否有待刷新波形
        self._last_plot_refresh = 0.0  ## 上次波形刷新时刻
        self._last_axis_refresh = 0.0  ## 上次坐标轴重绘时刻
        self._last_axis_y_mid = 0.0  ## 上次绘制刻度时的 Y 中心
        self._last_axis_y_amp = 1.0  ## 上次绘制刻度时的 Y 半幅
        self._update_scene_rect()
        self.viewport().installEventFilter(self)  ## 在 viewport 捕获滚轮，避免被场景项吞掉

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            watched is self.viewport()
            and event.type() == QtCore.QEvent.Wheel
            and self._fixed_y_axis
            and self._wheel_axis_zone(event) is not None
        ):
            self._handle_wheel_event(event)
            return True  ## 吞掉滚轮，防止 QGraphicsView 默认滚动造成“平移”
        return super().eventFilter(watched, event)

    def _sync_plot_size_from_viewport(self) -> None:
        """按 viewport 实际像素设置绘图区，使场景与显示框同宽高比。"""
        vp = self.viewport().size()
        plot_w = max(vp.width() - AXIS_MARGIN_LEFT - AXIS_MARGIN_RIGHT, MIN_PLOT_WIDTH)
        plot_h = max(vp.height() - AXIS_MARGIN_TOP - AXIS_MARGIN_BOTTOM, MIN_PLOT_HEIGHT)
        changed = int(plot_w) != int(self._plot_width) or int(plot_h) != int(self._plot_height)
        self._plot_width = float(plot_w)
        self._plot_height = float(plot_h)
        self._update_scene_rect()
        if changed:
            self._path_dirty = True
            self._refresh_plot(force=True)

    def set_sample_rate(self, sample_rate: float) -> None:
        self._sample_rate = max(sample_rate, 1.0)  ## 更新 fs，最小 1 Hz

    def configure_display(
        self,
        *,
        legend: str,
        sample_rate: float,
        max_points: int,
        line_color: str = "#1976D2",
        fixed_y_axis: bool = False,
        y_mid: Optional[float] = None,
        y_amp: Optional[float] = None,
        use_full_plot_height: bool = False,
        y_axis_label: str = "幅值",
        min_y_amp: float = MIN_Y_AMP,
        max_y_amp: float = MAX_Y_AMP,
        multi_series: Optional[Dict[str, str]] = None,
    ) -> None:
        """切换 Alpha / 原始波形显示参数；multi_series 非空时启用多曲线。"""
        self._legend_text = legend  ## 更新图例
        self._y_axis_label = y_axis_label
        self._min_y_amp = min_y_amp
        self._max_y_amp = max_y_amp
        self._sample_rate = max(sample_rate, 1.0)  ## 更新时间轴 fs
        self._max_points = max(max_points, 2)  ## 更新缓冲长度
        self._fixed_y_axis = fixed_y_axis
        self._use_full_plot_height = use_full_plot_height
        if fixed_y_axis:
            self._x_time_zoom = 1.0  ## 切换模式时重置时间缩放
            self._x_pan_ratio = 0.0
        if y_mid is not None:
            self._y_mid = y_mid
        if y_amp is not None:
            self._y_amp = float(max(y_amp, self._min_y_amp))
        if multi_series:
            self._enable_multi_series(multi_series)
        else:
            self._disable_multi_series()
            self._points = deque(maxlen=self._max_points)  ## 重建环缓冲
            self._reject_flags = deque(maxlen=self._max_points)
            self._path_item.setPen(_make_wave_pen(line_color))  ## 更新颜色与线宽
        self._path_dirty = True
        self._sync_plot_size_from_viewport()
        self._update_axes_if_needed(self._point_count(), force=True)
        self._fit_scene()
        self._refresh_plot(force=True)  ## 重绘

    def update_legend(self, legend: str) -> None:
        if legend == self._legend_text:
            return
        self._legend_text = legend
        self._path_dirty = True
        self._refresh_plot(force=True)

    def _point_count(self) -> int:
        if self._multi_mode and self._series_buffers:
            return max(len(buf) for buf in self._series_buffers.values())
        return len(self._points)

    def _disable_multi_series(self) -> None:
        for item in self._series_path_items.values():
            self._scene.removeItem(item)
        self._series_path_items.clear()
        self._series_buffers.clear()
        self._multi_mode = False
        self._path_item.setVisible(True)

    def _enable_multi_series(self, series_colors: Dict[str, str]) -> None:
        self._disable_multi_series()
        self._multi_mode = True
        self._path_item.setVisible(False)
        for name, color in series_colors.items():
            self._series_buffers[name] = deque(maxlen=self._max_points)
            path_item = self._scene.addPath(
                QtGui.QPainterPath(),
                pen=_make_wave_pen(color),
            )
            path_item.setZValue(2)
            path_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self._series_path_items[name] = path_item

    def _scene_width(self) -> float:
        return self._plot_left + self._plot_width + AXIS_MARGIN_RIGHT  ## 场景总宽

    def _scene_height(self) -> float:
        return self._plot_top + self._plot_height + AXIS_MARGIN_BOTTOM  ## 场景总高

    def _update_scene_rect(self) -> None:
        self._scene.setSceneRect(0, 0, self._scene_width(), self._scene_height())  ## 设置矩形

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.refresh_layout()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)  ## 父类处理
        self._sync_plot_size_from_viewport()
        self._fit_scene()  ## 窗口缩放时铺满 viewport

    def clear(self) -> None:
        self._points.clear()  ## 清空数据
        self._reject_flags.clear()
        for buf in self._series_buffers.values():
            buf.clear()
        self._path_dirty = True
        self._last_plot_refresh = 0.0
        self._last_axis_refresh = 0.0
        self._refresh_plot(force=True)  ## 重绘空波形

    def append_alpha(self, alpha: float, rejected: bool = False) -> None:
        self._points.append(alpha)  ## 追加点
        self._reject_flags.append(1 if rejected else 0)
        self._path_dirty = True
        self._refresh_plot()

    def append_alphas(
        self,
        alphas: Iterable[float],
        reject_flags: Optional[Iterable[bool]] = None,
    ) -> None:
        if reject_flags is None:
            for alpha in alphas:
                self._points.append(alpha)  ## 批量追加
                self._reject_flags.append(0)
        else:
            for alpha, rejected in zip(alphas, reject_flags):
                self._points.append(alpha)  ## 批量追加
                self._reject_flags.append(1 if rejected else 0)
        self._path_dirty = True
        self._refresh_plot()

    def append_multi(self, values: Dict[str, float]) -> None:
        for name, value in values.items():
            buf = self._series_buffers.get(name)
            if buf is not None:
                buf.append(value)
        self._path_dirty = True
        self._refresh_plot()

    def append_multi_batch(self, batches: Iterable[Dict[str, float]]) -> None:
        for values in batches:
            for name, value in values.items():
                buf = self._series_buffers.get(name)
                if buf is not None:
                    buf.append(value)
        self._path_dirty = True
        self._refresh_plot()

    def flush_plot(self) -> None:
        """强制刷新尚未绘制的波形（窗口关闭前等场景）。"""
        self._refresh_plot(force=True)

    def _plot_height_frac(self) -> float:
        return 0.5 if self._use_full_plot_height else 0.45

    def _y_scale(self) -> float:
        return (self._plot_height * self._plot_height_frac()) / self._y_amp

    def _value_at_scene_y(self, scene_y: float) -> float:
        y_center = self._plot_top + self._plot_height * 0.5
        return self._y_mid + (y_center - scene_y) / self._y_scale()

    def _zoom_y_at(self, scene_y: float, delta: int) -> None:
        """以鼠标所在高度为锚点缩放 Y 量程（刻度范围变宽/变窄）。"""
        step = WHEEL_Y_ZOOM_STEP if delta > 0 else 1.0 / WHEEL_Y_ZOOM_STEP
        anchor = self._value_at_scene_y(scene_y)
        new_amp = float(max(self._min_y_amp, min(self._max_y_amp, self._y_amp / step)))
        if abs(new_amp - self._y_amp) < 1e-6:
            return
        y_center = self._plot_top + self._plot_height * 0.5
        new_scale = (self._plot_height * self._plot_height_frac()) / new_amp
        self._y_amp = new_amp
        self._y_mid = anchor - (y_center - scene_y) / new_scale

    def _pan_y(self, delta: int) -> None:
        shift = self._y_amp * 0.08 * (1 if delta > 0 else -1)
        self._y_mid += shift

    def _reset_view_scroll(self) -> None:
        """清除 QGraphicsView 内部滚动偏移，避免滚轮后整图平移。"""
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)

    def _fit_scene(self) -> None:
        self.resetTransform()
        self.fitInView(
            self.sceneRect(),
            QtCore.Qt.IgnoreAspectRatio,
        )
        self.viewport().update()

    def refresh_layout(self) -> None:
        """窗口显示/缩放后同步场景尺寸并重绘坐标轴与波形。"""
        self._sync_plot_size_from_viewport()
        self._update_axes_if_needed(self._point_count(), force=True)
        self._fit_scene()
        self._path_dirty = True
        self._refresh_plot(force=True)

    def _use_x_window(self) -> bool:
        """raw 模式是否启用 X 时间窗缩放/平移（默认全缓冲线性映射）。"""
        return self._fixed_y_axis and (
            self._x_time_zoom > 1.001 or self._x_pan_ratio > 1e-9
        )

    def _wheel_axis_zone(self, event: QtGui.QWheelEvent) -> Optional[str]:
        """Return y/x target for wheel zoom; plot area defaults to x zoom."""
        vp = self.viewport()
        vp_pos = vp.mapFromGlobal(event.globalPos())
        scene_pos = self.mapToScene(self.mapFromGlobal(event.globalPos()))
        left = self._plot_left
        top = self._plot_top
        right = left + self._plot_width
        bottom = top + self._plot_height

        in_y_margin = vp_pos.x() < AXIS_MARGIN_LEFT + Y_AXIS_WHEEL_BAND
        in_y_axis_line = (
            left - 12 <= scene_pos.x() <= left + Y_AXIS_WHEEL_BAND
            and top <= scene_pos.y() <= bottom
        )
        in_x_margin = vp_pos.y() >= vp.height() - AXIS_MARGIN_BOTTOM
        in_x_axis_line = (
            bottom - X_AXIS_WHEEL_BAND <= scene_pos.y() <= bottom + 12
            and left <= scene_pos.x() <= left + self._plot_width
        )

        in_y = in_y_margin or in_y_axis_line
        in_x = in_x_margin or in_x_axis_line
        if in_y and in_x:
            return "y" if vp_pos.x() < vp_pos.y() else "x"
        if in_y:
            return "y"
        if in_x:
            return "x"
        if left <= scene_pos.x() <= right and top <= scene_pos.y() <= bottom:
            return "x"
        return None

    def _handle_wheel_event(self, event: QtGui.QWheelEvent) -> bool:
        """Wheel: plot/x-axis zooms time, y-axis/Ctrl zooms amplitude, Shift pans."""
        if not self._fixed_y_axis:
            return False
        zone = self._wheel_axis_zone(event)
        if zone is None:
            return False
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        x_step = WHEEL_X_ZOOM_STEP if delta > 0 else 1.0 / WHEEL_X_ZOOM_STEP
        scene_pos = self.mapToScene(self.mapFromGlobal(event.globalPos()))
        modifiers = event.modifiers()
        shift_pan = bool(modifiers & QtCore.Qt.ShiftModifier)
        if modifiers & QtCore.Qt.ControlModifier:
            zone = "y"
        if zone == "y":
            if shift_pan:
                self._pan_y(delta)
            else:
                self._zoom_y_at(scene_pos.y(), delta)
        elif shift_pan:
            pan = 0.06 * (1 if delta > 0 else -1)
            self._x_pan_ratio = float(max(0.0, min(1.0, self._x_pan_ratio + pan)))
        else:
            self._x_time_zoom = float(
                max(
                    MIN_X_TIME_ZOOM,
                    min(self._max_x_time_zoom(), self._x_time_zoom * x_step),
                )
            )
        self._path_dirty = True
        self._refresh_plot(force=True)
        self._update_axes_if_needed(len(self._points), force=True)
        self._reset_view_scroll()
        return True

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._fixed_y_axis and self._wheel_axis_zone(event) is not None:
            self._handle_wheel_event(event)
            event.accept()
            return
        event.ignore()

    def _max_x_time_zoom(self) -> float:
        """X 最大缩放，保证可见点数不少于 MIN_X_VISIBLE_POINTS。"""
        return max(1.0, self._max_points / MIN_X_VISIBLE_POINTS)

    def _x_visible_points(self) -> float:
        """当前 X 方向可见采样点数。"""
        zoomed = self._max_points / self._x_time_zoom
        return max(2.0, min(float(self._max_points), zoomed))

    def _x_window(self, count: int) -> tuple[int, int]:
        """可见时间窗对应的整数索引区间 [start, end)。"""
        if count <= 0:
            return 0, 0
        visible = min(int(self._x_visible_points()), count)
        max_start = max(0, count - visible)
        start = int(max_start * (1.0 - self._x_pan_ratio) + 0.5)
        start = max(0, min(start, max_start))
        end = min(count, start + visible)
        if end <= start:
            end = min(count, start + 1)
        return start, end

    def _x_view_start_index(self, count: int) -> float:
        """可见时间窗左边界对应的数据索引。"""
        return float(self._x_window(count)[0])

    def _x_visible_duration(self) -> float:
        """当前可见时间窗长度 (s)。"""
        count = self._point_count()
        if count <= 1:
            return 1.0 / self._sample_rate
        if self._use_x_window():
            start, end = self._x_window(count)
            return max(end - start - 1, 0) / self._sample_rate
        return (count - 1) / self._sample_rate

    def _value_to_y(self, value: float) -> float:
        y_center = self._plot_top + self._plot_height * 0.5  ## 垂直中心
        return y_center - (value - self._y_mid) * self._y_scale()  ## 上正下负

    def _index_to_x(self, index: int, count: int) -> float:
        if count <= 1:
            return self._plot_left + self._plot_width * 0.5
        if self._use_x_window():
            start, end = self._x_window(count)
            span = max(end - start - 1, 1)
            rel = max(0.0, min(1.0, (index - start) / span))
            return self._plot_left + rel * (self._plot_width - 1.0)
        return self._plot_left + index * (self._plot_width - 1.0) / (count - 1)

    def _clear_axes(self) -> None:
        for item in self._axis_items:
            self._scene.removeItem(item)  ## 移除旧坐标元素
        self._axis_items.clear()

    def _add_axis_item(self, item: QtWidgets.QGraphicsItem) -> None:
        item.setZValue(1)  ## 网格在曲线下方
        item.setAcceptedMouseButtons(QtCore.Qt.NoButton)  ## 不拦截滚轮
        self._scene.addItem(item)
        self._axis_items.append(item)  ## 记录以便下次清除

    def _draw_axes(self, count: int) -> None:
        self._clear_axes()  ## 先清旧轴

        left = self._plot_left
        top = self._plot_top
        right = left + self._plot_width - 1  ## 绘图区右边界
        bottom = top + self._plot_height  ## 绘图区下边界

        pen_axis = QtGui.QPen(QtGui.QColor("#555555"), 1.2)  ## 边框笔
        pen_grid = QtGui.QPen(QtGui.QColor("#DDDDDD"), 1.0, QtCore.Qt.DashLine)  ## 网格笔

        self._add_axis_item(  ## 绘制区外框
            self._scene.addRect(
                left,
                top,
                self._plot_width - 1,
                self._plot_height,
                pen_axis,
            )
        )

        y_ticks = [  ## Y 轴 5 档刻度值
            self._y_mid + self._y_amp,
            self._y_mid + self._y_amp * 0.5,
            self._y_mid,
            self._y_mid - self._y_amp * 0.5,
            self._y_mid - self._y_amp,
        ]
        for val in y_ticks:
            y = self._value_to_y(val)  ## 值转坐标
            self._add_axis_item(
                self._scene.addLine(left, y, right, y, pen_grid)  ## 水平网格线
            )
            label = self._scene.addText(_format_tick(val))  ## 刻度文字
            label.setDefaultTextColor(QtGui.QColor("#444444"))
            label.setPos(4, y - 10)
            self._add_axis_item(label)

        plot_count = count if count > 0 else self._point_count()
        if plot_count <= 1:
            duration = (
                self._x_visible_duration()
                if self._fixed_y_axis
                else 1.0 / self._sample_rate
            )
            x_times = [(0.0, left), (duration, right)]
        else:
            if self._fixed_y_axis:
                duration = self._x_visible_duration()  ## raw：随滚轮变化的时间窗
            else:
                duration = (plot_count - 1) / self._sample_rate
            x_times = [  ## 0/25/50/75/100% 时间点
                (0.0, left),
                (duration * 0.25, left + (right - left) * 0.25),
                (duration * 0.5, left + (right - left) * 0.5),
                (duration * 0.75, left + (right - left) * 0.75),
                (duration, right),
            ]

        for t_sec, x_pos in x_times:
            self._add_axis_item(
                self._scene.addLine(x_pos, top, x_pos, bottom, pen_grid)  ## 竖网格线
            )
            label = self._scene.addText(f"{t_sec:.2f}s")  ## 时间标签
            label.setDefaultTextColor(QtGui.QColor("#444444"))
            label.setPos(x_pos - 18, bottom + 6)
            self._add_axis_item(label)

        x_title = self._scene.addText("时间 (s)")  ## X 轴标题
        x_title.setDefaultTextColor(QtGui.QColor("#333333"))
        x_title.setPos(left + self._plot_width * 0.42, bottom + 28)
        self._add_axis_item(x_title)

        y_title = self._scene.addText(self._y_axis_label)  ## Y 轴标题
        y_title.setDefaultTextColor(QtGui.QColor("#333333"))
        y_title.setPos(8, top + self._plot_height * 0.45)
        self._add_axis_item(y_title)

    def _update_axes_if_needed(self, count: int, *, force: bool = False) -> None:
        if self._fixed_y_axis:
            if not force:
                return  ## raw 固定刻度：仅滚轮或切换模式时重绘坐标轴
            self._draw_axes(count)
            self._last_axis_refresh = time.monotonic()
            self._last_axis_y_mid = self._y_mid
            self._last_axis_y_amp = self._y_amp
            return
        now = time.monotonic()
        y_changed = (
            abs(self._y_mid - self._last_axis_y_mid) > self._y_amp * Y_AXIS_REDRAW_REL_EPS
            or abs(self._y_amp - self._last_axis_y_amp)
            / max(self._last_axis_y_amp, 1e-6)
            > Y_AXIS_REDRAW_REL_EPS
        )
        if not force and not y_changed and (now - self._last_axis_refresh) < AXIS_REDRAW_INTERVAL_S:
            return  ## 刻度变化不大且未到重绘间隔则跳过
        self._draw_axes(count)
        self._last_axis_refresh = now
        self._last_axis_y_mid = self._y_mid
        self._last_axis_y_amp = self._y_amp

    def _clear_reject_items(self) -> None:
        for item in self._reject_items:
            self._scene.removeItem(item)
        self._reject_items.clear()

    def _draw_reject_flags(self, count: int) -> None:
        self._clear_reject_items()
        if self._multi_mode or count <= 1 or not self._reject_flags:
            return
        flags = list(self._reject_flags)
        if len(flags) < count:
            flags = [0] * (count - len(flags)) + flags
        elif len(flags) > count:
            flags = flags[-count:]

        if self._use_x_window():
            visible_start, visible_end = self._x_window(count)
        else:
            visible_start, visible_end = 0, count

        top = self._plot_top
        height = self._plot_height
        brush = QtGui.QBrush(QtGui.QColor(211, 47, 47, 46))
        pen = QtGui.QPen()
        pen.setStyle(QtCore.Qt.NoPen)
        i = visible_start
        while i < visible_end:
            if not flags[i]:
                i += 1
                continue
            start = i
            while i < visible_end and flags[i]:
                i += 1
            end = min(i, visible_end - 1)
            x0 = self._index_to_x(start, count)
            x1 = self._index_to_x(end, count)
            if x1 <= x0:
                x1 = x0 + 1.0
            rect = self._scene.addRect(x0, top, x1 - x0, height, pen, brush)
            rect.setZValue(1.5)
            rect.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self._reject_items.append(rect)

    def _refresh_plot(self, *, force: bool = False) -> None:
        if not self._path_dirty and not force:
            return
        now = time.monotonic()
        if not force and (now - self._last_plot_refresh) < PLOT_REFRESH_INTERVAL_S:
            return  ## 限帧，避免每包串口数据都整屏重绘
        self._last_plot_refresh = now
        self._path_dirty = False
        self._rebuild_path(force_axes=force)

    def _rebuild_path(self, *, force_axes: bool = False) -> None:
        count = self._point_count()
        self._legend.setPlainText(self._legend_text)  ## 更新图例
        self._legend.setPos(self._plot_left + 8, self._plot_top + 6)

        if count == 0:  ## 无数据：空路径
            self._path = QtGui.QPainterPath()
            self._path_item.setPath(self._path)
            self._clear_reject_items()
            for path_item in self._series_path_items.values():
                path_item.setPath(QtGui.QPainterPath())
            if not self._fixed_y_axis:
                self._y_mid = 0.0
                self._y_amp = 1.0
            self._update_axes_if_needed(0, force=True)
            return

        if self._multi_mode:
            self._clear_reject_items()
            all_values: list[float] = []
            for name, buf in self._series_buffers.items():
                values = list(buf)
                path = QtGui.QPainterPath()
                for i, value in enumerate(values):
                    x = self._index_to_x(i, count)
                    y = self._value_to_y(value)
                    if i == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)
                    all_values.append(value)
                self._series_path_items[name].setPath(path)
            if not self._fixed_y_axis and all_values:
                self._y_mid = sum(all_values) / len(all_values)
                amplitude = max(abs(v - self._y_mid) for v in all_values)
                if amplitude < 1e-6:
                    amplitude = 1.0
                self._y_amp = amplitude * 1.15
            self._update_axes_if_needed(count, force=force_axes)
            return

        values = list(self._points)  ## 拷贝为列表
        finite_values = [value for value in values if np.isfinite(value)]
        if not self._fixed_y_axis:  ## Alpha 模式才自动跟踪 Y 量程
            if not finite_values:
                self._y_mid = 0.0
                self._y_amp = 1.0
                self._path_item.setPath(QtGui.QPainterPath())
                self._update_axes_if_needed(count, force=force_axes)
                return
            self._y_mid = sum(finite_values) / len(finite_values)  ## 均值作 Y 中心
            amplitude = max(abs(v - self._y_mid) for v in finite_values)  ## 最大偏离
            if amplitude < 1e-6:
                amplitude = 1.0  ## 避免除零
            self._y_amp = amplitude * 1.15  ## 留 15% 边距

        self._draw_reject_flags(count)
        path = QtGui.QPainterPath()
        drawing = False
        for i, value in enumerate(values):
            if not np.isfinite(value):
                drawing = False
                continue
            x = self._index_to_x(i, count)
            y = self._value_to_y(value)
            if not drawing:
                path.moveTo(x, y)
                drawing = True
            else:
                path.lineTo(x, y)

        self._path = path
        self._path_item.setPath(self._path)  ## 仅更新曲线路径
        self._update_axes_if_needed(count, force=force_axes)  ## 坐标轴低频/按需重绘


class MultiChannelEegView(QtWidgets.QWidget):
    """Six-channel EEG split view; each channel owns one waveform widget."""

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._views: List[AlphaWaveformView] = [
            AlphaWaveformView(sample_rate=sample_rate, parent=self)
            for _ in range(EEG_MULTI_CHANNEL_COUNT)
        ]
        self._mode = "raw"
        self._active_channels: tuple[int, ...] = tuple(range(EEG_MULTI_CHANNEL_COUNT))
        self._last_config: tuple = ()

    def setGeometry(self, rect: QtCore.QRect) -> None:  # type: ignore[override]
        super().setGeometry(rect)
        self._layout_views()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout_views()

    def _layout_views(self) -> None:
        gap = 6
        active = getattr(self, "_active_channels", tuple(range(EEG_MULTI_CHANNEL_COUNT)))
        rows = max(1, len(active))
        width = max(1, self.width())
        height = max(1, self.height())
        cell_h = max(1, (height - gap * (rows - 1)) // rows)
        active_set = set(active)
        visible_row = 0
        for index, view in enumerate(self._views):
            if index not in active_set:
                view.hide()
                continue
            y = visible_row * (cell_h + gap)
            cell_height = max(1, height - y) if visible_row == rows - 1 else cell_h
            view.setGeometry(QtCore.QRect(0, y, width, cell_height))
            view.show()
            view.refresh_layout()
            visible_row += 1

    def configure_display(
        self,
        *,
        mode: str,
        sample_rate: float,
        max_points: int,
        line_color: str,
        fixed_y_axis: bool,
        y_mid: float,
        y_amp: float,
        y_axis_label: str = "Amplitude",
        min_y_amp: float = MIN_Y_AMP,
        max_y_amp: float = MAX_Y_AMP,
        active_channels: Optional[Iterable[int]] = None,
    ) -> None:
        channels = tuple(active_channels) if active_channels is not None else tuple(range(EEG_MULTI_CHANNEL_COUNT))
        if not channels:
            channels = (0,)
        self._active_channels = tuple(ch for ch in channels if 0 <= ch < EEG_MULTI_CHANNEL_COUNT)
        config = (mode, sample_rate, max_points, line_color, fixed_y_axis, y_mid, y_amp, y_axis_label, min_y_amp, max_y_amp, self._active_channels)
        if config == self._last_config:
            return
        self._mode = mode
        self._last_config = config
        for index, view in enumerate(self._views):
            view.configure_display(
                legend=f"CH{index + 1} {mode.upper()}",
                sample_rate=sample_rate,
                max_points=max_points,
                line_color=line_color,
                fixed_y_axis=fixed_y_axis,
                y_mid=y_mid,
                y_amp=y_amp,
                use_full_plot_height=True,
                y_axis_label=y_axis_label,
                min_y_amp=min_y_amp,
                max_y_amp=max_y_amp,
            )
        self._layout_views()

    def clear(self) -> None:
        for view in self._views:
            view.clear()

    def refresh_layout(self) -> None:
        self._layout_views()
        for view in self._views:
            view.refresh_layout()

    def append_channel_values_batch(self, batches: Iterable[Iterable[float]]) -> None:
        per_channel = [[] for _ in range(EEG_MULTI_CHANNEL_COUNT)]
        active_set = set(self._active_channels)
        for values in batches:
            for index, value in enumerate(values):
                if index < EEG_MULTI_CHANNEL_COUNT and index in active_set:
                    per_channel[index].append(value)
        for index, values in enumerate(per_channel):
            if values:
                self._views[index].append_alphas_plain(values)


class Ks1082MainWindow(QtWidgets.QMainWindow):
    """主窗口：串口采集 + 波形显示。"""

    _audio_stopped = QtCore.pyqtSignal()

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.ui = Ui_MainWindow()  ## UI 对象
        self.ui.setupUi(self)  ## 加载 Designer 布局
        self.ui.checkBox_6.setChecked(True)  ## 默认 EEG raw data 显示
        self.ui.checkBox_11.setChecked(True)  ## 默认振子 M_Fre 显示
        self.setWindowTitle("EEG 实时波形 · CH1 · RAW")  ## 窗口标题

        if port:
            self.ui.lineEdit_5.setText(port)  ## 命令行传入时写入 UI
        elif not self.ui.lineEdit_5.text().strip():
            self.ui.lineEdit_5.setText(_resolve_serial_port())  ## UI 为空则用默认
        self.ui.lineEdit_5.setPlaceholderText("COM6")  ## EEG 串口输入框提示
        if not self.ui.lineEdit_4.text().strip():
            self.ui.lineEdit_4.setText(_resolve_osc_serial_port())
        self.ui.lineEdit_4.setPlaceholderText("COM7")  ## 振子串口输入框提示

        self._port = ""  ## 当前已连接的 EEG 串口
        self._osc_port = ""  ## 当前已连接的振子串口
        self._baudrate = baudrate  ## 波特率
        self._rebuild_control_panels(baudrate)
        self._link: Optional[Ks1082Serial] = None  ## EEG 串口
        self._osc_link: Optional[OscillatorSerial] = None  ## 振子串口连接
        self._rhythm = RhythmStreamProcessor(sample_rate=sample_rate)  ## EEG 节律流式滤波
        self._multi_rhythm = [
            RhythmStreamProcessor(sample_rate=sample_rate)
            for _ in range(EEG_MULTI_CHANNEL_COUNT)
        ]
        self._alpha_rejector = RealtimeAlphaThresholdRejector(sample_rate)
        self._quality_gate = RealtimeQualityGate(sample_rate)
        self._alpha_display_total_points = 0
        self._alpha_display_rejected_points = 0
        self._eeg_base_legend = "RAW CH1"
        self._osc_proc = OscStreamProcessor(sample_rate=OSC_SAMPLE_RATE)  ## 振子流式滤波
        self._audio_stopped.connect(self._on_audio_stopped)
        self._audio_controller = StereoAudioController(
            on_stopped=self._audio_stopped.emit,
        )
        self._sleep_aid_tracker = IAFEcHTPhaseTracker(sample_rate=sample_rate)
        try:
            self._sleep_aid_controller = SleepAidStimulusController(
                on_triggered=self._on_sleep_aid_burst,
            )
        except ImportError:
            self._sleep_aid_controller = None
        self._setup_audio_ui_defaults()
        self._setup_timed_test_ui()
        self._setup_sleep_aid_window_ui()
        self._setup_session_name_ui()
        self._eeg_save_root: Optional[Path] = None  ## None → 默认 Result
        self._setup_compare_segments_ui()
        self._setup_offline_viewer_ui()
        self._last_eeg_session_dir: Optional[Path] = None
        self._offline_csv_path: Optional[Path] = None
        self._offline_view_active = False  ## 离线分层波形查看中
        self._analysis_plot_active = False  ## 功率对比图显示中
        self._sleep_aid_last_warm_sec = -1
        self._sleep_aid_burst_count = 0
        self._sleep_aid_burst_record: List[dict] = []
        self._sleep_aid_timed_auto = False  ## 定时记录自动启停助眠
        self._running = False  ## 默认不采集
        self._test_duration_sec: Optional[float] = None  ## 实际记录时长 (s)
        self._test_started_at: Optional[float] = None  ## 采集开始时刻 (monotonic)
        self._eeg_raw_record: List[int] = []  ## 定时测试期间记录的 CH1 raw
        self._long_record_active = False  ## 本次测试是否启用长时记录模式
        self._long_session_dir: Optional[Path] = None  ## 长时记录固定会话目录
        self._long_chunks_saved = 0  ## 已自动保存的完整 5 分钟段数
        self._waveform_display_until: Optional[float] = None  ## 波形显示截止 (monotonic)
        self._waveform_sleep_logged = False  ## 是否已提示波形休眠
        self._sample_count = 0  ## EEG 已处理样本数
        self._osc_sample_count = 0  ## 振子已处理样本数
        self._no_data_ticks = 0  ## EEG 无数据 poll 计数
        self._osc_no_data_ticks = 0  ## 振子无数据 poll 计数
        self._last_status_update = 0.0  ## 上次刷新状态栏时刻
        self._display_mode = ""  ## 当前 EEG 显示模式
        self._eeg_channel_mode = "single"  ## single | multi
        self._osc_display_mode = ""  ## 当前振子显示模式
        self._osc_display_kind = "band"  ## band | axis
        self._osc_axis_display_key: tuple[str, ...] = ()
        self._decim_counter = 0  ## EEG 500 Hz→100 Hz 降采样计数
        self._osc_decim_counter = 0  ## 振子 1000 Hz→100 Hz 降采样计数
        self._active_view = "eeg"  ## 当前波形页：eeg | osc
        self._display_checkboxes = {  ## EEG UI 勾选 ↔ 显示模式
            "delta": self.ui.checkBox_3,
            "theta": self.ui.checkBox_4,
            "alpha": self.ui.checkBox,
            "beta": self.ui.checkBox_2,
            "gamma": self.ui.checkBox_5,
            "raw": self.ui.checkBox_6,
        }
        self._osc_display_checkboxes = {  ## 振子 UI 勾选 ↔ 显示模式
            "delta": self.ui.checkBox_12,
            "theta": self.ui.checkBox_8,
            "alpha": self.ui.checkBox_10,
            "beta": self.ui.checkBox_7,
            "gamma": self.ui.checkBox_9,
            "m_freq": self.ui.checkBox_11,
        }
        self._osc_axis_checkboxes = {  ## 振子三轴加速度
            "x": self.ui.checkBox_13,
            "y": self.ui.checkBox_14,
            "z": self.ui.checkBox_15,
        }
        self._eeg_channel_checkboxes = {
            0: self.ui.checkBox_eeg_ch1,
            1: self.ui.checkBox_eeg_ch2,
            2: self.ui.checkBox_eeg_ch3,
            3: self.ui.checkBox_eeg_ch4,
            4: self.ui.checkBox_eeg_ch5,
            5: self.ui.checkBox_eeg_ch6,
        }

        self._osc_graphics_view = QtWidgets.QGraphicsView(self.ui.page_2)  ## 振子页占位
        self._osc_graphics_view.setGeometry(self.ui.graphicsView.geometry())
        self._osc_graphics_view.hide()

        self._waveform = AlphaWaveformView(  ## EEG：覆盖在 page 内 graphicsView 位置
            sample_rate=sample_rate,
            parent=self.ui.page,
        )
        self._offline_view = OfflineRhythmStackView(parent=self.ui.page)
        self._offline_view.hide()
        self._multi_waveform = MultiChannelEegView(
            sample_rate=sample_rate,
            parent=self.ui.page,
        )
        self._multi_waveform.hide()
        self._analysis_plot = AnalysisPlotView(parent=self.ui.page)
        self._analysis_plot.hide()
        self._osc_waveform = AlphaWaveformView(  ## 振子：覆盖在 page_2
            sample_rate=OSC_SAMPLE_RATE,
            parent=self.ui.page_2,
        )
        self._sync_waveform_geometry()
        self._sync_multi_waveform_geometry()
        self._sync_offline_geometry()
        self._sync_analysis_plot_geometry()
        self._sync_osc_waveform_geometry()
        self._waveform.show()
        self._waveform.raise_()
        self._multi_waveform.hide()
        self._osc_waveform.show()
        self._osc_waveform.raise_()
        self.ui.graphicsView.hide()  ## 用自定义波形控件替代占位 QGraphicsView

        self.ui.pushButton.pressed.connect(self.on_toggle_capture)  ## 启动/停止按钮
        self.ui.pushButton_2.pressed.connect(self._switch_to_eeg_view)  ## 节律图页
        self.ui.pushButton_6.pressed.connect(self._switch_to_osc_view)  ## 振子页
        self.ui.pushButton_7.pressed.connect(self.on_toggle_audio)  ## 音频开始/停止
        self.ui.pushButton_8.pressed.connect(self.on_toggle_sleep_aid)  ## 助眠闭环 burst
        self._poll_timer = QtCore.QTimer(self)  ## 串口轮询定时器
        self._poll_timer.setInterval(SERIAL_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self.on_poll_serial)

        for mode, checkbox in self._display_checkboxes.items():
            checkbox.toggled.connect(
                lambda checked, m=mode: self._on_display_checkbox_toggled(m, checked)
            )
        self.ui.comboBox_eeg_channel_mode.currentIndexChanged.connect(
            self._on_eeg_channel_mode_changed
        )
        for channel, checkbox in self._eeg_channel_checkboxes.items():
            checkbox.toggled.connect(
                lambda checked, ch=channel: self._on_eeg_channel_checkbox_toggled(ch, checked)
            )
        self._set_eeg_channel_checkboxes_enabled(False)
        for mode, checkbox in self._osc_display_checkboxes.items():
            checkbox.toggled.connect(
                lambda checked, m=mode: self._on_osc_checkbox_toggled(m, checked)
            )
        for axis, checkbox in self._osc_axis_checkboxes.items():
            checkbox.toggled.connect(
                lambda checked, a=axis: self._on_osc_axis_checkbox_toggled(a, checked)
            )
        self._open_serial()  ## 打开 EEG 串口
        self._open_osc_serial()  ## 打开振子串口
        self._poll_timer.start()  ## 开始轮询
        QtCore.QTimer.singleShot(0, self._log_available_ports)
        self._apply_display_mode("raw")  ## 默认 EEG raw 波形参数
        self._apply_osc_display_mode("m_freq")  ## 默认振子 M_Fre 波形
        if self._link is not None and self._link.is_open:
            self._running = True  ## 串口可用则自动采集，动态刷新 raw 波形
            self._link._parser.reset()
            self._rhythm.reset()
            self._alpha_rejector.reset()
            self._reset_alpha_display_stats()
            self._decim_counter = 0
            self._waveform.clear()
            self._multi_waveform.clear()
        if self._osc_link is not None and self._osc_link.is_open:
            self._running = True
            self._osc_proc.reset()
            self._osc_decim_counter = 0
            self._osc_waveform.clear()
        if self._running:
            self._begin_timed_test_if_configured()
        self._refresh_capture_button()
        if self._running:
            self._log("默认 raw 模式，已自动开始采集")

    @staticmethod
    def _panel_label(text: str, parent: QtWidgets.QWidget) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, parent)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumHeight(28)
        return label

    @staticmethod
    def _set_combo_text(combo: QtWidgets.QComboBox, text: str) -> None:
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _rebuild_control_panels(self, baudrate: int) -> None:
        """Apply Designer-defined layout defaults and wire runtime-only behavior."""
        for name in (
            "textEdit",
            "textEdit_2",
            "textEdit_3",
            "textEdit_4",
            "textEdit_5",
            "textEdit_6",
            "textEdit_7",
            "textEdit_8",
            "textEdit_9",
            "textEdit_10",
        ):
            widget = getattr(self.ui, name, None)
            if widget is not None:
                widget.hide()

        self.ui.groupBox.setTitle("功能控制")
        self.ui.groupBox_2.setTitle("助眠音频")
        self.ui.functionScrollArea.setWidgetResizable(True)
        self.ui.functionScrollArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ui.functionScrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ui.functionScrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self.ui.pushButton_2.setFixedHeight(34)
        self.ui.pushButton_6.setFixedHeight(34)
        self.ui.groupBox_eeg_display.setMinimumHeight(210)
        self.ui.groupBox_osc_display.setMinimumHeight(210)
        self.ui.groupBox_2.setMinimumHeight(285)
        self.ui.pushButton_7.setMinimumSize(92, 42)
        self.ui.pushButton_8.setMinimumSize(180, 42)
        self.ui.pushButton_8.setStyleSheet(
            "QPushButton { background:#2F7D3B; color:white; font-weight:bold; border:none; border-radius:6px; }"
            "QPushButton:hover { background:#256A31; }"
        )

        self.ui.plainTextEdit.setMinimumSize(420, 82)
        self.ui.plainTextEdit.setMaximumHeight(96)
        self.ui.timeEdit.setMaximumHeight(40)
        self.ui.lcdNumber.setMinimumHeight(36)
        self.ui.lineEdit_13.setMaximumHeight(34)
        self.ui.lineEdit_3.setMaximumHeight(34)
        self.ui.lineEdit_sleep_aid_end.setMaximumHeight(34)
        self.ui.groupBox_save.setTitle("助眠时段")
        self.ui.groupBox_save.setToolTip(
            "仅定时记录生效：时间为记录阶段内秒数（不含 10 s 预热）；"
            f"助眠暖机 {SLEEP_AID_WARMUP_SEC:g}s 会提前启动"
        )
        self.ui.groupBox_session_name.setTitle("保存命名")
        self.ui.groupBox_session_name.setToolTip(
            "填写 XXX 时子目录为 时间戳_XXX/；留空则为 时间戳/。"
            "点「保存位置...」选根目录，取消则仍用默认 Result。"
        )
        self.ui.pushButton_save_location.setToolTip(
            "选择保存根目录；取消则默认工程下 Result/"
        )
        try:
            self.ui.pushButton_save_location.clicked.disconnect()
        except TypeError:
            pass
        self.ui.pushButton_save_location.clicked.connect(self._on_choose_save_location)

        self.ui.pushButton_serial_toggle.setMinimumSize(96, 32)
        self.ui.pushButton_serial_toggle.setText("打开串口")
        self.ui.pushButton_serial_toggle.setStyleSheet(
            "QPushButton { background:#2563EB; color:white; font-weight:bold; border:none; border-radius:6px; }"
            "QPushButton:hover { background:#1D4ED8; }"
        )
        for edit in (self.ui.lineEdit_5, self.ui.lineEdit_4):
            edit.setMinimumWidth(92)
        self.ui.lineEdit_session_name.setMinimumWidth(120)

        try:
            self.ui.pushButton_browse_offline.clicked.disconnect()
        except TypeError:
            pass
        self.ui.pushButton_browse_offline.clicked.connect(self._browse_offline_eeg_csv)
        try:
            self.ui.pushButton_load_offline.clicked.disconnect()
        except TypeError:
            pass
        self.ui.pushButton_load_offline.clicked.connect(self._load_offline_eeg_csv)
        try:
            self.ui.pushButton_clear_offline.clicked.disconnect()
        except TypeError:
            pass
        self.ui.pushButton_clear_offline.clicked.connect(self._exit_offline_view)
        try:
            self.ui.pushButton_5.clicked.disconnect()
        except TypeError:
            pass
        self.ui.pushButton_5.clicked.connect(self._on_power_compare_clicked)
        try:
            self.ui.pushButton_serial_toggle.clicked.disconnect()
        except TypeError:
            pass
        self.ui.pushButton_serial_toggle.clicked.connect(self._toggle_serial_connection)

        self.ui.pushButton.setMinimumSize(130, 78)
        self.ui.bottomPanel.raise_()
        self._layout_main_regions()

    def _layout_main_regions(self) -> None:
        """Resize the plot, right controls, and bottom controls with the window."""
        cw = self.ui.centralwidget
        width = max(640, cw.width())
        height = max(520, cw.height())
        margin = 10
        gap = 10
        bottom_h = max(175, min(225, int(height * 0.2)))
        top_h = max(320, height - bottom_h - margin * 2 - gap)
        right_w = max(620, min(760, int(width * 0.36)))
        left_w = width - right_w - margin * 2 - gap
        if left_w < 520:
            left_w = max(320, int(width * 0.58))
            right_w = max(620, width - left_w - margin * 2 - gap)

        plot_rect = QtCore.QRect(margin, margin, left_w, top_h)
        self.ui.stackedWidget.setGeometry(plot_rect)
        view_margin = 20
        view_w = max(200, plot_rect.width() - view_margin * 2)
        view_h = max(160, plot_rect.height() - view_margin * 2)
        self.ui.graphicsView.setGeometry(QtCore.QRect(view_margin, view_margin, view_w, view_h))

        right_x = margin + left_w + gap
        self.ui.groupBox.setGeometry(QtCore.QRect(right_x, margin, right_w, top_h))
        self.ui.bottomPanel.setGeometry(
            QtCore.QRect(
                margin,
                margin + top_h + gap,
                max(1, width - margin * 2),
                bottom_h,
            )
        )

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._layout_main_regions()
        self._sync_waveform_geometry()
        self._sync_multi_waveform_geometry()
        self._sync_offline_geometry()
        self._sync_analysis_plot_geometry()
        self._sync_osc_waveform_geometry()
        if self._analysis_plot_active:
            self._waveform.hide()
            self._multi_waveform.hide()
            self._offline_view.hide()
            self._analysis_plot.show()
            self._analysis_plot.raise_()
        elif self._offline_view_active:
            self._waveform.hide()
            self._multi_waveform.hide()
            self._analysis_plot.hide()
            self._offline_view.show()
            self._offline_view.raise_()
        else:
            self._offline_view.hide()
            self._analysis_plot.hide()
            self._waveform.show()
            self._waveform.raise_()
        self._osc_waveform.show()
        self._osc_waveform.raise_()
        self._waveform.refresh_layout()
        self._multi_waveform.refresh_layout()
        self._osc_waveform.refresh_layout()

    def _setup_offline_viewer_ui(self) -> None:
        """离线波形与分钟功率同一套路径B：拼 full → 统一剔坏段 → 可选按分钟截取。"""
        edit = self.ui.lineEdit_offline_path
        if not edit.text().strip():
            edit.setPlaceholderText("选会话内任一 CSV（路径B：full拼接→剔坏）")
        edit.setToolTip(
            "选会话目录中任一 EEG CSV 即可。"
            "加载与分钟功率相同（路径B）：拼接 eeg_chunk_*_full → 统一剔坏段 → 显示；"
            "填写分钟则在剔坏后时间轴上截取。"
        )
        self.ui.lineEdit_offline_min_start.setToolTip(
            "起始分钟（含）。在 full拼接并统一剔坏后的时间轴上截取，与功率图分钟对齐。"
        )
        self.ui.lineEdit_offline_min_end.setToolTip(
            "结束分钟（含）。例如 3 与 9 → 显示剔坏后第 3～9 分钟。"
        )
        self.ui.pushButton_load_offline.setToolTip(
            "路径B：拼 full → 剔坏段 →（可选）截取分钟后显示"
        )
        self.ui.pushButton_clear_offline.setToolTip("退出离线查看，回到实时波形")

    def _parse_offline_minute_range(self) -> Optional[Tuple[int, int]]:
        """解析分钟起止；都空返回 None；只填一侧或非法则抛 ValueError。"""
        start_text = self.ui.lineEdit_offline_min_start.text().strip()
        end_text = self.ui.lineEdit_offline_min_end.text().strip()
        if not start_text and not end_text:
            return None
        if not start_text or not end_text:
            raise ValueError("请同时填写起始分钟与结束分钟，或都留空显示剔坏后全部")
        try:
            start_m = int(float(start_text))
            end_m = int(float(end_text))
        except ValueError as exc:
            raise ValueError("分钟须为数字，例如起 3、止 9") from exc
        if start_m < 1 or end_m < 1:
            raise ValueError("分钟从 1 开始计数")
        if end_m < start_m:
            raise ValueError(f"结束分钟 {end_m} 不能小于起始分钟 {start_m}")
        return start_m, end_m

    def _offline_path_b_source(self, path: Path) -> Path:
        """路径B数据源：会话有 chunk full 则用目录；否则用对应 full 或所选文件。"""
        import re

        from LongRecordNormalReport import list_chunk_full_csvs

        parent = path.parent
        if list_chunk_full_csvs(parent):
            return parent

        if re.fullmatch(r"eeg_chunk_\d+\.csv", path.name, flags=re.IGNORECASE):
            full = parent / path.name.replace(".csv", "_full.csv")
            if full.is_file():
                return full
        if path.name.lower() == "eeg_raw.csv":
            full = parent / "eeg_raw_full.csv"
            if full.is_file():
                return full
        return path

    def _load_offline_selected_slice(
        self,
        path: Path,
        minute_range: Optional[Tuple[int, int]],
    ) -> tuple[np.ndarray, float, str, float]:
        """路径B：与分钟功率一致——full（可多chunk拼接）→ 统一剔坏段 → 可选截取分钟。"""
        from LongRecordMinuteBandPower import load_raw_for_minute_analysis

        source = self._offline_path_b_source(path)
        raw, fs, source_desc = load_raw_for_minute_analysis(source)
        quality = build_threshold_rejection(raw.astype(np.float64), float(fs))
        cleaned, _, n_removed = clean_raw_signal(raw.astype(np.int64), quality)
        cleaned = cleaned.astype(np.float64)
        cleaned_min = cleaned.size / float(fs) / 60.0 if fs > 0 else 0.0
        base_title = (
            f"路径B·{source_desc}·剔坏后"
            f"（删{n_removed}点，约{cleaned_min:.1f}分钟）"
        )

        if minute_range is None:
            if cleaned.size < 8:
                raise ValueError("剔坏后有效样本过少")
            return cleaned, float(fs), base_title, 0.0

        start_m, end_m = minute_range
        n_per_min = max(1, int(round(float(fs) * 60.0)))
        i0 = (start_m - 1) * n_per_min
        i1 = end_m * n_per_min
        if i0 >= cleaned.size:
            raise ValueError(
                f"起始分钟 {start_m} 超出剔坏后长度（约 {cleaned_min:.1f} 分钟）；"
                f"与功率图同一时间轴"
            )
        segment = cleaned[i0 : min(i1, int(cleaned.size))]
        if segment.size < 8:
            raise ValueError(f"截取后样本过少（{segment.size} 点）")
        time_offset_s = float((start_m - 1) * 60)
        title = f"{base_title} · 分钟{start_m}–{end_m}"
        return segment, float(fs), title, time_offset_s

    def _browse_offline_eeg_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 EEG CSV",
            str((_ROOT / "Result").resolve()),
            "CSV Files (*.csv *.txt);;All Files (*)",
        )
        if path:
            self.ui.lineEdit_offline_path.setText(path)
            self._log(f"已选择 EEG CSV: {path}")
            self._load_offline_eeg_csv()

    def _resolve_offline_eeg_csv_path(self, text: str) -> Optional[Path]:
        raw = text.strip().strip('"').strip("'")
        if not raw:
            return None
        path = Path(raw)
        if path.is_file():
            return path.resolve()
        if not path.is_absolute():
            for base in (_ROOT, _ROOT / "Result", Path.cwd()):
                candidate = (base / path).resolve()
                if candidate.is_file():
                    return candidate
        resolved = path.resolve()
        return resolved if resolved.is_file() else None

    def _ui_offline_eeg_csv_path(self) -> str:
        return self.ui.lineEdit_offline_path.text().strip()

    @QtCore.pyqtSlot()
    def _load_offline_eeg_csv(self) -> None:
        path = self._resolve_offline_eeg_csv_path(self._ui_offline_eeg_csv_path())
        if path is None:
            self._log("请先选择有效的 EEG CSV 文件")
            return
        try:
            minute_range = self._parse_offline_minute_range()
        except ValueError as exc:
            self._log(str(exc))
            return
        try:
            raw, fs, title, time_offset_s = self._load_offline_selected_slice(
                path, minute_range
            )
            n, fs = self._offline_view.load_raw(
                raw,
                fs,
                source_name=title,
                time_offset_s=time_offset_s,
            )
        except Exception as exc:
            self._log(f"离线加载失败 ({path.name}): {exc}")
            return
        self._offline_csv_path = path
        self._analysis_plot_active = False
        self._analysis_plot.hide()
        self._offline_view_active = True
        # 离线：raw 必显；其它节律默认不勾，按需勾选
        for mode, checkbox in self._display_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(mode == "raw")
            checkbox.blockSignals(False)
        self._refresh_offline_visible_channels()
        self._switch_to_eeg_view()
        self._waveform.hide()
        self._sync_offline_geometry()
        self._offline_view.show()
        self._offline_view.raise_()
        self.setWindowTitle(f"EEG 离线查看 · {title}")
        range_tip = (
            f"；截取分钟 {minute_range[0]}–{minute_range[1]}（墙钟约 "
            f"{time_offset_s:.0f}–{time_offset_s + n / fs:.0f} s）"
            if minute_range is not None
            else ""
        )
        self._log(
            f"离线查看已加载: {title}（{n} 点 @ {fs:.0f} Hz"
            f"{range_tip}）；raw 始终显示，勾选其它节律叠加分层波形"
        )
        self._update_status_bar()

    @QtCore.pyqtSlot()
    def _exit_offline_view(self) -> None:
        if not self._offline_view_active and not self._offline_view.has_data:
            return
        self._offline_view_active = False
        self._offline_view.clear()
        self._offline_view.hide()
        if not self._analysis_plot_active:
            self._waveform.show()
            self._waveform.raise_()
            # 恢复实时互斥：仅 raw
            for mode, checkbox in self._display_checkboxes.items():
                checkbox.blockSignals(True)
                checkbox.setChecked(mode == "raw")
                checkbox.blockSignals(False)
            self._display_mode = ""
            self._apply_display_mode("raw")
        self._log("已退出离线查看，回到实时波形")
        self._update_status_bar()

    def _refresh_offline_visible_channels(self) -> None:
        """raw 始终显示；其它节律仅勾选才显示。"""
        names = ["raw"]
        for mode in CHANNEL_ORDER:
            if mode == "raw":
                continue
            cb = self._display_checkboxes.get(mode)
            if cb is not None and cb.isChecked():
                names.append(mode)
        # 同步 UI：raw 勾选框保持勾选
        raw_cb = self._display_checkboxes.get("raw")
        if raw_cb is not None and not raw_cb.isChecked():
            raw_cb.blockSignals(True)
            raw_cb.setChecked(True)
            raw_cb.blockSignals(False)
        self._offline_view.set_visible_channels(names)

    def _setup_timed_test_ui(self) -> None:
        """timeEdit / lineEdit_13 测试时长、lcdNumber 倒计时；长时记录与波形唤醒。"""
        self.ui.timeEdit.setDisplayFormat("HH:mm:ss")
        self.ui.timeEdit.setTime(QtCore.QTime(0, 0, 0))
        self.ui.lineEdit_13.setPlaceholderText("秒(优先)")
        self.ui.lcdNumber.setDigitCount(6)
        self.ui.lcdNumber.display(0)
        self.ui.checkBox_long_record.setToolTip(
            "勾选后：每 5 分钟只保存 full；结束时路径B统一剔坏，再按段写出 "
            "eeg_chunk_XXX.csv（对应各 XXX_full）；"
            "并出正常段报告 + 每分钟五节律绝对功率图；"
            "不画 FFT/波形/段对比；波形仅显示开测后 5 分钟，可用「波形唤醒」"
        )
        self.ui.pushButton_wave_wake.setToolTip(
            "长时记录模式下，每次点击将动态波形再显示 5 分钟"
        )
        self.ui.pushButton_wave_wake.clicked.connect(self.on_waveform_wake)
        self.ui.checkBox_long_record.toggled.connect(self._on_long_record_toggled)
        self._refresh_wave_wake_button()

    def _on_long_record_toggled(self, _checked: bool) -> None:
        self._refresh_wave_wake_button()

    def _refresh_wave_wake_button(self) -> None:
        enabled = bool(self.ui.checkBox_long_record.isChecked())
        self.ui.pushButton_wave_wake.setEnabled(enabled)

    @QtCore.pyqtSlot()
    def on_waveform_wake(self) -> None:
        """长时记录：将动态波形再唤醒 WAVEFORM_WAKE_SEC 秒。"""
        if not self.ui.checkBox_long_record.isChecked():
            self._log("请先勾选「长时记录」再唤醒波形")
            return
        if not self._running or self._test_duration_sec is None:
            self._log("请先启动定时记录后再唤醒波形")
            return
        if not self._long_record_active:
            self._log("本次采集未以长时记录模式启动，无法唤醒波形窗口")
            return
        self._arm_waveform_display(WAVEFORM_WAKE_SEC, clear=True)
        self._log(f"波形显示已唤醒 {WAVEFORM_WAKE_SEC / 60.0:.0f} 分钟")

    def _arm_waveform_display(self, duration_sec: float, *, clear: bool = False) -> None:
        self._waveform_display_until = time.monotonic() + float(duration_sec)
        self._waveform_sleep_logged = False
        if clear:
            self._waveform.clear()
            self._osc_waveform.clear()

    def _is_waveform_display_active(self) -> bool:
        if not self._long_record_active:
            return True
        until = self._waveform_display_until
        if until is None:
            return False
        if time.monotonic() < until:
            return True
        if not self._waveform_sleep_logged:
            self._waveform_sleep_logged = True
            self._waveform.clear()
            self._osc_waveform.clear()
            self._log("波形显示已休眠（可点「波形唤醒」再开 5 分钟）")
        return False

    def _setup_sleep_aid_window_ui(self) -> None:
        """lineEdit_3 / lineEdit_sleep_aid_end：定时记录内的助眠 burst 起止秒。"""
        self.ui.lineEdit_3.setPlaceholderText("记录内")
        self.ui.lineEdit_sleep_aid_end.setPlaceholderText("记录内")
        self.ui.lineEdit_3.setToolTip(
            "定时记录开始后（不含 10 s 预热）从第几秒允许发 burst"
        )
        self.ui.lineEdit_sleep_aid_end.setToolTip(
            "定时记录开始后（不含 10 s 预热）在第几秒停止 burst"
        )

    def _setup_session_name_ui(self) -> None:
        """lineEdit_session_name：会话子文件夹命名后缀；根目录由「保存位置...」选择。"""
        edit = self.ui.lineEdit_session_name
        edit.setPlaceholderText("留空→仅时间戳")
        edit.setToolTip(
            "填写 XXX 时子目录为 时间戳_XXX/；留空则为 时间戳/。"
            "根目录点「保存位置...」选择，取消则默认 Result。"
        )

    @QtCore.pyqtSlot()
    def _on_choose_save_location(self) -> None:
        """直接打开文件夹选择；取消则保持默认 Result。命名仍用上方输入框。"""
        start = (
            str(self._eeg_save_root)
            if self._eeg_save_root is not None
            else str(DEFAULT_EEG_CSV_DIR.resolve())
        )
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择保存根目录", start
        )
        if not chosen:
            return
        root = Path(chosen).expanduser()
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._log(f"保存位置无效，已保持原设置: {exc}")
            return
        self._eeg_save_root = root.resolve()
        root_desc = str(self._eeg_save_root)
        tag = self._read_session_name_tag()
        tag_tip = f"时间戳_{tag}" if tag else "时间戳"
        self._log(f"保存根目录: {root_desc}；子目录命名: {tag_tip}")
        self.ui.groupBox_session_name.setToolTip(
            f"当前根目录: {root_desc}\n"
            f"子目录: {tag_tip}/\n"
            "点「保存位置...」可改目录；命名用上方输入框。"
        )

    @staticmethod
    def _sanitize_session_name_tag(raw: str) -> str:
        cleaned = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in raw.strip()
        ).strip("_")
        return cleaned

    def _read_session_name_tag(self) -> str:
        return self._sanitize_session_name_tag(self.ui.lineEdit_session_name.text())

    def _make_session_dir_name(self, stamp: Optional[str] = None) -> str:
        if stamp is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = self._read_session_name_tag()
        if not tag:
            return stamp
        return f"{stamp}_{tag}"

    def _save_root_dir(self) -> Path:
        """保存根目录：对话框所选，或默认 Result。"""
        if self._eeg_save_root is not None:
            return self._eeg_save_root
        return DEFAULT_EEG_CSV_DIR

    def _resolve_session_dir(self, stamp: Optional[str] = None) -> Path:
        root = self._save_root_dir()
        root.mkdir(parents=True, exist_ok=True)
        session_dir = root / self._make_session_dir_name(stamp)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _setup_compare_segments_ui(self) -> None:
        """段A–D：各列上格为起始秒、下格为结束秒；默认全空，按填写段数对比/显示。"""
        fields = (
            (self.ui.lineEdit, "A起"),
            (self.ui.lineEdit_2, "A止"),
            (self.ui.lineEdit_11, "B起"),
            (self.ui.lineEdit_12, "B止"),
            (self.ui.lineEdit_15, "C起"),
            (self.ui.lineEdit_16, "C止"),
            (self.ui.lineEdit_17, "D起"),
            (self.ui.lineEdit_18, "D止"),
        )
        for edit, hint in fields:
            edit.clear()
            edit.setPlaceholderText(hint)
            edit.setToolTip(
                "填写起始+结束秒后点「功率对比」："
                "仅段A→显示该时段 band_power；"
                "多段→显示功率对比图。"
                "测试结束时：未填写则不保存 band_power / 段对比图。"
            )

        if hasattr(self.ui, "label_time1"):
            self.ui.label_time1.hide()
        if hasattr(self.ui, "label_time2"):
            self.ui.label_time2.hide()
        if hasattr(self.ui, "label_time3"):
            self.ui.label_time3.hide()
        if hasattr(self.ui, "label_time4"):
            self.ui.label_time4.hide()
        self.ui.pushButton_5.setToolTip(
            "用当前填写的段时间，在左侧显示区绘制单段功率或分段对比图"
        )

        if not hasattr(self.ui, "checkBox_minute_abs"):
            self.ui.checkBox_minute_abs = QtWidgets.QCheckBox(
                "绝对功率", self.ui.groupBox_compare
            )
            self.ui.checkBox_minute_abs.setObjectName("checkBox_minute_abs")
            self.ui.checkBox_minute_abs.setToolTip(
                "勾选后在左侧显示每分钟五节律绝对功率曲线；"
                "长时多 chunk 会先按序号拼接再计算"
            )
            self.ui.gridLayout_compare.addWidget(self.ui.checkBox_minute_abs, 3, 0, 1, 2)
            self.ui.checkBox_minute_rel = QtWidgets.QCheckBox(
                "相对功率", self.ui.groupBox_compare
            )
            self.ui.checkBox_minute_rel.setObjectName("checkBox_minute_rel")
            self.ui.checkBox_minute_rel.setToolTip(
                "勾选后在左侧显示每分钟五节律相对功率曲线；"
                "可与「绝对功率」同时勾选（上下两图）"
            )
            self.ui.gridLayout_compare.addWidget(self.ui.checkBox_minute_rel, 3, 2, 1, 3)
            self.ui.checkBox_minute_abs.toggled.connect(self._on_minute_power_toggled)
            self.ui.checkBox_minute_rel.toggled.connect(self._on_minute_power_toggled)

        self.ui.checkBox_minute_abs.setToolTip(
            "勾选后在左侧显示按「窗长」切片的五节律绝对功率曲线；"
            "长时多 chunk 会先按序号拼接再计算"
        )
        self.ui.checkBox_minute_rel.setToolTip(
            "勾选后在左侧显示按「窗长」切片的五节律相对功率曲线；"
            "可与「绝对功率」同时勾选（上下两图）"
        )
        if not self.ui.lineEdit_power_window_sec.text().strip():
            self.ui.lineEdit_power_window_sec.setText("60")
        self.ui.lineEdit_power_window_sec.setPlaceholderText("默认60")
        self.ui.lineEdit_power_window_sec.setToolTip(
            "绝对/相对功率按此时长切片。"
            "填 30 → 每 30 秒一窗；"
            "填 60 → 每 60 秒一窗。留空同 60。"
        )
        self.ui.checkBox_reject_mask_power.setToolTip(
            "勾选后：离线查看 *_full 时用红色标出坏段；"
            "绝对/相对功率在连续好段内按窗长计算，不跨坏段硬拼接。"
        )
        try:
            self.ui.checkBox_minute_abs.toggled.disconnect()
        except TypeError:
            pass
        try:
            self.ui.checkBox_minute_rel.toggled.disconnect()
        except TypeError:
            pass
        try:
            self.ui.lineEdit_power_window_sec.editingFinished.disconnect()
        except TypeError:
            pass
        try:
            self.ui.checkBox_reject_mask_power.toggled.disconnect()
        except TypeError:
            pass
        self.ui.checkBox_minute_abs.toggled.connect(self._on_minute_power_toggled)
        self.ui.checkBox_minute_rel.toggled.connect(self._on_minute_power_toggled)
        self.ui.lineEdit_power_window_sec.editingFinished.connect(
            self._on_power_window_sec_edited
        )
        self.ui.checkBox_reject_mask_power.toggled.connect(
            self._on_reject_mask_power_toggled
        )

    def _want_reject_mask_power(self) -> bool:
        return bool(self.ui.checkBox_reject_mask_power.isChecked())

    @QtCore.pyqtSlot(bool)
    def _on_reject_mask_power_toggled(self, _checked: bool = False) -> None:
        if self._offline_view_active and self._offline_csv_path is not None:
            self._load_offline_eeg_csv()
        if (
            self.ui.checkBox_minute_abs.isChecked()
            or self.ui.checkBox_minute_rel.isChecked()
        ):
            self._on_minute_power_toggled()

    def _read_power_window_sec(self) -> float:
        """读取功率切片窗长（秒）；非法或空 → 60。"""
        text = self.ui.lineEdit_power_window_sec.text().strip()
        if not text:
            return 60.0
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError("窗长须为数字，例如 30 或 60") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError("窗长须为正数（秒）")
        if value < 2.0:
            raise ValueError("窗长过短（建议 ≥ 2 秒，以便 Welch 估计）")
        return float(value)

    @QtCore.pyqtSlot()
    def _on_power_window_sec_edited(self) -> None:
        """窗长改完后，若已勾选绝对/相对功率则按新窗长重算。"""
        try:
            self._read_power_window_sec()
        except ValueError as exc:
            self._log(f"窗长无效: {exc}")
            return
        want = bool(
            self.ui.checkBox_minute_abs.isChecked()
            or self.ui.checkBox_minute_rel.isChecked()
        )
        if want:
            self._on_minute_power_toggled()

    def _compare_segment_edit_pairs(self) -> List[Tuple[QtWidgets.QLineEdit, QtWidgets.QLineEdit]]:
        return [
            (self.ui.lineEdit, self.ui.lineEdit_2),
            (self.ui.lineEdit_11, self.ui.lineEdit_12),
            (self.ui.lineEdit_15, self.ui.lineEdit_16),
            (self.ui.lineEdit_17, self.ui.lineEdit_18),
        ]

    def _reset_alpha_display_stats(self) -> None:
        self._alpha_display_total_points = 0
        self._alpha_display_rejected_points = 0

    def _alpha_reject_status_legend(self, base_legend: str) -> str:
        if self._display_mode != "alpha":
            return base_legend
        if self._alpha_display_total_points <= 0:
            status = "alpha阈值: 等待数据"
        else:
            ratio = (
                self._alpha_display_rejected_points
                / max(self._alpha_display_total_points, 1)
            )
            status = (
                f"alpha阈值={self._alpha_rejector.threshold:.1f} "
                f"| 拒绝={ratio:.1%} ({self._alpha_display_rejected_points}/"
                f"{self._alpha_display_total_points}) | 仅标注"
            )
        return f"{base_legend}  |  {status}"

    def _read_compare_segments_from_ui(
        self,
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        """读取段A–D。

        返回:
          () — 全部留空
          ((s,e), ...) — 至少 1 段有效
          None — 填写不完整/非法
        """
        segments: List[Tuple[float, float]] = []
        for start_edit, end_edit in self._compare_segment_edit_pairs():
            start_text = start_edit.text().strip()
            end_text = end_edit.text().strip()
            if not start_text and not end_text:
                continue
            if not start_text or not end_text:
                return None
            try:
                start = float(start_text)
                end = float(end_text)
            except ValueError:
                return None
            if start >= end:
                return None
            segments.append((start, end))
            if len(segments) >= MAX_COMPARE_SEGMENTS:
                break
        return tuple(segments)

    def _run_post_test_power_analysis(
        self,
        full_csv_path: Path,
        cleaned_csv_path: Path,
        sample_rate: float,
        reject_rate: float,
    ) -> None:
        """测试结束后自动运行 power_cal.run_analysis（删减前/后各一份）。"""
        if reject_rate > EEG_REJECT_RATE_WARN:
            self._log(
                f"拒绝率 {reject_rate:.1%} 超过 {EEG_REJECT_RATE_WARN:.0%}，跳过自动功率分析"
            )
            return
        compare_segments = self._read_compare_segments_from_ui()
        if compare_segments is None:
            self._log("段时间填写无效（须成对填写起止秒），跳过 band_power / 段对比出图")
            save_plot = False
            save_segment_compare = False
            segments_for_compare = None
        elif len(compare_segments) == 0:
            self._log("未填写段时间：不生成 band_power / 功率对比图，仍导出离线 CSV 等")
            save_plot = False
            save_segment_compare = False
            segments_for_compare = None
        else:
            labels = ("A", "B", "C", "D")
            bits = [
                f"段{lab} {rng[0]:g}–{rng[1]:g}s"
                for lab, rng in zip(labels, compare_segments)
            ]
            save_plot = True
            save_segment_compare = len(compare_segments) >= 2
            segments_for_compare = (
                compare_segments if len(compare_segments) >= 2 else None
            )
            self._log(
                f"开始 power_cal 双份分析；段时间: " + ", ".join(bits)
                + ("；将保存段对比图" if save_segment_compare else "；仅保存 band_power")
            )
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                run_analysis(
                    full_csv_path,
                    sample_rate,
                    show_plot=False,
                    save_plot=save_plot,
                    show_waveform=False,
                    save_waveform=True,
                    waveform_seconds=None,
                    waveform_time_range=None,
                    show_fft=False,
                    save_fft=True,
                    fft_time_range=None,
                    compare_segments=segments_for_compare,
                    show_segment_compare=False,
                    save_segment_compare=save_segment_compare,
                    enable_alpha_suspicious=True,
                    save_model_window_table=True,
                    save_offline_waveform_data=True,
                    dual_power_analysis=True,
                )
            report = buffer.getvalue().strip()
            if report:
                for line in report.splitlines():
                    self._log(line)
                report_path = full_csv_path.parent / "eeg_analysis_report.txt"
                report_path.write_text(report + "\n", encoding="utf-8")
                self._log(f"分析报告已保存: {report_path}")
            self._log(
                f"分析输出已保存至: {full_csv_path.parent}"
                + (
                    "（含删减前/后 band_power）"
                    if save_plot
                    else "（未保存 band_power / 段对比图）"
                )
            )
        except Exception as exc:
            self._log(f"power_cal 分析失败: {exc}")

    def _resolve_power_compare_csv(self) -> Optional[Path]:
        """功率对比数据源：离线已加载文件 > 最近会话 full csv > 浏览选择。"""
        if self._offline_csv_path is not None and self._offline_csv_path.is_file():
            return self._offline_csv_path
        if self._last_eeg_session_dir is not None:
            for name in (
                "eeg_raw_full.csv",
                "eeg_raw.csv",
            ):
                candidate = self._last_eeg_session_dir / name
                if candidate.is_file():
                    return candidate
            for path in sorted(self._last_eeg_session_dir.glob("eeg_chunk_*_full.csv")):
                return path
            for path in sorted(self._last_eeg_session_dir.glob("*_offline_full_raw.csv")):
                return path
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择用于功率对比的 EEG CSV",
            str((_ROOT / "Result").resolve()),
            "CSV Files (*.csv *.txt);;All Files (*)",
        )
        return Path(path) if path else None

    @QtCore.pyqtSlot()
    def _on_power_compare_clicked(self) -> None:
        segments = self._read_compare_segments_from_ui()
        if segments is None:
            self._log("段时间填写无效：每段须同时填起始与结束秒，且起始 < 结束")
            return
        if len(segments) == 0:
            self._log("请至少填写段A的起始与结束时间")
            return
        csv_path = self._resolve_power_compare_csv()
        if csv_path is None:
            self._log("未选择 EEG CSV，无法做功率对比")
            return
        try:
            raw, sample_rate = load_eeg_csv_with_rate(csv_path)
        except Exception as exc:
            self._log(f"读取 CSV 失败 ({csv_path.name}): {exc}")
            return
        try:
            if len(segments) == 1:
                start, end = segments[0]
                seg = slice_signal(raw, sample_rate, start, end)
                analysis = compute_band_powers(seg, sample_rate=sample_rate)
                plot_band_powers(
                    analysis,
                    title=f"段A 节律功率 · {start:g}–{end:g} s · {csv_path.name}",
                    show=False,
                    figure=self._analysis_plot.figure,
                )
                self._log(
                    f"已显示段A band_power: {start:g}–{end:g} s（{csv_path.name}）"
                )
            else:
                comparison = compare_multi_segment_band_powers(
                    raw, sample_rate, segments
                )
                plot_segment_power_comparison(
                    comparison,
                    title=f"多段节律功率对比 · {csv_path.name}",
                    show=False,
                    figure=self._analysis_plot.figure,
                )
                labels = ("A", "B", "C", "D")
                bits = [
                    f"{lab} {rng[0]:g}–{rng[1]:g}s"
                    for lab, rng in zip(labels, segments)
                ]
                self._log(f"已显示多段功率对比: " + ", ".join(bits))
        except Exception as exc:
            self._log(f"功率对比失败: {exc}")
            return
        self._show_analysis_plot_view()
        self._analysis_plot.refresh()

    def _resolve_minute_power_source(self) -> Optional[Path]:
        """分钟功率数据源：离线文件所在会话 / 最近会话 / 选目录或 CSV。"""
        if self._offline_csv_path is not None and self._offline_csv_path.is_file():
            parent = self._offline_csv_path.parent
            try:
                from LongRecordNormalReport import list_chunk_full_csvs

                if list_chunk_full_csvs(parent):
                    return parent
            except Exception:
                pass
            return self._offline_csv_path
        if self._last_eeg_session_dir is not None and self._last_eeg_session_dir.is_dir():
            return self._last_eeg_session_dir
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择会话内 EEG CSV（长时请选任一 chunk，或选 eeg_raw_full.csv）",
            str((_ROOT / "Result").resolve()),
            "CSV Files (*.csv *.txt);;All Files (*)",
        )
        if not path:
            return None
        chosen = Path(path)
        parent = chosen.parent
        try:
            from LongRecordNormalReport import list_chunk_full_csvs

            if list_chunk_full_csvs(parent):
                return parent
        except Exception:
            pass
        return chosen

    @QtCore.pyqtSlot(bool)
    def _on_minute_power_toggled(self, _checked: bool = False) -> None:
        want_abs = bool(
            getattr(self.ui, "checkBox_minute_abs", None)
            and self.ui.checkBox_minute_abs.isChecked()
        )
        want_rel = bool(
            getattr(self.ui, "checkBox_minute_rel", None)
            and self.ui.checkBox_minute_rel.isChecked()
        )
        if not want_abs and not want_rel:
            if self._analysis_plot_active:
                self._analysis_plot.clear()
                self._log("已取消每分钟功率图显示（未勾选绝对/相对功率）")
            return

        source = self._resolve_minute_power_source()
        if source is None:
            self._log("未选择数据，无法绘制每分钟功率图")
            # 取消勾选，避免反复弹窗
            for name in ("checkBox_minute_abs", "checkBox_minute_rel"):
                cb = getattr(self.ui, name, None)
                if cb is not None:
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)
            return

        try:
            window_sec = self._read_power_window_sec()
        except ValueError as exc:
            self._log(f"窗长无效: {exc}")
            return

        no_splice = self._want_reject_mask_power()

        try:
            from LongRecordMinuteBandPower import (
                plot_minute_band_powers,
                prepare_cleaned_minute_powers,
            )

            minutes, absolute, relative, meta = prepare_cleaned_minute_powers(
                source, window_sec=window_sec, no_splice=no_splice
            )
        except Exception as exc:
            self._log(f"每分钟功率计算失败: {exc}")
            return

        if meta["n_minutes"] <= 0:
            self._log("有效数据不足 1 分钟，无法绘制每分钟功率图")
            return

        kinds = []
        if want_abs:
            kinds.append("绝对")
        if want_rel:
            kinds.append("相对")
        title = (
            f"每分钟节律{'/'.join(kinds)}功率（坏段剔除后，N={meta['n_minutes']}）"
            f" · {meta['source_desc']}"
        )
        try:
            plot_minute_band_powers(
                minutes,
                absolute=absolute if want_abs else None,
                relative=relative if want_rel else None,
                title=title,
                window_sec=float(meta.get("window_sec", window_sec)),
                x_as_time_s=bool(meta.get("x_as_time_s", False)),
                bad_spans=meta.get("bad_spans") if no_splice else None,
                short_good_spans=meta.get("short_good_spans") if no_splice else None,
                gated_spans=meta.get("gated_spans") if no_splice else None,
                total_duration_s=meta.get("total_duration_s") if no_splice else None,
                figure=self._analysis_plot.figure,
            )
        except Exception as exc:
            self._log(f"每分钟功率绘图失败: {exc}")
            return

        self._show_analysis_plot_view()
        self._analysis_plot.refresh()
        self._log(
            f"已显示每分钟{'/'.join(kinds)}功率："
            f"{meta['source_desc']}，N={meta['n_minutes']} 分钟"
            f"（剔除 {meta['n_removed']} 点）"
        )

    def _show_analysis_plot_view(self) -> None:
        self._analysis_plot_active = True
        self._offline_view_active = False
        self._offline_view.hide()
        self._waveform.hide()
        self._sync_analysis_plot_geometry()
        self._analysis_plot.show()
        self._analysis_plot.raise_()
        self._switch_to_eeg_view()
        self.setWindowTitle("EEG 功率对比")
        self._update_status_bar()

    def _exit_analysis_plot_view(self) -> None:
        if not self._analysis_plot_active:
            return
        self._analysis_plot_active = False
        self._analysis_plot.hide()
        self._analysis_plot.clear()
        if not self._offline_view_active:
            self._waveform.show()
            self._waveform.raise_()
            self._display_mode = ""
            self._apply_display_mode(self._current_display_mode())
        self._update_status_bar()

    def _finalize_timed_test(self) -> None:
        """保存 EEG raw CSV；普通模式再跑功率分析，长时模式存盘并输出正常段报告。"""
        if self._long_record_active:
            self._flush_long_record_buffer(final=True)
            session_dir = self._long_session_dir
            if session_dir is not None:
                self._run_long_record_postprocess(session_dir)
            self._reset_timed_test_state()
            if session_dir is not None:
                self._last_eeg_session_dir = session_dir
                self._log(
                    f"长时记录结束：已保存至 {session_dir}"
                    f"（路径B分段删减 eeg_chunk_XXX.csv + 正常段报告 + 每分钟绝对功率图；"
                    f"跳过 FFT/波形等其它图）"
                )
            return

        saved = self._save_eeg_raw_csv()
        self._reset_timed_test_state()
        if saved is not None:
            cleaned_path, full_path, sample_rate, reject_rate = saved
            session_dir = full_path.parent
            self._last_eeg_session_dir = session_dir
            self._run_post_test_power_analysis(
                full_path,
                cleaned_path,
                sample_rate,
                reject_rate,
            )
            self._maybe_run_trough_calibration(session_dir)
            self._maybe_run_burst_alpha_power_stats(session_dir)

    def _maybe_run_trough_calibration(self, session_dir: Path) -> None:
        """定时测试结束后：若有 burst 记录，运行波谷对齐标定（方法 B）。"""
        if not (session_dir / "sleep_aid_bursts.csv").is_file():
            return
        if not TROUGH_CAL_SCRIPT.is_file():
            self._log(f"未找到波谷标定脚本: {TROUGH_CAL_SCRIPT}")
            return
        try:
            from TroughCalibrator import run_calibration

            result = run_calibration(session_dir, save_plots=True)
            for line in result.report_text.splitlines():
                self._log(line)
            self._log(
                f"波谷标定完成，建议 total_latency "
                f"{result.suggested_total_latency_sec * 1000:.1f} ms"
            )
        except Exception as exc:
            self._log(f"波谷标定失败: {exc}")

    def _maybe_run_burst_alpha_power_stats(self, session_dir: Path) -> None:
        """定时测试结束后：短窗 Alpha 功率（0-100 ms）+ burst 锁定平均。"""
        if not (session_dir / "sleep_aid_bursts.csv").is_file():
            return
        try:
            from BurstAlphaPowerStats import run_burst_alpha_power_stats

            result = run_burst_alpha_power_stats(session_dir, save_outputs=True)
            for line in result.report_text.splitlines():
                self._log(line)
            self._log(
                f"刺激短窗 Alpha 统计完成: "
                f"0-100ms ↑{result.n_up} ↓{result.n_down} →{result.n_flat} "
                f"| 50-150ms ↑{result.n_up_erp} ↓{result.n_down_erp} "
                f"(有效 {result.n_valid}/{result.n_bursts})"
            )
        except Exception as exc:
            self._log(f"刺激前后 Alpha 功率统计失败: {exc}")

    def _read_time_edit_duration_sec(self) -> Optional[float]:
        """读取 timeEdit；00:00:00 表示未设置。"""
        t = self.ui.timeEdit.time()
        total = t.hour() * 3600 + t.minute() * 60 + t.second()
        if total <= 0:
            return None
        return float(total)

    @staticmethod
    def _parse_seconds_text(text: str) -> Optional[float]:
        raw = text.strip()
        if not raw:
            return None
        try:
            seconds = float(raw)
        except ValueError:
            return None
        if seconds <= 0:
            return None
        return seconds

    def _read_test_duration_sec(self) -> Optional[float]:
        """读取测试记录时长：lineEdit_13(秒) 与 timeEdit 均有值时以 lineEdit_13 为准。"""
        line13_sec = self._parse_seconds_text(self.ui.lineEdit_13.text())
        time_edit_sec = self._read_time_edit_duration_sec()
        if line13_sec is not None and time_edit_sec is not None:
            return line13_sec
        if line13_sec is not None:
            return line13_sec
        return time_edit_sec

    def _read_sleep_aid_window_from_ui(self) -> Optional[Tuple[float, float]]:
        """读取助眠 burst 起止秒（相对记录阶段，不含 TEST_WARMUP_SEC）。"""
        start = self._parse_seconds_text(self.ui.lineEdit_3.text())
        end = self._parse_seconds_text(self.ui.lineEdit_sleep_aid_end.text())
        if start is None or end is None:
            return None
        if start < 0 or end <= start:
            return None
        return (start, end)

    def _effective_sleep_aid_window_sec(self) -> Optional[Tuple[float, float]]:
        """仅定时记录进行中且起止有效时返回窗口。"""
        if self._test_duration_sec is None:
            return None
        window = self._read_sleep_aid_window_from_ui()
        if window is None:
            return None
        start, end = window
        if end > self._test_duration_sec:
            return None
        return (start, end)

    def _recording_elapsed_sec(self) -> float:
        """记录阶段已过去秒数（不含 TEST_WARMUP_SEC）。"""
        if self._test_started_at is None:
            return 0.0
        return max(0.0, self._test_total_elapsed_sec() - TEST_WARMUP_SEC)

    def _is_in_sleep_aid_burst_window(self) -> bool:
        window = self._effective_sleep_aid_window_sec()
        if window is None:
            return True
        start, end = window
        elapsed = self._recording_elapsed_sec()
        return start <= elapsed < end

    def _sleep_aid_schedule_bounds_total_sec(self) -> Optional[Tuple[float, float]]:
        """返回 [采集开始] 时刻轴上助眠应运行/发 burst 的起止秒。"""
        window = self._effective_sleep_aid_window_sec()
        if window is None:
            return None
        start, end = window
        warmup = (
            self._sleep_aid_controller.params.warmup_sec
            if self._sleep_aid_controller is not None
            else SLEEP_AID_WARMUP_SEC
        )
        run_start = max(0.0, TEST_WARMUP_SEC + start - warmup)
        run_end = TEST_WARMUP_SEC + end
        return (run_start, run_end)

    def _is_test_recording_phase(self) -> bool:
        """预热结束且仍在定时测试窗口内。"""
        if (
            self._test_duration_sec is None
            or self._test_started_at is None
            or not self._running
        ):
            return False
        elapsed = time.monotonic() - self._test_started_at
        return TEST_WARMUP_SEC <= elapsed < TEST_WARMUP_SEC + self._test_duration_sec

    def _test_total_elapsed_sec(self) -> float:
        if self._test_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._test_started_at)

    def _format_duration_hms(self, duration: float) -> str:
        total = int(duration)
        return (
            f"{total // 3600:02d}:"
            f"{(total % 3600) // 60:02d}:"
            f"{total % 60:02d}"
        )

    def _begin_timed_test_if_configured(self) -> None:
        """启动采集时：若设置了时长，则 10 s 预热后再计时并保存 EEG raw。"""
        duration = self._read_test_duration_sec()
        self._test_duration_sec = duration
        self._test_started_at = time.monotonic() if duration is not None else None
        self._sleep_aid_timed_auto = False
        self._eeg_raw_record.clear()
        self._long_record_active = False
        self._long_session_dir = None
        self._long_chunks_saved = 0
        self._waveform_display_until = None
        self._waveform_sleep_logged = False
        if duration is None:
            self.ui.lcdNumber.display(0)
            if self.ui.checkBox_long_record.isChecked():
                self._log("已勾选长时记录，但未设置测试时长，本次按普通采集运行")
            return
        self._long_record_active = bool(self.ui.checkBox_long_record.isChecked())
        if self._long_record_active:
            self._long_session_dir = self._resolve_session_dir()
            self._arm_waveform_display(WAVEFORM_WAKE_SEC, clear=False)
            self._log(
                f"长时记录模式: 前 {TEST_WARMUP_SEC:.0f}s 不保存，"
                f"之后记录 {self._format_duration_hms(duration)}；"
                f"每 {LONG_RECORD_CHUNK_SEC / 60.0:.0f} 分钟自动存盘；"
                f"不跑功率对比/FFT；波形先显示 {WAVEFORM_WAKE_SEC / 60.0:.0f} 分钟"
            )
            self._log(f"长时记录目录: {self._long_session_dir}")
        self._update_test_countdown_lcd()
        if not self._long_record_active:
            self._log(
                f"定时测试: 前 {TEST_WARMUP_SEC:.0f}s 不保存，"
                f"之后记录 {self._format_duration_hms(duration)}，"
                f"到时自动停止并保存；手动停止不保存"
            )
        window = self._read_sleep_aid_window_from_ui()
        if window is None:
            return
        start_sec, end_sec = window
        if end_sec > duration:
            self._log(
                f"助眠时段无效: 结束 {end_sec:g}s 超过测试时长 {duration:g}s，"
                "本次定时记录不启用助眠时段"
            )
            return
        warmup = SLEEP_AID_WARMUP_SEC
        if start_sec < warmup:
            self._log(
                f"助眠起始 {start_sec:g}s 小于暖机 {warmup:g}s："
                f"将在采集开始后尽快启动暖机，burst 最早约在记录 {warmup:g}s 发出"
            )
        pre_record = max(0.0, start_sec - warmup)
        self._log(
            f"助眠时段: 记录 {start_sec:g}–{end_sec:g}s 发 burst（不含暖机）；"
            f"暖机提前至记录第 {pre_record:g}s 前启动"
        )

    def _reset_timed_test_state(self) -> None:
        self._test_duration_sec = None
        self._test_started_at = None
        self._sleep_aid_timed_auto = False
        self._eeg_raw_record.clear()
        self._long_record_active = False
        self._long_session_dir = None
        self._long_chunks_saved = 0
        self._waveform_display_until = None
        self._waveform_sleep_logged = False
        self.ui.lcdNumber.display(0)

    def _update_test_countdown_lcd(self) -> None:
        if self._test_duration_sec is None or self._test_started_at is None:
            self.ui.lcdNumber.display(0)
            return
        elapsed = self._test_total_elapsed_sec()
        if elapsed < TEST_WARMUP_SEC:
            warmup_left = int(TEST_WARMUP_SEC - elapsed + 0.999)
            self.ui.lcdNumber.display(warmup_left)
            return
        recording_elapsed = elapsed - TEST_WARMUP_SEC
        remaining = max(0.0, self._test_duration_sec - recording_elapsed)
        secs = int(remaining + 0.999)
        hh = secs // 3600
        mm = (secs % 3600) // 60
        ss = secs % 60
        self.ui.lcdNumber.display(hh * 10000 + mm * 100 + ss)

    def _is_timed_test_finished(self) -> bool:
        if self._test_duration_sec is None or self._test_started_at is None:
            return False
        return self._test_total_elapsed_sec() >= TEST_WARMUP_SEC + self._test_duration_sec

    def _resolve_eeg_csv_path(self) -> Path:
        """在 Result 下按 时间戳 或 时间戳_命名 创建子文件夹并写入 CSV。"""
        return self._resolve_session_dir() / "eeg_raw.csv"

    def _measured_eeg_sample_rate(self) -> float:
        sample_rate = float(MCU_SAMPLE_RATE)
        measured = self._rhythm.measured_sample_rate
        if measured is not None and measured > 0:
            sample_rate = measured
        return sample_rate

    def _save_eeg_raw_csv(
        self,
        *,
        session_dir: Optional[Path] = None,
        name_stem: str = "eeg_raw",
        save_bursts: bool = True,
        quiet_empty: bool = False,
        save_cleaned: bool = True,
    ) -> Optional[Tuple[Optional[Path], Path, float, float]]:
        """保存当前缓冲。

        save_cleaned=True：同时写 {stem}.csv（本段当场剔坏，仅适合单段短时测试）。
        长时 chunk 应 save_cleaned=False，只写 *_full.csv；剔坏文件改由路径B在结束后统一生成。
        """
        if not self._eeg_raw_record:
            if not quiet_empty:
                self._log("定时测试结束，但没有 EEG raw 数据可保存")
            return None
        if session_dir is None:
            session_dir = self._resolve_session_dir()
        else:
            session_dir = Path(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = session_dir / f"{name_stem}.csv"
        full_path = session_dir / f"{name_stem}_full.csv"
        sample_rate = self._measured_eeg_sample_rate()
        raw = np.asarray(self._eeg_raw_record, dtype=np.int64)
        quality = build_threshold_rejection(raw.astype(np.float64), sample_rate)
        reject_rate = quality.reject_rate
        suspicious_rate = quality.suspicious_rate
        try:
            with full_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "time_s", "ch1_raw"])
                for index, value in enumerate(raw):
                    writer.writerow([index, index / sample_rate, int(value)])
            self._log(
                f"EEG raw 全量已保存: {full_path} ({raw.size} 点 @ {sample_rate:.0f} Hz)"
            )

            out_cleaned: Optional[Path] = None
            if save_cleaned:
                cleaned_raw, _, removed_points = clean_raw_signal(raw, quality)
                with cleaned_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["index", "time_s", "ch1_raw"])
                    for index, value in enumerate(cleaned_raw):
                        writer.writerow([index, index / sample_rate, int(value)])
                out_cleaned = cleaned_path
                self._log(
                    f"EEG raw 删减后已保存: {cleaned_path} "
                    f"({cleaned_raw.size}/{raw.size} 点, 剔除 {removed_points} 点)"
                )
            self._log(f"阈值拒绝率: {reject_rate:.1%}")
            self._log(f"可疑片段率: {suspicious_rate:.1%}")
            if reject_rate > EEG_REJECT_RATE_WARN:
                self._log("拒绝率过高，本次数据不建议用于样本分析")
            self._last_eeg_session_dir = session_dir
            if save_bursts:
                self._save_sleep_aid_bursts_csv(session_dir, sample_rate)
            return out_cleaned, full_path, sample_rate, reject_rate
        except OSError as exc:
            self._log(f"保存 EEG raw CSV 失败: {exc}")
            return None

    def _save_path_b_cleaned_csv(self, session_dir: Path) -> Optional[List[Path]]:
        """路径B：拼所有 *_full → 统一剔坏 → 按各 full 原始区间拆成对应分段 clean 文件。

        例：eeg_chunk_001_full.csv → eeg_chunk_001.csv（内容来自整晚统一剔坏后再切回）。
        """
        from analysis_plot_view import load_eeg_csv_with_rate
        from LongRecordNormalReport import list_chunk_full_csvs

        session_dir = Path(session_dir)
        full_files = list_chunk_full_csvs(session_dir)
        if not full_files:
            self._log("路径B删减保存跳过：目录中无 eeg_chunk_*_full.csv")
            return None
        try:
            parts: List[np.ndarray] = []
            rates: List[float] = []
            # (full_path, concat_start, concat_end)
            spans: List[Tuple[Path, int, int]] = []
            offset = 0
            for full_path in full_files:
                raw_i, fs_i = load_eeg_csv_with_rate(full_path)
                raw_i = np.asarray(raw_i, dtype=np.float64)
                n_i = int(raw_i.size)
                parts.append(raw_i)
                rates.append(float(fs_i))
                spans.append((full_path, offset, offset + n_i))
                offset += n_i

            raw = np.concatenate(parts)
            sample_rate = float(np.median(np.asarray(rates, dtype=np.float64)))
            quality = build_threshold_rejection(raw, sample_rate)
            cleaned, kept_index, n_removed = clean_raw_signal(
                raw.astype(np.int64), quality
            )
            cleaned = np.asarray(cleaned)
            kept_index = np.asarray(kept_index, dtype=np.int64)

            saved: List[Path] = []
            for full_path, i0, i1 in spans:
                mask = (kept_index >= i0) & (kept_index < i1)
                seg = cleaned[mask]
                name = full_path.name
                if name.lower().endswith("_full.csv"):
                    out_name = name[: -len("_full.csv")] + ".csv"
                else:
                    out_name = f"{full_path.stem}_cleaned.csv"
                out_path = session_dir / out_name
                with out_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["index", "time_s", "ch1_raw"])
                    for index, value in enumerate(seg):
                        writer.writerow(
                            [index, index / sample_rate, int(value)]
                        )
                saved.append(out_path)
                self._log(
                    f"路径B分段删减: {full_path.name} → {out_name} "
                    f"({seg.size}/{i1 - i0} 点保留)"
                )

            self._log(
                f"路径B删减完成：{len(saved)} 段（{len(full_files)} 个 full 统一剔坏后拆回 → "
                f"共保留 {cleaned.size}/{raw.size}，剔除 {n_removed}；"
                f"拒绝率 {quality.reject_rate:.1%}）"
            )
            return saved
        except Exception as exc:
            self._log(f"路径B删减 CSV 保存失败: {exc}")
            return None

    def _write_long_record_normal_report(self, session_dir: Path) -> None:
        """长时记录结束：扫描各 chunk，输出 raw 正常段报告。"""
        try:
            from LongRecordNormalReport import write_normal_segments_report

            result = write_normal_segments_report(
                session_dir,
                raw_min=LONG_NORMAL_RAW_MIN,
                raw_max=LONG_NORMAL_RAW_MAX,
                min_duration_sec=LONG_NORMAL_MIN_DURATION_SEC,
            )
            for line in result.report_text.splitlines():
                self._log(line)
            if result.report_txt_path is not None:
                self._log(f"正常段报告已保存: {result.report_txt_path}")
            if result.report_csv_path is not None:
                self._log(f"正常段 CSV 已保存: {result.report_csv_path}")
        except Exception as exc:
            self._log(f"正常段报告生成失败: {exc}")

    def _run_long_record_minute_band_power(self, session_dir: Path) -> None:
        """长时记录结束：坏段剔除后按分钟算五节律绝对功率，只画一张折线图。"""
        try:
            from LongRecordMinuteBandPower import run_minute_band_power_analysis

            try:
                window_sec = self._read_power_window_sec()
            except ValueError:
                window_sec = 60.0
            no_splice = self._want_reject_mask_power()
            result = run_minute_band_power_analysis(
                session_dir,
                save_outputs=True,
                window_sec=window_sec,
                no_splice=no_splice,
            )
            for line in result.report_text.splitlines():
                self._log(line)
            if result.plot_path is not None:
                self._log(f"每分钟绝对功率图已保存: {result.plot_path}")
        except Exception as exc:
            self._log(f"每分钟节律绝对功率分析失败: {exc}")

    def _run_long_record_postprocess(self, session_dir: Path) -> None:
        """长时记录结束后：路径B删减CSV + 正常段报告 + 每分钟绝对功率图。"""
        self._save_path_b_cleaned_csv(session_dir)
        self._write_long_record_normal_report(session_dir)
        self._run_long_record_minute_band_power(session_dir)

    def _flush_long_record_buffer(self, *, final: bool = False) -> None:
        """长时记录：把当前缓冲写成下一段；final 时额外写 burst。"""
        if self._long_session_dir is None:
            return
        if not self._eeg_raw_record:
            if final:
                self._save_sleep_aid_bursts_csv(
                    self._long_session_dir, self._measured_eeg_sample_rate()
                )
            return
        self._long_chunks_saved += 1
        stem = f"eeg_chunk_{self._long_chunks_saved:03d}"
        # 长时只存 full；删减版由结束后路径B统一剔坏再拆成 eeg_chunk_XXX.csv
        saved = self._save_eeg_raw_csv(
            session_dir=self._long_session_dir,
            name_stem=stem,
            save_bursts=False,
            quiet_empty=True,
            save_cleaned=False,
        )
        self._eeg_raw_record.clear()
        if saved is not None:
            self._log(
                f"长时记录已保存第 {self._long_chunks_saved} 段 full"
                + ("（收尾）" if final else "")
            )
        if final:
            self._save_sleep_aid_bursts_csv(
                self._long_session_dir, self._measured_eeg_sample_rate()
            )

    def _maybe_save_long_record_chunk(self) -> None:
        """记录阶段每满 LONG_RECORD_CHUNK_SEC 自动存一段并清空缓冲。"""
        if not self._long_record_active or self._long_session_dir is None:
            return
        if not self._is_test_recording_phase():
            return
        elapsed = self._recording_elapsed_sec()
        due = int(elapsed // LONG_RECORD_CHUNK_SEC)
        while self._long_chunks_saved < due:
            if not self._eeg_raw_record:
                self._long_chunks_saved = due
                break
            self._flush_long_record_buffer(final=False)

    def _save_sleep_aid_bursts_csv(self, session_dir: Path, sample_rate: float) -> None:
        if not self._sleep_aid_burst_record:
            return
        path = Path(session_dir) / "sleep_aid_bursts.csv"
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "burst_index",
                        "sample_index",
                        "time_s",
                        "total_latency_sec",
                        "seconds_to_trough",
                        "phase_rad",
                        "inst_freq_hz",
                    ]
                )
                for row in self._sleep_aid_burst_record:
                    writer.writerow(
                        [
                            row["burst_index"],
                            row["sample_index"],
                            row["sample_index"] / sample_rate,
                            row["total_latency_sec"],
                            row["seconds_to_trough"],
                            row["phase_rad"],
                            row["inst_freq_hz"],
                        ]
                    )
            self._log(
                f"助眠 burst 事件已保存: {path} ({len(self._sleep_aid_burst_record)} 条)"
            )
        except OSError as exc:
            self._log(f"保存 sleep_aid_bursts.csv 失败: {exc}")

    def _record_sleep_aid_burst(
        self,
        *,
        sample_index: int,
        snapshot: AlphaPhaseSnapshot,
    ) -> None:
        if self._sleep_aid_controller is None:
            return
        self._sleep_aid_burst_record.append(
            {
                "burst_index": len(self._sleep_aid_burst_record) + 1,
                "sample_index": int(sample_index),
                "total_latency_sec": float(
                    self._sleep_aid_controller.total_latency_sec
                ),
                "seconds_to_trough": float(snapshot.seconds_to_trough),
                "phase_rad": float(snapshot.phase_rad),
                "inst_freq_hz": float(snapshot.inst_freq_hz),
            }
        )

    def _stop_capture_due_to_timeout(self) -> None:
        """定时到时：停止采集并保存 CSV。"""
        if not self._running:
            return
        self._running = False
        if self._link is not None:
            self._link.discard_pending_input()
        if self._osc_link is not None:
            self._osc_link.discard_pending_input()
        if (
            self._sleep_aid_controller is not None
            and self._sleep_aid_controller.is_active
        ):
            self._sleep_aid_controller.stop()
            self._sleep_aid_timed_auto = False
            self._refresh_sleep_aid_button()
        if self._audio_controller.is_playing:
            self._audio_controller.stop()
            self._refresh_audio_button()
        self._play_timed_test_end_alert()
        if self._test_duration_sec is not None:
            self._finalize_timed_test()
        else:
            self._reset_timed_test_state()
        self._refresh_capture_button()
        self._update_status_bar()
        self._log("定时测试到时，已自动停止采集")

    def _play_timed_test_end_alert(self) -> None:
        """定时测试结束提示音（双音叮咚，非阻塞）。"""
        try:
            from audiio import play_alert_chime

            if play_alert_chime():
                self._log("已播放测试结束提示音")
            else:
                self._log("结束提示音不可用（未安装 sounddevice）")
        except Exception as exc:
            self._log(f"结束提示音播放失败: {exc}")

    def _setup_audio_ui_defaults(self) -> None:
        """音频区默认值：左 lineEdit_7 / 右 lineEdit_6，相位 lineEdit_9/10，时长 lineEdit_8。"""
        defaults = (
            (self.ui.lineEdit_7, AUDIO_DEFAULT_LEFT_FREQ_HZ, "左频率 Hz"),
            (self.ui.lineEdit_6, AUDIO_DEFAULT_RIGHT_FREQ_HZ, "右频率 Hz"),
            (self.ui.lineEdit_8, AUDIO_DEFAULT_DURATION_SEC, "秒"),
            (self.ui.lineEdit_9, AUDIO_DEFAULT_LEFT_PHASE_DEG, "左相位 °"),
            (self.ui.lineEdit_10, AUDIO_DEFAULT_RIGHT_PHASE_DEG, "右相位 °"),
        )
        for edit, value, hint in defaults:
            if not edit.text().strip():
                edit.setText(value)
            edit.setPlaceholderText(hint)
        self._refresh_audio_button()
        self._refresh_sleep_aid_button()

    def _refresh_sleep_aid_button(self) -> None:
        btn = self.ui.pushButton_8
        active = (
            self._sleep_aid_controller is not None
            and self._sleep_aid_controller.is_active
        )
        if active:
            warm = self._sleep_aid_controller.warmup_remaining()
            if warm > 0.0:
                btn.setText(f"暖机 {warm:.0f}s")
            else:
                count = self._sleep_aid_burst_count
                btn.setText(f"停止助眠 ({count})" if count > 0 else "停止助眠")
            btn.setStyleSheet(
                "QPushButton { font-size:14px; font-weight:bold; border:none;"
                "border-radius:8px; min-height:41px;"
                "background-color:#C62828; color:#FFFFFF; }"
            )
        else:
            btn.setText("助眠音效")
            btn.setStyleSheet(
                "QPushButton { font-size:14px; font-weight:bold; border:none;"
                "border-radius:8px; min-height:41px;"
                "background-color:#2E7D32; color:#FFFFFF; }"
            )

    @QtCore.pyqtSlot()
    def on_toggle_sleep_aid(self) -> None:
        """pushButton_8：Alpha 波谷相位锁定短 burst。"""
        if self._sleep_aid_controller is None:
            self._log("助眠模块不可用: 请 pip install sounddevice")
            return
        if self._sleep_aid_controller.is_active:
            self._stop_sleep_aid_stimulus(manual=True)
            return
        if not self._running:
            self._log("请先开始 EEG 采集后再开启助眠音效")
            return
        self._start_sleep_aid_stimulus(manual=True)

    def _start_sleep_aid_stimulus(self, *, manual: bool) -> None:
        if self._sleep_aid_controller is None:
            return
        if self._audio_controller.is_playing:
            self._audio_controller.stop()
            self._refresh_audio_button()
            self._log("已停止连续音频，避免与助眠 burst 冲突")
        self._sleep_aid_timed_auto = not manual
        self._sleep_aid_tracker.reset()
        self._rhythm.reset()
        self._alpha_rejector.reset()
        self._quality_gate.reset()
        self._reset_alpha_display_stats()
        if manual or not self._sleep_aid_burst_record:
            self._sleep_aid_burst_count = 0
            self._sleep_aid_burst_record.clear()
        self._sleep_aid_controller.start()
        self._apply_display_mode("alpha")
        self.ui.checkBox.blockSignals(True)
        self.ui.checkBox.setChecked(True)
        self.ui.checkBox.blockSignals(False)
        self._refresh_sleep_aid_button()
        p = self._sleep_aid_controller.params
        mode = "手动" if manual else "定时"
        self._log(
            f"助眠音效已启动({mode}): "
            f"暖机 {p.warmup_sec:g}s，最小间隔 {p.min_interval_sec:g}s，"
            f"刺激效应延迟 {p.stimulus_effect_latency_sec * 1000:g} ms，"
            f"粉噪 burst {p.burst_duration_ms:g} ms"
        )
        window = self._effective_sleep_aid_window_sec()
        if window is not None and not manual:
            start, end = window
            self._log(f"助眠 burst 窗口: 记录 {start:g}–{end:g}s")

    def _stop_sleep_aid_stimulus(self, *, manual: bool) -> None:
        if self._sleep_aid_controller is None or not self._sleep_aid_controller.is_active:
            return
        if manual:
            self._sleep_aid_timed_auto = False
        trigger_count = self._sleep_aid_controller.trigger_count
        self._sleep_aid_controller.stop()
        self._refresh_sleep_aid_button()
        skip = self._quality_gate.skip_count
        reasons = self._quality_gate.skip_reasons
        reason_text = (
            ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
            if reasons
            else "无"
        )
        stop_mode = "手动" if manual else "定时"
        self._log(
            f"助眠音效已停止({stop_mode})，"
            f"共触发 {trigger_count} 次，"
            f"门控跳过 {skip} 次（{reason_text}）"
        )
        if not manual:
            self._sleep_aid_timed_auto = False
        if self._sleep_aid_burst_record:
            if self._test_duration_sec is not None and self._test_started_at is not None:
                self._log(
                    f"助眠 burst 事件 {len(self._sleep_aid_burst_record)} 条，"
                    "将在定时测试结束时与 EEG 一并保存"
                )
            elif manual:
                sample_rate = float(MCU_SAMPLE_RATE)
                measured = self._rhythm.measured_sample_rate
                if measured is not None and measured > 0:
                    sample_rate = measured
                session_dir = self._resolve_session_dir()
                self._save_sleep_aid_bursts_csv(session_dir, sample_rate)

    def _maybe_manage_timed_sleep_aid(self) -> None:
        """定时记录 + 助眠时段：提前暖机、到点自动启停。"""
        if (
            self._sleep_aid_controller is None
            or not self._running
            or self._test_duration_sec is None
            or self._test_started_at is None
        ):
            return
        bounds = self._sleep_aid_schedule_bounds_total_sec()
        if bounds is None:
            return
        run_start, run_end = bounds
        total = self._test_total_elapsed_sec()
        if run_start <= total < run_end:
            if not self._sleep_aid_controller.is_active:
                self._start_sleep_aid_stimulus(manual=False)
        elif self._sleep_aid_controller.is_active and self._sleep_aid_timed_auto:
            self._stop_sleep_aid_stimulus(manual=False)

    def _on_sleep_aid_burst(self, count: float) -> None:
        """主线程触发回调：推迟 UI 更新，避免与波形重绘同 tick 争抢。"""
        QtCore.QTimer.singleShot(0, lambda c=int(count): self._update_sleep_aid_burst_ui(c))

    def _update_sleep_aid_burst_ui(self, count: int) -> None:
        self._sleep_aid_burst_count = count
        if count == 1 or count % SLEEP_AID_BURST_LOG_EVERY == 0:
            self._log(f"助眠 burst #{count}")
        self._refresh_sleep_aid_button()

    def _process_sleep_aid_sample(
        self, raw: int, alpha: float, sample_index: int
    ) -> None:
        if self._sleep_aid_controller is None or not self._sleep_aid_controller.is_active:
            return

        gate = self._quality_gate
        was_in_bad = gate.in_bad_segment
        result = gate.push(raw)
        if result.is_bad:
            if not was_in_bad:
                self._sleep_aid_tracker.reset()
            return

        if was_in_bad:
            self._sleep_aid_tracker.reset()

        snapshot = _to_alpha_phase_snapshot(self._sleep_aid_tracker.push(alpha))
        if not self._is_in_sleep_aid_burst_window():
            warm = self._sleep_aid_controller.warmup_remaining()
            if warm > 0.0 and int(warm) != getattr(self, "_sleep_aid_last_warm_sec", -1):
                self._sleep_aid_last_warm_sec = int(warm)
                self._refresh_sleep_aid_button()
            return

        params = self._sleep_aid_controller.params
        guard_sec = (
            self._sleep_aid_controller.total_latency_sec
            + params.stimulus_effect_latency_sec
            + params.burst_duration_ms / 1000.0
            + params.quality_lookahead_sec
        )
        triggered = self._sleep_aid_controller.process_snapshot(
            snapshot,
            is_stimulus_ok=lambda: gate.is_stimulus_window_clean(guard_sec),
            on_skip=gate.record_skip,
        )
        if triggered:
            self._record_sleep_aid_burst(
                sample_index=sample_index,
                snapshot=snapshot,
            )
        warm = self._sleep_aid_controller.warmup_remaining()
        if warm > 0.0 and int(warm) != getattr(self, "_sleep_aid_last_warm_sec", -1):
            self._sleep_aid_last_warm_sec = int(warm)
            self._refresh_sleep_aid_button()
        elif warm <= 0.0 and getattr(self, "_sleep_aid_last_warm_sec", -1) != 0:
            self._sleep_aid_last_warm_sec = 0
            self._refresh_sleep_aid_button()

    def _read_audio_params(self) -> StereoAudioParams:
        """读取 groupBox_2 中频率/相位/时长并校验。"""
        return StereoAudioController.parse_params(
            left_frequency_hz=self.ui.lineEdit_7.text(),
            right_frequency_hz=self.ui.lineEdit_6.text(),
            left_phase_deg=self.ui.lineEdit_9.text(),
            right_phase_deg=self.ui.lineEdit_10.text(),
            duration_sec=self.ui.lineEdit_8.text(),
        )

    def _refresh_audio_button(self) -> None:
        btn = self.ui.pushButton_7
        if self._audio_controller.is_playing:
            btn.setText("停止音频")
            btn.setStyleSheet(
                "QPushButton { font-size:14px; font-weight:bold; border:none;"
                "border-radius:8px; min-height:48px;"
                "background-color:#C62828; color:#FFFFFF; }"
            )
        else:
            btn.setText("音频")
            btn.setStyleSheet(
                "QPushButton { font-size:14px; font-weight:bold; border:none;"
                "border-radius:8px; min-height:48px;"
                "background-color:#1565C0; color:#FFFFFF; }"
            )

    @QtCore.pyqtSlot()
    def on_toggle_audio(self) -> None:
        """pushButton_7：开始/停止立体声正弦波输出。"""
        if self._sleep_aid_controller is not None and self._sleep_aid_controller.is_active:
            self._stop_sleep_aid_stimulus(manual=True)
        if self._audio_controller.is_playing:
            self._audio_controller.stop()
            self._refresh_audio_button()
            self._log("音频已手动停止")
            return
        try:
            params = self._read_audio_params()
        except ValueError as exc:
            self._log(f"音频参数错误: {exc}")
            return
        except ImportError as exc:
            self._log(f"音频模块不可用: {exc}（请 pip install sounddevice）")
            return
        try:
            self._audio_controller.start(params)
        except Exception as exc:
            self._log(f"音频启动失败: {exc}")
            self._refresh_audio_button()
            return
        self._refresh_audio_button()
        duration_text = (
            f"时长 {params.duration_sec:g}s"
            if params.duration_sec > 0
            else "持续播放，再次点击停止"
        )
        self._log(
            "音频输出: "
            f"左 {params.left_frequency_hz:g} Hz ∠{params.left_phase_deg:g}°, "
            f"右 {params.right_frequency_hz:g} Hz ∠{params.right_phase_deg:g}°, "
            f"{duration_text}"
        )

    @QtCore.pyqtSlot()
    def _on_audio_stopped(self) -> None:
        """定时播放结束后的 UI 刷新（主线程）。"""
        self._refresh_audio_button()
        self._log("音频播放结束")

    def _switch_to_eeg_view(self) -> None:
        """切换到 EEG 波形页（stackedWidget page）。"""
        self.ui.stackedWidget.setCurrentIndex(0)
        self._active_view = "eeg"
        if self._analysis_plot_active:
            self._waveform.hide()
            self._multi_waveform.hide()
            self._offline_view.hide()
            self._sync_analysis_plot_geometry()
            self._analysis_plot.show()
            self._analysis_plot.raise_()
            self.setWindowTitle("EEG 功率对比")
            self._update_status_bar()
            return
        if self._offline_view_active:
            self._waveform.hide()
            self._multi_waveform.hide()
            self._analysis_plot.hide()
            self._sync_offline_geometry()
            self._offline_view.show()
            self._offline_view.raise_()
            self.setWindowTitle(
                f"EEG 离线查看 · {self._offline_view.source_name or 'CSV'}"
            )
            self._update_status_bar()
            return
        self._offline_view.hide()
        self._analysis_plot.hide()
        self._show_current_eeg_waveform()
        self._apply_display_mode(self._current_display_mode())

    def _is_multi_eeg_mode(self) -> bool:
        return self._eeg_channel_mode == "multi"

    def _selected_eeg_channels(self) -> tuple[int, ...]:
        if not self._is_multi_eeg_mode():
            return (0,)
        selected = tuple(
            channel
            for channel, checkbox in self._eeg_channel_checkboxes.items()
            if checkbox.isChecked()
        )
        return selected or (0,)

    def _set_eeg_channel_checkboxes_enabled(self, enabled: bool) -> None:
        for channel, checkbox in self._eeg_channel_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setEnabled(enabled)
            checkbox.setChecked(enabled or channel == 0)
            checkbox.blockSignals(False)

    def _show_current_eeg_waveform(self) -> None:
        if self._is_multi_eeg_mode():
            self._waveform.hide()
            self._sync_multi_waveform_geometry()
            self._multi_waveform.show()
            self._multi_waveform.raise_()
        else:
            self._multi_waveform.hide()
            self._sync_waveform_geometry()
            self._waveform.show()
            self._waveform.raise_()

    @QtCore.pyqtSlot(int)
    def _on_eeg_channel_mode_changed(self, index: int) -> None:
        mode = "multi" if index == 1 else "single"
        if mode == self._eeg_channel_mode:
            return
        self._eeg_channel_mode = mode
        self._set_eeg_channel_checkboxes_enabled(mode == "multi")
        self._reset_eeg_display_after_channel_change()
        if self._link is not None:
            self._link.set_channel_count(EEG_MULTI_CHANNEL_COUNT if mode == "multi" else 1)
        self._log("EEG serial protocol switched to six-channel" if mode == "multi" else "EEG serial protocol switched to single-channel")

    @QtCore.pyqtSlot(int, bool)
    def _on_eeg_channel_checkbox_toggled(self, channel: int, checked: bool) -> None:
        if not self._is_multi_eeg_mode():
            return
        if not self._selected_eeg_channels():
            checkbox = self._eeg_channel_checkboxes[channel]
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
            return
        self._reset_eeg_display_after_channel_change()

    def _reset_eeg_display_after_channel_change(self) -> None:
        self._display_mode = ""
        self._decim_counter = 0
        self._rhythm.reset()
        for processor in self._multi_rhythm:
            processor.reset()
        self._waveform.clear()
        self._multi_waveform.clear()
        self._apply_display_mode(self._current_display_mode())
        if self._active_view == "eeg" and not (self._offline_view_active or self._analysis_plot_active):
            self._show_current_eeg_waveform()

    def _switch_to_osc_view(self) -> None:
        """切换到振子波形页（stackedWidget page_2）。"""
        self.ui.stackedWidget.setCurrentIndex(1)
        self._active_view = "osc"
        self._refresh_osc_display()

    def _uncheck_osc_axis_checkboxes(self) -> None:
        for checkbox in self._osc_axis_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def _uncheck_osc_band_checkboxes(self) -> None:
        for checkbox in self._osc_display_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def _current_osc_selected_axes(self) -> tuple[str, ...]:
        return tuple(
            axis
            for axis, checkbox in self._osc_axis_checkboxes.items()
            if checkbox.isChecked()
        )

    def _refresh_osc_display(self) -> None:
        axes = self._current_osc_selected_axes()
        if axes:
            self._apply_osc_axis_display(axes)
        else:
            self._apply_osc_display_mode(self._current_osc_display_mode())

    @QtCore.pyqtSlot(str, bool)
    def _on_osc_checkbox_toggled(self, mode: str, checked: bool) -> None:
        """勾选振子节律或 M_Fre 后立即切换显示，各选项互斥。"""
        if checked:
            self._uncheck_osc_axis_checkboxes()
            for other, checkbox in self._osc_display_checkboxes.items():
                if other != mode:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)
            self._apply_osc_display_mode(mode)
            if self._active_view != "osc":
                self._switch_to_osc_view()
        elif not any(cb.isChecked() for cb in self._osc_display_checkboxes.values()):
            if self._current_osc_selected_axes():
                return
            m_cb = self._osc_display_checkboxes["m_freq"]
            m_cb.blockSignals(True)
            m_cb.setChecked(True)
            m_cb.blockSignals(False)
            self._apply_osc_display_mode("m_freq")

    @QtCore.pyqtSlot(str, bool)
    def _on_osc_axis_checkbox_toggled(self, axis: str, checked: bool) -> None:
        """勾选 X/Y/Z 时显示三轴加速度，可多选；与节律选项互斥。"""
        if checked:
            self._uncheck_osc_band_checkboxes()
        axes = self._current_osc_selected_axes()
        if axes:
            self._apply_osc_axis_display(axes)
            if self._active_view != "osc":
                self._switch_to_osc_view()
        elif not any(cb.isChecked() for cb in self._osc_display_checkboxes.values()):
            m_cb = self._osc_display_checkboxes["m_freq"]
            m_cb.blockSignals(True)
            m_cb.setChecked(True)
            m_cb.blockSignals(False)
            self._apply_osc_display_mode("m_freq")

    @QtCore.pyqtSlot(str, bool)
    def _on_display_checkbox_toggled(self, mode: str, checked: bool) -> None:
        """实时：节律/raw 互斥；离线：raw 必显，其它节律可多选叠加。"""
        if self._offline_view_active:
            if mode == "raw" and not checked:
                raw_cb = self._display_checkboxes["raw"]
                raw_cb.blockSignals(True)
                raw_cb.setChecked(True)
                raw_cb.blockSignals(False)
                self._log("离线查看时 raw data 始终显示，不可取消勾选")
            self._refresh_offline_visible_channels()
            if self._active_view != "eeg":
                self._switch_to_eeg_view()
            self._update_status_bar()
            return
        if checked:
            for other, checkbox in self._display_checkboxes.items():
                if other != mode:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)
            self._apply_display_mode(mode)
            if self._active_view != "eeg":
                self._switch_to_eeg_view()
        elif not any(cb.isChecked() for cb in self._display_checkboxes.values()):
            raw_cb = self._display_checkboxes["raw"]
            raw_cb.blockSignals(True)
            raw_cb.setChecked(True)
            raw_cb.blockSignals(False)
            self._apply_display_mode("raw")

    def _sync_waveform_geometry(self) -> None:
        """EEG 波形控件与 Designer 中 graphicsView 同位置同大小。"""
        self._waveform.setGeometry(self.ui.graphicsView.geometry())

    def _sync_multi_waveform_geometry(self) -> None:
        self._multi_waveform.setGeometry(self.ui.graphicsView.geometry())

    def _sync_offline_geometry(self) -> None:
        self._offline_view.setGeometry(self.ui.graphicsView.geometry())

    def _sync_analysis_plot_geometry(self) -> None:
        self._analysis_plot.setGeometry(self.ui.graphicsView.geometry())

    def _sync_osc_waveform_geometry(self) -> None:
        """振子波形控件与 EEG 绘图区同尺寸。"""
        self._osc_graphics_view.setGeometry(self.ui.graphicsView.geometry())
        self._osc_waveform.setGeometry(self.ui.graphicsView.geometry())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self.ui, "bottomPanel"):
            self._layout_main_regions()
        if not hasattr(self, "_waveform") or not hasattr(self, "_osc_waveform"):
            return
        self._sync_waveform_geometry()
        self._sync_multi_waveform_geometry()
        self._sync_offline_geometry()
        self._sync_analysis_plot_geometry()
        self._sync_osc_waveform_geometry()
        self._waveform.refresh_layout()
        self._multi_waveform.refresh_layout()
        self._osc_waveform.refresh_layout()

    def _current_display_mode(self) -> str:
        for mode, checkbox in self._display_checkboxes.items():
            if checkbox.isChecked():
                return mode
        return "raw"

    def _current_osc_display_mode(self) -> str:
        for mode, checkbox in self._osc_display_checkboxes.items():
            if checkbox.isChecked():
                return mode
        return "m_freq"

    def _apply_display_mode(self, mode: str) -> None:
        """按勾选配置原始或各节律波形（显示均为 100 Hz）。"""
        if mode == self._display_mode:
            return
        self._display_mode = mode
        self._decim_counter = 0
        wheel_hint = "滚轮:时间轴缩放  Ctrl+滚轮:Y缩放  Shift+滚轮:平移"
        if mode == "raw":
            legend = f"RAW CH1  |  {wheel_hint}"
            line_color = "#BF360C"
            y_mid = RAW_DEFAULT_Y_MID
            y_amp = RAW_DEFAULT_Y_AMP
            title = "EEG 实时波形 · CH1 · RAW"
        else:
            low_hz, high_hz = EEG_BANDS[mode]
            label = BAND_LABELS[mode]
            legend = f"{label} {low_hz:g}-{high_hz:g} Hz  |  {wheel_hint}"
            line_color = RHYTHM_PLOT_COLORS[mode]
            y_mid = RHYTHM_DEFAULT_Y_MID
            y_amp = RHYTHM_DEFAULT_Y_AMP
            title = f"EEG 实时波形 · CH1 · {label} ({low_hz:g}-{high_hz:g} Hz)"
        self._eeg_base_legend = legend
        if mode == "alpha":
            legend = self._alpha_reject_status_legend(legend)
        if self._is_multi_eeg_mode():
            selected_channels = self._selected_eeg_channels()
            self._multi_waveform.configure_display(
                mode=mode,
                sample_rate=RAW_DISPLAY_RATE,
                max_points=RAW_PLOT_MAX_POINTS,
                line_color=line_color,
                fixed_y_axis=True,
                y_mid=y_mid,
                y_amp=y_amp,
                min_y_amp=MIN_Y_AMP,
                max_y_amp=MAX_Y_AMP,
                active_channels=selected_channels,
            )
            channel_text = "+".join(f"CH{channel + 1}" for channel in selected_channels)
            title = title.replace("CH1", channel_text)
        else:
            self._waveform.configure_display(
                legend=legend,
                sample_rate=RAW_DISPLAY_RATE,
                max_points=RAW_PLOT_MAX_POINTS,
                line_color=line_color,
                fixed_y_axis=True,
                y_mid=y_mid,
                y_amp=y_amp,
                use_full_plot_height=True,
                min_y_amp=MIN_Y_AMP,
                max_y_amp=MAX_Y_AMP,
            )
        if self._active_view == "eeg":
            self.setWindowTitle(title)

    def _apply_osc_display_mode(self, mode: str) -> None:
        """按勾选配置振子各频段或 M_Fre 波形。"""
        if mode == self._osc_display_mode and self._osc_display_kind == "band":
            return
        self._osc_display_kind = "band"
        self._osc_display_mode = mode
        self._osc_axis_display_key = ()
        self._osc_decim_counter = 0
        wheel_hint = "滚轮:时间轴缩放  Ctrl+滚轮:Y缩放  Shift+滚轮:平移"
        if mode == "m_freq":
            legend = f"M_Fre 主振加速度 ({ACCEL_DISPLAY_UNIT})  |  {wheel_hint}"
            line_color = M_FREQ_PLOT_COLOR
            title = f"振子 · 主振频率加速度 ({ACCEL_DISPLAY_UNIT})"
        else:
            low_hz, high_hz = OSC_BANDS[mode]
            label = OSC_BAND_LABELS[mode]
            legend = (
                f"振子 {label} {low_hz:g}-{high_hz:g} Hz "
                f"加速度 ({ACCEL_DISPLAY_UNIT})  |  {wheel_hint}"
            )
            line_color = RHYTHM_PLOT_COLORS[mode]
            title = f"振子 · {label} 加速度 ({ACCEL_DISPLAY_UNIT})"
        self._osc_waveform.configure_display(
            legend=legend,
            sample_rate=OSC_DISPLAY_RATE,
            max_points=OSC_PLOT_MAX_POINTS,
            line_color=line_color,
            fixed_y_axis=True,
            y_mid=OSC_DEFAULT_Y_MID,
            y_amp=OSC_DEFAULT_Y_AMP,
            use_full_plot_height=True,
            y_axis_label=OSC_Y_AXIS_LABEL,
            min_y_amp=OSC_MIN_Y_AMP,
            max_y_amp=OSC_MAX_Y_AMP,
        )
        if self._active_view == "osc":
            self.setWindowTitle(title)

    def _apply_osc_axis_display(self, axes: tuple[str, ...]) -> None:
        """按勾选显示 X/Y/Z 三轴高通加速度，可多选叠加。"""
        key = tuple(sorted(axes))
        if key == self._osc_axis_display_key and self._osc_display_kind == "axis":
            return
        self._osc_display_kind = "axis"
        self._osc_axis_display_key = key
        self._osc_display_mode = ""
        self._osc_decim_counter = 0
        wheel_hint = "滚轮:时间轴缩放  Ctrl+滚轮:Y缩放  Shift+滚轮:平移"
        axis_text = "+".join(axis.upper() for axis in axes)
        legend = (
            f"振子 {axis_text} 加速度 ({ACCEL_DISPLAY_UNIT})  |  {wheel_hint}"
        )
        title = f"振子 · {axis_text} 加速度 ({ACCEL_DISPLAY_UNIT})"
        self._osc_waveform.configure_display(
            legend=legend,
            sample_rate=OSC_DISPLAY_RATE,
            max_points=OSC_PLOT_MAX_POINTS,
            fixed_y_axis=True,
            y_mid=OSC_DEFAULT_Y_MID,
            y_amp=OSC_DEFAULT_Y_AMP,
            use_full_plot_height=True,
            y_axis_label=OSC_Y_AXIS_LABEL,
            min_y_amp=OSC_MIN_Y_AMP,
            max_y_amp=OSC_MAX_Y_AMP,
            multi_series={axis: OSC_AXIS_COLORS[axis] for axis in axes},
        )
        if self._active_view == "osc":
            self.setWindowTitle(title)

    def _refresh_capture_button(self) -> None:
        btn = self.ui.pushButton
        if self._running:
            btn.setStyleSheet(  ## 采集中：绿色
                "QPushButton { font-size:14px; font-weight:bold; border:none;"
                "border-radius:8px; min-height:48px;"
                "background-color:#388E3C; color:#FFFFFF; }"
            )
            btn.setText("采集中 · 停止")
        else:
            btn.setStyleSheet(  ## 已停止：灰色
                "QPushButton { font-size:14px; font-weight:bold; border:none;"
                "border-radius:8px; min-height:48px;"
                "background-color:#9E9E9E; color:#F5F5F5; }"
            )
            btn.setText("已停止 · 启动")
        self._refresh_serial_button()

    def _log_available_ports(self) -> None:
        try:
            for dev in list_ports():  ## 枚举 COM 口
                self._log(f"  可用: {dev}")
        except Exception as exc:
            self._log(f"枚举串口失败: {exc}")

    def _log(self, message: str) -> None:
        edit = self.ui.plainTextEdit  ## 底部日志框
        edit.appendPlainText(message)
        doc = edit.document()
        if doc.blockCount() > LOG_MAX_LINES:  ## 超出则删最旧一行
            cursor = QtGui.QTextCursor(doc)
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.select(QtGui.QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _serial_settings_from_ui(self) -> Dict[str, object]:
        baudrate = self._baudrate
        return {
            "baudrate": baudrate,
            "bytesize": DEFAULT_SERIAL_BYTESIZE,
            "parity": DEFAULT_SERIAL_PARITY,
            "stopbits": DEFAULT_SERIAL_STOPBITS,
        }

    @staticmethod
    def _serial_settings_match(link: object, settings: Dict[str, object]) -> bool:
        return (
            getattr(link, "baudrate", None) == settings["baudrate"]
            and getattr(link, "bytesize", None) == settings["bytesize"]
            and getattr(link, "parity", None) == settings["parity"]
            and getattr(link, "stopbits", None) == settings["stopbits"]
        )

    @staticmethod
    def _format_serial_settings(settings: Dict[str, object]) -> str:
        return (
            f"{settings['baudrate']}bps "
            f"{settings['bytesize']}{settings['parity']}{settings['stopbits']}"
        )

    def _refresh_serial_button(self) -> None:
        btn = getattr(self.ui, "pushButton_serial_toggle", None)
        if btn is None:
            return
        eeg_open = isinstance(self._link, Ks1082Serial) and self._link.is_open
        osc_open = self._osc_link is not None and self._osc_link.is_open
        if eeg_open or osc_open:
            btn.setText("关闭串口")
            btn.setStyleSheet(
                "QPushButton { background:#B91C1C; color:white; font-weight:bold; border:none; border-radius:6px; }"
                "QPushButton:hover { background:#991B1B; }"
            )
        else:
            btn.setText("打开串口")
            btn.setStyleSheet(
                "QPushButton { background:#2563EB; color:white; font-weight:bold; border:none; border-radius:6px; }"
                "QPushButton:hover { background:#1D4ED8; }"
            )

    def _toggle_serial_connection(self) -> None:
        eeg_open = isinstance(self._link, Ks1082Serial) and self._link.is_open
        osc_open = self._osc_link is not None and self._osc_link.is_open
        if eeg_open or osc_open:
            if self._link is not None:
                self._link.close()
            if self._osc_link is not None:
                self._osc_link.close()
            self._running = False
            self._refresh_capture_button()
            self._refresh_serial_button()
            self._update_status_bar()
            self._log("串口已关闭")
            return

        eeg_ok = self._open_serial()
        osc_ok = self._open_osc_serial()
        self._refresh_serial_button()
        if eeg_ok or osc_ok:
            self._log("串口已打开，可点击启动开始采集")
        else:
            self._log("串口打开失败，请检查端口与参数")

    def _ui_serial_port(self) -> str:
        """从 lineEdit_5 读取 EEG 串口号。"""
        port = self.ui.lineEdit_5.text().strip()
        if not port:
            port = DEFAULT_SERIAL_PORT
        return port

    def _ui_osc_serial_port(self) -> str:
        """从 lineEdit_4 读取振子串口号。"""
        port = self.ui.lineEdit_4.text().strip()
        if not port:
            port = DEFAULT_OSC_SERIAL_PORT
        return port

    def _open_serial(self) -> bool:
        """打开 EEG 串口。"""
        port = self._ui_serial_port()
        serial_settings = self._serial_settings_from_ui()
        try:
            if (
                isinstance(self._link, Ks1082Serial)
                and self._link.is_open
                and self._link.port == port
                and self._serial_settings_match(self._link, serial_settings)
            ):
                return True
            if self._link is not None:
                self._link.close()
            self._link = Ks1082Serial(port, timeout=0.05, **serial_settings)
            self._link.open()
            self._port = port
            self._log(f"EEG 串口已连接: {self._port} @ {self._format_serial_settings(serial_settings)}")
            self._log(f"EEG 采样率 {self._rhythm.sample_rate:.0f} Hz，显示 {RAW_DISPLAY_RATE} Hz")
            self._refresh_serial_button()
            return True
        except Exception as exc:
            self._link = None
            self._log(f"EEG 串口打开失败 ({port}): {exc}")
            self._refresh_serial_button()
            return False

    def _open_osc_serial(self) -> bool:
        """按 UI 串口框打开振子串口；成功返回 True。"""
        port = self._ui_osc_serial_port()
        serial_settings = self._serial_settings_from_ui()
        try:
            if (
                self._osc_link is not None
                and self._osc_link.is_open
                and self._osc_link.port == port
                and self._serial_settings_match(self._osc_link, serial_settings)
            ):
                return True
            if self._osc_link is not None:
                self._osc_link.close()
            self._osc_link = OscillatorSerial(port, timeout=0.05, **serial_settings)
            self._osc_link.open()
            self._osc_port = port
            self._log(f"振子串口已连接: {self._osc_port} @ {self._format_serial_settings(serial_settings)}")
            self._log(
                f"振子 MCU 采样 {OSC_SAMPLE_RATE:.0f} Hz，"
                f"每批 {OSC_BUFFER_SIZE} 点上传，"
                f"算法按 {self._osc_proc.sample_rate:.0f} Hz 逐点处理，"
                f"界面显示 {OSC_DISPLAY_RATE} Hz"
            )
            self._log(
                f"振子标定: {SENSOR_UNITS_PER_G:g} 计数/g，"
                f"g={GRAVITY_M_S2} m/s²，纵轴 {ACCEL_DISPLAY_UNIT}"
            )
            self._refresh_serial_button()
            return True
        except Exception as exc:
            self._osc_link = None
            self._log(f"振子串口打开失败 ({port}): {exc}")
            self._refresh_serial_button()
            return False

    def _update_status_bar(self) -> None:
        """根据当前页面刷新状态栏。"""
        if self._active_view == "osc":
            if self._osc_display_kind == "axis":
                axes = self._current_osc_selected_axes()
                mode_text = "+".join(axis.upper() for axis in axes) if axes else "—"
            else:
                dom_f = self._osc_proc.dominant_freq_hz
                dom_b = OSC_BAND_LABELS.get(
                    self._osc_proc.dominant_band, self._osc_proc.dominant_band
                )
                mode = self._current_osc_display_mode()
                mode_text = (
                    f"M_Fre≈{dom_f:.1f}Hz ({dom_b})"
                    if mode == "m_freq"
                    else f"{OSC_BAND_LABELS.get(mode, mode)}"
                )
            self.ui.statusbar.showMessage(
                f"振子 {mode_text}  |  "
                f"采样 {OSC_SAMPLE_RATE:.0f} Hz · {OSC_BUFFER_SIZE}点/批 · "
                f"显示 {OSC_DISPLAY_RATE} Hz"
            )
        else:
            if self._analysis_plot_active:
                self.ui.statusbar.showMessage(
                    "功率对比图  |  工具栏可拖动横/纵轴  |  填写段时间后点「功率对比」刷新"
                )
                return
            if self._offline_view_active:
                bands = [
                    BAND_LABELS.get(mode, mode)
                    for mode, cb in self._display_checkboxes.items()
                    if mode != "raw" and cb.isChecked()
                ]
                labels = "raw" + (("+" + ",".join(bands)) if bands else "")
                self.ui.statusbar.showMessage(
                    f"离线查看 {self._offline_view.source_name}  |  "
                    f"通道 [{labels}]  |  "
                    f"{self._offline_view.sample_rate:.0f} Hz  |  "
                    f"工具栏可拖动横/纵轴"
                )
                return
            mode = self._current_display_mode()
            rx = self._rhythm.measured_sample_rate
            rx_text = f"{rx:.0f}" if rx is not None else "—"
            if mode == "raw":
                self.ui.statusbar.showMessage(
                    f"RAW CH1 (串口)  |  显示≈{RAW_DISPLAY_RATE} Hz  |  接收≈{rx_text} Hz"
                )
            else:
                label = BAND_LABELS[mode]
                low_hz, high_hz = EEG_BANDS[mode]
                self.ui.statusbar.showMessage(
                    f"{label} {low_hz:g}-{high_hz:g} Hz (串口)  |  "
                    f"显示≈{RAW_DISPLAY_RATE} Hz  |  接收≈{rx_text} Hz"
                )

    @QtCore.pyqtSlot()
    def on_toggle_capture(self) -> None:
        """启动/停止采集。"""
        stopping = self._running
        self._running = not self._running
        if self._running:
            eeg_ok = self._open_serial()
            osc_ok = self._open_osc_serial()
            if not eeg_ok and not osc_ok:
                self._running = False
                self._refresh_capture_button()
                self.ui.statusbar.showMessage("串口未连接")
                return
        if self._link is not None:
            if self._running:
                flushed = self._link.flush_input_buffer()
                self._link.bytes_received = 0
                self._sample_count = 0
                self._no_data_ticks = 0
                self._decim_counter = 0
                self._last_status_update = 0.0
                self._rhythm.reset()
                self._alpha_rejector.reset()
                self._reset_alpha_display_stats()
                self._waveform.clear()
                if flushed:
                    self._log(f"EEG 已丢弃暂停期间积压的 {flushed} 字节")
            else:
                self._link.discard_pending_input()
        if self._osc_link is not None:
            if self._running:
                flushed = self._osc_link.flush_input_buffer()
                self._osc_link.bytes_received = 0
                self._osc_sample_count = 0
                self._osc_no_data_ticks = 0
                self._osc_decim_counter = 0
                self._osc_proc.reset()
                self._osc_waveform.clear()
                if flushed:
                    self._log(f"振子已丢弃暂停期间积压的 {flushed} 字节")
            else:
                self._osc_link.discard_pending_input()
        if self._running:
            self._begin_timed_test_if_configured()
        elif stopping:
            if (
                self._sleep_aid_controller is not None
                and self._sleep_aid_controller.is_active
            ):
                self._sleep_aid_controller.stop()
                self._sleep_aid_timed_auto = False
                self._refresh_sleep_aid_button()
            if self._test_duration_sec is not None:
                if self._long_record_active:
                    session_dir = self._long_session_dir
                    self._flush_long_record_buffer(final=True)
                    if session_dir is not None:
                        self._run_long_record_postprocess(session_dir)
                    self._sleep_aid_burst_record.clear()
                    self._reset_timed_test_state()
                    if session_dir is not None:
                        self._last_eeg_session_dir = session_dir
                        self._log(
                            f"手动停止长时记录：已保存剩余数据至 {session_dir}"
                        )
                    else:
                        self._log("手动停止长时记录：无数据可保存")
                else:
                    self._eeg_raw_record.clear()
                    self._sleep_aid_burst_record.clear()
                    self._reset_timed_test_state()
                    self._log("手动停止测试：未保存任何记录")
        self._refresh_capture_button()
        self._update_status_bar()
        view_label = "振子" if self._active_view == "osc" else "EEG"
        self._log(f"采集状态: {'运行' if self._running else '暂停'} ({view_label})")

    def _poll_eeg_serial(self) -> None:
        """读 EEG 串口并刷新 page 波形。"""
        if self._link is None or not self._link.is_open:
            return
        if not self._running:
            try:
                self._link.discard_pending_input()
            except Exception as exc:
                self._log(f"EEG 暂停丢弃缓冲异常: {exc}")
            return
        prev_bytes = self._link.bytes_received
        samples = self._link.poll_once()
        got_bytes = self._link.bytes_received - prev_bytes

        if samples:
            if len(samples) > MAX_SAMPLES_PER_POLL:
                samples = samples[-MAX_SAMPLES_PER_POLL:]
            if self._is_multi_eeg_mode() and not (self._offline_view_active or self._analysis_plot_active):
                mode = self._current_display_mode()
                self._apply_display_mode(mode)
                band = None if mode == "raw" else mode
                plot_batches: list[list[float]] = []
                for sample in samples:
                    channels = sample.channels[:EEG_MULTI_CHANNEL_COUNT]
                    if len(channels) < EEG_MULTI_CHANNEL_COUNT:
                        continue
                    if self._is_test_recording_phase():
                        self._eeg_raw_record.append(sample.channel1)
                    self._sample_count += 1
                    self._decim_counter += 1
                    if band is None:
                        values = [float(value) for value in channels]
                    else:
                        values = [
                            self._multi_rhythm[index].push(value, band)
                            for index, value in enumerate(channels)
                        ]
                    if self._decim_counter >= RAW_DECIM_FACTOR:
                        self._decim_counter = 0
                        plot_batches.append(values)
                if self._is_waveform_display_active() and plot_batches:
                    self._multi_waveform.append_channel_values_batch(plot_batches)
                self._no_data_ticks = 0
                return
            if self._offline_view_active or self._analysis_plot_active:
                mode = "raw"
            else:
                mode = self._current_display_mode()
                self._apply_display_mode(mode)
            plot_values: list[float] = []
            plot_reject_flags: list[bool] = []
            band = None if mode == "raw" else mode
            decim_rejected = False
            sleep_aid = (
                self._sleep_aid_controller is not None
                and self._sleep_aid_controller.is_active
            )
            if sleep_aid:
                band = "alpha"

            for sample in samples:
                if self._is_test_recording_phase():
                    self._eeg_raw_record.append(sample.channel1)
                self._sample_count += 1
                self._decim_counter += 1
                value = self._rhythm.push(sample.channel1, band)
                rejected = False
                if band == "alpha":
                    rejected = self._alpha_rejector.push(value)
                    decim_rejected = decim_rejected or rejected
                if sleep_aid:
                    self._process_sleep_aid_sample(
                        sample.channel1, value, self._sample_count - 1
                    )
                if self._offline_view_active or self._analysis_plot_active:
                    if self._decim_counter >= RAW_DECIM_FACTOR:
                        self._decim_counter = 0
                        decim_rejected = False
                    continue
                if self._decim_counter >= RAW_DECIM_FACTOR:
                    self._decim_counter = 0
                    display_rejected = decim_rejected
                    decim_rejected = False
                    if band == "alpha":
                        self._alpha_display_total_points += 1
                        if display_rejected:
                            self._alpha_display_rejected_points += 1
                    plot_values.append(value)
                    plot_reject_flags.append(display_rejected)

            if self._offline_view_active or self._analysis_plot_active:
                self._no_data_ticks = 0
            elif self._is_waveform_display_active():
                if band == "alpha":
                    self._waveform.update_legend(
                        self._alpha_reject_status_legend(self._eeg_base_legend)
                    )
                if plot_values:
                    self._waveform.append_alphas(
                        plot_values,
                        plot_reject_flags if band == "alpha" else None,
                    )
            else:
                plot_values.clear()
            if not (self._offline_view_active or self._analysis_plot_active):
                self._no_data_ticks = 0
        elif got_bytes == 0:
            self._no_data_ticks += 1
        else:
            self._no_data_ticks += 1

        if self._no_data_ticks == 50:
            total = self._link.bytes_received
            if total == 0:
                self._log("EEG 串口无任何数据: 请检查 COM 口与接线")
            else:
                self._log(f"EEG 已收到 {total} 字节但未解析出波形")
            self._no_data_ticks = 0

    def _poll_osc_serial(self) -> None:
        """读振子串口并刷新 page_2 波形。"""
        if self._osc_link is None or not self._osc_link.is_open:
            return
        if not self._running:
            try:
                self._osc_link.discard_pending_input()
            except Exception as exc:
                self._log(f"振子暂停丢弃缓冲异常: {exc}")
            return
        prev_bytes = self._osc_link.bytes_received
        batches = self._osc_link.poll_once()
        got_bytes = self._osc_link.bytes_received - prev_bytes

        if batches:
            axes = self._current_osc_selected_axes()
            plot_values: list[float] = []
            plot_batch: list[dict[str, float]] = []

            if axes:
                self._apply_osc_axis_display(axes)
            else:
                mode = self._current_osc_display_mode()
                self._apply_osc_display_mode(mode)

            for batch in batches:
                ## 每批 100 点为 MCU 缓冲上传；仍按 1000 Hz 顺序逐点送入滤波器
                for point in batch.points:
                    self._osc_sample_count += 1
                    self._osc_decim_counter += 1
                    if axes:
                        accel = self._osc_proc.push_axes(point.x, point.y, point.z)
                        if self._osc_decim_counter >= OSC_DECIM_FACTOR:
                            self._osc_decim_counter = 0
                            plot_batch.append({axis: accel[axis] for axis in axes})
                    elif mode == "m_freq":
                        value = self._osc_proc.push_m_freq(point.x, point.y, point.z)
                        if self._osc_decim_counter >= OSC_DECIM_FACTOR:
                            self._osc_decim_counter = 0
                            plot_values.append(value)
                    else:
                        value = self._osc_proc.push(point.x, point.y, point.z, band=mode)
                        if self._osc_decim_counter >= OSC_DECIM_FACTOR:
                            self._osc_decim_counter = 0
                            plot_values.append(value)

            if self._is_waveform_display_active():
                if plot_batch:
                    self._osc_waveform.append_multi_batch(plot_batch)
                elif plot_values:
                    self._osc_waveform.append_alphas(plot_values)
                if not axes and mode == "m_freq" and self._osc_proc.dominant_freq_hz > 0:
                    dom_f = self._osc_proc.dominant_freq_hz
                    dom_b = OSC_BAND_LABELS.get(
                        self._osc_proc.dominant_band, self._osc_proc.dominant_band
                    )
                    self._osc_waveform._legend_text = (
                        f"M_Fre {dom_f:.1f} Hz ({dom_b}) "
                        f"{ACCEL_DISPLAY_UNIT}"
                    )
            self._osc_no_data_ticks = 0
        elif got_bytes == 0:
            self._osc_no_data_ticks += 1
        else:
            self._osc_no_data_ticks += 1

        if self._osc_no_data_ticks == 50:
            total = self._osc_link.bytes_received
            if total == 0:
                self._log("振子串口无任何数据: 请检查 COM 口与接线")
            else:
                self._log(f"振子已收到 {total} 字节但未解析出批次")
            self._osc_no_data_ticks = 0

    @QtCore.pyqtSlot()
    def on_poll_serial(self) -> None:
        """定时读串口并刷新波形。"""
        try:
            self._poll_eeg_serial()
            self._poll_osc_serial()
            if (
                self._running
                and self._test_duration_sec is not None
                and self._is_timed_test_finished()
            ):
                self._stop_capture_due_to_timeout()
                return
            if self._running and self._test_duration_sec is not None:
                self._update_test_countdown_lcd()
                self._maybe_manage_timed_sleep_aid()
                self._maybe_save_long_record_chunk()
            now = time.monotonic()
            if now - self._last_status_update >= 0.2:
                self._last_status_update = now
                self._update_status_bar()
        except Exception as exc:
            self._log(f"接收异常: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._poll_timer.stop()
        self._audio_controller.shutdown()
        if self._sleep_aid_controller is not None:
            self._sleep_aid_controller.shutdown()
        if self._link is not None:
            self._link.close()
        if self._osc_link is not None:
            self._osc_link.close()
        super().closeEvent(event)


def _resolve_serial_port() -> str:
    env_port = os.environ.get("KS1082_SERIAL_PORT", "").strip()
    if env_port:
        return env_port
    return DEFAULT_SERIAL_PORT


def _resolve_osc_serial_port() -> str:
    env_port = os.environ.get("OSC_SERIAL_PORT", "").strip()
    if env_port:
        return env_port
    return DEFAULT_OSC_SERIAL_PORT


def run_app(
    port: Optional[str] = None,
    baudrate: int = 115200,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
) -> None:
    app = QtWidgets.QApplication(sys.argv)  ## 创建 Qt 应用
    window = Ks1082MainWindow(  ## 主窗口
        port=port,
        baudrate=baudrate,
        sample_rate=sample_rate,
    )
    window.showMaximized()  ## 默认最大化显示窗口
    sys.exit(app.exec_())  ## 进入事件循环


if __name__ == "__main__":
    run_app()  ## 直接运行本文件时启动
