"""主窗口：串口采集、节律/原始波形显示、UI 事件绑定。"""

from __future__ import annotations  ## 前向引用类型

import csv  ## 定时测试 EEG raw 导出
import io  ## 捕获 power_cal 分析输出
import math
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
from scipy.signal import hilbert

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
    build_threshold_rejection,
    run_analysis,
)
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
    EegCsvReplay,
    Ks1082Serial,
    MCU_SAMPLE_RATE,
    RAW_TYPICAL_AMP,
    RAW_TYPICAL_MID,
    list_ports,
)
from oscillator_serial import OscillatorSerial, BUFFER_SIZE as OSC_BUFFER_SIZE
from controller import (
    AlphaPhaseSnapshot,
    SleepAidStimulusController,
    StereoAudioController,
    StereoAudioParams,
)


DEFAULT_SERIAL_PORT = "COM6"  ## EEG 默认串口
DEFAULT_OSC_SERIAL_PORT = "COM7"  ## 振子默认串口
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
DEFAULT_SEGMENT_A_START = "20"  ## 段 A 起始 (s)，对应 compare_segments[0][0]
DEFAULT_SEGMENT_A_END = "30"  ## 段 A 结束 (s)
DEFAULT_SEGMENT_B_START = "100"  ## 段 B 起始 (s)
DEFAULT_SEGMENT_B_END = "110"  ## 段 B 结束 (s)
EEG_REJECT_SEGMENT_SEC = 1.0  ## 阈值拒绝按 1 s 切片打标签，不删除/拼接原始点
EEG_REJECT_RATE_WARN = 0.20  ## 拒绝率超过 20% 时提示本次采集不宜用于分析
EEG_RAW_MIN_VALID = 50  ## ADC 贴近下电源轨视为饱和/脱落风险
EEG_RAW_MAX_VALID = 4045  ## ADC 贴近上电源轨视为饱和/脱落风险
EEG_SEGMENT_MAX_PTP = 1200.0  ## 片内峰峰值阈值，单位 ADC count
EEG_SEGMENT_MAX_DEVIATION = 800.0  ## 相对片内中位数的最大偏移阈值
RAW_DISPLAY_RATE = 100  ## 波形显示约 100 点/秒
RAW_DECIM_FACTOR = max(1, int(MCU_SAMPLE_RATE / RAW_DISPLAY_RATE))  ## 500→100 降采样比
RAW_PLOT_WINDOW_SECONDS = 60.0  ## 波形时间窗 (s)
RAW_PLOT_MAX_POINTS = int(RAW_DISPLAY_RATE * RAW_PLOT_WINDOW_SECONDS)  ## 缓冲约 200 点
SERIAL_POLL_INTERVAL_MS = max(2, int(1000 / MCU_SAMPLE_RATE))  ## 串口轮询 2 ms
MAX_SAMPLES_PER_POLL = 80  ## 单次 poll 最多处理的原始样本数，防止恢复后积压卡顿
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
SLEEP_AID_HILBERT_WINDOW = int(MCU_SAMPLE_RATE * 0.5)  ## Hilbert 窗 0.5 s
SLEEP_AID_MIN_HILBERT_SAMPLES = max(64, SLEEP_AID_HILBERT_WINDOW // 2)
SLEEP_AID_ALPHA_FREQ_MIN_HZ = 7.0
SLEEP_AID_ALPHA_FREQ_MAX_HZ = 14.0
SLEEP_AID_TROUGH_PHASE_RAD = math.pi


def _build_eeg_rejection_tags(
    values: List[int],
    sample_rate: float,
) -> Tuple[List[int], List[int], List[str], float, List[int], List[str], float]:
    """按固定时间片做质量标记，返回坏段/可疑段 tag 与比例。"""
    count = len(values)
    if count == 0:
        return [], [], [], 0.0, [], [], 0.0
    segment_size = max(1, int(round(sample_rate * EEG_REJECT_SEGMENT_SEC)))
    quality = build_threshold_rejection(np.asarray(values, dtype=np.float64), sample_rate)
    segment_ids = [index // segment_size for index in range(count)]
    reject_flags = quality.reject_mask.astype(int).tolist()
    reject_reasons = quality.reject_reasons or [""] * count
    suspicious_mask = (
        quality.suspicious_mask
        if quality.suspicious_mask is not None
        else np.zeros(count, dtype=bool)
    )
    suspicious_flags = suspicious_mask.astype(int).tolist()
    suspicious_reasons = quality.suspicious_reasons or [""] * count
    return (
        segment_ids,
        reject_flags,
        reject_reasons,
        quality.reject_rate,
        suspicious_flags,
        suspicious_reasons,
        quality.suspicious_rate,
    )


class AlphaPhaseTracker:
    """Alpha 带通序列 → Hilbert 瞬时相位与下一波谷时间。"""

    def __init__(
        self,
        sample_rate: float,
        window: int = SLEEP_AID_HILBERT_WINDOW,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.window = int(window)
        self._buf: Deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, alpha: float) -> AlphaPhaseSnapshot:
        self._buf.append(float(alpha))
        return self.snapshot()

    def snapshot(self) -> AlphaPhaseSnapshot:
        if len(self._buf) < SLEEP_AID_MIN_HILBERT_SAMPLES:
            return AlphaPhaseSnapshot(ready=False)

        arr = np.asarray(self._buf, dtype=np.float64)
        analytic = hilbert(arr)
        phases = np.unwrap(np.angle(analytic))
        phase_rad = float(phases[-1] % (2.0 * math.pi))

        tail = min(50, len(phases))
        if tail >= 2:
            dphase = phases[-1] - phases[-tail]
            dt = (tail - 1) / self.sample_rate
            inst_freq_hz = abs(dphase / (2.0 * math.pi * dt))
            inst_freq_hz = float(
                np.clip(
                    inst_freq_hz,
                    SLEEP_AID_ALPHA_FREQ_MIN_HZ,
                    SLEEP_AID_ALPHA_FREQ_MAX_HZ,
                )
            )
        else:
            inst_freq_hz = 10.0

        delta = (SLEEP_AID_TROUGH_PHASE_RAD - phase_rad) % (2.0 * math.pi)
        seconds_to_trough = delta / (2.0 * math.pi * inst_freq_hz)
        return AlphaPhaseSnapshot(
            ready=True,
            phase_rad=phase_rad,
            inst_freq_hz=inst_freq_hz,
            seconds_to_trough=seconds_to_trough,
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
            self._path_item.setPen(_make_wave_pen(line_color))  ## 更新颜色与线宽
        self._path_dirty = True
        self._sync_plot_size_from_viewport()
        self._update_axes_if_needed(self._point_count(), force=True)
        self._fit_scene()
        self._refresh_plot(force=True)  ## 重绘

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
        for buf in self._series_buffers.values():
            buf.clear()
        self._path_dirty = True
        self._last_plot_refresh = 0.0
        self._last_axis_refresh = 0.0
        self._refresh_plot(force=True)  ## 重绘空波形

    def append_alpha(self, alpha: float) -> None:
        self._points.append(alpha)  ## 追加点
        self._path_dirty = True
        self._refresh_plot()

    def append_alphas(self, alphas: Iterable[float]) -> None:
        for alpha in alphas:
            self._points.append(alpha)  ## 批量追加
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
        """根据滚轮位置判断所在区域：y=Y轴区，x=X轴区，None=绘图区。"""
        vp = self.viewport()
        vp_pos = vp.mapFromGlobal(event.globalPos())
        scene_pos = self.mapToScene(self.mapFromGlobal(event.globalPos()))
        left = self._plot_left
        top = self._plot_top
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
        return None

    def _handle_wheel_event(self, event: QtGui.QWheelEvent) -> bool:
        """raw 模式滚轮：Y/X 轴区缩放；Shift+滚轮平移。"""
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
        shift_pan = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
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
            for path_item in self._series_path_items.values():
                path_item.setPath(QtGui.QPainterPath())
            if not self._fixed_y_axis:
                self._y_mid = 0.0
                self._y_amp = 1.0
            self._update_axes_if_needed(0, force=True)
            return

        if self._multi_mode:
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
        if not self._fixed_y_axis:  ## Alpha 模式才自动跟踪 Y 量程
            self._y_mid = sum(values) / count  ## 均值作 Y 中心
            amplitude = max(abs(v - self._y_mid) for v in values)  ## 最大偏离
            if amplitude < 1e-6:
                amplitude = 1.0  ## 避免除零
            self._y_amp = amplitude * 1.15  ## 留 15% 边距

        path = QtGui.QPainterPath()
        for i, value in enumerate(values):
            x = self._index_to_x(i, count)
            y = self._value_to_y(value)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        self._path = path
        self._path_item.setPath(self._path)  ## 仅更新曲线路径
        self._update_axes_if_needed(count, force=force_axes)  ## 坐标轴低频/按需重绘


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
        self._link: Optional[Ks1082Serial | EegCsvReplay] = None  ## EEG 串口或 CSV 模拟
        self._osc_link: Optional[OscillatorSerial] = None  ## 振子串口连接
        self._rhythm = RhythmStreamProcessor(sample_rate=sample_rate)  ## EEG 节律流式滤波
        self._osc_proc = OscStreamProcessor(sample_rate=OSC_SAMPLE_RATE)  ## 振子流式滤波
        self._audio_stopped.connect(self._on_audio_stopped)
        self._audio_controller = StereoAudioController(
            on_stopped=self._audio_stopped.emit,
        )
        self._sleep_aid_tracker = AlphaPhaseTracker(sample_rate=sample_rate)
        try:
            self._sleep_aid_controller = SleepAidStimulusController(
                on_triggered=self._on_sleep_aid_burst,
            )
        except ImportError:
            self._sleep_aid_controller = None
        self._setup_audio_ui_defaults()
        self._setup_timed_test_ui()
        self._setup_compare_segments_ui()
        self._setup_eeg_replay_ui()
        self._sleep_aid_last_warm_sec = -1
        self._running = False  ## 默认不采集
        self._test_duration_sec: Optional[float] = None  ## 实际记录时长 (s)
        self._test_started_at: Optional[float] = None  ## 采集开始时刻 (monotonic)
        self._eeg_raw_record: List[int] = []  ## 定时测试期间记录的 CH1 raw
        self._sample_count = 0  ## EEG 已处理样本数
        self._osc_sample_count = 0  ## 振子已处理样本数
        self._no_data_ticks = 0  ## EEG 无数据 poll 计数
        self._osc_no_data_ticks = 0  ## 振子无数据 poll 计数
        self._last_status_update = 0.0  ## 上次刷新状态栏时刻
        self._display_mode = ""  ## 当前 EEG 显示模式
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

        self._osc_graphics_view = QtWidgets.QGraphicsView(self.ui.page_2)  ## 振子页占位
        self._osc_graphics_view.setGeometry(self.ui.graphicsView.geometry())
        self._osc_graphics_view.hide()

        self._waveform = AlphaWaveformView(  ## EEG：覆盖在 page 内 graphicsView 位置
            sample_rate=sample_rate,
            parent=self.ui.page,
        )
        self._osc_waveform = AlphaWaveformView(  ## 振子：覆盖在 page_2
            sample_rate=OSC_SAMPLE_RATE,
            parent=self.ui.page_2,
        )
        self._sync_waveform_geometry()
        self._sync_osc_waveform_geometry()
        self._waveform.show()
        self._waveform.raise_()
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
            self._running = True  ## 串口/CSV 可用则自动采集，动态刷新 raw 波形
            if isinstance(self._link, Ks1082Serial):
                self._link._parser.reset()
            else:
                self._link.reset_playback()
            self._rhythm.reset()
            self._decim_counter = 0
            self._waveform.clear()
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

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._sync_waveform_geometry()
        self._sync_osc_waveform_geometry()
        self._waveform.show()
        self._waveform.raise_()
        self._osc_waveform.show()
        self._osc_waveform.raise_()
        self._waveform.refresh_layout()
        self._osc_waveform.refresh_layout()

    def _setup_eeg_replay_ui(self) -> None:
        """lineEdit_14：填写 eeg_raw.csv 路径时以文件模拟串口输入。"""
        edit = self.ui.lineEdit_14
        if not edit.text().strip():
            edit.setPlaceholderText("eeg_raw.csv 路径")
        edit.setToolTip(
            "填写已保存的 EEG CSV（如 Result/eeg_xxx/eeg_raw.csv）时，"
            "不读 EEG 串口，按 500 Hz 回放模拟输入"
        )

    def _resolve_eeg_replay_csv_path(self, text: str) -> Optional[Path]:
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

    def _ui_eeg_csv_path(self) -> str:
        return self.ui.lineEdit_14.text().strip()

    def _setup_timed_test_ui(self) -> None:
        """timeEdit / lineEdit_13 测试时长、lcdNumber 倒计时、lineEdit_3 保存路径。"""
        self.ui.timeEdit.setDisplayFormat("HH:mm:ss")
        self.ui.timeEdit.setTime(QtCore.QTime(0, 0, 0))
        self.ui.lineEdit_13.setPlaceholderText("秒(优先)")
        self.ui.lcdNumber.setDigitCount(6)
        self.ui.lcdNumber.display(0)
        if not self.ui.lineEdit_3.text().strip():
            self.ui.lineEdit_3.setPlaceholderText("留空→Result/时间戳文件夹/")

    def _setup_compare_segments_ui(self) -> None:
        """lineEdit/lineEdit_2=段A起止，lineEdit_11/lineEdit_12=段B起止（秒）。"""
        defaults = (
            (self.ui.lineEdit, DEFAULT_SEGMENT_A_START, "A起"),
            (self.ui.lineEdit_2, DEFAULT_SEGMENT_A_END, "A止"),
            (self.ui.lineEdit_11, DEFAULT_SEGMENT_B_START, "B起"),
            (self.ui.lineEdit_12, DEFAULT_SEGMENT_B_END, "B止"),
        )
        for edit, value, hint in defaults:
            if not edit.text().strip():
                edit.setText(value)
            edit.setPlaceholderText(hint)

    def _read_compare_segments_from_ui(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """读取四段输入，对应 power_cal.compare_segments = ((A起,A止), (B起,B止))。"""
        edits = (
            self.ui.lineEdit,
            self.ui.lineEdit_2,
            self.ui.lineEdit_11,
            self.ui.lineEdit_12,
        )
        texts = [edit.text().strip() for edit in edits]
        if not all(texts):
            return None
        try:
            a_start, a_end, b_start, b_end = (float(text) for text in texts)
        except ValueError:
            return None
        if a_start >= a_end or b_start >= b_end:
            return None
        return ((a_start, a_end), (b_start, b_end))

    def _run_post_test_power_analysis(
        self,
        csv_path: Path,
        sample_rate: float,
        reject_rate: float,
    ) -> None:
        """测试结束后自动运行 power_cal.run_analysis（段间功率对比等）。"""
        if reject_rate > EEG_REJECT_RATE_WARN:
            self._log(
                f"拒绝率 {reject_rate:.1%} 超过 {EEG_REJECT_RATE_WARN:.0%}，跳过自动功率分析"
            )
            return
        compare_segments = self._read_compare_segments_from_ui()
        if compare_segments is None:
            self._log("段间对比时间未填全或无效，跳过 power_cal 分析")
            return
        range_a, range_b = compare_segments
        self._log(
            f"开始 power_cal 分析: 段A {range_a[0]:g}–{range_a[1]:g}s, "
            f"段B {range_b[0]:g}–{range_b[1]:g}s"
        )
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                run_analysis(
                    csv_path,
                    sample_rate,
                    show_plot=False,
                    save_plot=True,
                    show_waveform=False,
                    save_waveform=True,
                    waveform_seconds=None,
                    waveform_time_range=None,
                    show_fft=False,
                    save_fft=True,
                    fft_time_range=None,
                    compare_segments=compare_segments,
                    show_segment_compare=False,
                    save_segment_compare=True,
                )
            report = buffer.getvalue().strip()
            if report:
                for line in report.splitlines():
                    self._log(line)
                report_path = csv_path.parent / "eeg_analysis_report.txt"
                report_path.write_text(report + "\n", encoding="utf-8")
                self._log(f"分析报告已保存: {report_path}")
            self._log(f"分析图表已保存至: {csv_path.parent}")
        except Exception as exc:
            self._log(f"power_cal 分析失败: {exc}")

    def _finalize_timed_test(self) -> None:
        """保存 EEG raw CSV 并自动运行段间功率分析。"""
        saved = self._save_eeg_raw_csv()
        self._reset_timed_test_state()
        if saved is not None:
            csv_path, sample_rate, reject_rate = saved
            self._run_post_test_power_analysis(csv_path, sample_rate, reject_rate)

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
        self._eeg_raw_record.clear()
        if duration is None:
            self.ui.lcdNumber.display(0)
            return
        self._update_test_countdown_lcd()
        self._log(
            f"定时测试: 前 {TEST_WARMUP_SEC:.0f}s 不保存，"
            f"之后记录 {self._format_duration_hms(duration)}，"
            f"到时自动停止并保存 EEG raw CSV"
        )

    def _reset_timed_test_state(self) -> None:
        self._test_duration_sec = None
        self._test_started_at = None
        self._eeg_raw_record.clear()
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
        """每次在目录下新建时间戳子文件夹，再写入 CSV。"""
        raw = self.ui.lineEdit_3.text().strip()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir_name = f"eeg_{stamp}"
        csv_name = "eeg_raw.csv"

        if not raw:
            session_dir = DEFAULT_EEG_CSV_DIR / session_dir_name
            session_dir.mkdir(parents=True, exist_ok=True)
            return session_dir / csv_name

        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = _ROOT / path
        if path.suffix.lower() == ".csv":
            session_dir = path.parent / session_dir_name
            session_dir.mkdir(parents=True, exist_ok=True)
            return session_dir / path.name
        session_dir = path / session_dir_name
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / csv_name

    def _save_eeg_raw_csv(self) -> Optional[Tuple[Path, float, float]]:
        if not self._eeg_raw_record:
            self._log("定时测试结束，但没有 EEG raw 数据可保存")
            return None
        path = self._resolve_eeg_csv_path()
        sample_rate = float(MCU_SAMPLE_RATE)
        measured = self._rhythm.measured_sample_rate
        if measured is not None and measured > 0:
            sample_rate = measured
        (
            segment_ids,
            reject_flags,
            reject_reasons,
            reject_rate,
            suspicious_flags,
            suspicious_reasons,
            suspicious_rate,
        ) = _build_eeg_rejection_tags(self._eeg_raw_record, sample_rate)
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "index",
                        "time_s",
                        "ch1_raw",
                        "segment_id",
                        "is_rejected",
                        "reject_reason",
                        "reject_rate",
                        "is_suspicious",
                        "suspicious_reason",
                        "suspicious_rate",
                    ]
                )
                for index, value in enumerate(self._eeg_raw_record):
                    writer.writerow(
                        [
                            index,
                            index / sample_rate,
                            value,
                            segment_ids[index],
                            reject_flags[index],
                            reject_reasons[index],
                            f"{reject_rate:.6f}",
                            suspicious_flags[index],
                            suspicious_reasons[index],
                            f"{suspicious_rate:.6f}",
                        ]
                    )
            self._log(
                f"EEG raw 已保存: {path} "
                f"({len(self._eeg_raw_record)} 点 @ {sample_rate:.0f} Hz)"
            )
            self._log(f"阈值拒绝率: {reject_rate:.1%}")
            self._log(f"可疑片段率: {suspicious_rate:.1%}")
            if reject_rate > EEG_REJECT_RATE_WARN:
                self._log("拒绝率过高，本次数据不建议用于样本分析")
            return path, sample_rate, reject_rate
        except OSError as exc:
            self._log(f"保存 EEG raw CSV 失败: {exc}")
            return None

    def _stop_capture_due_to_timeout(self) -> None:
        """定时到时：停止采集并保存 CSV。"""
        if not self._running:
            return
        self._running = False
        if self._link is not None:
            self._link.discard_pending_input()
        if self._osc_link is not None:
            self._osc_link.discard_pending_input()
        if self._test_duration_sec is not None:
            self._finalize_timed_test()
        else:
            self._reset_timed_test_state()
        self._refresh_capture_button()
        self._update_status_bar()
        self._log("定时测试到时，已自动停止采集")

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
                btn.setText("停止助眠")
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
            self._sleep_aid_controller.stop()
            self._refresh_sleep_aid_button()
            self._log(
                f"助眠音效已停止，共触发 {self._sleep_aid_controller.trigger_count} 次"
            )
            return
        if self._audio_controller.is_playing:
            self._audio_controller.stop()
            self._refresh_audio_button()
            self._log("已停止连续音频，避免与助眠 burst 冲突")
        if not self._running:
            self._log("请先开始 EEG 采集后再开启助眠音效")
            return

        self._sleep_aid_tracker.reset()
        self._rhythm.reset()
        self._sleep_aid_controller.start()
        self._apply_display_mode("alpha")
        self.ui.checkBox.blockSignals(True)
        self.ui.checkBox.setChecked(True)
        self.ui.checkBox.blockSignals(False)
        self._refresh_sleep_aid_button()
        p = self._sleep_aid_controller.params
        self._log(
            "助眠音效已启动: "
            f"暖机 {p.warmup_sec:g}s，最小间隔 {p.min_interval_sec:g}s，"
            f"ERP 延迟 {p.erp_latency_sec * 1000:g} ms，"
            f"粉噪 burst {p.burst_duration_ms:g} ms"
        )

    def _on_sleep_aid_burst(self, count: float) -> None:
        self._log(f"助眠 burst #{int(count)}")

    def _process_sleep_aid_sample(self, alpha: float) -> None:
        if self._sleep_aid_controller is None or not self._sleep_aid_controller.is_active:
            return
        snapshot = self._sleep_aid_tracker.push(alpha)
        self._sleep_aid_controller.process_snapshot(snapshot)
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
            self._sleep_aid_controller.stop()
            self._refresh_sleep_aid_button()
            self._log("已停止助眠音效，避免与连续音频冲突")
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
        self._apply_display_mode(self._current_display_mode())

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
        """勾选节律或 raw 后立即切换显示，各选项互斥。"""
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

    def _sync_osc_waveform_geometry(self) -> None:
        """振子波形控件与 EEG 绘图区同尺寸。"""
        self._osc_graphics_view.setGeometry(self.ui.graphicsView.geometry())
        self._osc_waveform.setGeometry(self.ui.graphicsView.geometry())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._sync_waveform_geometry()
        self._sync_osc_waveform_geometry()
        self._waveform.refresh_layout()
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
        wheel_hint = "Y/X轴滚轮:缩放  Shift+滚轮:平移"
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
        wheel_hint = "Y/X轴滚轮:缩放  Shift+滚轮:平移"
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
        wheel_hint = "Y/X轴滚轮:缩放  Shift+滚轮:平移"
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
        """打开 EEG 数据源：lineEdit_14 有 CSV 则回放文件，否则读串口。"""
        csv_path = self._resolve_eeg_replay_csv_path(self._ui_eeg_csv_path())
        if csv_path is not None:
            try:
                if (
                    isinstance(self._link, EegCsvReplay)
                    and self._link.is_open
                    and Path(self._link.csv_path) == csv_path
                ):
                    return True
                if self._link is not None:
                    self._link.close()
                replay = EegCsvReplay(csv_path)
                replay.open()
                self._link = replay
                self._port = f"CSV:{csv_path.name}"
                self._log(
                    f"EEG CSV 模拟: {csv_path} "
                    f"({replay.sample_count} 点 @ {replay.sample_rate:.0f} Hz)"
                )
                self._log(f"EEG 采样率 {replay.sample_rate:.0f} Hz，显示 {RAW_DISPLAY_RATE} Hz")
                return True
            except Exception as exc:
                self._link = None
                self._log(f"EEG CSV 加载失败 ({csv_path}): {exc}")
                return False

        port = self._ui_serial_port()
        try:
            if (
                isinstance(self._link, Ks1082Serial)
                and self._link.is_open
                and self._link.port == port
            ):
                return True
            if self._link is not None:
                self._link.close()
            self._link = Ks1082Serial(port, baudrate=self._baudrate, timeout=0.05)
            self._link.open()
            self._port = port
            self._log(f"EEG 串口已连接: {self._port} @ {self._baudrate}")
            self._log(f"EEG 采样率 {self._rhythm.sample_rate:.0f} Hz，显示 {RAW_DISPLAY_RATE} Hz")
            return True
        except Exception as exc:
            self._link = None
            self._log(f"EEG 串口打开失败 ({port}): {exc}")
            return False

    def _open_osc_serial(self) -> bool:
        """按 UI 串口框打开振子串口；成功返回 True。"""
        port = self._ui_osc_serial_port()
        try:
            if (
                self._osc_link is not None
                and self._osc_link.is_open
                and self._osc_link.port == port
            ):
                return True
            if self._osc_link is not None:
                self._osc_link.close()
            self._osc_link = OscillatorSerial(port, baudrate=self._baudrate, timeout=0.05)
            self._osc_link.open()
            self._osc_port = port
            self._log(f"振子串口已连接: {self._osc_port} @ {self._baudrate}")
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
            return True
        except Exception as exc:
            self._osc_link = None
            self._log(f"振子串口打开失败 ({port}): {exc}")
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
            mode = self._current_display_mode()
            rx = self._rhythm.measured_sample_rate
            rx_text = f"{rx:.0f}" if rx is not None else "—"
            if mode == "raw":
                src = "CSV" if isinstance(self._link, EegCsvReplay) else "串口"
                self.ui.statusbar.showMessage(
                    f"RAW CH1 ({src})  |  显示≈{RAW_DISPLAY_RATE} Hz  |  接收≈{rx_text} Hz"
                )
            else:
                label = BAND_LABELS[mode]
                low_hz, high_hz = EEG_BANDS[mode]
                src = "CSV" if isinstance(self._link, EegCsvReplay) else "串口"
                self.ui.statusbar.showMessage(
                    f"{label} {low_hz:g}-{high_hz:g} Hz ({src})  |  "
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
        elif stopping and self._test_duration_sec is not None:
            self._finalize_timed_test()
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
            mode = self._current_display_mode()
            self._apply_display_mode(mode)
            plot_values: list[float] = []
            band = None if mode == "raw" else mode
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
                if sleep_aid:
                    self._process_sleep_aid_sample(value)
                if self._decim_counter >= RAW_DECIM_FACTOR:
                    self._decim_counter = 0
                    plot_values.append(value)

            if plot_values:
                self._waveform.append_alphas(plot_values)
            self._no_data_ticks = 0
        elif got_bytes == 0:
            self._no_data_ticks += 1
        else:
            self._no_data_ticks += 1

        if self._no_data_ticks == 50:
            if isinstance(self._link, EegCsvReplay):
                if self._link.finished:
                    self._log("EEG CSV 已播放完毕")
                else:
                    self._log("EEG CSV 回放等待中（请确认采集已启动）")
            else:
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
    window.show()  ## 显示窗口
    sys.exit(app.exec_())  ## 进入事件循环


if __name__ == "__main__":
    run_app()  ## 直接运行本文件时启动
