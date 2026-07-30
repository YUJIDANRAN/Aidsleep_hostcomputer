"""EEG 运动伪迹 / 质量段检测：阈值拒绝与可疑段标记。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Tuple
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# ===== EEG 阈值拒绝/可疑段参数 =====
# 检测按固定时长切片，目前每 1 秒判断一次；调小会更精细，但也更容易被瞬时噪声影响。
EEG_REJECT_SEGMENT_SEC = 1

# ADC 贴边饱和坏段阈值：原始值 <= 50 或 >= 4045 时标红。
# 适合 12 bit ADC (0-4095)。如果硬件有效范围变窄/变宽，再同步调整。
EEG_RAW_MIN_VALID = 50
EEG_RAW_MAX_VALID = 4045

# 原始 1 秒片段峰峰值坏段阈值：max(segment)-min(segment) 超过该值标红。
# 主要抓大幅运动伪迹、严重电极扰动、接触不良；调小会更严格。
EEG_SEGMENT_MAX_PTP = 500.0

# 原始 1 秒片段偏离中位数坏段阈值：单点离该秒中位数太远时标红。
# 主要抓特别大的尖峰/跳变；调小会更严格。
EEG_SEGMENT_MAX_DEVIATION = 250.0

# 自适应可疑段倍率：阈值 = 全部 1 秒片段的 median + N*MAD。
# 同时作用于 EEG_SUSPICIOUS_MIN_PTP 和 EEG_SUSPICIOUS_MIN_DIFF 对应规则。
# 调小更敏感、黄段更多；调大更保守、黄段更少。
EEG_ADAPTIVE_MAD_MULT = 5.0

# 原始 1 秒峰峰值可疑段最低阈值。
# 实际阈值取 max(该值, median + EEG_ADAPTIVE_MAD_MULT*MAD)。
# 主要抓比全局背景明显更“抖”的片段；调小会更容易标黄。
EEG_SUSPICIOUS_MIN_PTP = 200.0

# 原始 1 秒内最大相邻跳变可疑段最低阈值。
# 实际阈值取 max(该值, median + EEG_ADAPTIVE_MAD_MULT*MAD)。
# 主要抓短促尖峰、突然跳点；调小会更容易标黄。
EEG_SUSPICIOUS_MIN_DIFF = 60.0

# δ(0.5-4 Hz)滤波后 RMS 可疑段 MAD 倍率。
# 阈值候选 = delta_rms_median + N*delta_rms_MAD。
# 主要抓低频慢漂/电极扰动/运动造成的 δ 行突刺；只标黄，不标红。
# 调小更容易抓低频异常；调大更保守。
EEG_SUSPICIOUS_DELTA_RMS_MAD_MULT = 6.0

# δ RMS 相对背景的最低倍数门槛。
# 最终 δ RMS 阈值 = max(median*该倍数, median + MAD_MULT*MAD)。
# 防止 MAD 很小时阈值过低；调小更敏感，调大更保守。
EEG_SUSPICIOUS_DELTA_RMS_RATIO = 2.0

# 多节律同步尖峰（头动伪迹）：各窄带波形在同一 1 s 片内 ptp 同时偏高。
# 阈值/频段 = max(PTP_FLOOR, median×RATIO, median + MAD_MULT×MAD)，按各频段自适应。
EEG_MULTIBAND_PTP_FLOOR = 15.0
EEG_MULTIBAND_PTP_RATIO = 2.0
EEG_MULTIBAND_PTP_MAD_MULT = 5.0
EEG_MULTIBAND_SYNC_MIN_BANDS = 3  # 超阈频段数 → suspicious
EEG_MULTIBAND_SYNC_REJECT_MIN_BANDS = 4  # 超阈频段数 → rejected

# 剔坏后拼接：接头两侧局部中位对齐（方案A），窗长过短易被尖峰带偏。
EEG_SPLICE_ALIGN_WINDOW_SEC = 0.25
EEG_SPLICE_ALIGN_DEFAULT_FS = 500.0

BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0

# Alpha(8-13 Hz) RMS adaptive suspicious segment detection.
# This is intentionally a yellow/tag-only rule: strong alpha may be a real
# relaxed/eyes-closed state, so it should not directly reject data.
EEG_SUSPICIOUS_ALPHA_RMS_MAD_MULT = 6.0
EEG_SUSPICIOUS_ALPHA_RMS_RATIO = 2.0
EEG_SUSPICIOUS_ALPHA_RMS_FLOOR = 1.0

# Window-level model quality policy. A model can use this table to keep,
# downweight, or skip windows without pretending repaired EEG is real EEG.
MODEL_WINDOW_SEC = 5.0
MODEL_ALPHA_SUSPICIOUS_WARN_RATIO = 0.10
MODEL_ALPHA_SUSPICIOUS_DROP_RATIO = 0.30
MODEL_REJECT_DROP_RATIO = 0.20
MODEL_SUSPICIOUS_WARN_RATIO = 0.20
MODEL_SUSPICIOUS_DROP_RATIO = 0.50

# ===== 在线 α 实时幅值阈值拒绝（因果、逐点） =====
REALTIME_ALPHA_REJECT_HISTORY_SEC = 30.0
REALTIME_ALPHA_REJECT_MIN_HISTORY_SEC = 1.0
REALTIME_ALPHA_REJECT_MAD_MULT = 6.0
REALTIME_ALPHA_REJECT_RATIO = 2.0
REALTIME_ALPHA_REJECT_ABS_FLOOR = 20.0

_FILTER_ORDER = 4

# 与 power_cal.EEG_BANDS 一致的节律划分（供带通 RMS 检测使用）。
EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


@dataclass
class EegQualityInfo:
    reject_mask: np.ndarray
    reject_rate: float
    has_tag: bool = False
    source: str = "none"
    suspicious_mask: np.ndarray | None = None
    suspicious_rate: float = 0.0
    reject_reasons: list[str] | None = None
    suspicious_reasons: list[str] | None = None

    @property
    def has_rejection(self) -> bool:
        return bool(self.has_tag and self.reject_mask.size and np.any(self.reject_mask))

    @property
    def has_suspicious(self) -> bool:
        return bool(
            self.has_tag
            and self.suspicious_mask is not None
            and self.suspicious_mask.size
            and np.any(self.suspicious_mask)
        )


@dataclass(frozen=True)
class BandSuspiciousInfo:
    band_name: str
    mask: np.ndarray
    rate: float
    rms_threshold: float
    rms_median: float
    rms_mad: float
    segment_sec: float
    reasons: list[str]


class RealtimeAlphaThresholdRejector:
    """在线 α 幅值实时标记：基于滚动历史的自适应阈值，仅用于显示拒绝标记。"""

    def __init__(self, sample_rate: float) -> None:
        self._sample_rate = max(float(sample_rate), 1.0)
        self._history_max = max(
            1, int(round(self._sample_rate * REALTIME_ALPHA_REJECT_HISTORY_SEC))
        )
        self._min_history = max(
            1, int(round(self._sample_rate * REALTIME_ALPHA_REJECT_MIN_HISTORY_SEC))
        )
        self._update_interval = max(1, int(round(self._sample_rate * 0.1)))
        self._history: Deque[float] = deque(maxlen=self._history_max)
        self._threshold = REALTIME_ALPHA_REJECT_ABS_FLOOR
        self._samples_since_update = self._update_interval

    @property
    def threshold(self) -> float:
        return self._threshold

    def reset(self) -> None:
        self._history.clear()
        self._threshold = REALTIME_ALPHA_REJECT_ABS_FLOOR
        self._samples_since_update = self._update_interval

    def push(self, alpha: float) -> bool:
        abs_value = abs(float(alpha))
        if (
            len(self._history) >= self._min_history
            and self._samples_since_update >= self._update_interval
        ):
            values = np.fromiter(self._history, dtype=np.float64)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            self._threshold = max(
                REALTIME_ALPHA_REJECT_ABS_FLOOR,
                median * REALTIME_ALPHA_REJECT_RATIO,
                median + REALTIME_ALPHA_REJECT_MAD_MULT * mad,
            )
            self._samples_since_update = 0

        rejected = len(self._history) >= self._min_history and abs_value > self._threshold
        self._history.append(abs_value)
        self._samples_since_update += 1
        return rejected


# ===== 在线 raw 快检 QualityGate（助眠 burst 门控，方案 A） =====
REALTIME_QUALITY_WINDOW_SEC = 0.2
REALTIME_QUALITY_TIMELINE_SEC = 2.0
REALTIME_QUALITY_BAD_COOLDOWN_SEC = 0.4
REALTIME_QUALITY_LOOKAHEAD_SEC = 0.15
REALTIME_QUALITY_MIN_ADAPTIVE_SEC = 0.5
REALTIME_QUALITY_PTP_FLOOR = 400.0
REALTIME_QUALITY_DIFF_FLOOR = 80.0
REALTIME_QUALITY_DEVIATION_FLOOR = 450.0


@dataclass(frozen=True)
class QualitySampleResult:
    """单点 raw 快检结果。"""

    is_bad: bool
    reason: str = ""


class RealtimeQualityGate:
    """
    因果 raw 快检：饱和 / 跳变 / 短窗 ptp / 偏离中位数。
    维护 Bad 时间线，供助眠 burst 发射前窗复核与坏段内暂停相位检测。
    """

    def __init__(self, sample_rate: float) -> None:
        self._sample_rate = max(float(sample_rate), 1.0)
        self._window_samples = max(
            10, int(round(self._sample_rate * REALTIME_QUALITY_WINDOW_SEC))
        )
        self._timeline_max = max(
            self._window_samples,
            int(round(self._sample_rate * REALTIME_QUALITY_TIMELINE_SEC)),
        )
        self._min_adaptive = max(
            10, int(round(self._sample_rate * REALTIME_QUALITY_MIN_ADAPTIVE_SEC))
        )
        self._history_max = max(
            self._min_adaptive, int(round(self._sample_rate * 10.0))
        )
        self._raw_window: Deque[int] = deque(maxlen=self._window_samples)
        self._bad_timeline: Deque[bool] = deque(maxlen=self._timeline_max)
        self._ptp_history: Deque[float] = deque(maxlen=self._history_max)
        self._diff_history: Deque[float] = deque(maxlen=self._history_max)
        self._prev_raw: int | None = None
        self._in_bad_segment = False
        self._cooldown_until = 0.0
        self._skip_count = 0
        self._skip_reasons: Dict[str, int] = {}

    @property
    def in_bad_segment(self) -> bool:
        return self._in_bad_segment

    @property
    def skip_count(self) -> int:
        return self._skip_count

    @property
    def skip_reasons(self) -> Dict[str, int]:
        return dict(self._skip_reasons)

    def reset(self) -> None:
        self._raw_window.clear()
        self._bad_timeline.clear()
        self._ptp_history.clear()
        self._diff_history.clear()
        self._prev_raw = None
        self._in_bad_segment = False
        self._cooldown_until = 0.0
        self._skip_count = 0
        self._skip_reasons.clear()

    def in_cooldown(self, now: float | None = None) -> bool:
        t = time.monotonic() if now is None else now
        return t < self._cooldown_until

    def record_skip(self, reason: str) -> None:
        self._skip_count += 1
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1

    @staticmethod
    def _adaptive_threshold(history: Deque[float], floor: float) -> float:
        if len(history) < 10:
            return floor * 1.5
        arr = np.fromiter(history, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        return max(floor, median + EEG_ADAPTIVE_MAD_MULT * mad)

    def push(self, raw: int, now: float | None = None) -> QualitySampleResult:
        raw_i = int(raw)
        t = time.monotonic() if now is None else now
        reasons: list[str] = []

        if raw_i <= EEG_RAW_MIN_VALID or raw_i >= EEG_RAW_MAX_VALID:
            reasons.append("rail_saturation")

        diff = 0.0
        if self._prev_raw is not None:
            diff = abs(float(raw_i - self._prev_raw))
            diff_threshold = self._adaptive_threshold(
                self._diff_history, REALTIME_QUALITY_DIFF_FLOOR
            )
            if diff > diff_threshold:
                reasons.append(f"diff>{diff_threshold:.0f}")
        self._prev_raw = raw_i

        self._raw_window.append(raw_i)
        ptp = 0.0
        if len(self._raw_window) >= max(4, self._window_samples // 2):
            arr = np.fromiter(self._raw_window, dtype=np.float64)
            ptp = float(np.ptp(arr))
            ptp_threshold = self._adaptive_threshold(
                self._ptp_history, REALTIME_QUALITY_PTP_FLOOR
            )
            if ptp > ptp_threshold:
                reasons.append(f"ptp>{ptp_threshold:.0f}")
            median = float(np.median(arr))
            max_dev = float(np.max(np.abs(arr - median)))
            if max_dev > REALTIME_QUALITY_DEVIATION_FLOOR:
                reasons.append(f"deviation>{REALTIME_QUALITY_DEVIATION_FLOOR:.0f}")

        is_bad = bool(reasons)
        self._bad_timeline.append(is_bad)

        if (
            not is_bad
            and len(self._raw_window) >= max(4, self._window_samples // 2)
            and len(self._ptp_history) >= self._min_adaptive
        ):
            self._ptp_history.append(ptp)
            if diff > 0.0:
                self._diff_history.append(diff)

        was_bad = self._in_bad_segment
        self._in_bad_segment = is_bad
        if was_bad and not is_bad:
            self._cooldown_until = t + REALTIME_QUALITY_BAD_COOLDOWN_SEC

        return QualitySampleResult(is_bad=is_bad, reason=";".join(reasons))

    def window_has_bad(self, duration_sec: float) -> bool:
        n = min(
            len(self._bad_timeline),
            max(1, int(round(self._sample_rate * duration_sec))),
        )
        if n <= 0:
            return False
        return any(list(self._bad_timeline)[-n:])

    def is_stimulus_window_clean(
        self,
        guard_sec: float,
        now: float | None = None,
    ) -> bool:
        if self._in_bad_segment:
            return False
        if self.in_cooldown(now):
            return False
        check_sec = max(float(guard_sec), REALTIME_QUALITY_LOOKAHEAD_SEC)
        return not self.window_has_bad(check_sec)


def _bandpass_filter(
    signal: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = _FILTER_ORDER,
) -> np.ndarray:
    if low_hz <= 0:
        raise ValueError(f"低截止频率须 > 0，当前为 {low_hz}")
    if sample_rate <= 2 * high_hz:
        raise ValueError(
            f"采样率 {sample_rate} Hz 过低，无法设计 {high_hz} Hz 高通上限"
        )
    nyquist = sample_rate * 0.5
    sos = butter(
        order,
        (low_hz / nyquist, high_hz / nyquist),
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal, axis=0)


def _mad_threshold(
    metrics: np.ndarray,
    *,
    ratio: float,
    mad_mult: float,
    floor: float = 0.0,
) -> tuple[float, float, float]:
    if metrics.size == 0:
        return floor, 0.0, 0.0
    median = float(np.median(metrics))
    mad = float(np.median(np.abs(metrics - median)))
    threshold = max(floor, median * ratio, median + mad_mult * mad)
    return threshold, median, mad


def _extract_band_waveforms(
    signal: np.ndarray,
    sample_rate: float,
) -> Dict[str, np.ndarray]:
    """先 0.5–40 Hz 带通，再各节律窄带（与 power_cal.extract_band_waveforms 一致）。"""
    base = _bandpass_filter(
        signal.astype(np.float64, copy=False),
        sample_rate,
        BANDPASS_LOW_HZ,
        BANDPASS_HIGH_HZ,
    )
    return {
        name: _bandpass_filter(base, sample_rate, low_hz=low, high_hz=high)
        for name, (low, high) in EEG_BANDS.items()
    }


def _multiband_ptp_context(
    values: np.ndarray,
    sample_rate: float,
    segments: list[tuple[int, int, np.ndarray]],
) -> tuple[list[dict[str, float]] | None, dict[str, float]]:
    """各 1 s 片段、各节律窄带波形的片内 ptp 及自适应阈值。"""
    if not segments:
        return None, {}
    try:
        band_waveforms = _extract_band_waveforms(values, sample_rate)
    except ValueError:
        return None, {}

    band_segment_ptp: dict[str, list[float]] = {name: [] for name in EEG_BANDS}
    for start, end, _segment in segments:
        for name, waveform in band_waveforms.items():
            piece = waveform[start:end]
            band_segment_ptp[name].append(float(np.ptp(piece)) if piece.size else 0.0)

    thresholds: dict[str, float] = {}
    for name, ptp_values in band_segment_ptp.items():
        threshold, _, _ = _mad_threshold(
            np.asarray(ptp_values, dtype=np.float64),
            ratio=EEG_MULTIBAND_PTP_RATIO,
            mad_mult=EEG_MULTIBAND_PTP_MAD_MULT,
            floor=EEG_MULTIBAND_PTP_FLOOR,
        )
        thresholds[name] = threshold

    ptp_by_segment = [
        {name: band_segment_ptp[name][idx] for name in EEG_BANDS}
        for idx in range(len(segments))
    ]
    return ptp_by_segment, thresholds


def build_threshold_rejection(
    values: np.ndarray,
    sample_rate: float,
) -> EegQualityInfo:
    """按 1 s 片段打质量标签：严重坏段 rejected，可疑段 suspicious。

    含 raw 规则与多节律窄带片内 ptp 同步尖峰（multiband_ptp）规则。
    """
    count = values.size
    if count == 0:
        empty = np.zeros(0, dtype=bool)
        return EegQualityInfo(empty, 0.0, has_tag=False, suspicious_mask=empty)

    reject_mask = np.zeros(count, dtype=bool)
    suspicious_mask = np.zeros(count, dtype=bool)
    reject_reasons = [""] * count
    suspicious_reasons = [""] * count
    segment_size = max(1, int(round(sample_rate * EEG_REJECT_SEGMENT_SEC)))
    segments: list[tuple[int, int, np.ndarray]] = []
    ptp_values: list[float] = []
    diff_values: list[float] = []
    delta_rms_values: list[float] = []

    for start in range(0, count, segment_size):
        end = min(count, start + segment_size)
        segment = values[start:end].astype(np.float64, copy=False)
        segments.append((start, end, segment))
        ptp_values.append(float(np.ptp(segment)))
        diffs = np.diff(segment)
        diff_values.append(float(np.max(np.abs(diffs))) if diffs.size else 0.0)

    try:
        delta_signal = _bandpass_filter(
            values.astype(np.float64, copy=False),
            sample_rate,
            0.5,
            4.0,
        )
    except ValueError:
        delta_signal = None

    for start, end, _segment in segments:
        if delta_signal is None:
            delta_rms_values.append(0.0)
            continue
        delta_segment = delta_signal[start:end]
        delta_rms_values.append(float(np.sqrt(np.mean(delta_segment * delta_segment))))

    def _adaptive_threshold(metrics: list[float], floor: float) -> float:
        if not metrics:
            return floor
        arr = np.asarray(metrics, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        return max(floor, median + EEG_ADAPTIVE_MAD_MULT * mad)

    ptp_suspicious_threshold = _adaptive_threshold(
        ptp_values,
        EEG_SUSPICIOUS_MIN_PTP,
    )
    diff_suspicious_threshold = _adaptive_threshold(
        diff_values,
        EEG_SUSPICIOUS_MIN_DIFF,
    )
    if delta_rms_values:
        delta_rms_arr = np.asarray(delta_rms_values, dtype=np.float64)
        delta_rms_median = float(np.median(delta_rms_arr))
        delta_rms_mad = float(np.median(np.abs(delta_rms_arr - delta_rms_median)))
        delta_rms_suspicious_threshold = max(
            delta_rms_median * EEG_SUSPICIOUS_DELTA_RMS_RATIO,
            delta_rms_median + EEG_SUSPICIOUS_DELTA_RMS_MAD_MULT * delta_rms_mad,
        )
    else:
        delta_rms_suspicious_threshold = float("inf")

    multiband_ptp_by_segment, multiband_ptp_thresholds = _multiband_ptp_context(
        values.astype(np.float64, copy=False),
        sample_rate,
        segments,
    )
    n_rhythm_bands = len(EEG_BANDS)

    for idx, (start, end, segment) in enumerate(segments):
        reject_reason_parts: list[str] = []
        suspicious_reason_parts: list[str] = []
        if np.any(segment <= EEG_RAW_MIN_VALID) or np.any(segment >= EEG_RAW_MAX_VALID):
            reject_reason_parts.append("rail_saturation")
        ptp = ptp_values[idx]
        if ptp > EEG_SEGMENT_MAX_PTP:
            reject_reason_parts.append(f"ptp>{EEG_SEGMENT_MAX_PTP:g}")
        median = float(np.median(segment))
        max_dev = float(np.max(np.abs(segment - median)))
        if max_dev > EEG_SEGMENT_MAX_DEVIATION:
            reject_reason_parts.append(f"deviation>{EEG_SEGMENT_MAX_DEVIATION:g}")

        max_diff = diff_values[idx]
        if ptp > ptp_suspicious_threshold:
            suspicious_reason_parts.append(
                f"adaptive_ptp>{ptp_suspicious_threshold:g}"
            )
        if max_diff > diff_suspicious_threshold:
            suspicious_reason_parts.append(
                f"adaptive_diff>{diff_suspicious_threshold:g}"
            )
        delta_rms = delta_rms_values[idx]
        if delta_rms > delta_rms_suspicious_threshold:
            suspicious_reason_parts.append(
                f"delta_rms>{delta_rms_suspicious_threshold:g}"
            )

        if multiband_ptp_by_segment is not None:
            segment_ptp = multiband_ptp_by_segment[idx]
            spiking_bands = [
                name
                for name in EEG_BANDS
                if segment_ptp[name] > multiband_ptp_thresholds[name]
            ]
            n_spiking = len(spiking_bands)
            if n_spiking >= EEG_MULTIBAND_SYNC_REJECT_MIN_BANDS:
                band_list = ",".join(spiking_bands)
                reject_reason_parts.append(
                    f"multiband_ptp({n_spiking}/{n_rhythm_bands}:{band_list})"
                )
            elif n_spiking >= EEG_MULTIBAND_SYNC_MIN_BANDS:
                band_list = ",".join(spiking_bands)
                suspicious_reason_parts.append(
                    f"multiband_ptp({n_spiking}/{n_rhythm_bands}:{band_list})"
                )

        if reject_reason_parts:
            reject_mask[start:end] = True
            reason = ";".join(reject_reason_parts)
            for point_idx in range(start, end):
                reject_reasons[point_idx] = reason
        elif suspicious_reason_parts:
            suspicious_mask[start:end] = True
            reason = ";".join(suspicious_reason_parts)
            for point_idx in range(start, end):
                suspicious_reasons[point_idx] = reason

    reject_rate = float(np.mean(reject_mask)) if reject_mask.size else 0.0
    suspicious_rate = (
        float(np.mean(suspicious_mask)) if suspicious_mask.size else 0.0
    )
    return EegQualityInfo(
        reject_mask,
        reject_rate,
        has_tag=True,
        source="computed",
        suspicious_mask=suspicious_mask,
        suspicious_rate=suspicious_rate,
        reject_reasons=reject_reasons,
        suspicious_reasons=suspicious_reasons,
    )


def build_band_rms_suspicious(
    values: np.ndarray,
    sample_rate: float,
    *,
    band_name: str = "alpha",
    mad_mult: float = EEG_SUSPICIOUS_ALPHA_RMS_MAD_MULT,
    ratio: float = EEG_SUSPICIOUS_ALPHA_RMS_RATIO,
    floor: float = EEG_SUSPICIOUS_ALPHA_RMS_FLOOR,
) -> BandSuspiciousInfo:
    """Mark high-RMS band segments as suspicious without rejecting them."""
    count = values.size
    empty = np.zeros(count, dtype=bool)
    empty_reasons = [""] * count
    if count == 0:
        return BandSuspiciousInfo(
            band_name,
            empty,
            0.0,
            floor,
            0.0,
            0.0,
            EEG_REJECT_SEGMENT_SEC,
            empty_reasons,
        )
    if band_name not in EEG_BANDS:
        raise ValueError(f"未知节律: {band_name!r}")

    low, high = EEG_BANDS[band_name]
    try:
        band_signal = _bandpass_filter(
            values.astype(np.float64, copy=False),
            sample_rate,
            low,
            high,
        )
    except ValueError:
        return BandSuspiciousInfo(
            band_name,
            empty,
            0.0,
            float("inf"),
            0.0,
            0.0,
            EEG_REJECT_SEGMENT_SEC,
            empty_reasons,
        )

    segment_size = max(1, int(round(sample_rate * EEG_REJECT_SEGMENT_SEC)))
    segments: list[tuple[int, int]] = []
    rms_values: list[float] = []
    for start in range(0, count, segment_size):
        end = min(count, start + segment_size)
        segment = band_signal[start:end]
        segments.append((start, end))
        rms_values.append(float(np.sqrt(np.mean(segment * segment))))

    rms_arr = np.asarray(rms_values, dtype=np.float64)
    threshold, median, mad = _mad_threshold(
        rms_arr,
        ratio=ratio,
        mad_mult=mad_mult,
        floor=floor,
    )
    mask = np.zeros(count, dtype=bool)
    reasons = [""] * count
    for idx, (start, end) in enumerate(segments):
        rms = rms_values[idx]
        if rms > threshold:
            mask[start:end] = True
            reason = f"{band_name}_rms>{threshold:g}"
            for point_idx in range(start, end):
                reasons[point_idx] = reason

    rate = float(np.mean(mask)) if mask.size else 0.0
    return BandSuspiciousInfo(
        band_name,
        mask,
        rate,
        threshold,
        median,
        mad,
        EEG_REJECT_SEGMENT_SEC,
        reasons,
    )


def merge_quality_with_band_suspicious(
    quality: EegQualityInfo,
    band_info: BandSuspiciousInfo,
) -> EegQualityInfo:
    """Add band-level suspicious tags to the general yellow mask."""
    count = quality.reject_mask.size
    band_mask = band_info.mask
    if band_mask.size < count:
        band_mask = np.pad(band_mask, (0, count - band_mask.size), constant_values=False)
    elif band_mask.size > count:
        band_mask = band_mask[:count]

    base_suspicious = (
        quality.suspicious_mask.copy()
        if quality.suspicious_mask is not None
        else np.zeros(count, dtype=bool)
    )
    band_mask = band_mask & ~quality.reject_mask
    suspicious_mask = base_suspicious | band_mask
    suspicious_rate = float(np.mean(suspicious_mask)) if suspicious_mask.size else 0.0

    suspicious_reasons = (
        list(quality.suspicious_reasons)
        if quality.suspicious_reasons is not None
        else [""] * count
    )
    if len(suspicious_reasons) < count:
        suspicious_reasons.extend([""] * (count - len(suspicious_reasons)))
    elif len(suspicious_reasons) > count:
        suspicious_reasons = suspicious_reasons[:count]

    for idx, is_band_suspicious in enumerate(band_mask):
        if not is_band_suspicious:
            continue
        reason = band_info.reasons[idx] if idx < len(band_info.reasons) else ""
        if not reason:
            reason = f"{band_info.band_name}_rms"
        if suspicious_reasons[idx]:
            suspicious_reasons[idx] = f"{suspicious_reasons[idx]};{reason}"
        else:
            suspicious_reasons[idx] = reason

    return EegQualityInfo(
        quality.reject_mask,
        quality.reject_rate,
        has_tag=quality.has_tag or band_info.mask.size > 0,
        source=quality.source,
        suspicious_mask=suspicious_mask,
        suspicious_rate=suspicious_rate,
        reject_reasons=quality.reject_reasons,
        suspicious_reasons=suspicious_reasons,
    )


def read_eeg_quality(
    data_path: str | Path,
    expected_len: int,
    raw_values: np.ndarray | None = None,
    sample_rate: float = 500.0,
) -> EegQualityInfo:
    """读取 CSV 中的阈值拒绝 tag；旧文件没有 tag 时返回 0 拒绝率。"""
    path = Path(data_path)
    empty_mask = np.zeros(expected_len, dtype=bool)
    empty = EegQualityInfo(
        empty_mask,
        0.0,
        has_tag=False,
        suspicious_mask=empty_mask.copy(),
    )
    if path.suffix.lower() not in {".csv", ".txt"}:
        return empty

    frame = pd.read_csv(path)
    if "is_rejected" not in frame.columns:
        if raw_values is not None:
            return build_threshold_rejection(raw_values, sample_rate)
        return empty

    computed = (
        build_threshold_rejection(raw_values, sample_rate)
        if raw_values is not None
        else None
    )
    if computed is not None:
        return computed

    def _fit_mask(mask: np.ndarray) -> np.ndarray:
        if mask.size < expected_len:
            return np.pad(mask, (0, expected_len - mask.size), constant_values=False)
        if mask.size > expected_len:
            return mask[:expected_len]
        return mask

    flags = pd.to_numeric(frame["is_rejected"], errors="coerce").fillna(0)
    mask = _fit_mask(flags.to_numpy(dtype=np.float64) > 0)

    reject_rate = float(np.mean(mask)) if mask.size else 0.0
    if "reject_rate" in frame.columns:
        rates = pd.to_numeric(frame["reject_rate"], errors="coerce").dropna()
        if not rates.empty:
            reject_rate = float(rates.iloc[0])

    if "is_suspicious" in frame.columns:
        suspicious_flags = pd.to_numeric(
            frame["is_suspicious"], errors="coerce"
        ).fillna(0)
        suspicious_mask = _fit_mask(
            suspicious_flags.to_numpy(dtype=np.float64) > 0
        )
        suspicious_mask = suspicious_mask & ~mask
        suspicious_rate = (
            float(np.mean(suspicious_mask)) if suspicious_mask.size else 0.0
        )
        if "suspicious_rate" in frame.columns:
            rates = pd.to_numeric(
                frame["suspicious_rate"], errors="coerce"
            ).dropna()
            if not rates.empty:
                suspicious_rate = float(rates.iloc[0])
        suspicious_reasons = (
            frame["suspicious_reason"].fillna("").astype(str).to_list()
            if "suspicious_reason" in frame.columns
            else None
        )
    elif computed is not None and computed.suspicious_mask is not None:
        suspicious_mask = computed.suspicious_mask & ~mask
        suspicious_rate = (
            float(np.mean(suspicious_mask)) if suspicious_mask.size else 0.0
        )
        suspicious_reasons = computed.suspicious_reasons
    else:
        suspicious_mask = np.zeros(expected_len, dtype=bool)
        suspicious_rate = 0.0
        suspicious_reasons = None

    reject_reasons = (
        frame["reject_reason"].fillna("").astype(str).to_list()
        if "reject_reason" in frame.columns
        else None
    )
    return EegQualityInfo(
        mask,
        reject_rate,
        has_tag=True,
        source="csv",
        suspicious_mask=suspicious_mask,
        suspicious_rate=suspicious_rate,
        reject_reasons=reject_reasons,
        suspicious_reasons=suspicious_reasons,
    )


def rejected_spans(
    reject_mask: np.ndarray,
    sample_rate: float,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """把逐点 reject mask 合并成时间段，供波形图背景标注。"""
    if reject_mask.size == 0 or not np.any(reject_mask):
        return []

    total_duration = reject_mask.size / sample_rate
    view_start = max(0.0, start_seconds)
    view_end = total_duration if end_seconds is None else min(end_seconds, total_duration)
    if view_start >= view_end:
        return []

    i_start = max(0, int(view_start * sample_rate))
    i_end = min(reject_mask.size, int(np.ceil(view_end * sample_rate)))
    local = reject_mask[i_start:i_end]
    spans: list[tuple[float, float]] = []
    in_span = False
    span_start = 0
    for offset, rejected in enumerate(local):
        if rejected and not in_span:
            in_span = True
            span_start = offset
        elif not rejected and in_span:
            abs_start = i_start + span_start
            abs_end = i_start + offset
            spans.append((abs_start / sample_rate, abs_end / sample_rate))
            in_span = False
    if in_span:
        abs_start = i_start + span_start
        abs_end = i_start + local.size
        spans.append((abs_start / sample_rate, abs_end / sample_rate))
    return spans


def fit_bool_mask(mask: np.ndarray, target_len: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size < target_len:
        return np.pad(mask, (0, target_len - mask.size), constant_values=False)
    if mask.size > target_len:
        return mask[:target_len]
    return mask


def build_raw_remove_mask(
    quality: EegQualityInfo,
    count: int,
    *,
    remove_suspicious: bool = True,
) -> np.ndarray:
    """合并 reject / suspicious 得到应从 raw 序列中剔除的逐点 mask。"""
    reject_mask = fit_bool_mask(quality.reject_mask, count)
    if not remove_suspicious:
        return reject_mask
    suspicious_mask = (
        fit_bool_mask(quality.suspicious_mask, count)
        if quality.suspicious_mask is not None
        else np.zeros(count, dtype=bool)
    )
    return reject_mask | suspicious_mask


def _iter_kept_runs(remove_mask: np.ndarray):
    """yield 保留段 [start, end)（原序列下标，半开）。"""
    n = int(remove_mask.size)
    i = 0
    while i < n:
        if remove_mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and not remove_mask[j]:
            j += 1
        yield i, j
        i = j


def splice_kept_runs_align_median(
    raw: np.ndarray,
    remove_mask: np.ndarray,
    sample_rate: float,
    *,
    window_sec: float = EEG_SPLICE_ALIGN_WINDOW_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """删点后按保留段拼接：每段接头用局部中位数对齐后再粘（方案A）。

    返回 (拼接后序列, 保留点在原序列中的下标)。
    """
    raw_arr = np.asarray(raw)
    count = int(raw_arr.size)
    if count == 0:
        return raw_arr.copy(), np.zeros(0, dtype=np.int64)

    remove_mask = np.asarray(remove_mask, dtype=bool)
    if remove_mask.size != count:
        raise ValueError(
            f"remove_mask 长度 {remove_mask.size} 与 raw {count} 不一致"
        )

    kept_index = np.flatnonzero(~remove_mask)
    if kept_index.size == 0:
        return raw_arr[:0].copy(), kept_index

    fs = max(float(sample_rate), 1.0)
    window_n = max(1, int(round(fs * float(window_sec))))
    runs = list(_iter_kept_runs(remove_mask))
    if len(runs) <= 1:
        return raw_arr[kept_index].copy(), kept_index

    parts: list[np.ndarray] = []
    # 累积平移量：后段相对「已拼接波形」的基线对齐
    for run_i, (start, end) in enumerate(runs):
        piece = raw_arr[start:end].astype(np.float64, copy=True)
        if run_i == 0:
            parts.append(piece)
            continue
        prev = parts[-1]
        prev_tail = prev[-min(window_n, prev.size) :]
        next_head = piece[: min(window_n, piece.size)]
        shift = float(np.median(prev_tail) - np.median(next_head))
        piece += shift
        parts.append(piece)

    cleaned = np.concatenate(parts)
    # 写回整型 raw 时取整，避免 CSV 出现过多小数
    if np.issubdtype(raw_arr.dtype, np.integer):
        cleaned = np.rint(cleaned).astype(raw_arr.dtype, copy=False)
    else:
        cleaned = cleaned.astype(raw_arr.dtype, copy=False)
    return cleaned, kept_index


def clean_raw_signal(
    raw: np.ndarray,
    quality: EegQualityInfo,
    *,
    remove_suspicious: bool = True,
    sample_rate: float | None = None,
    align_median: bool = True,
    align_window_sec: float = EEG_SPLICE_ALIGN_WINDOW_SEC,
) -> tuple[np.ndarray, np.ndarray, int]:
    """去掉坏段/可疑段后的 raw 及保留点在原序列中的索引。

    align_median=True（默认）：保留段接头按局部中位数对齐后再拼接（方案A），
    减轻硬拼接台阶带来的假低频。
    """
    raw_arr = np.asarray(raw)
    count = raw_arr.size
    if count == 0:
        return raw_arr.copy(), np.zeros(0, dtype=np.int64), 0
    remove_mask = build_raw_remove_mask(
        quality,
        count,
        remove_suspicious=remove_suspicious,
    )
    n_removed = int(np.count_nonzero(remove_mask))
    if not align_median:
        kept_index = np.flatnonzero(~remove_mask)
        return raw_arr[kept_index], kept_index, n_removed

    fs = (
        float(sample_rate)
        if sample_rate is not None and sample_rate > 0
        else EEG_SPLICE_ALIGN_DEFAULT_FS
    )
    cleaned, kept_index = splice_kept_runs_align_median(
        raw_arr,
        remove_mask,
        fs,
        window_sec=align_window_sec,
    )
    return cleaned, kept_index, n_removed


def slice_clean_raw_segment(
    raw: np.ndarray,
    quality: EegQualityInfo,
    sample_rate: float,
    start_seconds: float,
    end_seconds: float,
    *,
    remove_suspicious: bool = True,
    align_median: bool = True,
) -> np.ndarray:
    """按原始时间范围切片，并去掉该段内的 reject/suspicious 点。"""
    raw_arr = np.asarray(raw)
    count = raw_arr.size
    if count == 0:
        return raw_arr.copy()
    total_duration = count / sample_rate
    start = max(0.0, start_seconds)
    end = min(end_seconds, total_duration)
    if start >= end:
        raise ValueError(
            f"无效时间范围: {start_seconds:g}–{end_seconds:g} s "
            f"(数据总长 {total_duration:.2f} s)"
        )
    i_start = min(count - 1, int(start * sample_rate))
    i_end = max(i_start + 1, min(count, int(end * sample_rate)))
    remove_mask = build_raw_remove_mask(
        quality,
        count,
        remove_suspicious=remove_suspicious,
    )
    seg_remove = remove_mask[i_start:i_end]
    segment = raw_arr[i_start:i_end]
    if not align_median:
        return segment[~seg_remove]
    cleaned, _ = splice_kept_runs_align_median(
        segment,
        seg_remove,
        float(sample_rate),
    )
    return cleaned


def export_offline_raw_csvs(
    data_path: str | Path,
    raw: np.ndarray,
    quality: EegQualityInfo,
    sample_rate: float,
    *,
    alpha_info: BandSuspiciousInfo | None = None,
    remove_suspicious: bool = True,
) -> tuple[Path, Path, int]:
    """保存带质量标签的全量 raw 与去掉坏段后的 raw（均不含各节律波形列）。"""
    path = Path(data_path)
    raw_arr = np.asarray(raw)
    count = raw_arr.size
    index = np.arange(count, dtype=np.int64)
    reject_mask = fit_bool_mask(quality.reject_mask, count)
    suspicious_mask = (
        fit_bool_mask(quality.suspicious_mask, count)
        if quality.suspicious_mask is not None
        else np.zeros(count, dtype=bool)
    )
    alpha_mask = (
        fit_bool_mask(alpha_info.mask, count)
        if alpha_info is not None
        else np.zeros(count, dtype=bool)
    )
    remove_mask = build_raw_remove_mask(
        quality,
        count,
        remove_suspicious=remove_suspicious,
    )

    full_data: dict[str, object] = {
        "index": index,
        "time_s": index / sample_rate,
        "ch1_raw": raw_arr[:count],
        "is_rejected": reject_mask.astype(np.int8),
        "is_suspicious": suspicious_mask.astype(np.int8),
        "is_alpha_suspicious": alpha_mask.astype(np.int8),
        "is_removed_in_cleaned": remove_mask.astype(np.int8),
    }
    if quality.reject_reasons is not None:
        full_data["reject_reason"] = quality.reject_reasons[:count]
    if quality.suspicious_reasons is not None:
        full_data["suspicious_reason"] = quality.suspicious_reasons[:count]

    full_path = path.parent / f"{path.stem}_offline_full_raw.csv"
    pd.DataFrame(full_data).to_csv(full_path, index=False, encoding="utf-8-sig")

    cleaned_raw, kept_index, removed_points = clean_raw_signal(
        raw_arr,
        quality,
        remove_suspicious=remove_suspicious,
        sample_rate=sample_rate,
    )
    clean_data: dict[str, object] = {
        "index": np.arange(cleaned_raw.size, dtype=np.int64),
        "time_s": np.arange(cleaned_raw.size, dtype=np.float64) / sample_rate,
        "ch1_raw": cleaned_raw,
        "original_index": kept_index,
        "original_time_s": kept_index / sample_rate,
    }
    clean_path = path.parent / f"{path.stem}_offline_cleaned_raw.csv"
    pd.DataFrame(clean_data).to_csv(clean_path, index=False, encoding="utf-8-sig")
    return full_path, clean_path, removed_points


def build_model_window_quality_table(
    quality: EegQualityInfo,
    alpha_info: BandSuspiciousInfo | None,
    sample_rate: float,
    *,
    window_sec: float = MODEL_WINDOW_SEC,
) -> pd.DataFrame:
    """Summarize quality tags per model input window."""
    count = quality.reject_mask.size
    if count == 0:
        return pd.DataFrame()

    window_size = max(1, int(round(sample_rate * window_sec)))
    reject_mask = fit_bool_mask(quality.reject_mask, count)
    suspicious_mask = (
        fit_bool_mask(quality.suspicious_mask, count)
        if quality.suspicious_mask is not None
        else np.zeros(count, dtype=bool)
    )
    alpha_mask = (
        fit_bool_mask(alpha_info.mask, count)
        if alpha_info is not None
        else np.zeros(count, dtype=bool)
    )

    rows: list[dict[str, object]] = []
    for window_id, start in enumerate(range(0, count, window_size)):
        end = min(count, start + window_size)
        n = end - start
        reject_ratio = float(np.mean(reject_mask[start:end]))
        suspicious_ratio = float(np.mean(suspicious_mask[start:end]))
        alpha_ratio = float(np.mean(alpha_mask[start:end]))

        action = "keep"
        confidence_weight = 1.0
        if (
            reject_ratio >= MODEL_REJECT_DROP_RATIO
            or suspicious_ratio >= MODEL_SUSPICIOUS_DROP_RATIO
            or alpha_ratio >= MODEL_ALPHA_SUSPICIOUS_DROP_RATIO
        ):
            action = "drop"
            confidence_weight = 0.0
        elif (
            reject_ratio > 0.0
            or suspicious_ratio >= MODEL_SUSPICIOUS_WARN_RATIO
            or alpha_ratio >= MODEL_ALPHA_SUSPICIOUS_WARN_RATIO
        ):
            action = "low_confidence"
            confidence_weight = 0.5
        elif alpha_ratio > 0.0 or suspicious_ratio > 0.0:
            action = "downweight"
            confidence_weight = 0.75

        rows.append(
            {
                "window_id": window_id,
                "start_s": start / sample_rate,
                "end_s": end / sample_rate,
                "n_samples": n,
                "reject_ratio": reject_ratio,
                "suspicious_ratio": suspicious_ratio,
                "alpha_suspicious_ratio": alpha_ratio,
                "action": action,
                "confidence_weight": confidence_weight,
            }
        )

    return pd.DataFrame(rows)
