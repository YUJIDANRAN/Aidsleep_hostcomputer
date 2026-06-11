"""EEG alpha-band (8-13 Hz) bandpass filter for streaming samples."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque, Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

ALPHA_LOW_HZ = 8.0
ALPHA_HIGH_HZ = 13.0
FILTER_ORDER = 4
DEFAULT_SAMPLE_RATE = float(os.environ.get("KS1082_SAMPLE_RATE", "500"))
# 下位机已标定 fs 时保持 False；仅调试可设 KS1082_AUTO_FS=1
AUTO_ESTIMATE_SAMPLE_RATE = os.environ.get("KS1082_AUTO_FS", "0").strip() in (
    "1",
    "true",
    "yes",
)


def design_alpha_bandpass(
    sample_rate: float,
    low_hz: float = ALPHA_LOW_HZ,
    high_hz: float = ALPHA_HIGH_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    if sample_rate <= 2 * high_hz:
        raise ValueError(
            f"sample_rate {sample_rate} Hz too low for {high_hz} Hz passband"
        )
    nyquist = sample_rate * 0.5
    return butter(
        order,
        (low_hz / nyquist, high_hz / nyquist),
        btype="bandpass",
        output="sos",
    )


class SampleRateEstimator:
    """Track receive rate = sample_count / elapsed wall time (for diagnostics)."""

    def __init__(self, min_elapsed: float = 0.2, min_samples: int = 20) -> None:
        self._min_elapsed = min_elapsed
        self._min_samples = min_samples
        self._start: Optional[float] = None
        self._count = 0

    def reset(self) -> None:
        self._start = None
        self._count = 0

    def add(self, count: int = 1) -> None:
        if count <= 0:
            return
        now = time.monotonic()
        if self._start is None:
            self._start = now
        self._count += count

    def current_rate(self) -> Optional[float]:
        if self._start is None or self._count < self._min_samples:
            return None
        elapsed = time.monotonic() - self._start
        if elapsed < self._min_elapsed:
            return None
        rate = self._count / elapsed
        if 50.0 <= rate <= 2000.0:
            return rate
        return None


class AlphaBandpassFilter:
    """Real-time SOS bandpass (single channel)."""

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = sample_rate
        self.sos = design_alpha_bandpass(sample_rate)
        self.reset()

    def reset(self) -> None:
        self._zi = sosfilt_zi(self.sos)

    def process(self, value: float) -> float:
        y, self._zi = sosfilt(self.sos, [value], zi=self._zi)
        return float(y[0])


class AlphaExtractor:
    """Alpha bandpass on CH1; fs fixed to configured rate unless KS1082_AUTO_FS=1."""

    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE) -> None:
        self._configured_rate = sample_rate
        self._estimator = SampleRateEstimator()
        self._filter: Optional[AlphaBandpassFilter] = None
        self._alpha: Deque[float] = deque(maxlen=800)
        self._auto_fs = AUTO_ESTIMATE_SAMPLE_RATE
        self._init_filter(sample_rate)

    @property
    def sample_rate(self) -> float:
        """滤波与波形时间轴使用的 fs（默认固定为下位机标定值）。"""
        return self._configured_rate

    @property
    def measured_sample_rate(self) -> Optional[float]:
        """主机实际收到的样本速率，用于对比是否丢样。"""
        return self._estimator.current_rate()

    def reset(self) -> None:
        self._estimator.reset()
        self._alpha.clear()
        self._init_filter(self._configured_rate)

    def _init_filter(self, sample_rate: float) -> None:
        self._filter = AlphaBandpassFilter(sample_rate)

    def _maybe_rebuild_filter(self) -> None:
        if not self._auto_fs:
            return
        estimated = self._estimator.current_rate()
        if estimated is None:
            return
        if abs(estimated - self._configured_rate) / self._configured_rate < 0.08:
            return
        self._configured_rate = estimated
        self._init_filter(estimated)

    def push_adc_sample(self, adc: int) -> float:
        assert self._filter is not None
        self._estimator.add(1)
        self._maybe_rebuild_filter()
        alpha = self._filter.process(float(adc))
        self._alpha.append(alpha)
        return alpha

    def alpha_power(self, window: int = 128) -> float:
        if not self._alpha:
            return 0.0
        chunk = list(self._alpha)[-window:]
        return float(np.sqrt(np.mean(np.square(chunk))))

    @property
    def alpha_values(self) -> Deque[float]:
        return self._alpha
