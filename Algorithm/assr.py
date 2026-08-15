"""听觉稳态反应（ASSR）刺激分析。

默认协议：1000 Hz 载波、40 Hz 幅度调制（皮层 40 Hz ASSR）。
分析：Welch PSD，取调制频率及其二次谐波处的功率，相对邻频噪声算 SNR。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import welch

ASSR_CARRIER_HZ = 1000.0
ASSR_MODULATION_HZ = 40.0
ASSR_MODULATION_DEPTH = 1.0
ASSR_DURATION_SEC = 60.0
ASSR_DISCARD_SEC = 2.0
ASSR_AMPLITUDE = 0.18
ASSR_NEIGHBOR_BINS = 8
ASSR_EXCLUDE_BINS = 2
ASSR_WELCH_SEC = 4.0
ASSR_PLOT_FMIN_HZ = 1.0
ASSR_PLOT_FMAX_HZ = 90.0


@dataclass(frozen=True)
class AssrParams:
    carrier_hz: float = ASSR_CARRIER_HZ
    modulation_hz: float = ASSR_MODULATION_HZ
    depth: float = ASSR_MODULATION_DEPTH
    duration_sec: float = ASSR_DURATION_SEC
    discard_sec: float = ASSR_DISCARD_SEC
    amplitude: float = ASSR_AMPLITUDE
    stimulus: str = "am"  ## am | click
    ear: str = "stereo"  ## stereo | left | right


@dataclass(frozen=True)
class AssrPeakResult:
    freq_hz: float
    power: float
    noise: float
    snr: float
    snr_db: float
    f_stat: float


@dataclass(frozen=True)
class AssrAnalysis:
    params: AssrParams
    sample_rate: float
    n_samples: int
    duration_used_sec: float
    freqs: np.ndarray
    psd: np.ndarray
    fundamental: AssrPeakResult
    harmonic: Optional[AssrPeakResult]
    passed: bool
    note: str = ""


def _nearest_bin(freqs: np.ndarray, target_hz: float) -> int:
    return int(np.argmin(np.abs(freqs - float(target_hz))))


def _peak_vs_neighbors(
    freqs: np.ndarray,
    psd: np.ndarray,
    target_hz: float,
    *,
    n_neighbors: int = ASSR_NEIGHBOR_BINS,
    exclude_bins: int = ASSR_EXCLUDE_BINS,
) -> Optional[AssrPeakResult]:
    if freqs.size < 8 or psd.size != freqs.size:
        return None
    center = _nearest_bin(freqs, target_hz)
    if abs(freqs[center] - target_hz) > max(0.6, target_hz * 0.03):
        return None
    lo = max(0, center - exclude_bins - n_neighbors)
    hi = min(psd.size, center + exclude_bins + n_neighbors + 1)
    noise_idx = np.r_[np.arange(lo, max(lo, center - exclude_bins)), np.arange(min(hi, center + exclude_bins + 1), hi)]
    if noise_idx.size < 4:
        return None
    signal = float(psd[center])
    noise = float(np.mean(psd[noise_idx]))
    if noise <= 0.0:
        noise = float(np.finfo(float).tiny)
    snr = signal / noise
    return AssrPeakResult(
        freq_hz=float(freqs[center]),
        power=signal,
        noise=noise,
        snr=float(snr),
        snr_db=float(10.0 * np.log10(max(snr, np.finfo(float).tiny))),
        f_stat=float(snr),  ## 功率比近似 F(2, 2N)
    )


def analyze_assr(
    raw: np.ndarray,
    sample_rate: float,
    params: AssrParams,
    *,
    welch_seconds: float = ASSR_WELCH_SEC,
    n_neighbors: int = ASSR_NEIGHBOR_BINS,
) -> AssrAnalysis:
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    fs = float(sample_rate)
    if fs <= 0 or arr.size < int(fs * 2.0):
        empty = np.zeros(0, dtype=np.float64)
        dummy = AssrPeakResult(0.0, 0.0, 1.0, 0.0, -np.inf, 0.0)
        return AssrAnalysis(
            params=params,
            sample_rate=fs,
            n_samples=int(arr.size),
            duration_used_sec=0.0,
            freqs=empty,
            psd=empty,
            fundamental=dummy,
            harmonic=None,
            passed=False,
            note="有效样本过少，至少需要约 2 s",
        )

    discard = max(0, int(round(params.discard_sec * fs)))
    if discard >= arr.size:
        discard = 0
    used = arr[discard:]
    used = used - float(np.mean(used))
    nperseg = int(min(used.size, max(256, round(fs * welch_seconds))))
    nperseg = min(nperseg, used.size)
    freqs, psd = welch(used, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    fund = _peak_vs_neighbors(freqs, psd, params.modulation_hz, n_neighbors=n_neighbors)
    harm = _peak_vs_neighbors(freqs, psd, 2.0 * params.modulation_hz, n_neighbors=n_neighbors)
    if fund is None:
        dummy = AssrPeakResult(params.modulation_hz, 0.0, 1.0, 0.0, -np.inf, 0.0)
        return AssrAnalysis(
            params=params,
            sample_rate=fs,
            n_samples=int(used.size),
            duration_used_sec=used.size / fs,
            freqs=freqs,
            psd=psd,
            fundamental=dummy,
            harmonic=harm,
            passed=False,
            note=f"未在 {params.modulation_hz:g} Hz 附近找到谱峰",
        )

    # 经验门槛：SNR ≥ 6 dB 且 F ≥ 11（约 p<0.01，邻频足够时）
    passed = fund.snr_db >= 6.0 and fund.f_stat >= 11.0
    note = (
        f"{params.modulation_hz:g} Hz SNR {fund.snr_db:.1f} dB"
        + (f"；{2 * params.modulation_hz:g} Hz SNR {harm.snr_db:.1f} dB" if harm is not None else "")
        + ("；判定：引出" if passed else "；判定：未稳定引出")
    )
    return AssrAnalysis(
        params=params,
        sample_rate=fs,
        n_samples=int(used.size),
        duration_used_sec=used.size / fs,
        freqs=freqs,
        psd=psd,
        fundamental=fund,
        harmonic=harm,
        passed=passed,
        note=note,
    )


def format_assr_report(result: AssrAnalysis) -> str:
    p = result.params
    lines = [
        f"ASSR | {p.stimulus} | 载波 {p.carrier_hz:g} Hz | 调制 {p.modulation_hz:g} Hz | {p.ear}",
        f"分析时长 {result.duration_used_sec:.1f} s（去起始 {p.discard_sec:g} s），N={result.n_samples} @ {result.sample_rate:.1f} Hz",
        (
            f"基频 {result.fundamental.freq_hz:.2f} Hz: "
            f"SNR {result.fundamental.snr_db:.2f} dB, F={result.fundamental.f_stat:.1f}"
        ),
    ]
    if result.harmonic is not None:
        lines.append(
            f"谐波 {result.harmonic.freq_hz:.2f} Hz: "
            f"SNR {result.harmonic.snr_db:.2f} dB, F={result.harmonic.f_stat:.1f}"
        )
    lines.append(result.note)
    return "\n".join(lines)
