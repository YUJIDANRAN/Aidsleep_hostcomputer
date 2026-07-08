"""EEG wavelet demo for saved eeg_raw.csv files.

This script is intentionally separate from the main analysis flow. It reads a
saved EEG CSV/XLSX file, produces CWT time-frequency plots, and gives a small
DWT decomposition/denoising demonstration.

Concept map:
  CWT = Continuous Wavelet Transform. It scans many frequencies over time, so
        it is useful for "when did alpha/theta/delta become stronger?".
  DWT = Discrete Wavelet Transform. It splits the signal into coarse-to-fine
        layers, so it is useful for denoising or extracting multi-scale
        features. This script uses PyWavelets when available, otherwise a small
        built-in Haar DWT fallback for demonstration.

Example:
  python tools/wavelet_eeg_demo.py Result/eeg_20260629_173342/eeg_raw.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from power_cal import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    DEFAULT_SAMPLE_RATE,
    EEG_BANDS,
    bandpass_filter,
    read_eeg_signal,
)


def _slice_signal(
    values: np.ndarray,
    sample_rate: float,
    start_sec: float,
    max_seconds: float | None,
) -> np.ndarray:
    start = max(0, int(round(start_sec * sample_rate)))
    stop = values.size
    if max_seconds is not None and max_seconds > 0:
        stop = min(stop, start + int(round(max_seconds * sample_rate)))
    sliced = values[start:stop]
    if sliced.size < int(sample_rate):
        raise ValueError("Selected segment is shorter than 1 second.")
    return sliced.astype(float, copy=False)


def _decimate_for_cwt(
    values: np.ndarray,
    sample_rate: float,
    target_rate: float,
) -> Tuple[np.ndarray, float]:
    if target_rate <= 0 or sample_rate <= target_rate:
        return values, sample_rate
    gcd = math.gcd(int(round(sample_rate)), int(round(target_rate)))
    up = int(round(target_rate)) // gcd
    down = int(round(sample_rate)) // gcd
    return resample_poly(values, up=up, down=down), float(target_rate)


def morlet_cwt_power(
    values: np.ndarray,
    sample_rate: float,
    freqs: np.ndarray,
    cycles: float = 6.0,
) -> np.ndarray:
    """Return CWT power with a complex Morlet wavelet.

    The implementation avoids an extra dependency and is good enough for
    visual inspection. More cycles improve frequency precision but smear time.
    """
    centered = values - np.median(values)
    power = np.empty((freqs.size, centered.size), dtype=np.float64)
    dt = 1.0 / sample_rate

    for index, freq in enumerate(freqs):
        sigma_t = cycles / (2.0 * np.pi * freq)
        half_width = max(int(round(4.0 * sigma_t * sample_rate)), 8)
        t = np.arange(-half_width, half_width + 1) * dt
        wavelet = np.exp(2j * np.pi * freq * t) * np.exp(
            -(t**2) / (2.0 * sigma_t**2)
        )
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
        coeff = np.convolve(centered, np.conj(wavelet[::-1]), mode="same")
        power[index] = np.abs(coeff) ** 2

    return power


def band_power_curves(
    cwt_power: np.ndarray,
    freqs: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    curves: Dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            curves[name] = np.mean(cwt_power[mask], axis=0)
    return curves


def _smooth(values: np.ndarray, sample_rate: float, seconds: float = 0.5) -> np.ndarray:
    width = max(1, int(round(sample_rate * seconds)))
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def plot_cwt_scalogram(
    times: np.ndarray,
    freqs: np.ndarray,
    cwt_power: np.ndarray,
    out_path: Path,
    mark_seconds: list[float] | None = None,
) -> None:
    db_power = 10.0 * np.log10(cwt_power + np.finfo(float).eps)
    vmin, vmax = np.percentile(db_power, [5, 98])

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    image = ax.pcolormesh(times, freqs, db_power, shading="auto", cmap="turbo")
    image.set_clim(vmin, vmax)
    ax.set_title("CWT scalogram: time-frequency energy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(freqs[0], freqs[-1])
    for low, high in EEG_BANDS.values():
        ax.axhline(low, color="white", alpha=0.18, linewidth=0.8)
        ax.axhline(high, color="white", alpha=0.18, linewidth=0.8)
    for sec in mark_seconds or []:
        ax.axvline(sec, color="white", linestyle="--", linewidth=1.4, alpha=0.9)
        ax.text(
            sec,
            freqs[-1],
            f" {sec:g}s",
            color="white",
            va="top",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
    fig.colorbar(image, ax=ax, label="Power (dB)")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cwt_band_curves(
    times: np.ndarray,
    curves: Dict[str, np.ndarray],
    sample_rate: float,
    out_path: Path,
    mark_seconds: list[float] | None = None,
) -> None:
    fig, axes = plt.subplots(
        len(curves),
        1,
        figsize=(13, 8),
        sharex=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (name, values) in zip(axes, curves.items()):
        smoothed = _smooth(values, sample_rate, seconds=0.5)
        ax.plot(times, smoothed, linewidth=1.0)
        low, high = EEG_BANDS[name]
        ax.set_ylabel(f"{name}\n{low:g}-{high:g} Hz")
        for sec in mark_seconds or []:
            ax.axvline(sec, color="0.25", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("CWT band energy curves")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _haar_decompose(values: np.ndarray, levels: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    n = 2 ** int(np.floor(np.log2(values.size)))
    current = values[:n].astype(float, copy=True)
    details: List[np.ndarray] = []
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for _ in range(levels):
        if current.size < 2:
            break
        even = current[0::2]
        odd = current[1::2]
        approx = (even + odd) * inv_sqrt2
        detail = (even - odd) * inv_sqrt2
        details.append(detail)
        current = approx
    return current, details


def _haar_reconstruct(approx: np.ndarray, details: Iterable[np.ndarray]) -> np.ndarray:
    current = approx
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for detail in reversed(list(details)):
        rebuilt = np.empty(detail.size * 2, dtype=float)
        rebuilt[0::2] = (current + detail) * inv_sqrt2
        rebuilt[1::2] = (current - detail) * inv_sqrt2
        current = rebuilt
    return current


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _dwt_with_pywavelets(
    values: np.ndarray,
    sample_rate: float,
    levels: int,
) -> Tuple[str, List[np.ndarray], np.ndarray]:
    import pywt  # type: ignore

    wavelet = "db4"
    max_level = pywt.dwt_max_level(values.size, pywt.Wavelet(wavelet).dec_len)
    levels = max(1, min(levels, max_level))
    coeffs = pywt.wavedec(values, wavelet=wavelet, level=levels, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
    threshold = sigma * np.sqrt(2.0 * np.log(values.size))
    denoise_coeffs = [coeffs[0]] + [
        pywt.threshold(detail, threshold, mode="soft") for detail in coeffs[1:]
    ]
    denoised = pywt.waverec(denoise_coeffs, wavelet=wavelet, mode="symmetric")
    return f"PyWavelets {wavelet}", coeffs[1:], denoised[: values.size]


def _dwt_with_haar_fallback(
    values: np.ndarray,
    sample_rate: float,
    levels: int,
) -> Tuple[str, List[np.ndarray], np.ndarray]:
    max_level = int(np.floor(np.log2(values.size))) - 1
    levels = max(1, min(levels, max_level))
    approx, details = _haar_decompose(values, levels)
    sigma = np.median(np.abs(details[0])) / 0.6745 if details and details[0].size else 0.0
    threshold = sigma * np.sqrt(2.0 * np.log(values.size))
    thresholded = [_soft_threshold(detail, threshold) for detail in details]
    denoised = _haar_reconstruct(approx, thresholded)
    return "built-in Haar fallback", details, denoised[: values.size]


def compute_dwt_demo(
    values: np.ndarray,
    sample_rate: float,
    levels: int,
) -> Tuple[str, List[np.ndarray], np.ndarray]:
    try:
        return _dwt_with_pywavelets(values, sample_rate, levels)
    except ImportError:
        return _dwt_with_haar_fallback(values, sample_rate, levels)


def plot_dwt_demo(
    values: np.ndarray,
    denoised: np.ndarray,
    details: List[np.ndarray],
    method: str,
    sample_rate: float,
    out_path: Path,
) -> None:
    seconds = min(10.0, values.size / sample_rate)
    count = int(round(seconds * sample_rate))
    times = np.arange(count) / sample_rate

    rows = min(4, len(details)) + 1
    fig, axes = plt.subplots(rows, 1, figsize=(13, 8), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].plot(times, values[:count], label="filtered EEG", linewidth=0.9)
    axes[0].plot(times, denoised[:count], label="DWT soft-threshold demo", linewidth=0.9)
    axes[0].set_title(f"DWT demo ({method})")
    axes[0].set_ylabel("ADC count")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    for level_index, detail in enumerate(details[: rows - 1], start=1):
        detail_rate = sample_rate / (2**level_index)
        detail_times = np.arange(detail.size) / detail_rate
        high = sample_rate / (2**level_index)
        low = sample_rate / (2 ** (level_index + 1))
        axes[level_index].plot(detail_times, detail, linewidth=0.75)
        axes[level_index].set_xlim(0, seconds)
        axes[level_index].set_ylabel(f"D{level_index}\n~{low:g}-{high:g} Hz")
        axes[level_index].grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary(
    out_path: Path,
    input_path: Path,
    sample_rate: float,
    cwt_rate: float,
    seconds: float,
    method: str,
) -> None:
    text = f"""EEG wavelet demo summary

Input: {input_path}
Original/analysis sample rate: {sample_rate:g} Hz
CWT display sample rate: {cwt_rate:g} Hz
Analyzed duration: {seconds:.2f} s
DWT method: {method}

Generated files:
  - cwt_scalogram.png
    CWT time-frequency map. Bright areas mean that frequency is strong at that
    time. This is the easiest plot for seeing transient alpha/theta/delta
    changes.

  - cwt_band_energy.png
    CWT power averaged into the existing EEG bands. Use it to see whether a
    band is continuously strong or only briefly bursts.

  - dwt_demo.png
    DWT multi-scale decomposition plus a simple soft-threshold denoising demo.
    Treat this as a teaching/inspection plot first, not as a clinical or model
    preprocessing decision.
"""
    out_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EEG wavelet demo plots.")
    parser.add_argument("input", type=Path, help="Path to eeg_raw.csv/xlsx.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input_dir>/wavelet_demo",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help=f"EEG sample rate in Hz. Default: {DEFAULT_SAMPLE_RATE:g}",
    )
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Limit analyzed duration. Default 0 means full file.",
    )
    parser.add_argument("--cwt-low", type=float, default=BANDPASS_LOW_HZ)
    parser.add_argument("--cwt-high", type=float, default=BANDPASS_HIGH_HZ)
    parser.add_argument("--cwt-freqs", type=int, default=80)
    parser.add_argument(
        "--mark-sec",
        type=float,
        action="append",
        default=[],
        help="Mark a vertical reference line on CWT plots. Can be repeated.",
    )
    parser.add_argument(
        "--cwt-rate",
        type=float,
        default=100.0,
        help="Downsample rate for CWT plotting. Keeps the demo responsive.",
    )
    parser.add_argument("--dwt-levels", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    out_dir = args.out_dir or (input_path.parent / "wavelet_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_seconds = None if args.max_seconds == 0 else args.max_seconds
    raw = read_eeg_signal(input_path)
    segment = _slice_signal(raw, args.sample_rate, args.start_sec, max_seconds)
    filtered = bandpass_filter(segment, args.sample_rate)

    cwt_signal, cwt_rate = _decimate_for_cwt(filtered, args.sample_rate, args.cwt_rate)
    freqs = np.linspace(args.cwt_low, args.cwt_high, args.cwt_freqs)
    cwt_power = morlet_cwt_power(cwt_signal, cwt_rate, freqs)
    cwt_times = np.arange(cwt_signal.size) / cwt_rate + args.start_sec

    plot_cwt_scalogram(
        cwt_times,
        freqs,
        cwt_power,
        out_dir / "cwt_scalogram.png",
        mark_seconds=args.mark_sec,
    )
    curves = band_power_curves(cwt_power, freqs, EEG_BANDS)
    plot_cwt_band_curves(
        cwt_times,
        curves,
        cwt_rate,
        out_dir / "cwt_band_energy.png",
        mark_seconds=args.mark_sec,
    )

    method, details, denoised = compute_dwt_demo(filtered, args.sample_rate, args.dwt_levels)
    plot_dwt_demo(
        filtered,
        denoised,
        details,
        method,
        args.sample_rate,
        out_dir / "dwt_demo.png",
    )

    write_summary(
        out_dir / "README_wavelet_demo.txt",
        input_path,
        args.sample_rate,
        cwt_rate,
        filtered.size / args.sample_rate,
        method,
    )
    print(f"Wavelet demo saved to: {out_dir}")
    print("Generated: cwt_scalogram.png, cwt_band_energy.png, dwt_demo.png")


if __name__ == "__main__":
    main()
