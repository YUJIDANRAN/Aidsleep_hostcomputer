from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from power_cal import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    compare_segment_band_powers,
    compute_band_powers,
    read_eeg_quality,
    read_eeg_signal,
)


def parse_segments(name: str, duration: float):
    match = re.search(r"open(\d+)_close(\d+)", name)
    if match:
        open_s = float(match.group(1))
        close_s = float(match.group(2))
        return "open", "close", (0.0, min(open_s, duration)), (
            open_s,
            min(open_s + close_s, duration),
        )
    match = re.search(r"close(\d+)open(\d+)", name)
    if match:
        close_s = float(match.group(1))
        open_s = float(match.group(2))
        return "close", "open", (0.0, min(close_s, duration)), (
            close_s,
            min(close_s + open_s, duration),
        )
    return None


def pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def main() -> None:
    for path in sorted((ROOT / "Sample").rglob("*_converted.csv")):
        if path.name == "eeg_raw_converted.csv":
            continue

        raw = read_eeg_signal(path)
        duration = len(raw) / DEFAULT_SAMPLE_RATE
        quality = read_eeg_quality(path, len(raw), raw, DEFAULT_SAMPLE_RATE)
        overall = compute_band_powers(raw, DEFAULT_SAMPLE_RATE).result.relative
        segments = parse_segments(path.parent.name, duration)

        print()
        print(f"FILE {path.relative_to(ROOT)}")
        print(
            f"duration={duration:.2f}s reject={quality.reject_rate:.1%} "
            f"suspicious={quality.suspicious_rate:.1%} "
            f"overall alpha={pct(overall['alpha'])} "
            f"delta={pct(overall['delta'])} beta={pct(overall['beta'])}"
        )
        if segments is None:
            print("segments=not parsed")
            continue

        label_a, label_b, range_a, range_b = segments
        if range_a[1] <= range_a[0] or range_b[1] <= range_b[0]:
            print(f"segments=invalid {range_a} {range_b}")
            continue

        comp = compare_segment_band_powers(raw, DEFAULT_SAMPLE_RATE, range_a, range_b)
        a_rel = comp.analysis_a.result.relative
        b_rel = comp.analysis_b.result.relative
        a_abs = comp.analysis_a.result.absolute
        b_abs = comp.analysis_b.result.absolute
        print(
            f"{label_a} {range_a[0]:g}-{range_a[1]:g}s: "
            f"alpha_rel={pct(a_rel['alpha'])} alpha_abs={a_abs['alpha']:.4g} "
            f"delta_rel={pct(a_rel['delta'])} beta_rel={pct(a_rel['beta'])}"
        )
        print(
            f"{label_b} {range_b[0]:g}-{range_b[1]:g}s: "
            f"alpha_rel={pct(b_rel['alpha'])} alpha_abs={b_abs['alpha']:.4g} "
            f"delta_rel={pct(b_rel['delta'])} beta_rel={pct(b_rel['beta'])}"
        )
        print(
            f"alpha_abs_ratio({label_b}/{label_a})="
            f"{comp.absolute_ratio['alpha']:.2f} "
            f"alpha_rel_ratio={comp.relative_ratio['alpha']:.2f}"
        )


if __name__ == "__main__":
    main()
