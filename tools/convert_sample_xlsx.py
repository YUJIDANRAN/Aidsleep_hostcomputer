from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from power_cal import DEFAULT_SAMPLE_RATE, build_threshold_rejection


def convert_one(src: Path, sample_rate: float) -> dict[str, object]:
    xl = pd.ExcelFile(src)
    sheet = xl.sheet_names[0]
    frame = pd.read_excel(src, sheet_name=sheet, header=None)

    numeric_counts = [
        pd.to_numeric(frame[col], errors="coerce").notna().sum()
        for col in frame.columns
    ]
    raw_col = int(np.argmax(numeric_counts))
    raw_series = pd.to_numeric(frame[raw_col], errors="coerce")
    valid = raw_series.notna()
    raw_values = raw_series[valid].round().astype(int).to_numpy()

    timestamp_col = None
    for col in frame.columns:
        if col == raw_col:
            continue
        if frame.loc[valid, col].notna().sum() > 0:
            timestamp_col = col
            break
    if timestamp_col is None:
        timestamps = [""] * len(raw_values)
    else:
        timestamps = (
            frame.loc[valid, timestamp_col]
            .astype(str)
            .str.replace("]", "", regex=False)
            .to_list()
        )

    quality = build_threshold_rejection(raw_values.astype(float), sample_rate)
    segment_size = max(1, int(round(sample_rate)))
    out = src.with_name(f"{src.stem}_converted.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "time_s",
                "ch1_raw",
                "source_timestamp",
                "segment_id",
                "is_rejected",
                "reject_reason",
                "reject_rate",
                "is_suspicious",
                "suspicious_reason",
                "suspicious_rate",
            ]
        )
        suspicious_mask = (
            quality.suspicious_mask
            if quality.suspicious_mask is not None
            else np.zeros(len(raw_values), dtype=bool)
        )
        reject_reasons = quality.reject_reasons or [""] * len(raw_values)
        suspicious_reasons = quality.suspicious_reasons or [""] * len(raw_values)
        for index, value in enumerate(raw_values):
            rejected = int(quality.reject_mask[index])
            suspicious = int(suspicious_mask[index])
            writer.writerow(
                [
                    index,
                    f"{index / sample_rate:.6f}",
                    int(value),
                    timestamps[index],
                    index // segment_size,
                    rejected,
                    reject_reasons[index],
                    f"{quality.reject_rate:.6f}",
                    suspicious,
                    suspicious_reasons[index],
                    f"{quality.suspicious_rate:.6f}",
                ]
            )

    return {
        "source": str(src),
        "output": str(out),
        "sheet": sheet,
        "raw_col": raw_col,
        "rows": len(raw_values),
        "duration_s": len(raw_values) / sample_rate,
        "min": int(np.min(raw_values)) if len(raw_values) else "",
        "max": int(np.max(raw_values)) if len(raw_values) else "",
        "mean": float(np.mean(raw_values)) if len(raw_values) else "",
        "std": float(np.std(raw_values, ddof=1)) if len(raw_values) > 1 else "",
        "reject_rate": quality.reject_rate,
        "suspicious_rate": quality.suspicious_rate,
    }


def main() -> None:
    sample_rate = float(DEFAULT_SAMPLE_RATE)
    sample_dir = Path("Sample")
    rows = []
    for src in sorted(sample_dir.rglob("*.xlsx")):
        if src.name.startswith("~$"):
            continue
        rows.append(convert_one(src, sample_rate))

    summary_path = sample_dir / "conversion_summary.csv"
    fields = [
        "source",
        "output",
        "sheet",
        "raw_col",
        "rows",
        "duration_s",
        "min",
        "max",
        "mean",
        "std",
        "reject_rate",
        "suspicious_rate",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for item in rows:
        print(
            f"{item['output']} | rows={item['rows']} "
            f"duration={item['duration_s']:.2f}s "
            f"range={item['min']}-{item['max']} "
            f"reject={item['reject_rate']:.1%} "
            f"suspicious={item['suspicious_rate']:.1%}"
        )
    print(f"summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
