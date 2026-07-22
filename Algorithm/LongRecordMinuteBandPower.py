"""
长时记录结束后：拼接各 chunk → 按离线规则剔除坏段 →
按分钟计算五节律绝对功率，只输出 CSV + 一张折线图。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ALGO_DIR = Path(__file__).resolve().parent
_ROOT = _ALGO_DIR.parent
for _p in (_ROOT, _ALGO_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from MovementArtifact import (  # noqa: E402
    build_threshold_rejection,
    clean_raw_signal,
)
from LongRecordNormalReport import list_chunk_full_csvs  # noqa: E402
from power_cal import (  # noqa: E402
    BAND_COLORS,
    BAND_LABELS,
    DEFAULT_SAMPLE_RATE,
    EEG_BANDS,
    compute_band_powers,
)

MINUTE_SEC = 60.0
CSV_NAME = "minute_band_absolute_power.csv"
PLOT_NAME = "minute_band_absolute_power.png"
REPORT_NAME = "minute_band_absolute_power_report.txt"


@dataclass
class MinuteBandPowerResult:
    session_dir: Path
    sample_rate: float
    n_raw: int
    n_cleaned: int
    n_removed: int
    n_minutes: int
    minutes: np.ndarray = field(repr=False)
    absolute: Dict[str, np.ndarray] = field(repr=False)
    csv_path: Optional[Path] = None
    plot_path: Optional[Path] = None
    report_text: str = ""


def _setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_concat_chunk_raw(session_dir: Path) -> tuple[np.ndarray, float]:
    files = list_chunk_full_csvs(session_dir)
    if not files:
        raise FileNotFoundError(f"会话目录无 eeg_chunk_*_full.csv: {session_dir}")

    parts: List[np.ndarray] = []
    rates: List[float] = []
    for path in files:
        df = pd.read_csv(path)
        if "ch1_raw" not in df.columns:
            raise ValueError(f"缺少 ch1_raw: {path}")
        raw = df["ch1_raw"].to_numpy(dtype=np.float64)
        parts.append(raw)
        if "time_s" in df.columns and len(df) >= 2:
            dt = np.diff(df["time_s"].to_numpy(dtype=np.float64))
            dt = dt[dt > 0]
            if dt.size:
                rates.append(float(1.0 / np.median(dt)))
    sample_rate = float(np.median(rates)) if rates else float(DEFAULT_SAMPLE_RATE)
    return np.concatenate(parts), sample_rate


def compute_minute_band_absolute_powers(
    cleaned: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """返回 minute 序号(1..N) 与各节律绝对功率数组。"""
    n_per_min = max(1, int(round(sample_rate * MINUTE_SEC)))
    n_minutes = int(cleaned.size // n_per_min)
    if n_minutes <= 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, {name: empty.copy() for name in EEG_BANDS}

    minutes = np.arange(1, n_minutes + 1, dtype=np.int32)
    absolute: Dict[str, np.ndarray] = {
        name: np.zeros(n_minutes, dtype=np.float64) for name in EEG_BANDS
    }
    for i in range(n_minutes):
        start = i * n_per_min
        end = start + n_per_min
        seg = cleaned[start:end]
        analysis = compute_band_powers(seg, sample_rate=sample_rate, welch_seconds=2.0)
        for name in EEG_BANDS:
            absolute[name][i] = float(analysis.result.absolute[name])
    return minutes, absolute


def _save_csv(
    path: Path,
    minutes: np.ndarray,
    absolute: Dict[str, np.ndarray],
) -> None:
    rows = []
    for i, minute in enumerate(minutes):
        row = {"minute": int(minute)}
        for name in EEG_BANDS:
            row[f"{name}_absolute"] = float(absolute[name][i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _save_plot(
    path: Path,
    minutes: np.ndarray,
    absolute: Dict[str, np.ndarray],
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    for name in EEG_BANDS:
        ax.plot(
            minutes,
            absolute[name],
            color=BAND_COLORS[name],
            lw=1.8,
            marker="o",
            markersize=3.5,
            label=f"{BAND_LABELS[name]} ({EEG_BANDS[name][0]:g}-{EEG_BANDS[name][1]:g} Hz)",
        )
    ax.set_xlabel("分钟")
    ax.set_ylabel("绝对功率")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if minutes.size:
        ax.set_xlim(0.5, float(minutes[-1]) + 0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_minute_band_power_analysis(
    session_dir: Path,
    *,
    save_outputs: bool = True,
) -> MinuteBandPowerResult:
    session_dir = Path(session_dir)
    raw, sample_rate = load_concat_chunk_raw(session_dir)
    quality = build_threshold_rejection(raw, sample_rate)
    cleaned, _, n_removed = clean_raw_signal(raw.astype(np.int64), quality)
    cleaned = cleaned.astype(np.float64)

    minutes, absolute = compute_minute_band_absolute_powers(cleaned, sample_rate)
    n_minutes = int(minutes.size)
    cleaned_sec = float(cleaned.size / sample_rate) if sample_rate > 0 else 0.0

    lines = [
        "长时记录 · 每分钟节律绝对功率",
        f"会话目录: {session_dir.resolve()}",
        f"拼接 raw: {raw.size} 点 @ {sample_rate:.3f} Hz",
        f"坏段剔除后: {cleaned.size} 点（剔除 {n_removed}），约 {cleaned_sec / 60.0:.2f} 分钟",
        f"完整分钟数 N: {n_minutes}（按 60 s 整分切片，不足 1 分钟的尾部丢弃）",
        "处理与离线一致: build_threshold_rejection + clean_raw_signal + Welch 带功率",
        "仅输出每分钟绝对功率曲线图，不生成 FFT/波形/段对比等其它图",
    ]
    if n_minutes == 0:
        lines.append("有效数据不足 1 分钟，未生成曲线。")

    csv_path = None
    plot_path = None
    if save_outputs and n_minutes > 0:
        csv_path = session_dir / CSV_NAME
        plot_path = session_dir / PLOT_NAME
        _save_csv(csv_path, minutes, absolute)
        _save_plot(
            plot_path,
            minutes,
            absolute,
            title=f"每分钟节律绝对功率（坏段剔除后，N={n_minutes}）",
        )
        lines.append(f"CSV: {csv_path.name}")
        lines.append(f"图: {plot_path.name}")

    report_text = "\n".join(lines)
    if save_outputs:
        (session_dir / REPORT_NAME).write_text(report_text + "\n", encoding="utf-8")

    return MinuteBandPowerResult(
        session_dir=session_dir,
        sample_rate=sample_rate,
        n_raw=int(raw.size),
        n_cleaned=int(cleaned.size),
        n_removed=int(n_removed),
        n_minutes=n_minutes,
        minutes=minutes,
        absolute=absolute,
        csv_path=csv_path,
        plot_path=plot_path,
        report_text=report_text,
    )
