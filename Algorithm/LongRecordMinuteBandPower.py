"""
长时/单文件 EEG：坏段剔除后按分钟计算五节律绝对/相对功率。

长时多 chunk：按序号拼接 eeg_chunk_*_full.csv，再统一剔坏段、按 60 s 切片。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    relative: Dict[str, np.ndarray] = field(repr=False)
    csv_path: Optional[Path] = None
    plot_path: Optional[Path] = None
    report_text: str = ""


def _setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_concat_chunk_raw(session_dir: Path) -> tuple[np.ndarray, float]:
    """按 chunk 序号拼接长时记录 full CSV。"""
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


def load_raw_for_minute_analysis(source: Path) -> tuple[np.ndarray, float, str]:
    """从会话目录或单文件加载 raw。

    - 目录含 eeg_chunk_*_full.csv → 按序拼接（长时）
    - 目录含 eeg_raw_full.csv / eeg_raw.csv → 读文件
    - 直接给 CSV 文件 → 读该文件
    返回 (raw, sample_rate, 数据来源说明)
    """
    source = Path(source)
    if source.is_file():
        df = pd.read_csv(source)
        if "ch1_raw" not in df.columns:
            raise ValueError(f"缺少 ch1_raw: {source}")
        raw = df["ch1_raw"].to_numpy(dtype=np.float64)
        sample_rate = float(DEFAULT_SAMPLE_RATE)
        if "time_s" in df.columns and len(df) >= 2:
            dt = np.diff(df["time_s"].to_numpy(dtype=np.float64))
            dt = dt[dt > 0]
            if dt.size:
                sample_rate = float(1.0 / np.median(dt))
        return raw, sample_rate, source.name

    if not source.is_dir():
        raise FileNotFoundError(f"路径不存在: {source}")

    chunks = list_chunk_full_csvs(source)
    if chunks:
        raw, sample_rate = load_concat_chunk_raw(source)
        return raw, sample_rate, f"{len(chunks)} 个 chunk 已按序拼接"

    for name in ("eeg_raw_full.csv", "eeg_raw.csv"):
        path = source / name
        if path.is_file():
            raw, sample_rate, _ = load_raw_for_minute_analysis(path)
            return raw, sample_rate, name

    offline = sorted(source.glob("*_offline_full_raw.csv"))
    if offline:
        raw, sample_rate, _ = load_raw_for_minute_analysis(offline[0])
        return raw, sample_rate, offline[0].name

    raise FileNotFoundError(
        f"目录中无可用 EEG：需 eeg_chunk_*_full.csv 或 eeg_raw_full.csv: {source}"
    )


def compute_minute_band_powers(
    cleaned: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """返回 minute 序号、各节律绝对功率、相对功率。"""
    n_per_min = max(1, int(round(sample_rate * MINUTE_SEC)))
    n_minutes = int(cleaned.size // n_per_min)
    if n_minutes <= 0:
        empty = np.zeros(0, dtype=np.float64)
        z = {name: empty.copy() for name in EEG_BANDS}
        return empty, z, {name: empty.copy() for name in EEG_BANDS}

    minutes = np.arange(1, n_minutes + 1, dtype=np.int32)
    absolute: Dict[str, np.ndarray] = {
        name: np.zeros(n_minutes, dtype=np.float64) for name in EEG_BANDS
    }
    relative: Dict[str, np.ndarray] = {
        name: np.zeros(n_minutes, dtype=np.float64) for name in EEG_BANDS
    }
    for i in range(n_minutes):
        start = i * n_per_min
        end = start + n_per_min
        seg = cleaned[start:end]
        analysis = compute_band_powers(seg, sample_rate=sample_rate, welch_seconds=2.0)
        for name in EEG_BANDS:
            absolute[name][i] = float(analysis.result.absolute[name])
            relative[name][i] = float(analysis.result.relative[name])
    return minutes, absolute, relative


def plot_minute_band_powers(
    minutes: np.ndarray,
    *,
    absolute: Optional[Dict[str, np.ndarray]] = None,
    relative: Optional[Dict[str, np.ndarray]] = None,
    title: str = "",
    save_path: Optional[Path] = None,
    figure=None,
) -> None:
    """绘制每分钟绝对/相对功率折线；可嵌入 Qt figure。"""
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    panels: List[Tuple[str, Dict[str, np.ndarray], str]] = []
    if absolute is not None:
        panels.append(("绝对功率", absolute, "绝对功率"))
    if relative is not None:
        panels.append(("相对功率 (%)", relative, "相对功率 (%)"))
    if not panels:
        raise ValueError("须至少提供 absolute 或 relative")

    own_figure = figure is None
    n = len(panels)
    if own_figure:
        fig, axes = plt.subplots(n, 1, figsize=(10, 4.2 * n), dpi=120, squeeze=False)
    else:
        fig = figure
        fig.clear()
        axes = fig.subplots(n, 1, squeeze=False)

    for row, (ylabel, series, _kind) in enumerate(panels):
        ax = axes[row, 0]
        for name in EEG_BANDS:
            y = series[name]
            if ylabel.startswith("相对"):
                y = y * 100.0
            ax.plot(
                minutes,
                y,
                color=BAND_COLORS[name],
                lw=1.8,
                marker="o",
                markersize=3.5,
                label=(
                    f"{BAND_LABELS[name]} "
                    f"({EEG_BANDS[name][0]:g}-{EEG_BANDS[name][1]:g} Hz)"
                ),
            )
        ax.set_xlabel("分钟")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        if minutes.size:
            ax.set_xlim(0.5, float(minutes[-1]) + 0.5)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")

    if not own_figure:
        return
    plt.close(fig)


def _save_csv(
    path: Path,
    minutes: np.ndarray,
    absolute: Dict[str, np.ndarray],
    relative: Dict[str, np.ndarray],
) -> None:
    rows = []
    for i, minute in enumerate(minutes):
        row = {"minute": int(minute)}
        for name in EEG_BANDS:
            row[f"{name}_absolute"] = float(absolute[name][i])
            row[f"{name}_relative"] = float(relative[name][i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def prepare_cleaned_minute_powers(
    source: Path,
) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """加载 → 剔坏段 → 按分钟算绝对/相对功率。

    返回 (minutes, absolute, relative, meta)
    """
    raw, sample_rate, source_desc = load_raw_for_minute_analysis(source)
    quality = build_threshold_rejection(raw, sample_rate)
    cleaned, _, n_removed = clean_raw_signal(
        raw.astype(np.int64), quality, sample_rate=sample_rate
    )
    cleaned = cleaned.astype(np.float64)
    minutes, absolute, relative = compute_minute_band_powers(cleaned, sample_rate)
    meta = {
        "sample_rate": float(sample_rate),
        "n_raw": int(raw.size),
        "n_cleaned": int(cleaned.size),
        "n_removed": int(n_removed),
        "n_minutes": int(minutes.size),
        "source_desc": source_desc,
    }
    return minutes, absolute, relative, meta


def run_minute_band_power_analysis(
    session_dir: Path,
    *,
    save_outputs: bool = True,
) -> MinuteBandPowerResult:
    session_dir = Path(session_dir)
    minutes, absolute, relative, meta = prepare_cleaned_minute_powers(session_dir)
    n_minutes = int(meta["n_minutes"])
    sample_rate = float(meta["sample_rate"])
    cleaned_sec = (
        float(meta["n_cleaned"] / sample_rate) if sample_rate > 0 else 0.0
    )

    lines = [
        "长时记录 · 每分钟节律绝对/相对功率",
        f"会话目录: {session_dir.resolve()}",
        f"数据来源: {meta['source_desc']}",
        f"拼接 raw: {meta['n_raw']} 点 @ {sample_rate:.3f} Hz",
        f"坏段剔除后: {meta['n_cleaned']} 点（剔除 {meta['n_removed']}），"
        f"约 {cleaned_sec / 60.0:.2f} 分钟",
        f"完整分钟数 N: {n_minutes}（按 60 s 整分切片，不足 1 分钟的尾部丢弃）",
        "处理: chunk拼接(若有) + build_threshold_rejection + clean_raw_signal + Welch",
    ]
    if n_minutes == 0:
        lines.append("有效数据不足 1 分钟，未生成曲线。")

    csv_path = None
    plot_path = None
    if save_outputs and n_minutes > 0:
        csv_path = session_dir / CSV_NAME
        plot_path = session_dir / PLOT_NAME
        _save_csv(csv_path, minutes, absolute, relative)
        plot_minute_band_powers(
            minutes,
            absolute=absolute,
            title=f"每分钟节律绝对功率（坏段剔除后，N={n_minutes}）",
            save_path=plot_path,
        )
        lines.append(f"CSV: {csv_path.name}")
        lines.append(f"图: {plot_path.name}")

    report_text = "\n".join(lines)
    if save_outputs:
        (session_dir / REPORT_NAME).write_text(report_text + "\n", encoding="utf-8")

    return MinuteBandPowerResult(
        session_dir=session_dir,
        sample_rate=sample_rate,
        n_raw=int(meta["n_raw"]),
        n_cleaned=int(meta["n_cleaned"]),
        n_removed=int(meta["n_removed"]),
        n_minutes=n_minutes,
        minutes=minutes,
        absolute=absolute,
        relative=relative,
        csv_path=csv_path,
        plot_path=plot_path,
        report_text=report_text,
    )
