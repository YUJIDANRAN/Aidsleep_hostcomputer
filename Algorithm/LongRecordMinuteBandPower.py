"""
长时/单文件 EEG：坏段剔除后按固定窗长（默认 60 s，可改）计算五节律绝对/相对功率。

长时多 chunk：按序号拼接 eeg_chunk_*_full.csv，再统一剔坏段、按窗长切片。
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
    build_raw_remove_mask,
    build_threshold_rejection,
    clean_raw_signal,
    _iter_kept_runs,
)
from LongRecordNormalReport import list_chunk_full_csvs  # noqa: E402
from power_cal import (  # noqa: E402
    BAND_COLORS,
    BAND_LABELS,
    BANDPASS_LOW_HZ,
    DEFAULT_SAMPLE_RATE,
    EEG_BANDS,
    FILTER_ORDER,
    compute_band_powers,
    compute_band_powers_with_metrics,
)

MINUTE_SEC = 60.0
CSV_NAME = "minute_band_absolute_power.csv"
PLOT_NAME = "minute_band_absolute_power.png"
REPORT_NAME = "minute_band_absolute_power_report.txt"

# 功率窗两端去边：减轻 filtfilt 边界暂态（每侧秒数）
POWER_EDGE_TRIM_FRAC = 0.15
POWER_EDGE_TRIM_MIN_SEC = 0.5
POWER_EDGE_TRIM_MAX_SEC = 3.0
# 按窗质量门控：相对稳健基线的 MAD 倍数
POWER_GATE_MAD_MULT = 5.0
POWER_GATE_MAD_ITERS = 3


def _power_edge_trim_sec(window_sec: float) -> float:
    """按窗长与滤波器低频估计去边秒数（每侧）。"""
    win = float(window_sec)
    # ~阶数/f_low 量级，再夹在 [min, max] 与窗长比例内
    by_filter = float(FILTER_ORDER) / max(float(BANDPASS_LOW_HZ), 0.1)
    by_frac = win * POWER_EDGE_TRIM_FRAC
    trim = min(by_filter, by_frac, POWER_EDGE_TRIM_MAX_SEC)
    return float(max(POWER_EDGE_TRIM_MIN_SEC, trim))


def _robust_high_outlier_mask(
    values: np.ndarray,
    *,
    mad_mult: float = POWER_GATE_MAD_MULT,
    max_iter: int = POWER_GATE_MAD_ITERS,
) -> np.ndarray:
    """迭代 MAD：标记显著偏高的离群点（True=离群）。"""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(n, dtype=bool)
    for _ in range(max(1, int(max_iter))):
        v = x[keep]
        if v.size < 3:
            break
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        # 1.4826*MAD ≈ σ；MAD≈0 时用相对地板，避免全拒或全留
        scale = mad * 1.4826 if mad > 1e-15 else max(abs(med) * 0.05, 1e-12)
        thresh = med + float(mad_mult) * scale
        new_keep = keep & (x <= thresh)
        if bool(np.array_equal(new_keep, keep)):
            break
        keep = new_keep
    return ~keep


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
    *,
    window_sec: float = MINUTE_SEC,
) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """按固定时长切片计算各节律绝对/相对功率。

    window_sec: 每窗秒数，默认 60；例如 30 → 按 30 s 一片。
    返回窗序号（从 1 起）、绝对功率、相对功率。
    """
    win = float(window_sec)
    if not np.isfinite(win) or win <= 0:
        raise ValueError(f"window_sec 须为正数，当前为 {window_sec}")
    n_per_win = max(1, int(round(sample_rate * win)))
    n_windows = int(cleaned.size // n_per_win)
    if n_windows <= 0:
        empty = np.zeros(0, dtype=np.float64)
        z = {name: empty.copy() for name in EEG_BANDS}
        return empty, z, {name: empty.copy() for name in EEG_BANDS}

    # Welch 窗不超过切片时长的一半，且至少 0.5 s
    welch_seconds = float(min(2.0, max(0.5, win * 0.5)))
    edge_trim = _power_edge_trim_sec(win)

    minutes = np.arange(1, n_windows + 1, dtype=np.int32)
    absolute: Dict[str, np.ndarray] = {
        name: np.zeros(n_windows, dtype=np.float64) for name in EEG_BANDS
    }
    relative: Dict[str, np.ndarray] = {
        name: np.zeros(n_windows, dtype=np.float64) for name in EEG_BANDS
    }
    for i in range(n_windows):
        start = i * n_per_win
        end = start + n_per_win
        seg = cleaned[start:end]
        analysis = compute_band_powers(
            seg,
            sample_rate=sample_rate,
            welch_seconds=welch_seconds,
            edge_trim_sec=edge_trim,
        )
        for name in EEG_BANDS:
            absolute[name][i] = float(analysis.result.absolute[name])
            relative[name][i] = float(analysis.result.relative[name])
    return minutes, absolute, relative


def compute_band_powers_good_runs(
    raw: np.ndarray,
    remove_mask: np.ndarray,
    sample_rate: float,
    *,
    window_sec: float = MINUTE_SEC,
) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """不硬拼接：仅在连续好段内按 window_sec 切片算功率。

    每窗：带通后两端去边再 Welch；再对候选窗做 RMS/总功率 MAD 门控，
    丢掉滤波振铃等异常抬高的窗。

    同时收集坏段、过短好段、门控丢弃窗的墙钟区间，供绘图标注。
    返回 (墙钟起始秒数组, absolute, relative, extra_meta)。
    """
    raw_arr = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(remove_mask, dtype=bool)
    if mask.size != raw_arr.size:
        raise ValueError(
            f"remove_mask 长度 {mask.size} 与 raw {raw_arr.size} 不一致"
        )
    win = float(window_sec)
    if not np.isfinite(win) or win <= 0:
        raise ValueError(f"window_sec 须为正数，当前为 {window_sec}")
    fs = float(sample_rate)
    n_per_win = max(1, int(round(fs * win)))
    welch_seconds = float(min(2.0, max(0.5, win * 0.5)))
    edge_trim = _power_edge_trim_sec(win)
    dt = 1.0 / fs
    total_s = float(raw_arr.size) * dt

    def _span_s(i0: int, i1: int) -> Tuple[float, float]:
        t0 = float(i0) * dt
        t1 = float(i1) * dt
        return t0, min(t1, total_s)

    bad_spans: List[Tuple[float, float]] = []
    short_good_spans: List[Tuple[float, float]] = []
    n_good_runs = 0
    n_skipped_tail = 0

    # 坏段区间
    for bad_start, bad_end in _iter_kept_runs(~mask):
        bad_spans.append(_span_s(bad_start, bad_end))

    # 候选窗：先算功率与质量指标，再统一门控
    cand_i0: List[int] = []
    cand_i1: List[int] = []
    cand_rms: List[float] = []
    cand_ptp: List[float] = []
    cand_total: List[float] = []
    cand_abs: List[Dict[str, float]] = []
    cand_rel: List[Dict[str, float]] = []

    for run_start, run_end in _iter_kept_runs(mask):
        n_good_runs += 1
        run_len = run_end - run_start
        n_wins = int(run_len // n_per_win)
        tail = int(run_len - n_wins * n_per_win)
        n_skipped_tail += tail
        if n_wins <= 0:
            short_good_spans.append(_span_s(run_start, run_end))
            continue
        for i in range(n_wins):
            i0 = run_start + i * n_per_win
            i1 = i0 + n_per_win
            seg = raw_arr[i0:i1]
            analysis, rms, ptp = compute_band_powers_with_metrics(
                seg,
                sample_rate=fs,
                welch_seconds=welch_seconds,
                edge_trim_sec=edge_trim,
            )
            cand_i0.append(i0)
            cand_i1.append(i1)
            cand_rms.append(rms)
            cand_ptp.append(ptp)
            cand_total.append(float(analysis.result.total_power))
            cand_abs.append(dict(analysis.result.absolute))
            cand_rel.append(dict(analysis.result.relative))
        if tail > 0:
            short_good_spans.append(_span_s(run_end - tail, run_end))

    n_cand = len(cand_i0)
    gated_spans: List[Tuple[float, float]] = []
    if n_cand == 0:
        empty = np.zeros(0, dtype=np.float64)
        z = {name: empty.copy() for name in EEG_BANDS}
        extra = {
            "n_good_runs": n_good_runs,
            "n_skipped_tail_samples": n_skipped_tail,
            "n_removed": int(np.count_nonzero(mask)),
            "n_candidates": 0,
            "n_gated": 0,
            "edge_trim_sec": edge_trim,
            "x_as_time_s": True,
            "bad_spans": bad_spans,
            "short_good_spans": short_good_spans,
            "gated_spans": gated_spans,
            "total_duration_s": total_s,
        }
        return empty, z, {name: empty.copy() for name in EEG_BANDS}, extra

    rms_arr = np.asarray(cand_rms, dtype=np.float64)
    ptp_arr = np.asarray(cand_ptp, dtype=np.float64)
    tot_arr = np.asarray(cand_total, dtype=np.float64)
    outlier = (
        _robust_high_outlier_mask(rms_arr)
        | _robust_high_outlier_mask(ptp_arr)
        | _robust_high_outlier_mask(tot_arr)
    )

    starts_s: List[float] = []
    abs_lists: Dict[str, List[float]] = {name: [] for name in EEG_BANDS}
    rel_lists: Dict[str, List[float]] = {name: [] for name in EEG_BANDS}
    for k in range(n_cand):
        if outlier[k]:
            gated_spans.append(_span_s(cand_i0[k], cand_i1[k]))
            continue
        starts_s.append(float(cand_i0[k]) / fs)
        for name in EEG_BANDS:
            abs_lists[name].append(float(cand_abs[k][name]))
            rel_lists[name].append(float(cand_rel[k][name]))

    x_axis = np.asarray(starts_s, dtype=np.float64)
    absolute = {
        name: np.asarray(abs_lists[name], dtype=np.float64) for name in EEG_BANDS
    }
    relative = {
        name: np.asarray(rel_lists[name], dtype=np.float64) for name in EEG_BANDS
    }
    extra = {
        "n_good_runs": n_good_runs,
        "n_skipped_tail_samples": n_skipped_tail,
        "n_removed": int(np.count_nonzero(mask)),
        "n_candidates": n_cand,
        "n_gated": int(np.count_nonzero(outlier)),
        "edge_trim_sec": edge_trim,
        "x_as_time_s": True,
        "bad_spans": bad_spans,
        "short_good_spans": short_good_spans,
        "gated_spans": gated_spans,
        "total_duration_s": total_s,
    }
    return x_axis, absolute, relative, extra


def plot_minute_band_powers(
    minutes: np.ndarray,
    *,
    absolute: Optional[Dict[str, np.ndarray]] = None,
    relative: Optional[Dict[str, np.ndarray]] = None,
    title: str = "",
    save_path: Optional[Path] = None,
    figure=None,
    window_sec: float = MINUTE_SEC,
    x_as_time_s: bool = False,
    bad_spans: Optional[List[Tuple[float, float]]] = None,
    short_good_spans: Optional[List[Tuple[float, float]]] = None,
    gated_spans: Optional[List[Tuple[float, float]]] = None,
    total_duration_s: Optional[float] = None,
) -> None:
    """绘制按固定窗长切片的绝对/相对功率折线；可嵌入 Qt figure。

    x_as_time_s=True 时 minutes 视为墙钟起始秒。
    bad_spans / short_good_spans / gated_spans：不拼接模式下的墙钟标注区间 (t0, t1)。
    """
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    panels: List[Tuple[str, Dict[str, np.ndarray], str]] = []
    if absolute is not None:
        panels.append(("绝对功率", absolute, "绝对功率"))
    if relative is not None:
        panels.append(("相对功率 (%)", relative, "相对功率 (%)"))
    # 无功率点但有区间标注时仍画空面板，便于看坏段/过短段
    if not panels:
        if x_as_time_s and (bad_spans or short_good_spans or gated_spans):
            panels.append(("绝对功率", {name: np.zeros(0) for name in EEG_BANDS}, "绝对功率"))
        else:
            raise ValueError("须至少提供 absolute 或 relative")

    own_figure = figure is None
    n = len(panels)
    if own_figure:
        fig, axes = plt.subplots(n, 1, figsize=(10, 4.2 * n), dpi=120, squeeze=False)
    else:
        fig = figure
        fig.clear()
        axes = fig.subplots(n, 1, squeeze=False)

    win = float(window_sec)
    x = np.asarray(minutes, dtype=np.float64)
    bad_spans = list(bad_spans or [])
    short_good_spans = list(short_good_spans or [])
    gated_spans = list(gated_spans or [])

    # 墙钟横轴：总时长 < 10 分钟用秒；≥ 10 分钟用分钟
    use_minutes_axis = False
    time_scale = 1.0
    if x_as_time_s:
        dur_s = 0.0
        if total_duration_s is not None and total_duration_s > 0:
            dur_s = float(total_duration_s)
        else:
            candidates = []
            if x.size:
                candidates.append(float(x[-1]) + win)
            for t0, t1 in bad_spans + short_good_spans + gated_spans:
                candidates.extend([float(t0), float(t1)])
            if candidates:
                dur_s = max(candidates)
        use_minutes_axis = dur_s >= 600.0  # 10 分钟
        time_scale = 1.0 / 60.0 if use_minutes_axis else 1.0
        if use_minutes_axis:
            xlabel = f"墙钟时间 (min)，功率窗每 {win:g} s"
        else:
            xlabel = f"墙钟时间 (s)，功率窗每 {win:g} s"
    elif abs(win - 60.0) < 1e-6:
        xlabel = "分钟"
    else:
        xlabel = f"窗序号（每 {win:g} s）"

    def _tx(t: float) -> float:
        return float(t) * time_scale

    x_plot = x * time_scale if x_as_time_s and x.size else x
    win_plot = win * time_scale if x_as_time_s else win

    for row, (ylabel, series, _kind) in enumerate(panels):
        ax = axes[row, 0]
        # 先画区间底色，再画折线
        labeled_bad = False
        labeled_short = False
        labeled_gated = False
        for t0, t1 in bad_spans:
            ax.axvspan(
                _tx(t0),
                _tx(t1),
                color="#E53935",
                alpha=0.22,
                lw=0,
                label="坏段(跳过)" if not labeled_bad else None,
            )
            labeled_bad = True
        for t0, t1 in short_good_spans:
            ax.axvspan(
                _tx(t0),
                _tx(t1),
                color="#F9A825",
                alpha=0.28,
                lw=0,
                label=f"好段过短(<{win:g}s)" if not labeled_short else None,
            )
            labeled_short = True
        for t0, t1 in gated_spans:
            ax.axvspan(
                _tx(t0),
                _tx(t1),
                color="#8E24AA",
                alpha=0.22,
                lw=0,
                label="质量门控丢弃" if not labeled_gated else None,
            )
            labeled_gated = True

        for name in EEG_BANDS:
            y = series[name]
            if y.size == 0 or x_plot.size == 0:
                continue
            if ylabel.startswith("相对"):
                y = y * 100.0
            ax.plot(
                x_plot,
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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        # 横轴范围：覆盖功率点与标注区间
        x_candidates: List[float] = []
        if x_plot.size:
            x_candidates.extend([float(x_plot[0]), float(x_plot[-1])])
            if x_as_time_s:
                x_candidates.append(float(x_plot[-1]) + win_plot)
        for t0, t1 in bad_spans + short_good_spans + gated_spans:
            x_candidates.extend([_tx(t0), _tx(t1)])
        if total_duration_s is not None and total_duration_s > 0:
            x_candidates.extend([0.0, _tx(float(total_duration_s))])
        if x_candidates:
            xmin, xmax = min(x_candidates), max(x_candidates)
            pad = max(win_plot * 0.05, 0.02 if use_minutes_axis else 1.0) if x_as_time_s else 0.5
            if x_as_time_s:
                ax.set_xlim(max(0.0, xmin - pad), xmax + pad)
            else:
                ax.set_xlim(0.5, float(x[-1]) + 0.5 if x.size else xmax + pad)

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
    *,
    window_sec: float = MINUTE_SEC,
    x_as_time_s: bool = False,
) -> None:
    rows = []
    for i, minute in enumerate(minutes):
        if x_as_time_s:
            row = {
                "window": i + 1,
                "start_time_s": float(minute),
                "window_sec": float(window_sec),
            }
        else:
            row = {"window": int(minute), "window_sec": float(window_sec)}
        for name in EEG_BANDS:
            row[f"{name}_absolute"] = float(absolute[name][i])
            row[f"{name}_relative"] = float(relative[name][i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def prepare_cleaned_minute_powers(
    source: Path,
    *,
    window_sec: float = MINUTE_SEC,
    no_splice: bool = False,
    skip_rejection: bool = False,
) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """加载 →（可选）质量标记 → 按窗长算绝对/相对功率。

    skip_rejection=True：不做坏段剔除，直接在全量 raw 上切窗。
    no_splice=False：剔坏硬拼（中位对齐）后再切窗。
    no_splice=True：不拼接，仅在连续好段内切窗；横轴为墙钟起始秒。

    返回 (x_axis, absolute, relative, meta)
    """
    raw, sample_rate, source_desc = load_raw_for_minute_analysis(source)
    win = float(window_sec)

    if skip_rejection:
        minutes, absolute, relative = compute_minute_band_powers(
            raw.astype(np.float64), sample_rate, window_sec=win
        )
        meta = {
            "sample_rate": float(sample_rate),
            "n_raw": int(raw.size),
            "n_cleaned": int(raw.size),
            "n_removed": 0,
            "n_minutes": int(minutes.size),
            "window_sec": win,
            "source_desc": source_desc,
            "no_splice": False,
            "x_as_time_s": False,
            "skip_rejection": True,
        }
        return minutes, absolute, relative, meta

    quality = build_threshold_rejection(raw, sample_rate)
    remove_mask = build_raw_remove_mask(quality, int(raw.size), remove_suspicious=True)

    if no_splice:
        x_axis, absolute, relative, extra = compute_band_powers_good_runs(
            raw.astype(np.float64),
            remove_mask,
            sample_rate,
            window_sec=win,
        )
        n_removed = int(extra["n_removed"])
        meta = {
            "sample_rate": float(sample_rate),
            "n_raw": int(raw.size),
            "n_cleaned": int(raw.size) - n_removed,
            "n_removed": n_removed,
            "n_minutes": int(x_axis.size),
            "window_sec": win,
            "source_desc": source_desc,
            "no_splice": True,
            "x_as_time_s": True,
            "n_good_runs": int(extra["n_good_runs"]),
            "n_candidates": int(extra.get("n_candidates", x_axis.size)),
            "n_gated": int(extra.get("n_gated", 0)),
            "edge_trim_sec": float(extra.get("edge_trim_sec", 0.0)),
            "bad_spans": extra.get("bad_spans", []),
            "short_good_spans": extra.get("short_good_spans", []),
            "gated_spans": extra.get("gated_spans", []),
            "total_duration_s": float(extra.get("total_duration_s", 0.0)),
        }
        return x_axis, absolute, relative, meta

    cleaned, _, n_removed = clean_raw_signal(
        raw.astype(np.int64), quality, sample_rate=sample_rate
    )
    cleaned = cleaned.astype(np.float64)
    minutes, absolute, relative = compute_minute_band_powers(
        cleaned, sample_rate, window_sec=win
    )
    meta = {
        "sample_rate": float(sample_rate),
        "n_raw": int(raw.size),
        "n_cleaned": int(cleaned.size),
        "n_removed": int(n_removed),
        "n_minutes": int(minutes.size),
        "window_sec": win,
        "source_desc": source_desc,
        "no_splice": False,
        "x_as_time_s": False,
    }
    return minutes, absolute, relative, meta


def run_minute_band_power_analysis(
    session_dir: Path,
    *,
    save_outputs: bool = True,
    window_sec: float = MINUTE_SEC,
    no_splice: bool = False,
    skip_rejection: bool = False,
) -> MinuteBandPowerResult:
    session_dir = Path(session_dir)
    win = float(window_sec)
    minutes, absolute, relative, meta = prepare_cleaned_minute_powers(
        session_dir,
        window_sec=win,
        no_splice=no_splice,
        skip_rejection=skip_rejection,
    )
    n_minutes = int(meta["n_minutes"])
    sample_rate = float(meta["sample_rate"])
    cleaned_sec = (
        float(meta["n_cleaned"] / sample_rate) if sample_rate > 0 else 0.0
    )
    x_as_time = bool(meta.get("x_as_time_s", False))
    if skip_rejection:
        mode_desc = "全量不剔坏后切窗"
    elif no_splice:
        mode_desc = "不拼接·连续好段内切窗"
    else:
        mode_desc = "剔坏硬拼后再切窗"

    lines = [
        f"长时记录 · 每 {win:g} s 节律绝对/相对功率（{mode_desc}）",
        f"会话目录: {session_dir.resolve()}",
        f"数据来源: {meta['source_desc']}",
        f"拼接 raw: {meta['n_raw']} 点 @ {sample_rate:.3f} Hz",
        f"坏段标记: 剔除/跳过 {meta['n_removed']} 点；"
        f"有效约 {cleaned_sec / 60.0:.2f} 分钟",
        f"完整窗数 N: {n_minutes}（按 {win:g} s；不足一窗的尾部丢弃）",
        (
            f"处理: chunk拼接(若有) + 全量不剔坏 + Welch"
            if skip_rejection
            else f"处理: chunk拼接(若有) + build_threshold_rejection + {mode_desc} + Welch"
        ),
    ]
    if no_splice:
        lines.append(f"连续好段数: {meta.get('n_good_runs', 0)}")
        lines.append(
            f"坏段区间数: {len(meta.get('bad_spans', []))}；"
            f"过短好段/尾部数: {len(meta.get('short_good_spans', []))}"
        )
        lines.append(
            f"窗两端去边: 每侧 {float(meta.get('edge_trim_sec', 0.0)):g} s；"
            f"候选窗 {int(meta.get('n_candidates', n_minutes))}，"
            f"质量门控丢弃 {int(meta.get('n_gated', 0))}，"
            f"保留 N={n_minutes}"
        )
    if n_minutes == 0:
        lines.append(f"有效数据不足 {win:g} s，未生成功率点。")

    csv_path = None
    plot_path = None
    has_regions = bool(
        meta.get("bad_spans")
        or meta.get("short_good_spans")
        or meta.get("gated_spans")
    )
    if save_outputs and (n_minutes > 0 or (no_splice and has_regions)):
        plot_path = session_dir / PLOT_NAME
        if n_minutes > 0:
            csv_path = session_dir / CSV_NAME
            _save_csv(
                csv_path,
                minutes,
                absolute,
                relative,
                window_sec=win,
                x_as_time_s=x_as_time,
            )
            lines.append(f"CSV: {csv_path.name}")
        plot_minute_band_powers(
            minutes,
            absolute=absolute if n_minutes > 0 else None,
            title=(
                f"每 {win:g} s 节律绝对功率（{mode_desc}，N={n_minutes}）"
            ),
            save_path=plot_path,
            window_sec=win,
            x_as_time_s=x_as_time,
            bad_spans=meta.get("bad_spans") if no_splice else None,
            short_good_spans=meta.get("short_good_spans") if no_splice else None,
            gated_spans=meta.get("gated_spans") if no_splice else None,
            total_duration_s=meta.get("total_duration_s") if no_splice else None,
        )
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
