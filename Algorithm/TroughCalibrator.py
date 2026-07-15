"""
助眠粉噪闭环：离线 Alpha 波谷对齐标定（方法 B）。

用法:
    python TroughCalibrator.py --session Result/eeg_20260707_175433
    python TroughCalibrator.py --sweep Result/run_92ms Result/run_95ms
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

_ALGO_DIR = Path(__file__).resolve().parent
_ROOT = _ALGO_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from MovementArtifact import build_raw_remove_mask, build_threshold_rejection  # noqa: E402
from iaf_echt import (  # noqa: E402
    estimate_iaf_from_series,
    find_echt_trough_indices,
    run_tracker_series,
)
from power_cal import DEFAULT_SAMPLE_RATE, RhythmStreamProcessor  # noqa: E402

BURST_CSV_NAME = "sleep_aid_bursts.csv"
EEG_FULL_CSV_NAME = "eeg_raw_full.csv"
REPORT_NAME = "calibration_report.txt"
HIST_PLOT_NAME = "burst_trough_delta_hist.png"


@dataclass
class BurstEvent:
    burst_index: int
    sample_index: int
    time_s: float
    total_latency_sec: float
    seconds_to_trough: float
    phase_rad: float
    inst_freq_hz: float


@dataclass
class TroughAlignment:
    burst_index: int
    sample_index: int
    predicted_trough_index: int
    nearest_trough_index: int
    delta_samples: int
    delta_sec: float
    valid: bool
    reason: str = ""


@dataclass
class CalibrationResult:
    session_dir: Path
    sample_rate: float
    n_bursts: int
    n_valid_alignments: int
    total_latency_sec: float
    mean_trough_delta_sec: float
    std_trough_delta_sec: float
    median_trough_delta_sec: float
    suggested_total_latency_sec: float
    reject_rate: float
    alignments: List[TroughAlignment] = field(default_factory=list)
    report_text: str = ""


def _setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def infer_sample_rate(time_s: np.ndarray, fallback: float = DEFAULT_SAMPLE_RATE) -> float:
    if time_s.size < 2:
        return float(fallback)
    dt = np.diff(time_s.astype(np.float64))
    dt = dt[dt > 0]
    if dt.size == 0:
        return float(fallback)
    return float(1.0 / np.median(dt))


def read_eeg_csv(path: Path) -> tuple[np.ndarray, float]:
    df = pd.read_csv(path)
    if "ch1_raw" not in df.columns:
        raise ValueError(f"EEG CSV 缺少 ch1_raw 列: {path}")
    raw = df["ch1_raw"].to_numpy(dtype=np.float64)
    if "time_s" in df.columns:
        fs = infer_sample_rate(df["time_s"].to_numpy())
    else:
        fs = float(DEFAULT_SAMPLE_RATE)
    return raw, fs


def read_burst_csv(path: Path) -> List[BurstEvent]:
    df = pd.read_csv(path)
    required = {
        "burst_index",
        "sample_index",
        "time_s",
        "total_latency_sec",
        "seconds_to_trough",
        "phase_rad",
        "inst_freq_hz",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"burst CSV 缺少列 {missing}: {path}")
    events: List[BurstEvent] = []
    for row in df.itertuples(index=False):
        events.append(
            BurstEvent(
                burst_index=int(row.burst_index),
                sample_index=int(row.sample_index),
                time_s=float(row.time_s),
                total_latency_sec=float(row.total_latency_sec),
                seconds_to_trough=float(row.seconds_to_trough),
                phase_rad=float(row.phase_rad),
                inst_freq_hz=float(row.inst_freq_hz),
            )
        )
    return events


def filter_alpha_stream(raw: np.ndarray, sample_rate: float) -> np.ndarray:
    """与在线 RhythmStreamProcessor(band='alpha') 等价的因果逐点滤波。"""
    proc = RhythmStreamProcessor(sample_rate=sample_rate)
    out = np.empty(raw.size, dtype=np.float64)
    for i, value in enumerate(raw):
        out[i] = proc.push(int(value), "alpha")
    return out


def find_alpha_trough_indices(alpha: np.ndarray, min_distance: int) -> np.ndarray:
    if alpha.size < 3:
        return np.array([], dtype=np.int64)
    troughs: list[int] = []
    for i in range(1, alpha.size - 1):
        if alpha[i] <= alpha[i - 1] and alpha[i] < alpha[i + 1]:
            if not troughs or i - troughs[-1] >= min_distance:
                troughs.append(i)
    return np.asarray(troughs, dtype=np.int64)


def nearest_trough_index(
    target_index: int,
    trough_indices: np.ndarray,
    max_distance: int,
) -> Optional[int]:
    if trough_indices.size == 0:
        return None
    dist = np.abs(trough_indices - target_index)
    idx = int(np.argmin(dist))
    if dist[idx] > max_distance:
        return None
    return int(trough_indices[idx])


def analyze_trough_alignment(
    bursts: Sequence[BurstEvent],
    trough_indices: np.ndarray,
    sample_rate: float,
    remove_mask: np.ndarray,
    n_samples: int,
    snapshots: Sequence | None = None,
) -> List[TroughAlignment]:
    """
    预测波谷优先用离线重放 IAFEcHTPhaseTracker（与在线同算法）；
    该点未 ready 时回退 CSV 的 seconds_to_trough。
    """
    max_trough_search = int(round(0.25 * sample_rate))
    results: List[TroughAlignment] = []

    for event in bursts:
        idx = event.sample_index
        if idx < 0 or idx >= n_samples or remove_mask[idx]:
            results.append(
                TroughAlignment(
                    burst_index=event.burst_index,
                    sample_index=idx,
                    predicted_trough_index=-1,
                    nearest_trough_index=-1,
                    delta_samples=0,
                    delta_sec=0.0,
                    valid=False,
                    reason="bad_or_out_of_range",
                )
            )
            continue

        seconds_to_trough = float(event.seconds_to_trough)
        if (
            snapshots is not None
            and 0 <= idx < len(snapshots)
            and getattr(snapshots[idx], "ready", False)
        ):
            seconds_to_trough = float(snapshots[idx].seconds_to_trough)

        predicted = int(round(idx + seconds_to_trough * sample_rate))
        predicted = int(np.clip(predicted, 0, n_samples - 1))
        nearest = nearest_trough_index(predicted, trough_indices, max_trough_search)
        if nearest is None:
            results.append(
                TroughAlignment(
                    burst_index=event.burst_index,
                    sample_index=idx,
                    predicted_trough_index=predicted,
                    nearest_trough_index=-1,
                    delta_samples=0,
                    delta_sec=0.0,
                    valid=False,
                    reason="no_near_trough",
                )
            )
            continue

        delta_samples = predicted - nearest
        results.append(
            TroughAlignment(
                burst_index=event.burst_index,
                sample_index=idx,
                predicted_trough_index=predicted,
                nearest_trough_index=nearest,
                delta_samples=delta_samples,
                delta_sec=delta_samples / sample_rate,
                valid=True,
            )
        )
    return results


def run_calibration(
    session_dir: Path,
    *,
    eeg_path: Optional[Path] = None,
    bursts_path: Optional[Path] = None,
    save_plots: bool = True,
) -> CalibrationResult:
    session_dir = Path(session_dir)
    eeg_path = Path(eeg_path or session_dir / EEG_FULL_CSV_NAME)
    bursts_path = Path(bursts_path or session_dir / BURST_CSV_NAME)

    if not eeg_path.is_file():
        raise FileNotFoundError(f"找不到 EEG 文件: {eeg_path}")
    if not bursts_path.is_file():
        raise FileNotFoundError(f"找不到 burst 文件: {bursts_path}")

    raw, sample_rate = read_eeg_csv(eeg_path)
    bursts = read_burst_csv(bursts_path)
    if not bursts:
        raise ValueError("burst 记录为空")

    quality = build_threshold_rejection(raw, sample_rate)
    remove_mask = build_raw_remove_mask(quality, raw.size)
    # 与在线一致：因果 Alpha → IAFEcHTPhaseTracker (IAF + ecHT)
    alpha = filter_alpha_stream(raw, sample_rate)
    snapshots = run_tracker_series(alpha, sample_rate)
    trough_indices = find_echt_trough_indices(snapshots, sample_rate)
    dominant_hz = estimate_iaf_from_series(alpha, sample_rate, bursts=bursts)
    alignments = analyze_trough_alignment(
        bursts,
        trough_indices,
        sample_rate,
        remove_mask,
        alpha.size,
        snapshots=snapshots,
    )
    valid_alignments = [a for a in alignments if a.valid]
    deltas = [a.delta_sec for a in valid_alignments]

    mean_delta = float(np.mean(deltas)) if deltas else float("nan")
    std_delta = float(np.std(deltas)) if deltas else float("nan")
    median_delta = float(np.median(deltas)) if deltas else float("nan")
    total_latency = float(np.median([b.total_latency_sec for b in bursts]))
    suggested = total_latency + (mean_delta if np.isfinite(mean_delta) else 0.0)

    lines = [
        f"会话目录: {session_dir.resolve()}",
        f"EEG: {eeg_path.name}  ({raw.size} 点 @ {sample_rate:.3f} Hz)",
        f"burst 数: {len(bursts)}  有效波谷对齐: {len(valid_alignments)}",
        f"IAF: {dominant_hz:.1f} Hz（Welch+1/f 去趋势，与在线一致）",
        f"检测到波谷数: {trough_indices.size}",
        f"伪迹拒绝率: {quality.reject_rate:.1%}",
        f"当前 total_latency 中位数: {total_latency * 1000:.1f} ms",
        "",
        "波谷对齐（预测=离线重放 IAF+ecHT；实际谷=同款相位越过 π）:",
        f"  mean(Δ):   {mean_delta * 1000:+.2f} ms",
        f"  median(Δ): {median_delta * 1000:+.2f} ms",
        f"  std(Δ):    {std_delta * 1000:.2f} ms",
        "  Δ > 0 → 预测波谷偏晚 → 建议增大 total_latency",
        "  Δ < 0 → 预测波谷偏早 → 建议减小 total_latency",
        "",
        f"建议 total_latency: {suggested * 1000:.1f} ms",
        "说明: 在线已改为 IAF + ecHT；评估用离线同款重放，口径一致。",
    ]
    report_text = "\n".join(lines)
    result = CalibrationResult(
        session_dir=session_dir,
        sample_rate=sample_rate,
        n_bursts=len(bursts),
        n_valid_alignments=len(valid_alignments),
        total_latency_sec=total_latency,
        mean_trough_delta_sec=mean_delta,
        std_trough_delta_sec=std_delta,
        median_trough_delta_sec=median_delta,
        suggested_total_latency_sec=suggested,
        reject_rate=quality.reject_rate,
        alignments=alignments,
        report_text=report_text,
    )

    (session_dir / REPORT_NAME).write_text(report_text + "\n", encoding="utf-8")
    if save_plots and valid_alignments:
        _save_delta_histogram(session_dir, valid_alignments)
    return result


def _save_delta_histogram(
    session_dir: Path, alignments: Sequence[TroughAlignment]
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    deltas_ms = [a.delta_sec * 1000.0 for a in alignments]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.hist(
        deltas_ms,
        bins=min(30, max(5, len(deltas_ms) // 2)),
        color="#E65100",
        alpha=0.85,
    )
    ax.axvline(0.0, color="#212121", linestyle="--", linewidth=0.8)
    ax.axvline(
        float(np.mean(deltas_ms)),
        color="#1565C0",
        linestyle="-",
        linewidth=1.0,
        label=f"均值 {np.mean(deltas_ms):+.1f} ms",
    )
    ax.set_title("预测波谷 - 实际波谷 (Δ ms)")
    ax.set_xlabel("Δ (ms)")
    ax.set_ylabel("试次数")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(session_dir / HIST_PLOT_NAME, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_latency_sweep(session_dirs: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for directory in session_dirs:
        result = run_calibration(directory, save_plots=True)
        rows.append(
            {
                "session": str(directory),
                "total_latency_ms": result.total_latency_sec * 1000.0,
                "suggested_latency_ms": result.suggested_total_latency_sec * 1000.0,
                "mean_delta_ms": result.mean_trough_delta_sec * 1000.0,
                "abs_mean_delta_ms": abs(result.mean_trough_delta_sec) * 1000.0,
                "n_bursts": result.n_bursts,
                "n_valid_alignments": result.n_valid_alignments,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        best_idx = int(df["abs_mean_delta_ms"].idxmin())
        df.loc[best_idx, "recommended"] = True
    return df


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha 波谷对齐 / total_latency 标定")
    parser.add_argument("--session", type=Path, help="会话目录")
    parser.add_argument("--eeg", type=Path, help="EEG CSV 路径")
    parser.add_argument("--bursts", type=Path, help="burst CSV 路径")
    parser.add_argument("--sweep", type=Path, nargs="+", help="多会话比较")
    parser.add_argument("--no-plots", action="store_true", help="不保存图表")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.sweep:
        df = compare_latency_sweep(args.sweep)
        out = Path(args.sweep[0]).parent / "latency_sweep_summary.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(df.to_string(index=False))
        print(f"\n汇总已保存: {out.resolve()}")
        return 0

    if args.session:
        result = run_calibration(
            args.session,
            eeg_path=args.eeg,
            bursts_path=args.bursts,
            save_plots=not args.no_plots,
        )
    elif args.eeg and args.bursts:
        result = run_calibration(
            args.eeg.parent,
            eeg_path=args.eeg,
            bursts_path=args.bursts,
            save_plots=not args.no_plots,
        )
    else:
        print("请指定 --session 或 (--eeg 与 --bursts)")
        return 1

    print(result.report_text)
    print(f"\n报告已保存: {(result.session_dir / REPORT_NAME).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
