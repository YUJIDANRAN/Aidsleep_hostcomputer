"""
粉噪刺激前后短窗 Alpha 功率 + burst 锁定叠加。

默认窗口（更贴近瞬态效应，避免 200 ms 跨多拍冲淡）:
  前窗 baseline = [t-100 ms, t)
  后窗 early   = (t, t+100 ms]

另输出 burst 锁定平均 Alpha / |Alpha|（-200～+300 ms）。

用法:
    python BurstAlphaPowerStats.py --session Result/xxx
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_ALGO_DIR = Path(__file__).resolve().parent
_ROOT = _ALGO_DIR.parent
for _p in (_ROOT, _ALGO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from MovementArtifact import build_raw_remove_mask, build_threshold_rejection  # noqa: E402
from TroughCalibrator import (  # noqa: E402
    BURST_CSV_NAME,
    EEG_FULL_CSV_NAME,
    BurstEvent,
    filter_alpha_stream,
    read_burst_csv,
    read_eeg_csv,
)

PRE_WINDOW_SEC = (-0.100, 0.0)  ## [t+pre0, t+pre1)
POST_WINDOW_SEC = (0.0, 0.100)  ## (t+post0, t+post1] 用 (t+post0, t+post1) 取样
ERP_EARLY_SEC = (0.050, 0.150)  ## 额外报告：ERP 常见早期窗
LOCK_PRE_SEC = 0.200
LOCK_POST_SEC = 0.300
EPS_RATIO = 0.05  ## |ratio-1| < 5% 视为持平（短窗波动大）
MAX_BAD_FRACTION = 0.30
DETAIL_CSV_NAME = "burst_alpha_power_prepost.csv"
REPORT_NAME = "burst_alpha_power_report.txt"
HIST_PLOT_NAME = "burst_alpha_power_ratio_hist.png"
LOCK_PLOT_NAME = "burst_alpha_locked_average.png"


@dataclass
class BurstPowerTrial:
    burst_index: int
    sample_index: int
    time_s: float
    power_pre: float
    power_post: float
    power_erp: float
    ratio: float
    ratio_erp: float
    delta: float
    change: str  ## "up" | "down" | "flat"
    change_erp: str
    valid: bool
    reason: str = ""


@dataclass
class BurstPowerStatsResult:
    session_dir: Path
    sample_rate: float
    pre_window_sec: Tuple[float, float]
    post_window_sec: Tuple[float, float]
    erp_window_sec: Tuple[float, float]
    n_bursts: int
    n_valid: int
    n_up: int
    n_down: int
    n_flat: int
    n_up_erp: int
    n_down_erp: int
    n_flat_erp: int
    mean_ratio: float
    median_ratio: float
    mean_delta: float
    median_delta: float
    mean_power_pre: float
    mean_power_post: float
    median_ratio_erp: float
    mean_locked_peak_ms: float
    trials: List[BurstPowerTrial] = field(default_factory=list)
    report_text: str = ""


def _setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _sec_to_index_span(
    center: int,
    window: Tuple[float, float],
    sample_rate: float,
    n_total: int,
    *,
    exclusive_end: bool = True,
) -> Tuple[int, int]:
    start = int(round(center + window[0] * sample_rate))
    end = int(round(center + window[1] * sample_rate))
    if exclusive_end:
        end = max(start + 1, end)
    else:
        end = max(start + 1, end + 1)
    start = max(0, start)
    end = min(n_total, end)
    return start, end


def _window_power(
    alpha: np.ndarray,
    remove_mask: np.ndarray,
    start: int,
    end: int,
) -> tuple[Optional[float], str]:
    """[start, end) 上 Alpha 均方功率；坏点过多则失败。"""
    if start < 0 or end > alpha.size or start >= end:
        return None, "out_of_range"
    segment = alpha[start:end]
    bad = remove_mask[start:end]
    bad_frac = float(np.mean(bad)) if bad.size else 1.0
    if bad_frac > MAX_BAD_FRACTION:
        return None, f"bad_frac={bad_frac:.0%}"
    keep = ~bad
    if not np.any(keep):
        return None, "all_bad"
    vals = segment[keep]
    power = float(np.mean(vals * vals))
    if not np.isfinite(power) or power < 0:
        return None, "invalid_power"
    return power, ""


def classify_change(ratio: float, eps: float = EPS_RATIO) -> str:
    if ratio > 1.0 + eps:
        return "up"
    if ratio < 1.0 - eps:
        return "down"
    return "flat"


def analyze_burst_alpha_power(
    bursts: Sequence[BurstEvent],
    alpha: np.ndarray,
    sample_rate: float,
    remove_mask: np.ndarray,
    *,
    pre_window_sec: Tuple[float, float] = PRE_WINDOW_SEC,
    post_window_sec: Tuple[float, float] = POST_WINDOW_SEC,
    erp_window_sec: Tuple[float, float] = ERP_EARLY_SEC,
) -> List[BurstPowerTrial]:
    trials: List[BurstPowerTrial] = []
    n = alpha.size
    for event in bursts:
        idx = int(event.sample_index)
        pre_start, pre_end = _sec_to_index_span(idx, pre_window_sec, sample_rate, n)
        # 后窗从 idx+1 起，避开触发点本身
        post_start, post_end = _sec_to_index_span(
            idx, (max(post_window_sec[0], 1.0 / sample_rate), post_window_sec[1]), sample_rate, n
        )
        erp_start, erp_end = _sec_to_index_span(idx, erp_window_sec, sample_rate, n)

        power_pre, reason_pre = _window_power(alpha, remove_mask, pre_start, pre_end)
        power_post, reason_post = _window_power(alpha, remove_mask, post_start, post_end)
        power_erp, reason_erp = _window_power(alpha, remove_mask, erp_start, erp_end)

        if power_pre is None or power_post is None:
            reason = reason_pre or reason_post or reason_erp or "invalid"
            trials.append(
                BurstPowerTrial(
                    burst_index=event.burst_index,
                    sample_index=idx,
                    time_s=float(event.time_s),
                    power_pre=float("nan"),
                    power_post=float("nan"),
                    power_erp=float("nan"),
                    ratio=float("nan"),
                    ratio_erp=float("nan"),
                    delta=float("nan"),
                    change="",
                    change_erp="",
                    valid=False,
                    reason=reason,
                )
            )
            continue
        if power_pre <= 0:
            trials.append(
                BurstPowerTrial(
                    burst_index=event.burst_index,
                    sample_index=idx,
                    time_s=float(event.time_s),
                    power_pre=power_pre,
                    power_post=power_post,
                    power_erp=float(power_erp) if power_erp is not None else float("nan"),
                    ratio=float("nan"),
                    ratio_erp=float("nan"),
                    delta=power_post - power_pre,
                    change="",
                    change_erp="",
                    valid=False,
                    reason="pre_power_zero",
                )
            )
            continue

        ratio = power_post / power_pre
        ratio_erp = (
            (power_erp / power_pre)
            if power_erp is not None and power_erp >= 0
            else float("nan")
        )
        trials.append(
            BurstPowerTrial(
                burst_index=event.burst_index,
                sample_index=idx,
                time_s=float(event.time_s),
                power_pre=power_pre,
                power_post=power_post,
                power_erp=float(power_erp) if power_erp is not None else float("nan"),
                ratio=ratio,
                ratio_erp=ratio_erp,
                delta=power_post - power_pre,
                change=classify_change(ratio),
                change_erp=classify_change(ratio_erp) if np.isfinite(ratio_erp) else "",
                valid=True,
            )
        )
    return trials


def extract_locked_epochs(
    bursts: Sequence[BurstEvent],
    signal: np.ndarray,
    remove_mask: np.ndarray,
    sample_rate: float,
    *,
    pre_sec: float = LOCK_PRE_SEC,
    post_sec: float = LOCK_POST_SEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (times_ms, epochs[n_trial, n_samp])，坏段试次丢弃。"""
    n_pre = int(round(pre_sec * sample_rate))
    n_post = int(round(post_sec * sample_rate))
    width = n_pre + n_post + 1
    times_ms = (np.arange(width) - n_pre) * (1000.0 / sample_rate)
    rows: List[np.ndarray] = []
    for event in bursts:
        idx = int(event.sample_index)
        start = idx - n_pre
        end = idx + n_post + 1
        if start < 0 or end > signal.size:
            continue
        if float(np.mean(remove_mask[start:end])) > MAX_BAD_FRACTION:
            continue
        epoch = signal[start:end].astype(np.float64).copy()
        bad = remove_mask[start:end]
        if np.any(bad):
            # 坏点用线性插值，避免整段丢弃过多
            good = ~bad
            if np.count_nonzero(good) < max(8, width // 4):
                continue
            x = np.arange(width)
            epoch[bad] = np.interp(x[bad], x[good], epoch[good])
        # 相对基线（burst 前 100 ms）去均值，突出锁定响应
        bas0 = n_pre - int(round(0.100 * sample_rate))
        bas0 = max(0, bas0)
        epoch = epoch - float(np.mean(epoch[bas0:n_pre]))
        rows.append(epoch)
    if not rows:
        return times_ms, np.zeros((0, width), dtype=np.float64)
    return times_ms, np.vstack(rows)


def _save_detail_csv(path: Path, trials: Sequence[BurstPowerTrial]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "burst_index",
                "sample_index",
                "time_s",
                "power_pre",
                "power_post_0_100ms",
                "power_erp_50_150ms",
                "ratio_post_over_pre",
                "ratio_erp_over_pre",
                "delta_post_minus_pre",
                "change_0_100ms",
                "change_50_150ms",
                "valid",
                "reason",
            ]
        )
        for trial in trials:
            writer.writerow(
                [
                    trial.burst_index,
                    trial.sample_index,
                    f"{trial.time_s:.6f}",
                    "" if not np.isfinite(trial.power_pre) else f"{trial.power_pre:.8g}",
                    "" if not np.isfinite(trial.power_post) else f"{trial.power_post:.8g}",
                    "" if not np.isfinite(trial.power_erp) else f"{trial.power_erp:.8g}",
                    "" if not np.isfinite(trial.ratio) else f"{trial.ratio:.6f}",
                    "" if not np.isfinite(trial.ratio_erp) else f"{trial.ratio_erp:.6f}",
                    "" if not np.isfinite(trial.delta) else f"{trial.delta:.8g}",
                    trial.change,
                    trial.change_erp,
                    int(trial.valid),
                    trial.reason,
                ]
            )


def _save_ratio_hist(
    path: Path,
    ratios: np.ndarray,
    *,
    title: str,
    xlabel: str,
) -> None:
    if ratios.size == 0:
        return
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(ratios, bins=min(30, max(8, ratios.size // 3)), color="#4C78A8", edgecolor="white")
    ax.axvline(1.0, color="#E45756", ls="--", lw=1.2, label="ratio = 1")
    ax.axvline(float(np.median(ratios)), color="#F58518", ls="-", lw=1.2, label="median")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("次数")
    ax.set_title(title)
    ax.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_locked_average_plot(
    path: Path,
    times_ms: np.ndarray,
    epochs_alpha: np.ndarray,
    epochs_env: np.ndarray,
) -> float:
    """绘制锁定平均；返回 |alpha| 平均曲线在 0～150 ms 内峰值时刻(ms)。"""
    import matplotlib.pyplot as plt

    _setup_matplotlib()
    peak_ms = float("nan")
    if epochs_alpha.size == 0:
        return peak_ms

    mean_a = np.mean(epochs_alpha, axis=0)
    sem_a = np.std(epochs_alpha, axis=0, ddof=1) / np.sqrt(epochs_alpha.shape[0])
    mean_e = np.mean(epochs_env, axis=0)
    sem_e = np.std(epochs_env, axis=0, ddof=1) / np.sqrt(epochs_env.shape[0])

    mask = (times_ms >= 0.0) & (times_ms <= 150.0)
    if np.any(mask):
        local = mean_e[mask]
        local_t = times_ms[mask]
        peak_ms = float(local_t[int(np.argmax(local))])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    ax0, ax1 = axes
    ax0.plot(times_ms, mean_a, color="#4C78A8", lw=1.6, label="mean alpha")
    ax0.fill_between(times_ms, mean_a - sem_a, mean_a + sem_a, color="#4C78A8", alpha=0.25)
    ax0.axvline(0.0, color="#E45756", ls="--", lw=1.0, label="burst")
    ax0.axvspan(0.0, 100.0, color="#54A24B", alpha=0.08, label="post 0-100 ms")
    ax0.axvspan(50.0, 150.0, color="#F58518", alpha=0.08, label="ERP 50-150 ms")
    ax0.set_ylabel("Alpha (基线校正)")
    ax0.set_title(f"Burst 锁定平均 Alpha  n={epochs_alpha.shape[0]}")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1.plot(times_ms, mean_e, color="#B279A2", lw=1.6, label="mean |alpha|")
    ax1.fill_between(times_ms, mean_e - sem_e, mean_e + sem_e, color="#B279A2", alpha=0.25)
    ax1.axvline(0.0, color="#E45756", ls="--", lw=1.0)
    if np.isfinite(peak_ms):
        ax1.axvline(peak_ms, color="#F58518", ls=":", lw=1.2, label=f"peak {peak_ms:.0f} ms")
    ax1.set_xlabel("相对 burst 时间 (ms)")
    ax1.set_ylabel("|Alpha|")
    ax1.set_title("瞬时幅度包络锁定平均")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return peak_ms


def format_report(result: BurstPowerStatsResult) -> str:
    n = max(result.n_valid, 1)
    pre = result.pre_window_sec
    post = result.post_window_sec
    erp = result.erp_window_sec
    lines = [
        "粉噪刺激前后 Alpha 功率 / 锁定平均统计",
        f"会话: {result.session_dir}",
        f"采样率: {result.sample_rate:g} Hz",
        (
            f"基线窗: [{pre[0]*1000:.0f}, {pre[1]*1000:.0f}) ms  |  "
            f"后窗: ({post[0]*1000:.0f}, {post[1]*1000:.0f}] ms  |  "
            f"ERP窗: [{erp[0]*1000:.0f}, {erp[1]*1000:.0f}] ms"
        ),
        "功率定义: 因果 Alpha 滤波后均方值 mean(alpha^2)",
        "判读优先看 median 与增减次数；mean 比值易被离群值拉偏。",
        f"总 burst: {result.n_bursts}  |  有效: {result.n_valid}",
        "",
        f"[后窗 0-100 ms vs 基线]  ↑{result.n_up} ({result.n_up / n:.1%})  "
        f"↓{result.n_down} ({result.n_down / n:.1%})  "
        f"→{result.n_flat} ({result.n_flat / n:.1%})",
        f"  功率比 post/pre:  mean={result.mean_ratio:.3f}  median={result.median_ratio:.3f}",
        f"  功率差 post-pre:  mean={result.mean_delta:.4g}  median={result.median_delta:.4g}",
        f"  平均功率: pre={result.mean_power_pre:.4g}  post={result.mean_power_post:.4g}",
        "",
        f"[ERP窗 50-150 ms vs 基线]  ↑{result.n_up_erp} ({result.n_up_erp / n:.1%})  "
        f"↓{result.n_down_erp} ({result.n_down_erp / n:.1%})  "
        f"→{result.n_flat_erp} ({result.n_flat_erp / n:.1%})",
        f"  功率比 erp/pre median={result.median_ratio_erp:.3f}",
        "",
        f"锁定平均 |alpha| 在 0-150 ms 峰值时刻: {result.mean_locked_peak_ms:.1f} ms"
        if np.isfinite(result.mean_locked_peak_ms)
        else "锁定平均: 有效试次不足，未估计峰值",
    ]
    # 粗判
    median = result.median_ratio
    if result.n_valid >= 10 and np.isfinite(median):
        if abs(median - 1.0) < 0.08 and abs(result.n_up - result.n_down) / n < 0.20:
            lines.append("粗判: 未见稳定 Alpha 功率变化（接近无效应）。")
        elif median >= 1.08 and result.n_up > result.n_down:
            lines.append("粗判: 倾向刺激后短窗 Alpha 功率升高。")
        elif median <= 0.92 and result.n_down > result.n_up:
            lines.append("粗判: 倾向刺激后短窗 Alpha 功率降低。")
        else:
            lines.append("粗判: 方向不稳定，建议结合锁定平均图看瞬态形态。")
    return "\n".join(lines)


def run_burst_alpha_power_stats(
    session_dir: Path,
    *,
    eeg_path: Optional[Path] = None,
    bursts_path: Optional[Path] = None,
    pre_window_sec: Tuple[float, float] = PRE_WINDOW_SEC,
    post_window_sec: Tuple[float, float] = POST_WINDOW_SEC,
    erp_window_sec: Tuple[float, float] = ERP_EARLY_SEC,
    save_outputs: bool = True,
) -> BurstPowerStatsResult:
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
    alpha = filter_alpha_stream(raw, sample_rate)
    trials = analyze_burst_alpha_power(
        bursts,
        alpha,
        sample_rate,
        remove_mask,
        pre_window_sec=pre_window_sec,
        post_window_sec=post_window_sec,
        erp_window_sec=erp_window_sec,
    )

    valid = [t for t in trials if t.valid]
    ratios = np.asarray([t.ratio for t in valid], dtype=np.float64)
    ratios_erp = np.asarray(
        [t.ratio_erp for t in valid if np.isfinite(t.ratio_erp)], dtype=np.float64
    )
    deltas = np.asarray([t.delta for t in valid], dtype=np.float64)
    powers_pre = np.asarray([t.power_pre for t in valid], dtype=np.float64)
    powers_post = np.asarray([t.power_post for t in valid], dtype=np.float64)
    n_up = sum(1 for t in valid if t.change == "up")
    n_down = sum(1 for t in valid if t.change == "down")
    n_flat = sum(1 for t in valid if t.change == "flat")
    n_up_erp = sum(1 for t in valid if t.change_erp == "up")
    n_down_erp = sum(1 for t in valid if t.change_erp == "down")
    n_flat_erp = sum(1 for t in valid if t.change_erp == "flat")

    times_ms, epochs_a = extract_locked_epochs(
        bursts, alpha, remove_mask, sample_rate
    )
    epochs_env = np.abs(epochs_a) if epochs_a.size else epochs_a
    peak_ms = float("nan")

    result = BurstPowerStatsResult(
        session_dir=session_dir,
        sample_rate=sample_rate,
        pre_window_sec=pre_window_sec,
        post_window_sec=post_window_sec,
        erp_window_sec=erp_window_sec,
        n_bursts=len(bursts),
        n_valid=len(valid),
        n_up=n_up,
        n_down=n_down,
        n_flat=n_flat,
        n_up_erp=n_up_erp,
        n_down_erp=n_down_erp,
        n_flat_erp=n_flat_erp,
        mean_ratio=float(np.mean(ratios)) if ratios.size else float("nan"),
        median_ratio=float(np.median(ratios)) if ratios.size else float("nan"),
        mean_delta=float(np.mean(deltas)) if deltas.size else float("nan"),
        median_delta=float(np.median(deltas)) if deltas.size else float("nan"),
        mean_power_pre=float(np.mean(powers_pre)) if powers_pre.size else float("nan"),
        mean_power_post=float(np.mean(powers_post)) if powers_post.size else float("nan"),
        median_ratio_erp=float(np.median(ratios_erp)) if ratios_erp.size else float("nan"),
        mean_locked_peak_ms=peak_ms,
        trials=trials,
    )

    if save_outputs:
        _save_detail_csv(session_dir / DETAIL_CSV_NAME, trials)
        try:
            _save_ratio_hist(
                session_dir / HIST_PLOT_NAME,
                ratios,
                title=f"后窗0-100ms / 基线 功率比  n={result.n_valid}",
                xlabel="Alpha 功率比 (0-100 ms / 基线-100-0 ms)",
            )
        except Exception as exc:  # pragma: no cover
            print(f"保存直方图失败: {exc}")
        try:
            peak_ms = _save_locked_average_plot(
                session_dir / LOCK_PLOT_NAME,
                times_ms,
                epochs_a,
                epochs_env,
            )
            result.mean_locked_peak_ms = peak_ms
        except Exception as exc:  # pragma: no cover
            print(f"保存锁定平均图失败: {exc}")

    result.report_text = format_report(result)
    if save_outputs:
        (session_dir / REPORT_NAME).write_text(result.report_text + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="粉噪刺激前后 Alpha 短窗功率与锁定平均")
    parser.add_argument("--session", type=Path, required=True, help="会话目录")
    parser.add_argument("--pre-ms", type=float, default=100.0, help="基线窗长度(ms)，默认100")
    parser.add_argument("--post-ms", type=float, default=100.0, help="后窗长度(ms)，默认100")
    args = parser.parse_args()
    pre = max(10.0, args.pre_ms) / 1000.0
    post = max(10.0, args.post_ms) / 1000.0
    result = run_burst_alpha_power_stats(
        args.session,
        pre_window_sec=(-pre, 0.0),
        post_window_sec=(0.0, post),
    )
    print(result.report_text)


if __name__ == "__main__":
    main()
