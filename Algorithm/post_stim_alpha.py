"""粉噪刺激前后 α 功率对比（过程指标）。

刺激与伪 Sham 均以 α 正峰为时间零点（刺激侧：在粉噪 onset 邻近找对应正峰），
比较峰前/后短窗带内功率（包络平方）；按 P_pre 强度匹配伪 Sham。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert, iirnotch

from erp import (
    ERP_NOTCH_HZ,
    ERP_NOTCH_Q,
    detect_pink_seq_events,
    load_eeg_pink_csv,
)

ALPHA_BANDPASS_HZ = (8.0, 13.0)
POST_STIM_MIN_EVENTS = 8
POST_STIM_FIXED_W_SEC = 0.080
POST_STIM_EPOCH_PRE_SEC = 0.200
POST_STIM_EPOCH_POST_SEC = 0.400
POST_STIM_SHAM_EXCL_SEC = 0.150
## 在粉噪 onset 邻近搜索对应 α 正峰（优先 onset 前/处的峰）
POST_STIM_PEAK_SEARCH_PRE_SEC = 0.150
POST_STIM_PEAK_SEARCH_POST_SEC = 0.030
## 匹配伪 Sham：P_pre 相对容差（±20%）与最少配对数
POST_STIM_MATCH_REL_TOL = 0.20
POST_STIM_MATCH_MIN = 8


@dataclass(frozen=True)
class PowerChangeStats:
    n: int
    median_db: float
    mean_db: float
    p25_db: float
    p75_db: float
    frac_pos: float
    values_db: np.ndarray


@dataclass(frozen=True)
class PostStimAlphaAnalysis:
    sample_rate: float
    duration_s: float
    n_events: int
    n_alpha_cycles: int
    pink_per_cycle: float
    isi_median_ms: float
    isi_p10_ms: float
    isi_p90_ms: float
    alpha_band_hz: Tuple[float, float]
    window_ms: float
    stim: PowerChangeStats
    sham_peak: PowerChangeStats
    matched_stim: PowerChangeStats
    matched_sham: PowerChangeStats
    match_rate: float
    match_pre_ratio_median: float
    match_delta_db_diff: float
    ## onset − 邻近正峰；>0 表示发声落在峰后下降沿
    onset_minus_peak_median_ms: float
    times_ms: np.ndarray
    envelope_average: np.ndarray
    envelope_sem: np.ndarray
    envelope_rel_pct: np.ndarray
    sham_envelope_average: np.ndarray
    sham_envelope_sem: np.ndarray
    sham_envelope_rel_pct: np.ndarray
    matched_envelope_average: np.ndarray
    matched_envelope_sem: np.ndarray
    matched_envelope_rel_pct: np.ndarray
    note: str = ""


def _notch(eeg: np.ndarray, sample_rate: float) -> np.ndarray:
    y = np.asarray(eeg, dtype=np.float64)
    fs = float(sample_rate)
    if fs <= 0 or y.size < 24:
        return y
    nyq = 0.5 * fs
    for f0 in ERP_NOTCH_HZ:
        if f0 <= 0 or f0 >= nyq * 0.95:
            continue
        b, a = iirnotch(f0 / nyq, ERP_NOTCH_Q)
        y = filtfilt(b, a, y)
    return y


def _alpha_bandpass(
    eeg: np.ndarray,
    sample_rate: float,
    band_hz: Tuple[float, float] = ALPHA_BANDPASS_HZ,
) -> np.ndarray:
    fs = float(sample_rate)
    lo, hi = float(band_hz[0]), float(band_hz[1])
    nyq = 0.5 * fs
    if hi >= nyq * 0.95:
        hi = max(lo + 1.0, nyq * 0.45)
    if lo <= 0 or hi <= lo or eeg.size < 48:
        return np.asarray(eeg, dtype=np.float64)
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, np.asarray(eeg, dtype=np.float64))


def _power_stats(values_db: np.ndarray) -> PowerChangeStats:
    v = np.asarray(values_db, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return PowerChangeStats(
            n=0,
            median_db=float("nan"),
            mean_db=float("nan"),
            p25_db=float("nan"),
            p75_db=float("nan"),
            frac_pos=float("nan"),
            values_db=v,
        )
    return PowerChangeStats(
        n=int(v.size),
        median_db=float(np.median(v)),
        mean_db=float(np.mean(v)),
        p25_db=float(np.percentile(v, 25)),
        p75_db=float(np.percentile(v, 75)),
        frac_pos=float(np.mean(v > 0)),
        values_db=v,
    )


def _pre_power(i0: int, power: np.ndarray, n: int) -> Optional[float]:
    a0, a1 = i0 - n, i0
    if a0 < 0 or i0 > power.size:
        return None
    pre = float(np.mean(power[a0:a1]))
    if pre <= 1e-12:
        return None
    return pre


def _delta_db_at(i0: int, power: np.ndarray, n: int) -> Optional[float]:
    a0, a1 = i0 - n, i0
    b0, b1 = i0, i0 + n
    if a0 < 0 or b1 > power.size:
        return None
    pre = float(np.mean(power[a0:a1]))
    post = float(np.mean(power[b0:b1]))
    if pre <= 1e-12:
        return None
    return float(10.0 * np.log10(post / pre))


def _near_event(i0: int, events: np.ndarray, half: int) -> bool:
    if events.size == 0:
        return False
    j = int(np.searchsorted(events, i0))
    for jj in (j - 1, j, j + 1):
        if 0 <= jj < events.size and abs(int(events[jj]) - i0) <= half:
            return True
    return False


def _peak_near_onset(
    onset: int,
    peaks: np.ndarray,
    *,
    pre_n: int,
    post_n: int,
) -> Optional[int]:
    """在 onset 邻近窗内取对应 α 正峰；优先 onset 前（含当日）最近峰。"""
    if peaks.size == 0:
        return None
    lo = int(onset) - int(pre_n)
    hi = int(onset) + int(post_n)
    cand = peaks[(peaks >= lo) & (peaks <= hi)]
    if cand.size == 0:
        return None
    before = cand[cand <= int(onset)]
    if before.size:
        return int(before[-1])
    return int(cand[0])


def _match_by_pre_power(
    stim_idx: np.ndarray,
    stim_pre: np.ndarray,
    stim_db: np.ndarray,
    cand_idx: np.ndarray,
    cand_pre: np.ndarray,
    cand_db: np.ndarray,
    *,
    rel_tol: float = POST_STIM_MATCH_REL_TOL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 P_pre 贪心 1:1 最近邻匹配；返回 matched stim/sham 的 idx 与 ΔdB。"""
    if stim_idx.size == 0 or cand_idx.size == 0:
        z = np.zeros(0, dtype=np.int64)
        zf = np.zeros(0, dtype=np.float64)
        return z, z, zf, zf

    order = np.argsort(-stim_pre)  ## 先配最强包，候选池更稀缺时优先
    used = np.zeros(cand_idx.size, dtype=bool)
    m_stim_i: List[int] = []
    m_sham_i: List[int] = []
    m_stim_db: List[float] = []
    m_sham_db: List[float] = []
    lo = 1.0 / (1.0 + float(rel_tol))
    hi = 1.0 + float(rel_tol)

    for oi in order:
        sp = float(stim_pre[oi])
        if not np.isfinite(sp) or sp <= 0:
            continue
        ratios = cand_pre / sp
        ok = (~used) & np.isfinite(ratios) & (ratios >= lo) & (ratios <= hi)
        if not np.any(ok):
            continue
        ## 在容差内选 |log(P_c/P_s)| 最小
        score = np.full(cand_pre.size, np.inf, dtype=np.float64)
        score[ok] = np.abs(np.log(ratios[ok]))
        best = int(np.argmin(score))
        if not np.isfinite(score[best]):
            continue
        used[best] = True
        m_stim_i.append(int(stim_idx[oi]))
        m_sham_i.append(int(cand_idx[best]))
        m_stim_db.append(float(stim_db[oi]))
        m_sham_db.append(float(cand_db[best]))

    return (
        np.asarray(m_stim_i, dtype=np.int64),
        np.asarray(m_sham_i, dtype=np.int64),
        np.asarray(m_stim_db, dtype=np.float64),
        np.asarray(m_sham_db, dtype=np.float64),
    )


def _locked_envelope(
    env: np.ndarray,
    anchors: np.ndarray,
    pre_n: int,
    post_n: int,
    fs: float,
    window_sec: float,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """以 anchors 为 0 点切包络 epoch，返回 n / times_ms / avg / sem / rel_pct。"""
    times_ms = (np.arange(-pre_n, post_n + 1, dtype=np.float64) / fs) * 1000.0
    epochs = []
    for a0 in anchors:
        i0 = int(a0)
        a = i0 - pre_n
        b = i0 + post_n + 1
        if a < 0 or b > env.size:
            continue
        epochs.append(env[a:b])
    if not epochs:
        z = np.zeros_like(times_ms)
        return 0, times_ms, z, z, z
    epochs_arr = np.asarray(epochs, dtype=np.float64)
    avg = np.mean(epochs_arr, axis=0)
    if epochs_arr.shape[0] > 1:
        sem = np.std(epochs_arr, axis=0, ddof=1) / np.sqrt(epochs_arr.shape[0])
    else:
        sem = np.zeros_like(avg)
    bas_mask = (times_ms >= -window_sec * 1000.0) & (times_ms < 0.0)
    bas = (
        float(np.mean(avg[bas_mask]))
        if np.any(bas_mask)
        else float(np.mean(avg[times_ms < 0]))
    )
    if bas > 1e-12:
        rel = 100.0 * (avg / bas - 1.0)
    else:
        rel = np.zeros_like(avg)
    return int(epochs_arr.shape[0]), times_ms, avg, sem, rel


def analyze_post_stim_alpha(
    eeg: np.ndarray,
    pink_seq: np.ndarray,
    sample_rate: float,
    *,
    band_hz: Tuple[float, float] = ALPHA_BANDPASS_HZ,
    window_sec: float = POST_STIM_FIXED_W_SEC,
    min_events: int = POST_STIM_MIN_EVENTS,
    match_rel_tol: float = POST_STIM_MATCH_REL_TOL,
) -> PostStimAlphaAnalysis:
    """对已对齐的 EEG / pink_seq 做刺激前后 α 功率对比。"""
    fs = float(sample_rate)
    if fs <= 0:
        raise ValueError(f"无效采样率: {sample_rate}")

    events = detect_pink_seq_events(pink_seq)
    n_events = int(events.size)
    if n_events < min_events:
        raise ValueError(
            f"有效粉噪事件过少 ({n_events} < {min_events})。"
            "请确认 CSV 含 pink_seq 且存在跳变（播放编号变化）。"
        )

    alpha = _alpha_bandpass(_notch(eeg - np.median(eeg), fs), fs, band_hz)
    env = np.abs(hilbert(alpha))
    power = env ** 2

    peaks, _ = find_peaks(
        alpha,
        distance=max(1, int(fs / 14.0)),
        prominence=0.3 * float(np.std(alpha)),
    )
    n_cycles = max(0, int(peaks.size) - 1)

    isi = np.diff(events) / fs if events.size > 1 else np.zeros(0, dtype=np.float64)
    isi_ms = isi * 1000.0

    w = float(window_sec)
    if w <= 0:
        w = POST_STIM_FIXED_W_SEC
    n_win = max(4, int(round(w * fs)))

    peak_pre_n = int(round(POST_STIM_PEAK_SEARCH_PRE_SEC * fs))
    peak_post_n = int(round(POST_STIM_PEAK_SEARCH_POST_SEC * fs))

    ## 刺激锚点 = 粉噪 onset 邻近的 α 正峰（与伪 Sham 同一类零点）
    stim_idx_list: List[int] = []
    stim_pre_list: List[float] = []
    stim_db_list: List[float] = []
    lag_ms_list: List[float] = []
    for ev in events:
        onset = int(ev)
        pk = _peak_near_onset(onset, peaks, pre_n=peak_pre_n, post_n=peak_post_n)
        if pk is None:
            continue
        pre = _pre_power(pk, power, n_win)
        d = _delta_db_at(pk, power, n_win)
        if pre is None or d is None:
            continue
        stim_idx_list.append(pk)
        stim_pre_list.append(pre)
        stim_db_list.append(d)
        lag_ms_list.append((onset - pk) * 1000.0 / fs)
    stim_idx = np.asarray(stim_idx_list, dtype=np.int64)
    stim_pre = np.asarray(stim_pre_list, dtype=np.float64)
    stim_db_arr = np.asarray(stim_db_list, dtype=np.float64)
    stim_stats = _power_stats(stim_db_arr)
    lag_ms = np.asarray(lag_ms_list, dtype=np.float64)
    onset_minus_peak_median_ms = (
        float(np.median(lag_ms)) if lag_ms.size else float("nan")
    )
    if stim_stats.n < min_events:
        raise ValueError(
            f"能关联到 α 正峰的粉噪事件过少 ({stim_stats.n} < {min_events})。"
            "请检查刺激是否过稀或 α 峰检测失败。"
        )

    half = int(round(POST_STIM_SHAM_EXCL_SEC * fs))
    sham_idx_list: List[int] = []
    sham_pre_list: List[float] = []
    sham_db_list: List[float] = []
    for p in peaks:
        i0 = int(p)
        if _near_event(i0, events, half):
            continue
        ## 已被用作刺激锚点的峰不再进对照池
        if stim_idx.size and np.any(np.abs(stim_idx.astype(np.int64) - i0) <= 1):
            continue
        pre = _pre_power(i0, power, n_win)
        d = _delta_db_at(i0, power, n_win)
        if pre is None or d is None:
            continue
        sham_idx_list.append(i0)
        sham_pre_list.append(pre)
        sham_db_list.append(d)
    sham_idx = np.asarray(sham_idx_list, dtype=np.int64)
    sham_pre = np.asarray(sham_pre_list, dtype=np.float64)
    sham_db_arr = np.asarray(sham_db_list, dtype=np.float64)
    sham_stats = _power_stats(sham_db_arr)

    m_stim_i, m_sham_i, m_stim_db, m_sham_db = _match_by_pre_power(
        stim_idx,
        stim_pre,
        stim_db_arr,
        sham_idx,
        sham_pre,
        sham_db_arr,
        rel_tol=match_rel_tol,
    )
    matched_stim = _power_stats(m_stim_db)
    matched_sham = _power_stats(m_sham_db)
    match_rate = (
        float(matched_stim.n / stim_stats.n) if stim_stats.n > 0 else float("nan")
    )
    if matched_stim.n > 0:
        stim_map = {int(i): float(p) for i, p in zip(stim_idx, stim_pre)}
        sham_map = {int(i): float(p) for i, p in zip(sham_idx, sham_pre)}
        ratios = []
        for si, hi in zip(m_stim_i, m_sham_i):
            sp = stim_map.get(int(si))
            hp = sham_map.get(int(hi))
            if sp and hp and sp > 0:
                ratios.append(hp / sp)
        match_pre_ratio_median = (
            float(np.median(ratios)) if ratios else float("nan")
        )
        match_delta_db_diff = float(matched_stim.median_db - matched_sham.median_db)
    else:
        match_pre_ratio_median = float("nan")
        match_delta_db_diff = float("nan")

    pre_n = int(round(POST_STIM_EPOCH_PRE_SEC * fs))
    post_n = int(round(POST_STIM_EPOCH_POST_SEC * fs))
    ## 包络图：刺激与匹配伪 Sham 都锁到 α 正峰
    env_stim_anchors = m_stim_i if m_stim_i.size >= POST_STIM_MATCH_MIN else stim_idx
    n_ep, times_ms, avg, sem, rel = _locked_envelope(
        env, env_stim_anchors, pre_n, post_n, fs, w
    )
    if n_ep <= 0:
        raise ValueError("无法切出有效刺激锁定 epoch（记录过短或事件靠边）")
    _, _, sham_avg, sham_sem, sham_rel = _locked_envelope(
        env, sham_idx, pre_n, post_n, fs, w
    )
    _, _, m_avg, m_sem, m_rel = _locked_envelope(
        env, m_sham_i, pre_n, post_n, fs, w
    )

    ratio = float(n_events / n_cycles) if n_cycles > 0 else float("nan")
    dense = bool(isi.size and float(np.median(isi)) < 0.15)
    lag_note = (
        f"onset−正峰中位 {onset_minus_peak_median_ms:+.0f} ms"
        f"（>0=发声在峰后）"
        if np.isfinite(onset_minus_peak_median_ms)
        else "onset−正峰未知"
    )
    match_note = (
        f"匹配伪Sham n={matched_stim.n}（容差±{match_rel_tol:.0%} P_pre，"
        f"配对率 {match_rate:.0%}，P_pre比中位 {match_pre_ratio_median:.2f}）"
        if matched_stim.n > 0
        else f"匹配伪Sham 不足（需远离粉噪且 P_pre±{match_rel_tol:.0%} 的正峰）"
    )
    note = (
        f"零点=α正峰（刺激取 onset 邻近峰）；"
        f"ΔdB=10·log10(P_post/P_pre)，P=α包络²；窗长={w*1000:.0f} ms；"
        f"{lag_note}；粉噪/周期≈{ratio:.2f}；"
        f"{'刺激较密，短窗会叠下一拍' if dense else '刺激相对稀疏'}；"
        f"{match_note}"
    )

    return PostStimAlphaAnalysis(
        sample_rate=fs,
        duration_s=float(eeg.size / fs),
        n_events=n_events,
        n_alpha_cycles=n_cycles,
        pink_per_cycle=ratio,
        isi_median_ms=float(np.median(isi_ms)) if isi_ms.size else float("nan"),
        isi_p10_ms=float(np.percentile(isi_ms, 10)) if isi_ms.size else float("nan"),
        isi_p90_ms=float(np.percentile(isi_ms, 90)) if isi_ms.size else float("nan"),
        alpha_band_hz=(float(band_hz[0]), float(band_hz[1])),
        window_ms=w * 1000.0,
        stim=stim_stats,
        sham_peak=sham_stats,
        matched_stim=matched_stim,
        matched_sham=matched_sham,
        match_rate=match_rate,
        match_pre_ratio_median=match_pre_ratio_median,
        match_delta_db_diff=match_delta_db_diff,
        onset_minus_peak_median_ms=onset_minus_peak_median_ms,
        times_ms=times_ms,
        envelope_average=avg,
        envelope_sem=sem,
        envelope_rel_pct=rel,
        sham_envelope_average=sham_avg,
        sham_envelope_sem=sham_sem,
        sham_envelope_rel_pct=sham_rel,
        matched_envelope_average=m_avg,
        matched_envelope_sem=m_sem,
        matched_envelope_rel_pct=m_rel,
        note=note,
    )


def analyze_post_stim_alpha_csv(
    path: Path,
    *,
    band_hz: Tuple[float, float] = ALPHA_BANDPASS_HZ,
    window_sec: float = POST_STIM_FIXED_W_SEC,
    min_events: int = POST_STIM_MIN_EVENTS,
    match_rel_tol: float = POST_STIM_MATCH_REL_TOL,
) -> PostStimAlphaAnalysis:
    eeg, pink, fs = load_eeg_pink_csv(path)
    return analyze_post_stim_alpha(
        eeg,
        pink,
        fs,
        band_hz=band_hz,
        window_sec=window_sec,
        min_events=min_events,
        match_rel_tol=match_rel_tol,
    )


def format_post_stim_alpha_report(result: PostStimAlphaAnalysis) -> str:
    lo, hi = result.alpha_band_hz
    stim = result.stim
    sham = result.sham_peak
    m_stim = result.matched_stim
    m_sham = result.matched_sham
    ratio = 10 ** (stim.median_db / 10.0) if np.isfinite(stim.median_db) else float("nan")
    lag = result.onset_minus_peak_median_ms
    lag_s = (
        f"onset−正峰中位={lag:+.1f} ms" if np.isfinite(lag) else "onset−正峰未知"
    )
    lines = [
        f"粉噪前后 α 功率: {result.n_events} 事件 @ {result.sample_rate:.1f} Hz，"
        f"时长 {result.duration_s:.1f} s",
        f"α 带通: {lo:.1f}–{hi:.1f} Hz；比较窗 {result.window_ms:.0f} ms；"
        f"零点=α正峰",
        f"ISI 中位 {result.isi_median_ms:.0f} ms（P10–P90: "
        f"{result.isi_p10_ms:.0f}–{result.isi_p90_ms:.0f}）；"
        f"粉噪/周期≈{result.pink_per_cycle:.2f}；{lag_s}",
        result.note,
        f"[刺激·正峰对齐] n={stim.n}  中位ΔdB={stim.median_db:.3f}  "
        f"均值={stim.mean_db:.3f}  IQR=[{stim.p25_db:.2f},{stim.p75_db:.2f}]  "
        f"增强占比={stim.frac_pos:.1%}",
        f"  → 中位功率比≈{ratio:.2%}（相对峰前）" if np.isfinite(ratio) else "",
        f"[无粉噪正峰·未匹配] n={sham.n}  中位ΔdB={sham.median_db:.3f}  "
        f"增强占比={sham.frac_pos:.1%}"
        if sham.n > 0
        else "[无粉噪正峰·未匹配] 样本不足（刺激过密时常见）",
    ]
    if m_stim.n >= POST_STIM_MATCH_MIN:
        lines.append(
            f"[匹配伪Sham·同为正峰零点] 配对数={m_stim.n}  "
            f"配对率={result.match_rate:.1%}  "
            f"P_pre比中位={result.match_pre_ratio_median:.3f}"
        )
        lines.append(
            f"  匹配刺激 中位ΔdB={m_stim.median_db:.3f}  增强占比={m_stim.frac_pos:.1%}"
        )
        lines.append(
            f"  匹配对照 中位ΔdB={m_sham.median_db:.3f}  增强占比={m_sham.frac_pos:.1%}"
        )
        lines.append(
            f"  → Stim−Sham 中位差={result.match_delta_db_diff:+.3f} dB"
            f"（负=同相位下打了比未打掉得更多）"
        )
    else:
        lines.append(
            f"[匹配伪Sham] 配对数不足（n={m_stim.n} < {POST_STIM_MATCH_MIN}；"
            "刺激过密或强包几乎都被打时常见）"
        )
    return "\n".join(x for x in lines if x)


def export_post_stim_alpha_summary_csv(path: Path, result: PostStimAlphaAnalysis) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["sample_rate_hz", f"{result.sample_rate:.6g}"])
        writer.writerow(["duration_s", f"{result.duration_s:.6g}"])
        writer.writerow(["n_events", result.n_events])
        writer.writerow(["n_alpha_cycles", result.n_alpha_cycles])
        writer.writerow(["pink_per_cycle", f"{result.pink_per_cycle:.6g}"])
        writer.writerow(["isi_median_ms", f"{result.isi_median_ms:.6g}"])
        writer.writerow(["window_ms", f"{result.window_ms:.6g}"])
        writer.writerow(["stim_n", result.stim.n])
        writer.writerow(["stim_median_db", f"{result.stim.median_db:.6g}"])
        writer.writerow(["stim_mean_db", f"{result.stim.mean_db:.6g}"])
        writer.writerow(["stim_frac_pos", f"{result.stim.frac_pos:.6g}"])
        writer.writerow(["sham_n", result.sham_peak.n])
        writer.writerow(["sham_median_db", f"{result.sham_peak.median_db:.6g}"])
        writer.writerow(["sham_frac_pos", f"{result.sham_peak.frac_pos:.6g}"])
        writer.writerow(["matched_n", result.matched_stim.n])
        writer.writerow(["match_rate", f"{result.match_rate:.6g}"])
        writer.writerow(
            ["match_pre_ratio_median", f"{result.match_pre_ratio_median:.6g}"]
        )
        writer.writerow(
            ["matched_stim_median_db", f"{result.matched_stim.median_db:.6g}"]
        )
        writer.writerow(
            ["matched_sham_median_db", f"{result.matched_sham.median_db:.6g}"]
        )
        writer.writerow(
            ["match_delta_db_diff", f"{result.match_delta_db_diff:.6g}"]
        )
        writer.writerow(
            [
                "onset_minus_peak_median_ms",
                f"{result.onset_minus_peak_median_ms:.6g}",
            ]
        )
        writer.writerow(["note", result.note])
        writer.writerow([])
        writer.writerow(["event_index", "stim_delta_db"])
        for i, v in enumerate(result.stim.values_db):
            writer.writerow([i, f"{float(v):.6f}"])
        writer.writerow([])
        writer.writerow(["pair_index", "matched_stim_delta_db", "matched_sham_delta_db"])
        n_pair = min(result.matched_stim.values_db.size, result.matched_sham.values_db.size)
        for i in range(n_pair):
            writer.writerow(
                [
                    i,
                    f"{float(result.matched_stim.values_db[i]):.6f}",
                    f"{float(result.matched_sham.values_db[i]):.6f}",
                ]
            )
        writer.writerow([])
        writer.writerow(
            [
                "time_ms",
                "envelope_average",
                "envelope_sem",
                "envelope_rel_pct",
                "sham_envelope_average",
                "sham_envelope_sem",
                "sham_envelope_rel_pct",
                "matched_envelope_average",
                "matched_envelope_sem",
                "matched_envelope_rel_pct",
            ]
        )
        for t, y, s, r, sy, ss, sr, my, ms, mr in zip(
            result.times_ms,
            result.envelope_average,
            result.envelope_sem,
            result.envelope_rel_pct,
            result.sham_envelope_average,
            result.sham_envelope_sem,
            result.sham_envelope_rel_pct,
            result.matched_envelope_average,
            result.matched_envelope_sem,
            result.matched_envelope_rel_pct,
        ):
            writer.writerow(
                [
                    f"{float(t):.6f}",
                    f"{float(y):.6g}",
                    f"{float(s):.6g}",
                    f"{float(r):.6g}",
                    f"{float(sy):.6g}",
                    f"{float(ss):.6g}",
                    f"{float(sr):.6g}",
                    f"{float(my):.6g}",
                    f"{float(ms):.6g}",
                    f"{float(mr):.6g}",
                ]
            )
    return path
