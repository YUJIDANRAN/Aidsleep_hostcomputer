"""
长时记录结束后：扫描各 chunk 的 raw，标记「正常段」。

正常定义：ch1_raw 全程落在 [raw_min, raw_max]，且连续时长 ≥ min_duration_sec。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NORMAL_RAW_MIN = 900
NORMAL_RAW_MAX = 1300
NORMAL_MIN_DURATION_SEC = 120.0
REPORT_TXT_NAME = "normal_segments_report.txt"
REPORT_CSV_NAME = "normal_segments.csv"


@dataclass(frozen=True)
class NormalSegment:
    file_name: str
    file_path: str
    chunk_index: int
    local_start_s: float
    local_end_s: float
    duration_s: float
    global_start_s: float
    global_end_s: float
    n_samples: int
    sample_rate: float
    raw_min: int
    raw_max: int


@dataclass
class NormalSegmentsResult:
    session_dir: Path
    segments: List[NormalSegment] = field(default_factory=list)
    report_text: str = ""
    report_txt_path: Optional[Path] = None
    report_csv_path: Optional[Path] = None


def _chunk_sort_key(path: Path) -> Tuple[int, str]:
    name = path.name
    # eeg_chunk_012_full.csv → 12
    try:
        stem = name.replace("_full.csv", "").replace(".csv", "")
        idx = int(stem.split("_")[-1])
        return (idx, name)
    except Exception:
        return (10**9, name)


def list_chunk_full_csvs(session_dir: Path) -> List[Path]:
    session_dir = Path(session_dir)
    files = sorted(session_dir.glob("eeg_chunk_*_full.csv"), key=_chunk_sort_key)
    if files:
        return files
    # 兼容只有删减版
    return sorted(session_dir.glob("eeg_chunk_*.csv"), key=_chunk_sort_key)


def _infer_sample_rate(time_s: np.ndarray, n: int, fallback: float = 500.0) -> float:
    if time_s.size >= 2:
        dt = np.diff(time_s.astype(np.float64))
        dt = dt[dt > 0]
        if dt.size:
            return float(1.0 / np.median(dt))
    if n > 0 and time_s.size == n and time_s[-1] > 0:
        return float((n - 1) / float(time_s[-1]))
    return float(fallback)


def find_in_range_runs(
    raw: np.ndarray,
    sample_rate: float,
    *,
    raw_min: int = NORMAL_RAW_MIN,
    raw_max: int = NORMAL_RAW_MAX,
    min_duration_sec: float = NORMAL_MIN_DURATION_SEC,
) -> List[Tuple[int, int]]:
    """返回满足条件的 [start_idx, end_idx) 样本下标区间（半开）。"""
    if raw.size == 0 or sample_rate <= 0:
        return []
    ok = (raw >= raw_min) & (raw <= raw_max)
    min_samples = max(1, int(np.ceil(min_duration_sec * sample_rate)))

    runs: List[Tuple[int, int]] = []
    i = 0
    n = int(ok.size)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i + 1
        while j < n and ok[j]:
            j += 1
        if (j - i) >= min_samples:
            runs.append((i, j))
        i = j
    return runs


def analyze_chunk_file(
    path: Path,
    *,
    global_time_offset_s: float,
    raw_min: int = NORMAL_RAW_MIN,
    raw_max: int = NORMAL_RAW_MAX,
    min_duration_sec: float = NORMAL_MIN_DURATION_SEC,
) -> Tuple[List[NormalSegment], float]:
    """分析单个 chunk；返回 (正常段列表, 本文件时长秒)。"""
    df = pd.read_csv(path)
    if "ch1_raw" not in df.columns:
        raise ValueError(f"缺少 ch1_raw 列: {path}")
    raw = df["ch1_raw"].to_numpy(dtype=np.int64)
    if "time_s" in df.columns:
        time_s = df["time_s"].to_numpy(dtype=np.float64)
    else:
        time_s = np.arange(raw.size, dtype=np.float64) / 500.0
    sample_rate = _infer_sample_rate(time_s, raw.size)
    duration = float(raw.size / sample_rate) if sample_rate > 0 else 0.0

    try:
        chunk_index = int(
            path.name.replace("_full.csv", "").replace(".csv", "").split("_")[-1]
        )
    except Exception:
        chunk_index = 0

    segments: List[NormalSegment] = []
    for start_i, end_i in find_in_range_runs(
        raw,
        sample_rate,
        raw_min=raw_min,
        raw_max=raw_max,
        min_duration_sec=min_duration_sec,
    ):
        local_start = float(start_i / sample_rate)
        local_end = float(end_i / sample_rate)
        seg_raw = raw[start_i:end_i]
        segments.append(
            NormalSegment(
                file_name=path.name,
                file_path=str(path.resolve()),
                chunk_index=chunk_index,
                local_start_s=local_start,
                local_end_s=local_end,
                duration_s=local_end - local_start,
                global_start_s=global_time_offset_s + local_start,
                global_end_s=global_time_offset_s + local_end,
                n_samples=int(end_i - start_i),
                sample_rate=sample_rate,
                raw_min=int(seg_raw.min()),
                raw_max=int(seg_raw.max()),
            )
        )
    return segments, duration


def build_report_text(
    session_dir: Path,
    segments: Sequence[NormalSegment],
    *,
    raw_min: int,
    raw_max: int,
    min_duration_sec: float,
    n_files: int,
) -> str:
    lines = [
        "长时记录 · 正常段报告",
        f"会话目录: {Path(session_dir).resolve()}",
        f"判定规则: raw ∈ [{raw_min}, {raw_max}]，连续时长 ≥ {min_duration_sec:g} s",
        f"扫描文件数: {n_files}",
        f"合格正常段数: {len(segments)}",
        "",
    ]
    if not segments:
        lines.append("未找到满足条件的正常段。")
        return "\n".join(lines)

    for i, seg in enumerate(segments, start=1):
        lines.extend(
            [
                f"[{i}] 文件: {seg.file_name}",
                f"    文件内时间: {seg.local_start_s:.1f} – {seg.local_end_s:.1f} s"
                f"（时长 {seg.duration_s:.1f} s）",
                f"    会话累计时间: {seg.global_start_s:.1f} – {seg.global_end_s:.1f} s",
                f"    样本数: {seg.n_samples} @ {seg.sample_rate:.1f} Hz",
                f"    段内 raw 实际范围: [{seg.raw_min}, {seg.raw_max}]",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_normal_segments_report(
    session_dir: Path,
    *,
    raw_min: int = NORMAL_RAW_MIN,
    raw_max: int = NORMAL_RAW_MAX,
    min_duration_sec: float = NORMAL_MIN_DURATION_SEC,
) -> NormalSegmentsResult:
    session_dir = Path(session_dir)
    files = list_chunk_full_csvs(session_dir)
    all_segments: List[NormalSegment] = []
    offset = 0.0
    for path in files:
        segs, duration = analyze_chunk_file(
            path,
            global_time_offset_s=offset,
            raw_min=raw_min,
            raw_max=raw_max,
            min_duration_sec=min_duration_sec,
        )
        all_segments.extend(segs)
        offset += duration

    report_text = build_report_text(
        session_dir,
        all_segments,
        raw_min=raw_min,
        raw_max=raw_max,
        min_duration_sec=min_duration_sec,
        n_files=len(files),
    )
    txt_path = session_dir / REPORT_TXT_NAME
    csv_path = session_dir / REPORT_CSV_NAME
    txt_path.write_text(report_text, encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "file_name",
                "chunk_index",
                "local_start_s",
                "local_end_s",
                "duration_s",
                "global_start_s",
                "global_end_s",
                "n_samples",
                "sample_rate",
                "raw_min",
                "raw_max",
            ]
        )
        for i, seg in enumerate(all_segments, start=1):
            writer.writerow(
                [
                    i,
                    seg.file_name,
                    seg.chunk_index,
                    f"{seg.local_start_s:.3f}",
                    f"{seg.local_end_s:.3f}",
                    f"{seg.duration_s:.3f}",
                    f"{seg.global_start_s:.3f}",
                    f"{seg.global_end_s:.3f}",
                    seg.n_samples,
                    f"{seg.sample_rate:.3f}",
                    seg.raw_min,
                    seg.raw_max,
                ]
            )

    return NormalSegmentsResult(
        session_dir=session_dir,
        segments=list(all_segments),
        report_text=report_text,
        report_txt_path=txt_path,
        report_csv_path=csv_path,
    )
