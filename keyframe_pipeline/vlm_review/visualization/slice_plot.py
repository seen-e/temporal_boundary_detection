from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from ..core.config import VisualizationConfig
from ..core.models import Event, ReviewSlice
from ..data.io_utils import write_json


EVENT_COLORS = {
    "MAX": "#d62728",
    "MIN": "#1f77b4",
    "PL": "#2ca02c",
    "PR": "#9467bd",
}


def render_slice(
    review_slice: ReviewSlice,
    time: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    output_root: str | Path,
    cfg: VisualizationConfig,
    show_raw: bool = True,
    show_smooth: bool = True,
    show_time: bool = True,
    show_type: bool = True,
    show_id: bool = True,
    output_subdir: str | None = None,
    local_filename: str | None = None,
    global_filename: str | None = None,
    show_candidates: bool = True,
    show_global_points: bool = True,
    render_global: bool = True,
    pass1_ticks: bool = False,
) -> ReviewSlice:
    out_dir = Path(output_root) / review_slice.episode_id / review_slice.arm
    if output_subdir:
        out_dir = out_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / (local_filename or f"{review_slice.slice_id}.png")
    global_image_path = out_dir / (global_filename or f"{review_slice.slice_id}_global.png")
    metadata_path = out_dir / f"{review_slice.slice_id}.metadata.json"

    t0, t1 = _window(review_slice, cfg)
    mask = (time >= t0) & (time <= t1) if len(time) else np.asarray([], dtype=bool)
    fig, ax = plt.subplots(figsize=(cfg.width, cfg.height), dpi=cfg.dpi)

    if len(time):
        if show_raw:
            ax.plot(time[mask], raw[mask], color="#8a8a8a", linewidth=1.0, alpha=0.45, label="raw")
        if show_smooth:
            ax.plot(time[mask], smooth[mask], color="#111111", linewidth=1.8, alpha=0.98, label="smoothed")

    visible_t0, visible_t1 = review_slice.time_range
    target_t0, target_t1 = _target_time_range(review_slice)
    if visible_t0 < target_t0:
        ax.axvspan(visible_t0, target_t0, color="#6baed6", alpha=0.11, zorder=0)
    if target_t1 < visible_t1:
        ax.axvspan(target_t1, visible_t1, color="#6baed6", alpha=0.11, zorder=0)
    ax.axvspan(target_t0, target_t1, color="#ffcc00", alpha=0.12, zorder=0)

    if show_candidates:
        _draw_candidates(ax, review_slice.context_left, "left_overlap", show_time, show_type, show_id)
        _draw_candidates(ax, review_slice.targets, "target", show_time, show_type, show_id)
        _draw_candidates(ax, review_slice.context_right, "right_overlap", show_time, show_type, show_id)
        for interval in review_slice.owned_intervals:
            ax.axvspan(interval.start_time, interval.end_time, color="#ffdd57", alpha=0.08, zorder=0)

    ax.axvline(target_t0, color="#b00020", linewidth=1.4, linestyle="--", alpha=0.9)
    ax.axvline(target_t1, color="#b00020", linewidth=1.4, linestyle="--", alpha=0.9)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("gripper state")
    if show_candidates:
        title_suffix = (
            f"targets {review_slice.target_display_ids[0] if review_slice.target_display_ids else '-'}-"
            f"{review_slice.target_display_ids[-1] if review_slice.target_display_ids else '-'}"
        )
    else:
        title_suffix = "target region"
    ax.set_title(f"{review_slice.episode_id} | {review_slice.arm} gripper | {review_slice.slice_id} | {title_suffix}")
    if pass1_ticks:
        _apply_time_grid(ax, t0, t1, cfg.pass1_local_major_xticks, cfg.pass1_local_minor_xticks)
    else:
        ax.grid(True, axis="both", linewidth=0.5, alpha=0.25)
        if cfg.dense_ticks:
            span = max(0.1, t1 - t0)
            step = 1.0 if span <= 20 else 5.0 if span <= 80 else 10.0
            ax.set_xticks(np.arange(np.floor(t0 / step) * step, t1 + step, step))
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(image_path)
    plt.close(fig)

    review_slice.image_path = str(image_path)
    review_slice.global_image_path = str(global_image_path) if render_global else None
    review_slice.metadata_path = str(metadata_path)
    if render_global:
        _render_global_context(
            review_slice=review_slice,
            time=time,
            raw=raw,
            smooth=smooth,
            image_path=global_image_path,
            show_raw=show_raw,
            show_smooth=show_smooth,
            show_points=show_global_points,
            pass1_ticks=pass1_ticks,
            cfg=cfg,
        )
    write_json(metadata_path, review_slice.metadata())
    return review_slice


def _render_global_context(
    review_slice: ReviewSlice,
    time: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    image_path: Path,
    show_raw: bool,
    show_smooth: bool,
    show_points: bool,
    pass1_ticks: bool,
    cfg: VisualizationConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 4.2), dpi=170)
    if len(time):
        if show_raw:
            ax.plot(time, raw, color="#8a8a8a", linewidth=0.9, alpha=0.35, label="raw full episode")
        if show_smooth:
            ax.plot(time, smooth, color="#111111", linewidth=1.35, alpha=0.98, label="smoothed full episode")

    visible_t0, visible_t1 = review_slice.time_range
    target_t0, target_t1 = _target_time_range(review_slice)
    if visible_t0 < target_t0:
        ax.axvspan(visible_t0, target_t0, color="#6baed6", alpha=0.18, label="left overlap context", zorder=0)
    if target_t1 < visible_t1:
        ax.axvspan(target_t1, visible_t1, color="#6baed6", alpha=0.18, label="right overlap context", zorder=0)
    ax.axvspan(target_t0, target_t1, color="#ffcc00", alpha=0.26, label="target region", zorder=0)
    if show_points:
        _draw_global_points(ax, review_slice.episode_events or review_slice.all_events)
    ax.axvline(target_t0, color="#b00020", linewidth=1.4, linestyle="--", alpha=0.9)
    ax.axvline(target_t1, color="#b00020", linewidth=1.4, linestyle="--", alpha=0.9)

    xmin, xmax = (float(time[0]), float(time[-1])) if len(time) else (0.0, 1.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("gripper state")
    ax.set_title(f"{review_slice.episode_id} | {review_slice.arm} gripper | full episode context for {review_slice.slice_id}")
    if pass1_ticks:
        _apply_time_grid(ax, xmin, xmax, cfg.pass1_global_major_xticks, cfg.pass1_global_minor_xticks)
    else:
        ax.grid(True, axis="both", linewidth=0.45, alpha=0.22)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(image_path)
    plt.close(fig)


def _apply_time_grid(ax, t0: float, t1: float, major_count: int, minor_count: int) -> None:
    major_count = max(2, int(major_count or 10))
    minor_count = max(0, int(minor_count or 0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=major_count, prune=None))
    if minor_count > 0:
        major_intervals = max(1, major_count - 1)
        subdivisions = max(2, int(round(minor_count / major_intervals)))
        ax.xaxis.set_minor_locator(AutoMinorLocator(subdivisions))
    ax.grid(True, axis="x", which="major", linewidth=0.55, alpha=0.28)
    ax.grid(True, axis="x", which="minor", linewidth=0.35, alpha=0.16)
    ax.grid(True, axis="y", which="major", linewidth=0.45, alpha=0.18)


def _draw_global_points(ax, events: Iterable[Event]) -> None:
    for event_type, color in EVENT_COLORS.items():
        selected = [event for event in events if event.event_type == event_type]
        if not selected:
            continue
        ax.scatter(
            [event.time for event in selected],
            [_event_y(event) for event in selected],
            color=color,
            s=16,
            alpha=0.62,
            marker="o",
            linewidths=0,
            label=f"{event_type} points",
            zorder=4,
        )


def _target_time_range(review_slice: ReviewSlice) -> Tuple[float, float]:
    if review_slice.targets:
        return review_slice.targets[0].time, review_slice.targets[-1].time
    return review_slice.time_range


def _window(review_slice: ReviewSlice, cfg: VisualizationConfig) -> Tuple[float, float]:
    t0, t1 = review_slice.time_range
    t0 -= cfg.padding_sec
    t1 += cfg.padding_sec
    if t1 - t0 < cfg.min_window_sec:
        mid = (t0 + t1) / 2.0
        t0 = mid - cfg.min_window_sec / 2.0
        t1 = mid + cfg.min_window_sec / 2.0
    return max(0.0, t0), max(t1, t0 + 0.5)


def _draw_candidates(ax, events: Iterable[Event], scope: str, show_time: bool, show_type: bool, show_id: bool) -> None:
    alpha = 0.66 if scope == "target" else 0.54
    edge = "#111111" if scope == "target" else "none"
    scope_tag = {"left_overlap": "L", "target": "T", "right_overlap": "R"}.get(scope, "")
    for event in events:
        color = EVENT_COLORS.get(event.event_type, "#666666")
        ax.scatter(
            [event.time],
            [_event_y(event)],
            color=color,
            s=54,
            alpha=alpha,
            marker="o",
            edgecolors=edge,
            linewidths=0.5 if scope == "target" else 0.0,
            zorder=5,
        )
        label_parts = []
        if show_id:
            label_parts.append(event.display_id)
        if scope_tag:
            label_parts.append(scope_tag)
        label = "/".join(label_parts)
        ax.text(
            event.time,
            _event_y(event),
            label,
            fontsize=7,
            color=color,
            alpha=0.82,
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "boxstyle": "round,pad=0.12"},
        )


def _draw_targets(ax, events: Iterable[Event], show_time: bool, show_type: bool, show_id: bool) -> None:
    for event in events:
        color = EVENT_COLORS.get(event.event_type, "#000000")
        ax.scatter([event.time], [_event_y(event)], color=color, s=72, marker="D", edgecolors="white", linewidths=0.8, zorder=5)
        label = event.display_id if show_id else ""
        ax.text(
            event.time,
            _event_y(event),
            label,
            fontsize=8,
            color=color,
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.86, "boxstyle": "round,pad=0.2"},
        )


def _event_y(event: Event) -> float:
    if event.value_smooth is not None:
        return float(event.value_smooth)
    if event.value_raw is not None:
        return float(event.value_raw)
    return 0.5
