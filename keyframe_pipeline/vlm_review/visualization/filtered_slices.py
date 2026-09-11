from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from keyframe_pipeline.vlm_review.core.config import ensure_config_defaults, load_config
from keyframe_pipeline.vlm_review.core.models import Event
from keyframe_pipeline.vlm_review.data.io_utils import (
    load_gripper_trajectory,
    normalize_events,
    read_json,
    write_json,
)
from keyframe_pipeline.vlm_review.slicing import build_review_slices


EVENT_COLORS = {
    "MAX": "#d62728",
    "MIN": "#1f77b4",
    "PL": "#2ca02c",
    "PR": "#9467bd",
}

REMOVED_STATUSES = {"removed", "merged_removed"}


def render_filtered_slices(
    config_path: str,
    keyframe_json: str,
    episode_id: str,
    episode_index: int,
    output_root: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_config_defaults(config_path)
    cfg = load_config(config_path)
    if output_root:
        cfg.paths.output_root = output_root

    data = read_json(keyframe_json)
    output_root_path = Path(cfg.paths.output_root)
    result: Dict[str, Any] = {"episode_id": episode_id, "arms": {}}

    for arm in cfg.run.arms:
        events = normalize_events(data, arm)
        time, raw, smooth = load_gripper_trajectory(
            data,
            arm,
            cfg.paths.phase_module_root,
            dataset_root=cfg.paths.dataset_root,
            episode_index=episode_index,
        )
        review_slices = build_review_slices(
            episode_id=episode_id,
            episode_index=episode_index,
            arm=arm,
            events=events,
            left_context_events=cfg.vlm_trajectory_review.left_context_events,
            target_events_per_slice=cfg.vlm_trajectory_review.target_events_per_slice,
            right_context_events=cfg.vlm_trajectory_review.right_context_events,
        )
        final_review = read_json(output_root_path / episode_id / arm / "final_review.json")
        final_by_id = {event["event_id"]: event for event in final_review["final_events"]}

        arm_dir = output_root_path / "visualizations" / episode_id / f"{arm}_filtered"
        arm_dir.mkdir(parents=True, exist_ok=True)
        rendered = []
        for review_slice in review_slices:
            image_path = arm_dir / f"{review_slice.slice_id}.png"
            kept_ids, removed_ids = _render_one_slice(
                image_path=image_path,
                review_slice=review_slice,
                final_by_id=final_by_id,
                time=time,
                raw=raw,
                smooth=smooth,
                cfg=cfg.visualization,
                show_raw=cfg.vlm_trajectory_review.show_raw_trajectory,
                show_smooth=cfg.vlm_trajectory_review.show_smoothed_trajectory,
            )
            metadata_path = arm_dir / f"{review_slice.slice_id}.metadata.json"
            write_json(
                metadata_path,
                {
                    **review_slice.metadata(),
                    "filtered_image_path": str(image_path),
                    "kept_event_ids_in_image": kept_ids,
                    "removed_event_ids_hidden": removed_ids,
                    "source": "filtered_by_final_review",
                },
            )
            rendered.append(str(image_path))

        result["arms"][arm] = {"num_slices": len(rendered), "paths": rendered}

    return result


def _render_one_slice(
    image_path: Path,
    review_slice,
    final_by_id: Dict[str, Dict[str, Any]],
    time: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    cfg,
    show_raw: bool,
    show_smooth: bool,
) -> Tuple[List[str], List[str]]:
    t0, t1 = _window(review_slice, cfg)
    mask = (time >= t0) & (time <= t1) if len(time) else np.asarray([], dtype=bool)
    kept_ids: List[str] = []
    removed_ids: List[str] = []

    fig, ax = plt.subplots(figsize=(cfg.width, cfg.height), dpi=cfg.dpi)
    if show_raw and len(time):
        ax.plot(time[mask], raw[mask], color="#a0a0a0", linewidth=1.0, alpha=0.55, label="raw")
    if show_smooth and len(time):
        ax.plot(time[mask], smooth[mask], color="#111111", linewidth=1.8, label="smoothed")

    kept_context_left, removed_context_left = _split_kept_removed(review_slice.context_left, final_by_id)
    kept_targets, removed_targets = _split_kept_removed(review_slice.targets, final_by_id)
    kept_context_right, removed_context_right = _split_kept_removed(review_slice.context_right, final_by_id)
    removed_ids.extend([event.event_id for event in [*removed_context_left, *removed_targets, *removed_context_right]])

    _draw_context(ax, kept_context_left)
    _draw_context(ax, kept_context_right)
    _draw_targets(ax, kept_targets)
    kept_ids.extend([event.event_id for event in [*kept_context_left, *kept_targets, *kept_context_right]])

    for interval in review_slice.owned_intervals:
        ax.axvspan(interval.start_time, interval.end_time, color="#ffdd57", alpha=0.06, zorder=0)

    ax.set_xlim(t0, t1)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("gripper state")
    ax.set_title(
        f"{review_slice.episode_id} | {review_slice.arm} gripper | {review_slice.slice_id} | "
        f"filtered after VLM | kept={len(kept_ids)} hidden_removed={len(removed_ids)}"
    )
    ax.grid(True, axis="both", linewidth=0.5, alpha=0.25)
    if cfg.dense_ticks:
        span = max(0.1, t1 - t0)
        step = 1.0 if span <= 20 else 5.0 if span <= 80 else 10.0
        ax.set_xticks(np.arange(np.floor(t0 / step) * step, t1 + step, step))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(image_path)
    plt.close(fig)
    return kept_ids, removed_ids


def _split_kept_removed(events: Iterable[Event], final_by_id: Dict[str, Dict[str, Any]]) -> Tuple[List[Event], List[Event]]:
    kept: List[Event] = []
    removed: List[Event] = []
    for event in events:
        final_status = final_by_id.get(event.event_id, {}).get("final_status", "kept")
        if final_status in REMOVED_STATUSES:
            removed.append(event)
        else:
            kept.append(event)
    return kept, removed


def _window(review_slice, cfg) -> Tuple[float, float]:
    t0, t1 = review_slice.time_range
    t0 -= cfg.padding_sec
    t1 += cfg.padding_sec
    if t1 - t0 < cfg.min_window_sec:
        mid = (t0 + t1) / 2.0
        t0 = mid - cfg.min_window_sec / 2.0
        t1 = mid + cfg.min_window_sec / 2.0
    return max(0.0, t0), max(t1, t0 + 0.5)


def _draw_context(ax, events: Iterable[Event]) -> None:
    for event in events:
        color = EVENT_COLORS.get(event.event_type, "#666666")
        ax.scatter([event.time], [_event_y(event)], color=color, s=36, alpha=0.55, marker="o")
        ax.text(event.time, _event_y(event), f" {event.event_id}", fontsize=7, color=color, alpha=0.55, va="bottom")


def _draw_targets(ax, events: Iterable[Event]) -> None:
    for event in events:
        color = EVENT_COLORS.get(event.event_type, "#000000")
        ax.scatter([event.time], [_event_y(event)], color=color, s=72, marker="D", edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(
            event.time,
            _event_y(event),
            event.event_id,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Render per-slice filtered visualizations after VLM review.")
    parser.add_argument("--config", default="/mnt/workspace/wrist_label_exp/config.yaml")
    parser.add_argument("--keyframe-json", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    result = render_filtered_slices(
        config_path=args.config,
        keyframe_json=args.keyframe_json,
        episode_id=args.episode_id,
        episode_index=args.episode_index,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
