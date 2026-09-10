from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data.io_utils import (
    find_keyframe_json,
    load_gripper_trajectory,
    normalize_events,
    read_json,
    write_json,
)


EVENT_COLORS = {"MAX": "#d62728", "MIN": "#1f77b4", "PL": "#2ca02c", "PR": "#9467bd"}


def plot_episode_three_stage(
    episode_index: int,
    keyframe_root: str,
    output_root: str,
    dataset_root: str,
    phase_module_root: str,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    keyframe_path = find_keyframe_json(keyframe_root, episode_index)
    if keyframe_path is None:
        raise FileNotFoundError(f"keyframe json not found for episode {episode_index}")
    keyframe_data = read_json(keyframe_path)
    episode_id = keyframe_data.get("episode_id") or f"episode_{episode_index:06d}"
    output_root_path = Path(output_root)

    fig, axes = plt.subplots(6, 1, figsize=(26, 17), dpi=170, sharex=True)
    summary: Dict[str, Any] = {"episode_id": episode_id, "episode_index": episode_index, "arms": {}}
    max_time = 0.0
    rows = []
    for arm in ["left", "right"]:
        final_path = output_root_path / episode_id / arm / "final_review.json"
        if not final_path.exists():
            continue
        final = read_json(final_path)
        original_events = [
            {
                "event_id": event.event_id,
                "type": event.event_type,
                "time_sec": event.time,
                "value_smooth": event.value_smooth,
                "source": "traditional",
            }
            for event in normalize_events(keyframe_data, arm)
        ]
        all_final_events = final.get("final_events", [])
        pass2_events = [
            event
            for event in all_final_events
            if event.get("final_status") not in {"removed", "merged_removed"} and event.get("source") != "pass3"
        ]
        pass3_events = [
            event
            for event in all_final_events
            if event.get("final_status") not in {"removed", "merged_removed"}
        ]
        rows.extend(
            [
                (arm, "A original candidates", original_events),
                (arm, "B Pass 2 cleaned", pass2_events),
                (arm, "C Pass 3 completed", pass3_events),
            ]
        )
        summary["arms"][arm] = {
            "original": _stage_summary(original_events),
            "pass2": _stage_summary(pass2_events),
            "pass3": _stage_summary(pass3_events),
            "pass3_added": _stage_summary([event for event in pass3_events if event.get("source") == "pass3"]),
        }

    for ax, row in zip(axes, rows):
        arm, stage, events = row
        time, raw, smooth = load_gripper_trajectory(
            keyframe_data,
            arm,
            phase_module_root,
            dataset_root=dataset_root,
            episode_index=episode_index,
        )
        if len(time):
            max_time = max(max_time, float(time[-1]))
            ax.plot(time, smooth, color="#111111", lw=1.15, alpha=0.98, label="smoothed")
        _draw_events(ax, time, smooth, events)
        _draw_pass3_additions(ax, time, smooth, [event for event in events if event.get("source") == "pass3"])
        ax.set_ylabel(f"{arm}\n{stage}", fontsize=9)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.22, lw=0.45)
        ax.legend(loc="upper right", ncol=8, fontsize=8)
        ax.set_title(f"{episode_id} | {arm} | {stage} | total={len(events)}", fontsize=10)

    for ax in axes[len(rows) :]:
        ax.axis("off")
    if max_time > 0:
        axes[-1].set_xlim(0, max_time)
    axes[-1].set_xlabel("time (sec)")
    fig.suptitle(f"{episode_id}: three-stage keyframe comparison", y=0.997, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.982])

    image_path = Path(out_path) if out_path else output_root_path / episode_id / "three_stage_before_after_comparison.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path)
    plt.close(fig)
    summary["image_path"] = str(image_path)
    write_json(image_path.with_suffix(".summary.json"), summary)
    return summary


def _draw_events(ax, time: np.ndarray, smooth: np.ndarray, events: List[Dict[str, Any]]) -> None:
    for event_type, color in EVENT_COLORS.items():
        selected = [event for event in events if event.get("type") == event_type and event.get("source") != "pass3"]
        if not selected:
            continue
        xs = [float(event["time_sec"]) for event in selected]
        ys = [_value_at(time, smooth, x, event.get("value_smooth")) for x, event in zip(xs, selected)]
        ax.scatter(xs, ys, s=15, color=color, alpha=0.86, label=f"{event_type} {len(selected)}", zorder=4)


def _draw_pass3_additions(ax, time: np.ndarray, smooth: np.ndarray, events: List[Dict[str, Any]]) -> None:
    for event_type, color in EVENT_COLORS.items():
        selected = [event for event in events if event.get("type") == event_type]
        if not selected:
            continue
        xs = [float(event["time_sec"]) for event in selected]
        ys = [_value_at(time, smooth, x, event.get("value_smooth")) for x, event in zip(xs, selected)]
        ax.scatter(
            xs,
            ys,
            s=72,
            color=color,
            marker="*",
            edgecolors="white",
            linewidths=0.8,
            alpha=0.96,
            label=f"Pass3 {event_type} +{len(selected)}",
            zorder=6,
        )


def _stage_summary(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    events = list(events)
    return {"total": len(events), "by_type": dict(Counter(event.get("type") for event in events))}


def _value_at(time: np.ndarray, smooth: np.ndarray, x: float, fallback: Any = None) -> float:
    if fallback is not None:
        return float(fallback)
    if len(time) and len(smooth):
        return float(smooth[int(np.argmin(np.abs(time - x)))])
    return 0.5


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot original -> Pass2 -> Pass3 episode-level comparison.")
    parser.add_argument("--episode-index", type=int, action="append", required=True)
    parser.add_argument("--keyframe-root", default="/mnt/data/chachaxu/dataset/abc_130k_train_key_frame")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", default="/mnt/data/yuluo/data/abc_130k_v3_train")
    parser.add_argument("--phase-module-root", default="/mnt/workspace/temporal_boundary_detection")
    args = parser.parse_args(argv)
    results = [
        plot_episode_three_stage(
            episode_index=episode_index,
            keyframe_root=args.keyframe_root,
            output_root=args.output_root,
            dataset_root=args.dataset_root,
            phase_module_root=args.phase_module_root,
        )
        for episode_index in args.episode_index
    ]
    aggregate_path = Path(args.output_root) / "three_stage_before_after_comparison.summary.json"
    write_json(aggregate_path, {"results": results})
    print(json.dumps({"summary_path": str(aggregate_path), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
