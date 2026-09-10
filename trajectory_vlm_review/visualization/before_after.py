from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data.io_utils import load_gripper_trajectory, normalize_events, read_json, write_json


EVENT_COLORS = {"MAX": "#d62728", "MIN": "#1f77b4", "PL": "#2ca02c", "PR": "#9467bd"}
STATUS_COLORS = {"removed": "#ff7f0e", "pending_relocalize": "#e377c2", "manual_review": "#8c564b"}


def plot_before_after(
    keyframe_json: str,
    output_root: str,
    episode_id: str,
    episode_index: int,
    dataset_root: str,
    phase_module_root: str,
    out_path: Optional[str] = None,
    show_changed_markers: bool = True,
) -> Dict[str, Any]:
    keyframe_data = read_json(keyframe_json)
    out_dir = Path(output_root) / episode_id
    fig, axes = plt.subplots(4, 1, figsize=(25, 13), dpi=170, sharex=True)
    rows = [("left", "before"), ("left", "after"), ("right", "before"), ("right", "after")]
    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    changed: Dict[str, List[Dict[str, Any]]] = {}
    max_time = 0.0

    for ax, (arm, stage) in zip(axes, rows):
        time, raw, smooth = load_gripper_trajectory(
            keyframe_data,
            arm,
            phase_module_root,
            dataset_root=dataset_root,
            episode_index=episode_index,
        )
        max_time = max(max_time, float(time[-1]) if len(time) else 0.0)
        ax.plot(time, smooth, color="#111111", lw=1.35, alpha=0.98, label="smoothed")

        if stage == "before":
            events = [
                {
                    "event_id": event.event_id,
                    "type": event.event_type,
                    "time_sec": event.time,
                    "value_smooth": event.value_smooth,
                    "final_status": "candidate",
                    "vlm_action": "candidate",
                }
                for event in normalize_events(keyframe_data, arm)
            ]
            draw_events = events
            title_stage = "Before VLM: traditional cleaned candidates"
        else:
            final = read_json(out_dir / arm / "final_review.json")
            events = final["final_events"]
            draw_events = [event for event in events if event.get("final_status") not in {"removed", "merged_removed"}]
            title_stage = "After VLM: final kept + pending relocalize"
            changed[arm] = [
                event
                for event in events
                if event.get("vlm_action") not in {"KEEP", None} or event.get("final_status") != "kept"
            ]

        summary.setdefault(arm, {})[stage] = dict(Counter(event.get("type") for event in draw_events))
        _draw_events(ax, time, smooth, draw_events)
        if stage == "after" and show_changed_markers:
            _draw_changed_events(ax, time, smooth, events)

        ax.set_ylabel(f"{arm}\n{stage}")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.22, lw=0.45)
        ax.legend(loc="upper right", ncol=7, fontsize=8)
        ax.set_title(f"{episode_id} {arm} | {title_stage} | total={sum(summary[arm][stage].values())}")

    axes[-1].set_xlabel("time (sec)")
    axes[-1].set_xlim(0, max_time)
    fig.suptitle(f"{episode_id}: VLM before/after keyframe review with changed events highlighted", y=0.996, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    image_path = Path(out_path) if out_path else out_dir / "vlm_before_after_comparison.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path)
    plt.close(fig)

    change_summary = {
        arm: [
            {
                "event_id": event.get("event_id"),
                "type": event.get("type"),
                "time_sec": event.get("time_sec"),
                "vlm_action": event.get("vlm_action"),
                "final_status": event.get("final_status"),
                "reason": (event.get("review") or {}).get("reason"),
                "relocalize": (event.get("review") or {}).get("relocalize"),
            }
            for event in events
        ]
        for arm, events in changed.items()
    }
    result = {
        "episode_id": episode_id,
        "episode_index": episode_index,
        "summary": summary,
        "changed_events": change_summary,
        "image_path": str(image_path),
    }
    write_json(image_path.with_suffix(".summary.json"), result)
    return result


def _draw_events(ax, time: np.ndarray, smooth: np.ndarray, events: List[Dict[str, Any]]) -> None:
    for event_type in ["MAX", "MIN", "PL", "PR"]:
        selected = [event for event in events if event.get("type") == event_type]
        xs = [float(event["time_sec"]) for event in selected]
        ys = [_value_at(time, smooth, x, event.get("value_smooth")) for x, event in zip(xs, selected)]
        ax.scatter(xs, ys, s=13, color=EVENT_COLORS[event_type], label=f"{event_type} {len(selected)}", zorder=4, alpha=0.9)


def _draw_changed_events(ax, time: np.ndarray, smooth: np.ndarray, events: List[Dict[str, Any]]) -> None:
    labels = set()
    for event in events:
        status = event.get("final_status")
        if status not in {"removed", "pending_relocalize", "manual_review"}:
            continue
        x = float(event["time_sec"])
        y = _value_at(time, smooth, x, event.get("value_smooth"))
        color = STATUS_COLORS[status]
        label = status if status not in labels else None
        labels.add(status)
        if status == "removed":
            ax.scatter([x], [y], s=95, color=color, marker="x", linewidths=2.2, label=label, zorder=6)
            ax.text(x, y + 0.06, f"{event['event_id']} REMOVE", fontsize=7, color=color, ha="center")
        else:
            ax.scatter([x], [y], s=95, facecolors="none", edgecolors=color, linewidths=2.0, marker="s", label=label, zorder=6)
            relocalize = (event.get("review") or {}).get("relocalize") or {}
            ax.text(x, y + 0.06, f"{event['event_id']} RELOC {relocalize.get('direction', '')}", fontsize=7, color=color, ha="center")


def _value_at(time: np.ndarray, smooth: np.ndarray, x: float, fallback: Any) -> float:
    if fallback is not None:
        return float(fallback)
    if len(time) and len(smooth):
        return float(smooth[int(np.argmin(np.abs(time - x)))])
    return 0.5


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot before/after VLM trajectory review comparison.")
    parser.add_argument("--keyframe-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--dataset-root", default="/mnt/data/yuluo/data/abc_130k_v3_train")
    parser.add_argument("--phase-module-root", default="/mnt/workspace/temporal_boundary_detection")
    parser.add_argument("--out-path")
    parser.add_argument("--hide-changed-markers", action="store_true")
    args = parser.parse_args(argv)
    kwargs = vars(args)
    kwargs["show_changed_markers"] = not kwargs.pop("hide_changed_markers")
    result = plot_before_after(**kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
