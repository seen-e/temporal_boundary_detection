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

from ..data import load_gripper_trajectory, read_json, write_json


EVENT_COLORS = {"MAX": "#d62728", "MIN": "#1f77b4", "PL": "#2ca02c", "PR": "#9467bd"}


def plot_final_overview(
    keyframe_json: str,
    output_root: str,
    episode_id: str,
    episode_index: int,
    dataset_root: str,
    phase_module_root: str,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    keyframe_data = read_json(keyframe_json)
    output_root_path = Path(output_root)
    fig, axes = plt.subplots(2, 1, figsize=(25, 8), dpi=170, sharex=True)
    summary: Dict[str, Any] = {}
    max_time = 0.0

    for ax, arm in zip(axes, ["left", "right"]):
        final = read_json(output_root_path / episode_id / arm / "final_review.json")
        time, raw, smooth = load_gripper_trajectory(
            keyframe_data,
            arm,
            phase_module_root,
            dataset_root=dataset_root,
            episode_index=episode_index,
        )
        max_time = max(max_time, float(time[-1]) if len(time) else 0.0)
        kept = [event for event in final["final_events"] if event.get("final_status") not in {"removed", "merged_removed"}]
        removed = [event for event in final["final_events"] if event.get("final_status") in {"removed", "merged_removed"}]

        ax.plot(time, smooth, color="#111111", lw=1.35, label="smoothed")
        _draw_events(ax, time, smooth, kept)
        add_requests = final.get("localization_requests", [])
        _draw_add_requests(ax, time, smooth, add_requests)

        summary[arm] = {
            "kept_total": len(kept),
            "removed_total": len(removed),
            "add_pending": len(add_requests),
            "kept_by_type": dict(Counter(event.get("type") for event in kept)),
        }
        ax.set_title(
            f"{episode_id} {arm} final VLM-reviewed trajectory | "
            f"kept={len(kept)} removed={len(removed)} add_pending={len(add_requests)}"
        )
        ax.set_ylabel(f"{arm} gripper")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.22, lw=0.45)
        ax.legend(loc="upper right", ncol=7, fontsize=8)

    axes[-1].set_xlabel("time (sec)")
    axes[-1].set_xlim(0, max_time)
    fig.suptitle(f"{episode_id}: final VLM-reviewed gripper trajectory segmentation", y=0.995, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    image_path = Path(out_path) if out_path else output_root_path / episode_id / "final_vlm_reviewed_trajectory_overview.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path)
    plt.close(fig)

    result = {"episode_id": episode_id, "episode_index": episode_index, "summary": summary, "image_path": str(image_path)}
    write_json(image_path.with_suffix(".summary.json"), result)
    return result


def _draw_events(ax, time: np.ndarray, smooth: np.ndarray, events: List[Dict[str, Any]]) -> None:
    for event_type in ["MAX", "MIN", "PL", "PR"]:
        selected = [event for event in events if event.get("type") == event_type]
        xs = [float(event["time_sec"]) for event in selected]
        ys = [_value_at(time, smooth, x, event.get("value_smooth")) for x, event in zip(xs, selected)]
        ax.scatter(xs, ys, s=18, color=EVENT_COLORS[event_type], label=f"{event_type} {len(selected)}", zorder=4, alpha=0.9)


def _draw_add_requests(ax, time: np.ndarray, smooth: np.ndarray, add_requests: List[Dict[str, Any]]) -> None:
    for idx, add in enumerate(add_requests):
        x = float(add.get("approx_time_sec") or 0.0)
        y = _value_at(time, smooth, x, None)
        ax.scatter(
            [x],
            [y],
            s=90,
            facecolors="none",
            edgecolors="#ff7f0e",
            marker="*",
            linewidths=1.6,
            label="pending ADD" if idx == 0 else None,
            zorder=6,
        )
        ax.text(x, y + 0.06, add.get("new_event_id", "NEW"), fontsize=7, color="#ff7f0e", ha="center")


def _value_at(time: np.ndarray, smooth: np.ndarray, x: float, fallback: Any) -> float:
    if len(time) and len(smooth):
        return float(smooth[int(np.argmin(np.abs(time - x)))])
    return float(fallback or 0.5)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot final VLM-reviewed trajectory overview.")
    parser.add_argument("--keyframe-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--dataset-root", default="/mnt/data/yuluo/data/abc_130k_v3_train")
    parser.add_argument("--phase-module-root", default="/mnt/workspace/temporal_boundary_detection")
    parser.add_argument("--out-path")
    args = parser.parse_args(argv)
    result = plot_final_overview(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
