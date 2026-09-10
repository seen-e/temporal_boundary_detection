"""Batch gripper phase segmentation for the extracted abc_40task dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import GripperPhaseSegmentationConfig
from .dual import segment_dual_gripper_trajectory
from .segmenter import segment_gripper_trajectory
from .visualize import plot_dual_gripper_phase_result, plot_gripper_phase_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run gripper phase segmentation for every abc_40task episode.")
    parser.add_argument("--dataset-root", default="/mnt/data/chachaxu/dataset/abc_40task")
    parser.add_argument("--config", default=None, help="Optional YAML config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max episodes for smoke testing.")
    parser.add_argument("--task-limit", type=int, default=None, help="Process only the first N task directories.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip episodes that already have all three PNG files.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    config = GripperPhaseSegmentationConfig.from_yaml(args.config) if args.config else GripperPhaseSegmentationConfig()
    task_dirs = sorted(path for path in dataset_root.glob("task_*") if path.is_dir())
    if args.task_limit is not None:
        task_dirs = task_dirs[: args.task_limit]
    trajectory_files = []
    for task_dir in task_dirs:
        trajectory_files.extend(sorted(task_dir.glob("episode_*/trajectory.parquet")))
    if args.limit is not None:
        trajectory_files = trajectory_files[: args.limit]
    if not trajectory_files:
        raise FileNotFoundError(f"no trajectory.parquet files found under {dataset_root}")

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "episodes_expected": len(trajectory_files),
        "episodes_processed": 0,
        "episodes_skipped": 0,
        "errors": [],
        "outputs_per_episode": {
            "png": [
                "left_gripper_phase.png",
                "right_gripper_phase.png",
                "dual_gripper_phase.png",
            ],
            "json_dir": "gripper_phase_segment",
        },
    }

    for idx, trajectory_path in enumerate(trajectory_files, 1):
        episode_dir = trajectory_path.parent
        png_paths = [
            episode_dir / "left_gripper_phase.png",
            episode_dir / "right_gripper_phase.png",
            episode_dir / "dual_gripper_phase.png",
        ]
        if args.skip_existing and all(path.exists() for path in png_paths):
            summary["episodes_skipped"] += 1
            continue

        try:
            process_episode(trajectory_path, config)
            summary["episodes_processed"] += 1
        except Exception as exc:  # noqa: BLE001 - batch should continue and summarize failures.
            summary["errors"].append(
                {
                    "trajectory": str(trajectory_path),
                    "error": repr(exc),
                }
            )

        if idx == 1 or idx % 25 == 0 or idx == len(trajectory_files):
            print(
                f"processed {idx}/{len(trajectory_files)} "
                f"ok={summary['episodes_processed']} skipped={summary['episodes_skipped']} "
                f"errors={len(summary['errors'])}",
                flush=True,
            )

    log_dir = dataset_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "gripper_phase_batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"summary: {summary_path}")
    if summary["errors"]:
        return 1
    return 0


def process_episode(trajectory_path: Path, config: GripperPhaseSegmentationConfig) -> None:
    episode_dir = trajectory_path.parent
    result_dir = episode_dir / "gripper_phase_segment"
    result_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(
        trajectory_path,
        columns=["timestamp", "frame_index", "left_gripper", "right_gripper"],
    )
    timestamps = df["timestamp"].to_numpy(dtype=float)
    frame_indices = df["frame_index"].to_numpy(dtype=int)
    left_values = df["left_gripper"].to_numpy(dtype=float)
    right_values = df["right_gripper"].to_numpy(dtype=float)

    left_result = segment_gripper_trajectory(
        timestamps=timestamps,
        gripper_values=left_values,
        config=config,
        frame_indices=frame_indices,
        side="left",
    )
    right_result = segment_gripper_trajectory(
        timestamps=timestamps,
        gripper_values=right_values,
        config=config,
        frame_indices=frame_indices,
        side="right",
    )
    dual_result = segment_dual_gripper_trajectory(
        timestamps=timestamps,
        left_gripper_values=left_values,
        right_gripper_values=right_values,
        config=config,
        frame_indices=frame_indices,
    )

    _write_json(result_dir / "left_boundaries.json", left_result.boundaries_dict())
    _write_json(result_dir / "left_segments.json", left_result.segments_dict())
    _write_json(result_dir / "left_extrema.json", left_result.extrema_dict())
    _write_json(result_dir / "left_diagnostics.json", left_result.diagnostics_dict())
    _write_json(result_dir / "right_boundaries.json", right_result.boundaries_dict())
    _write_json(result_dir / "right_segments.json", right_result.segments_dict())
    _write_json(result_dir / "right_extrema.json", right_result.extrema_dict())
    _write_json(result_dir / "right_diagnostics.json", right_result.diagnostics_dict())
    _write_json(result_dir / "global_boundaries.json", dual_result.global_boundaries_dict())
    _write_json(result_dir / "global_segments.json", dual_result.global_segments_dict())
    _write_json(result_dir / "fusion_diagnostics.json", dual_result.fusion_diagnostics)

    plot_gripper_phase_result(left_result, episode_dir / "left_gripper_phase.png", title="Left gripper")
    plot_gripper_phase_result(right_result, episode_dir / "right_gripper_phase.png", title="Right gripper")
    plot_dual_gripper_phase_result(dual_result, episode_dir / "dual_gripper_phase.png")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
