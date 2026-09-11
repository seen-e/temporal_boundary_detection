"""Command-line interface for gripper phase segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .dual import segment_dual_gripper_trajectory
from .segmenter import segment_gripper_trajectory
from .visualize import plot_dual_gripper_phase_result, plot_gripper_phase_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Segment a gripper trajectory into low-level motion phases.")
    parser.add_argument("--input", required=True, help="Input trajectory file: parquet, csv, or json/jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory where boundaries/segments/debug files are saved.")
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--frame-col", default="frame_index")
    parser.add_argument("--gripper-col", default=None, help="Explicit gripper column to segment.")
    parser.add_argument("--side", choices=["left", "right"], default="left", help="Column shortcut if --gripper-col is absent.")
    parser.add_argument("--dual", action="store_true", help="Run dual-gripper segmentation and boundary fusion.")
    parser.add_argument("--left-gripper-col", default="left_gripper")
    parser.add_argument("--right-gripper-col", default="right_gripper")
    parser.add_argument("--plot", action="store_true", help="Save debug PNG.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_table(input_path)
    if args.timestamp_col not in df:
        raise KeyError(f"missing timestamp column: {args.timestamp_col}")

    timestamps = df[args.timestamp_col].to_numpy(dtype=float)
    frame_indices = df[args.frame_col].to_numpy(dtype=int) if args.frame_col in df else np.arange(len(df), dtype=int)

    config = GripperPhaseSegmentationConfig.from_yaml(args.config) if args.config else GripperPhaseSegmentationConfig()
    if args.dual:
        left_values = df[args.left_gripper_col].to_numpy(dtype=float) if args.left_gripper_col in df else None
        right_values = df[args.right_gripper_col].to_numpy(dtype=float) if args.right_gripper_col in df else None
        if left_values is None and right_values is None:
            raise KeyError(f"missing dual gripper columns: {args.left_gripper_col}, {args.right_gripper_col}")
        result = segment_dual_gripper_trajectory(
            timestamps=timestamps,
            left_gripper_values=left_values,
            right_gripper_values=right_values,
            config=config,
            frame_indices=frame_indices,
        )
        _write_json(output_dir / "global_boundaries.json", result.global_boundaries_dict())
        _write_json(output_dir / "global_segments.json", result.global_segments_dict())
        _write_json(output_dir / "fusion_diagnostics.json", result.fusion_diagnostics)
        if result.left_result is not None:
            _write_json(output_dir / "left_boundaries.json", result.left_result.boundaries_dict())
            _write_json(output_dir / "left_segments.json", result.left_result.segments_dict())
            _write_json(output_dir / "left_extrema.json", result.left_result.extrema_dict())
            _write_json(output_dir / "left_diagnostics.json", result.left_result.diagnostics_dict())
        if result.right_result is not None:
            _write_json(output_dir / "right_boundaries.json", result.right_result.boundaries_dict())
            _write_json(output_dir / "right_segments.json", result.right_result.segments_dict())
            _write_json(output_dir / "right_extrema.json", result.right_result.extrema_dict())
            _write_json(output_dir / "right_diagnostics.json", result.right_result.diagnostics_dict())
        if args.plot or config.visualization.enabled:
            plot_dual_gripper_phase_result(result, output_dir / "dual_gripper_phase_debug.png")
        print(f"global_boundaries: {len(result.global_boundaries)}")
        print(f"global_segments: {len(result.global_segments)}")
        print(f"output_dir: {output_dir}")
        return 0

    gripper_col = args.gripper_col or f"{args.side}_gripper"
    if gripper_col not in df:
        raise KeyError(f"missing gripper column: {gripper_col}")
    gripper_values = df[gripper_col].to_numpy(dtype=float)
    result = segment_gripper_trajectory(
        timestamps=timestamps,
        gripper_values=gripper_values,
        config=config,
        frame_indices=frame_indices,
        side=args.side,
    )

    _write_json(output_dir / "boundaries.json", result.boundaries_dict())
    _write_json(output_dir / "segments.json", result.segments_dict())
    _write_json(output_dir / "extrema.json", result.extrema_dict())
    _write_json(output_dir / "diagnostics.json", result.diagnostics_dict())

    if args.plot or config.visualization.enabled:
        plot_gripper_phase_result(result, output_dir / "gripper_phase_debug.png", title=gripper_col)

    print(f"boundaries: {len(result.boundaries)}")
    print(f"segments: {len(result.segments)}")
    print(f"output_dir: {output_dir}")
    return 0


def _read_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported input extension: {suffix}")


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
