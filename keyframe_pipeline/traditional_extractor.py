from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from phase_segment.gripper_phase_segment.config import GripperPhaseSegmentationConfig
from phase_segment.gripper_phase_segment.segmenter import segment_gripper_trajectory

from .index_io import extract_gripper_values, json_safe
from .traditional.extrema_base_detector import ExtremaBaseConfig, annotate_extrema_bases
from .traditional.gripper_keypoint_filter import KeypointFilterConfig, filter_keyframes
from .traditional.post_base_filter import PostBaseFilterConfig, filter_after_extrema_bases


def extract_episode_from_index(
    file_df: pd.DataFrame,
    episode: Dict[str, Any],
    data_file: str | Path,
    phase_cfg: GripperPhaseSegmentationConfig,
    filter_cfg: KeypointFilterConfig,
    base_cfg: ExtremaBaseConfig,
    post_base_cfg: PostBaseFilterConfig,
    full_config: Dict[str, Any],
) -> Dict[str, Any]:
    left_segment = episode["left_gripper"]
    right_segment = episode["right_gripper"]
    row_start = int(left_segment["row_start"])
    row_end = int(left_segment["row_end"])
    ep_df = file_df.iloc[row_start:row_end].copy()
    if "episode_index" in ep_df.columns and not (ep_df["episode_index"].to_numpy() == int(episode["episode_index"])).all():
        raise ValueError(f"episode_index mismatch in {data_file} for episode {episode['episode_index']}")
    if "frame_index" in ep_df.columns:
        ep_df = ep_df.sort_values("frame_index", kind="mergesort")
    timestamps = ep_df["timestamp"].to_numpy(dtype=float)
    frame_indices = ep_df["frame_index"].to_numpy(dtype=int) if "frame_index" in ep_df.columns else np.arange(len(ep_df), dtype=int)
    global_indices = ep_df["index"].to_numpy(dtype=int) if "index" in ep_df.columns else None

    left_values = extract_gripper_values(file_df, left_segment)
    right_values = extract_gripper_values(file_df, right_segment)
    left_result = segment_gripper_trajectory(timestamps, left_values, config=phase_cfg, frame_indices=frame_indices, side="left")
    right_result = segment_gripper_trajectory(timestamps, right_values, config=phase_cfg, frame_indices=frame_indices, side="right")

    task_index = episode.get("task_index")
    if task_index is None and "task_index" in ep_df.columns:
        values = ep_df["task_index"].dropna().unique()
        task_index = int(values[0]) if len(values) else None

    return {
        "status": "complete",
        "schema_version": "2.0",
        "source": "index_traditional_then_vlm_pipeline",
        "dataset": full_config.get("dataset", {}).get("name") or Path(str(full_config["paths"]["index_json"])).stem,
        "episode_id": episode.get("episode_id") or f"episode_{int(episode['episode_index']):06d}",
        "episode_index": int(episode["episode_index"]),
        "task_index": task_index,
        "task": episode.get("task"),
        "instruction": episode.get("instruction"),
        "fps": float(episode.get("fps") or 0.0),
        "frame_count": int(episode.get("frame_count") or len(ep_df)),
        "data_file": str(data_file),
        "source_data_file": str(data_file),
        "index_json": str(full_config["paths"]["index_json"]),
        "phase_config": {"gripper_phase_segmentation": full_config.get("gripper_phase_segmentation", {})},
        "keyframe_types": ["local_maximum", "local_minimum", "plateau_left_endpoint", "plateau_right_endpoint"],
        "left_gripper": extract_side_keyframes("left", left_values, left_result, global_indices, filter_cfg, base_cfg, post_base_cfg),
        "right_gripper": extract_side_keyframes("right", right_values, right_result, global_indices, filter_cfg, base_cfg, post_base_cfg),
        "video_segments": episode.get("video_segments", {}),
        "index_episode": {
            "left_gripper": left_segment,
            "right_gripper": right_segment,
            "global_row_start": episode.get("global_row_start"),
            "global_row_end": episode.get("global_row_end"),
        },
    }


def extract_side_keyframes(
    side: str,
    raw_values: np.ndarray,
    result: Any,
    global_indices: Optional[np.ndarray],
    filter_cfg: KeypointFilterConfig,
    base_cfg: ExtremaBaseConfig,
    post_base_cfg: PostBaseFilterConfig,
) -> Dict[str, Any]:
    keyframes = []
    for point in result.extrema:
        keyframes.append(
            _keyframe_record(
                "extremum",
                "local_maximum" if point.type == "local_max" else "local_minimum",
                side,
                int(point.index),
                float(point.time),
                int(point.frame_index),
                raw_values,
                result.gripper_smooth,
                global_indices,
                {"prominence": float(point.prominence)},
            )
        )

    plateau_id = 0
    for segment in result.segments:
        if segment.motion_state != "PLATEAU":
            continue
        start_idx = _nearest_frame_index(result.frame_indices, int(segment.start_frame))
        end_idx = _nearest_frame_index(result.frame_indices, int(segment.end_frame))
        extra = {
            "plateau_id": plateau_id,
            "plateau_start_time": float(segment.start_time),
            "plateau_end_time": float(segment.end_time),
            "plateau_duration_sec": float(segment.duration_sec),
            "plateau_mean": float(segment.plateau_mean) if segment.plateau_mean is not None else None,
            "plateau_value_range": float(segment.value_range) if segment.value_range is not None else None,
        }
        keyframes.append(_keyframe_record("plateau_endpoint", "plateau_left_endpoint", side, start_idx, float(segment.start_time), int(segment.start_frame), raw_values, result.gripper_smooth, global_indices, extra))
        keyframes.append(_keyframe_record("plateau_endpoint", "plateau_right_endpoint", side, end_idx, float(segment.end_time), int(segment.end_frame), raw_values, result.gripper_smooth, global_indices, extra))
        plateau_id += 1

    keyframes.sort(key=lambda item: (item["time"], item["sample_index"], item["type"]))
    filtering = filter_keyframes(keyframes, result.timestamps, result.gripper_smooth, filter_cfg)
    cleaned = filtering.pop("cleaned_keyframes")
    cleaned, base_summary = annotate_extrema_bases(cleaned, result.timestamps, result.gripper_smooth, result.frame_indices, base_cfg)
    cleaned, post_summary = filter_after_extrema_bases(cleaned, post_base_cfg)
    return {
        "signal_index": 6 if side == "left" else 13,
        "num_keyframes": len(keyframes),
        "num_first_pass_cleaned_keyframes": int(filtering["cleaned_counts"]["total"]),
        "num_cleaned_keyframes": int(post_summary["output_counts"]["total"]),
        "num_extrema": len(result.extrema),
        "num_plateaus": int(result.diagnostics.get("num_plateaus", 0)),
        "num_candidate_plateaus": int(result.diagnostics.get("num_candidate_plateaus", result.diagnostics.get("num_plateaus", 0))),
        "num_suppressed_internal_plateaus": int(result.diagnostics.get("num_suppressed_internal_plateaus", 0)),
        "diagnostics": {
            "gripper_min": float(result.diagnostics.get("gripper_min", np.nan)),
            "gripper_max": float(result.diagnostics.get("gripper_max", np.nan)),
            "num_invalid_intervals": int(result.diagnostics.get("num_invalid_intervals", 0)),
            "plateau_thresholds": json_safe(result.diagnostics.get("plateau_thresholds", {})),
        },
        "keyframes": keyframes,
        "cleaned_keyframes": cleaned,
        "filtering": filtering,
        "extrema_base_detection": base_summary,
        "post_base_filtering": post_summary,
    }


def build_traditional_configs(config: Dict[str, Any]):
    phase_cfg = GripperPhaseSegmentationConfig.from_dict({"gripper_phase_segmentation": config.get("gripper_phase_segmentation", {})})
    filter_cfg = KeypointFilterConfig.from_mapping(config.get("keyframe_filter", {}))
    base_cfg = ExtremaBaseConfig.from_mapping(config.get("extrema_base_detection", {}))
    post_base_cfg = PostBaseFilterConfig.from_mapping(config.get("post_base_filter", {}))
    return phase_cfg, filter_cfg, base_cfg, post_base_cfg


def _keyframe_record(kind: str, event_type: str, side: str, sample_index: int, time: float, frame_index: int, raw_values: np.ndarray, smooth_values: np.ndarray, global_indices: Optional[np.ndarray], extra: Dict[str, Any]) -> Dict[str, Any]:
    sample_index = int(np.clip(sample_index, 0, len(smooth_values) - 1))
    record = {
        "type": event_type,
        "kind": kind,
        "side": side,
        "time": float(time),
        "time_sec": float(time),
        "frame_index": int(frame_index),
        "sample_index": sample_index,
        "global_index": int(global_indices[sample_index]) if global_indices is not None and len(global_indices) else None,
        "value_raw": float(raw_values[sample_index]),
        "value_smooth": float(smooth_values[sample_index]),
    }
    record.update(json_safe(extra))
    return record


def _nearest_frame_index(frame_indices: np.ndarray, frame: int) -> int:
    if len(frame_indices) == 0:
        return 0
    return int(np.argmin(np.abs(frame_indices.astype(int) - int(frame))))

