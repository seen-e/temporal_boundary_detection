"""Dual-gripper segmentation and cross-side boundary fusion."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .segmenter import segment_gripper_trajectory
from .types import (
    Boundary,
    DualGripperSegmentationResult,
    FusedBoundary,
    GlobalSegment,
    GripperPhaseSegmentationResult,
)
from .utils import estimate_sample_period


def segment_dual_gripper_trajectory(
    timestamps: np.ndarray,
    left_gripper_values: np.ndarray | None = None,
    right_gripper_values: np.ndarray | None = None,
    config: GripperPhaseSegmentationConfig | dict[str, Any] | None = None,
    frame_indices: np.ndarray | None = None,
) -> DualGripperSegmentationResult:
    """Segment left/right grippers independently, then fuse their boundaries."""

    if left_gripper_values is None and right_gripper_values is None:
        raise ValueError("at least one of left_gripper_values or right_gripper_values is required")

    cfg = config if isinstance(config, GripperPhaseSegmentationConfig) else GripperPhaseSegmentationConfig.from_dict(config)
    timestamps = np.asarray(timestamps, dtype=float)
    if frame_indices is None:
        frame_indices = np.arange(len(timestamps), dtype=int)
    else:
        frame_indices = np.asarray(frame_indices, dtype=int)

    left_result = None
    right_result = None
    if left_gripper_values is not None:
        left_result = segment_gripper_trajectory(
            timestamps=timestamps,
            gripper_values=np.asarray(left_gripper_values, dtype=float),
            config=cfg,
            frame_indices=frame_indices,
            side="left",
        )
    if right_gripper_values is not None:
        right_result = segment_gripper_trajectory(
            timestamps=timestamps,
            gripper_values=np.asarray(right_gripper_values, dtype=float),
            config=cfg,
            frame_indices=frame_indices,
            side="right",
        )

    global_boundaries, diagnostics = fuse_gripper_boundaries(
        left_result.boundaries if left_result else [],
        right_result.boundaries if right_result else [],
        timestamps=timestamps,
        frame_indices=frame_indices,
        config=cfg,
    )
    global_segments = build_global_segments(timestamps, frame_indices, global_boundaries)

    diagnostics.update(
        {
            "mode": "dual" if left_result is not None and right_result is not None else "single_side",
            "has_left": left_result is not None,
            "has_right": right_result is not None,
        }
    )

    return DualGripperSegmentationResult(
        left_result=left_result,
        right_result=right_result,
        global_boundaries=global_boundaries,
        global_segments=global_segments,
        fusion_diagnostics=diagnostics,
    )


def fuse_gripper_boundaries(
    left_boundaries: list[Boundary],
    right_boundaries: list[Boundary],
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    config: GripperPhaseSegmentationConfig,
) -> tuple[list[FusedBoundary], dict[str, Any]]:
    """Fuse clean single-side boundaries with one-to-one nearest matching."""

    merge_window = float(config.dual_gripper.fusion.merge_window_sec)
    time_strategy = config.dual_gripper.fusion.time_strategy
    confidence_strategy = config.dual_gripper.fusion.confidence_strategy

    if not config.dual_gripper.enabled or not left_boundaries or not right_boundaries:
        all_single = [
            _single_to_fused(boundary, timestamps, frame_indices)
            for boundary in [*left_boundaries, *right_boundaries]
        ]
        all_single.sort(key=lambda item: item.time)
        diagnostics = _fusion_diagnostics(
            left_boundaries,
            right_boundaries,
            merged_pairs=0,
            merge_window=merge_window,
            time_strategy=time_strategy,
            confidence_strategy=confidence_strategy,
        )
        return all_single, diagnostics

    pairs: list[tuple[float, int, int]] = []
    for left_idx, left in enumerate(left_boundaries):
        for right_idx, right in enumerate(right_boundaries):
            delta = abs(float(left.time) - float(right.time))
            if delta <= merge_window:
                pairs.append((delta, left_idx, right_idx))
    pairs.sort(key=lambda item: item[0])

    matched_left: set[int] = set()
    matched_right: set[int] = set()
    merged: list[FusedBoundary] = []

    for _, left_idx, right_idx in pairs:
        if left_idx in matched_left or right_idx in matched_right:
            continue
        left = left_boundaries[left_idx]
        right = right_boundaries[right_idx]
        merged.append(
            _merge_pair(
                left,
                right,
                timestamps=timestamps,
                frame_indices=frame_indices,
                time_strategy=time_strategy,
                confidence_strategy=confidence_strategy,
            )
        )
        matched_left.add(left_idx)
        matched_right.add(right_idx)

    for idx, boundary in enumerate(left_boundaries):
        if idx not in matched_left:
            merged.append(_single_to_fused(boundary, timestamps, frame_indices))
    for idx, boundary in enumerate(right_boundaries):
        if idx not in matched_right:
            merged.append(_single_to_fused(boundary, timestamps, frame_indices))

    merged.sort(key=lambda item: item.time)
    diagnostics = _fusion_diagnostics(
        left_boundaries,
        right_boundaries,
        merged_pairs=len(matched_left),
        merge_window=merge_window,
        time_strategy=time_strategy,
        confidence_strategy=confidence_strategy,
    )
    diagnostics["num_unmatched_left"] = len(left_boundaries) - len(matched_left)
    diagnostics["num_unmatched_right"] = len(right_boundaries) - len(matched_right)
    return merged, diagnostics


def build_global_segments(
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    global_boundaries: list[FusedBoundary],
) -> list[GlobalSegment]:
    """Build global segments from episode start/end and fused boundary times."""

    if len(timestamps) == 0:
        return []
    dt = estimate_sample_period(timestamps)
    split_times = [float(timestamps[0]), *[b.time for b in global_boundaries], float(timestamps[-1] + dt)]
    split_frames = [int(frame_indices[0]), *[b.frame_index for b in global_boundaries], int(frame_indices[-1] + 1)]

    segments: list[GlobalSegment] = []
    for idx in range(len(split_times) - 1):
        start_time = split_times[idx]
        end_time = split_times[idx + 1]
        start_frame = split_frames[idx]
        end_frame = split_frames[idx + 1] - 1
        segments.append(
            GlobalSegment(
                segment_id=idx,
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_sec=max(0.0, end_time - start_time),
            )
        )
    return segments


def _merge_pair(
    left: Boundary,
    right: Boundary,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    time_strategy: str,
    confidence_strategy: str,
) -> FusedBoundary:
    sources = [_boundary_source(left), _boundary_source(right)]
    times = np.asarray([left.time, right.time], dtype=float)
    confidences = np.asarray([left.confidence, right.confidence], dtype=float)
    time = _merge_time(times, confidences, time_strategy)
    confidence = _merge_confidence(confidences, confidence_strategy)
    return FusedBoundary(
        time=time,
        frame_index=_nearest_frame(time, timestamps, frame_indices),
        source="dual_gripper_fusion",
        confidence=confidence,
        source_sides=["left", "right"],
        sources=sources,
        spread_sec=float(np.max(times) - np.min(times)),
    )


def _single_to_fused(boundary: Boundary, timestamps: np.ndarray, frame_indices: np.ndarray) -> FusedBoundary:
    side = boundary.side if boundary.side in {"left", "right"} else "single"
    source = f"{side}_gripper" if side in {"left", "right"} else "gripper"
    return FusedBoundary(
        time=float(boundary.time),
        frame_index=_nearest_frame(float(boundary.time), timestamps, frame_indices),
        source=source,
        confidence=float(boundary.confidence),
        source_sides=[side],
        sources=[_boundary_source(boundary)],
        spread_sec=0.0,
    )


def _boundary_source(boundary: Boundary) -> dict[str, Any]:
    return {
        "side": boundary.side,
        "time": boundary.time,
        "frame_index": boundary.frame_index,
        "confidence": boundary.confidence,
        "from_state": boundary.from_state,
        "to_state": boundary.to_state,
        "velocity": boundary.velocity,
        "source": boundary.source,
    }


def _merge_time(times: np.ndarray, confidences: np.ndarray, strategy: str) -> float:
    strategy = strategy.lower()
    if strategy == "mean":
        return float(np.mean(times))
    if strategy == "median":
        return float(np.median(times))
    if strategy == "weighted_mean":
        valid = np.isfinite(confidences) & (confidences > 0)
        if not np.any(valid):
            return float(np.median(times))
        return float(np.sum(times[valid] * confidences[valid]) / np.sum(confidences[valid]))
    raise ValueError(f"unsupported dual fusion time_strategy: {strategy}")


def _merge_confidence(confidences: np.ndarray, strategy: str) -> float:
    strategy = strategy.lower()
    valid = confidences[np.isfinite(confidences)]
    if valid.size == 0:
        return 0.0
    if strategy == "max":
        return float(np.max(valid))
    if strategy in {"mean", "weighted_mean"}:
        return float(np.mean(valid))
    if strategy == "probabilistic_or":
        return float(1.0 - np.prod(1.0 - np.clip(valid, 0.0, 1.0)))
    raise ValueError(f"unsupported dual fusion confidence_strategy: {strategy}")


def _nearest_frame(time: float, timestamps: np.ndarray, frame_indices: np.ndarray) -> int:
    if len(timestamps) == 0:
        return 0
    idx = int(np.argmin(np.abs(timestamps.astype(float) - float(time))))
    return int(frame_indices[idx])


def _fusion_diagnostics(
    left_boundaries: list[Boundary],
    right_boundaries: list[Boundary],
    merged_pairs: int,
    merge_window: float,
    time_strategy: str,
    confidence_strategy: str,
) -> dict[str, Any]:
    return {
        "num_left_boundaries": len(left_boundaries),
        "num_right_boundaries": len(right_boundaries),
        "num_merged_pairs": merged_pairs,
        "num_unmatched_left": len(left_boundaries) - merged_pairs,
        "num_unmatched_right": len(right_boundaries) - merged_pairs,
        "merge_window_sec": merge_window,
        "time_strategy": time_strategy,
        "confidence_strategy": confidence_strategy,
        "one_to_one_matching": True,
    }

