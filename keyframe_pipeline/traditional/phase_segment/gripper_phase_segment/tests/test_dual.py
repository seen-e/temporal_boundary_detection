from __future__ import annotations

import numpy as np

from keyframe_pipeline.traditional.phase_segment.gripper_phase_segment.config import GripperPhaseSegmentationConfig
from keyframe_pipeline.traditional.phase_segment.gripper_phase_segment.dual import fuse_gripper_boundaries, segment_dual_gripper_trajectory
from keyframe_pipeline.traditional.phase_segment.gripper_phase_segment.types import Boundary


def cfg():
    config = GripperPhaseSegmentationConfig()
    config.smoothing.method = "moving_average"
    config.velocity_smoothing.method = "moving_average"
    config.motion_state.threshold_mode = "fixed"
    config.motion_state.velocity_enter_threshold = 0.5
    config.motion_state.velocity_exit_threshold = 0.15
    config.temporal.min_state_duration_sec = 0.10
    config.segment.min_segment_duration_sec = 0.20
    config.segment.min_motion_event_duration_sec = 0.08
    config.dual_gripper.fusion.merge_window_sec = 0.15
    config.dual_gripper.fusion.time_strategy = "weighted_mean"
    config.dual_gripper.fusion.confidence_strategy = "max"
    return config


def make_t(seconds=6.0, fps=30):
    return np.arange(0.0, seconds, 1.0 / fps)


def move(t, start, end, a, b):
    g = np.ones_like(t) * a
    mask = (t >= start) & (t <= end)
    g[t > end] = b
    g[mask] = np.linspace(a, b, mask.sum())
    return g


def boundary(time, side, confidence=0.9, from_state="STABLE", to_state="NEGATIVE"):
    return Boundary(
        time=time,
        frame_index=int(round(time * 30)),
        from_state=from_state,
        to_state=to_state,
        source=f"{side}_gripper",
        confidence=confidence,
        velocity=-1.0,
        score=confidence,
        side=side,
    )


def test_sync_left_right_boundaries_merge():
    t = make_t()
    result = segment_dual_gripper_trajectory(
        timestamps=t,
        left_gripper_values=move(t, 2.0, 2.4, 0.8, 0.2),
        right_gripper_values=move(t, 2.03, 2.43, 0.8, 0.2),
        config=cfg(),
    )
    assert result.fusion_diagnostics["num_merged_pairs"] == 2
    assert all(b.source == "dual_gripper_fusion" for b in result.global_boundaries)
    assert all(b.source_sides == ["left", "right"] for b in result.global_boundaries)


def test_async_left_right_boundaries_are_kept_separate():
    t = make_t()
    result = segment_dual_gripper_trajectory(
        timestamps=t,
        left_gripper_values=move(t, 2.0, 2.4, 0.8, 0.2),
        right_gripper_values=move(t, 4.0, 4.4, 0.8, 0.2),
        config=cfg(),
    )
    assert result.fusion_diagnostics["num_merged_pairs"] == 0
    assert len(result.global_boundaries) == 4
    assert {tuple(b.source_sides) for b in result.global_boundaries} == {("left",), ("right",)}


def test_left_only_degrades_to_single_side_mode():
    t = make_t()
    result = segment_dual_gripper_trajectory(t, left_gripper_values=move(t, 2.0, 2.4, 0.8, 0.2), config=cfg())
    assert result.left_result is not None
    assert result.right_result is None
    assert result.fusion_diagnostics["mode"] == "single_side"
    assert len(result.global_boundaries) == len(result.left_result.boundaries)
    assert all(b.source_sides == ["left"] for b in result.global_boundaries)


def test_right_only_degrades_to_single_side_mode():
    t = make_t()
    result = segment_dual_gripper_trajectory(t, right_gripper_values=move(t, 2.0, 2.4, 0.8, 0.2), config=cfg())
    assert result.right_result is not None
    assert result.left_result is None
    assert result.fusion_diagnostics["mode"] == "single_side"
    assert len(result.global_boundaries) == len(result.right_result.boundaries)
    assert all(b.source_sides == ["right"] for b in result.global_boundaries)


def test_weighted_merge_time_uses_confidence():
    t = make_t()
    frame = np.arange(len(t))
    fused, _ = fuse_gripper_boundaries(
        [boundary(5.00, "left", confidence=0.9)],
        [boundary(5.10, "right", confidence=0.3)],
        timestamps=t,
        frame_indices=frame,
        config=cfg(),
    )
    assert len(fused) == 1
    assert abs(fused[0].time - 5.025) < 1e-9
    assert fused[0].confidence == 0.9


def test_boundaries_beyond_merge_window_do_not_merge():
    t = make_t()
    frame = np.arange(len(t))
    fused, diagnostics = fuse_gripper_boundaries(
        [boundary(5.00, "left")],
        [boundary(5.30, "right")],
        timestamps=t,
        frame_indices=frame,
        config=cfg(),
    )
    assert len(fused) == 2
    assert diagnostics["num_merged_pairs"] == 0


def test_one_to_many_nearby_boundaries_use_one_to_one_matching():
    t = make_t()
    frame = np.arange(len(t))
    fused, diagnostics = fuse_gripper_boundaries(
        [boundary(5.00, "left"), boundary(5.10, "left", from_state="NEGATIVE", to_state="STABLE")],
        [boundary(5.05, "right")],
        timestamps=t,
        frame_indices=frame,
        config=cfg(),
    )
    assert len(fused) == 2
    assert diagnostics["num_merged_pairs"] == 1
    assert sum(b.source == "dual_gripper_fusion" for b in fused) == 1


def test_single_side_short_motion_is_not_deleted_by_dual_fusion():
    t = make_t(seconds=3.0)
    left = np.ones_like(t) * 0.8
    mask = (t >= 1.0) & (t < 1.13)
    left[t >= 1.13] = 0.15
    left[mask] = np.linspace(0.8, 0.15, mask.sum())
    right = np.ones_like(t) * 0.5
    config = cfg()
    config.smoothing.window_sec = 0.03
    config.velocity_smoothing.window_sec = 0.03
    result = segment_dual_gripper_trajectory(t, left_gripper_values=left, right_gripper_values=right, config=config)
    assert result.right_result is not None
    assert len(result.right_result.boundaries) == 0
    assert len(result.global_boundaries) == len(result.left_result.boundaries)
    assert len(result.global_boundaries) == 2
