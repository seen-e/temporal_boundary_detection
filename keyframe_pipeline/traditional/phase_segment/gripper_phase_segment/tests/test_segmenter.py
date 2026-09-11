from __future__ import annotations

import numpy as np

from keyframe_pipeline.traditional.phase_segment.gripper_phase_segment.config import GripperPhaseSegmentationConfig
from keyframe_pipeline.traditional.phase_segment.gripper_phase_segment.segmenter import segment_gripper_trajectory


def cfg(**overrides):
    config = GripperPhaseSegmentationConfig()
    config.outlier.enabled = True
    config.smoothing.method = "moving_average"
    config.velocity_smoothing.method = "moving_average"
    config.plateau.window_sec = 0.25
    config.plateau.range_ratio = 0.03
    config.plateau.normalized_slope_threshold = 0.03
    config.plateau.min_duration_sec = 0.30
    config.plateau.max_internal_gap_sec = 0.10
    config.plateau.merge_value_ratio = 0.03
    for path, value in overrides.items():
        section, name = path.split("__", 1)
        setattr(getattr(config, section), name, value)
    return config


def make_t(fps=30, seconds=4.0):
    return np.arange(0.0, seconds, 1.0 / fps)


def states(result):
    return [segment.motion_state for segment in result.segments]


def test_plateau_to_down_motion_to_plateau_has_two_main_boundaries():
    t = make_t()
    g = np.interp(t, [0.0, 1.0, 2.2, 4.0], [0.8, 0.8, 0.2, 0.2])
    result = segment_gripper_trajectory(t, g, cfg())
    assert states(result) == ["PLATEAU", "MOTION", "PLATEAU"]
    assert len(result.boundaries) == 2
    assert result.diagnostics["num_plateaus"] == 2


def test_down_motion_with_speed_changes_stays_one_motion_segment():
    t = make_t(seconds=5.0)
    g = np.ones_like(t) * 0.8
    fast1 = (t >= 1.0) & (t < 1.5)
    slow = (t >= 1.5) & (t < 2.4)
    fast2 = (t >= 2.4) & (t <= 3.0)
    g[fast1] = np.linspace(0.8, 0.55, fast1.sum())
    g[slow] = np.linspace(0.55, 0.45, slow.sum())
    g[fast2] = np.linspace(0.45, 0.2, fast2.sum())
    g[t > 3.0] = 0.2
    result = segment_gripper_trajectory(t, g, cfg())
    assert states(result) == ["PLATEAU", "MOTION", "PLATEAU"]


def test_short_pause_during_motion_does_not_form_plateau():
    t = make_t(seconds=4.0)
    g = np.ones_like(t) * 0.8
    first = (t >= 1.0) & (t < 1.8)
    pause = (t >= 1.8) & (t < 1.88)
    second = (t >= 1.88) & (t <= 2.8)
    g[first] = np.linspace(0.8, 0.45, first.sum())
    g[pause] = 0.45
    g[second] = np.linspace(0.45, 0.2, second.sum())
    g[t > 2.8] = 0.2
    result = segment_gripper_trajectory(t, g, cfg())
    assert states(result) == ["PLATEAU", "MOTION", "PLATEAU"]


def test_short_reversal_is_motion_metadata_not_main_boundary():
    t = make_t(seconds=4.0)
    g = np.interp(
        t,
        [0.0, 0.8, 1.5, 1.6, 2.5, 4.0],
        [0.8, 0.8, 0.30, 0.34, 0.20, 0.20],
    )
    result = segment_gripper_trajectory(
        t,
        g,
        cfg(smoothing__window_sec=0.03, extrema__auto_prominence_ratio=0.01),
    )
    assert states(result) == ["PLATEAU", "MOTION", "PLATEAU"]
    assert len(result.boundaries) == 2
    short_reversals = [event for event in result.diagnostics["anomaly_events"] if event["type"] == "short_reversal"]
    assert short_reversals


def test_same_direction_middle_plateau_is_suppressed():
    t = make_t(seconds=5.0)
    g = np.interp(t, [0.0, 0.8, 1.6, 2.3, 3.1, 5.0], [0.8, 0.8, 0.45, 0.45, 0.2, 0.2])
    result = segment_gripper_trajectory(t, g, cfg())
    assert states(result) == ["PLATEAU", "MOTION", "PLATEAU"]
    assert len(result.boundaries) == 2
    assert result.diagnostics["num_suppressed_internal_plateaus"] == 1


def test_opposite_direction_middle_plateau_splits_motion_plateau_motion():
    t = make_t(seconds=5.0)
    g = np.interp(t, [0.0, 0.8, 1.6, 2.3, 3.1, 5.0], [0.8, 0.8, 0.45, 0.45, 0.7, 0.7])
    result = segment_gripper_trajectory(t, g, cfg())
    assert states(result) == ["PLATEAU", "MOTION", "PLATEAU", "MOTION", "PLATEAU"]
    assert len(result.boundaries) == 4
    assert result.diagnostics["num_suppressed_internal_plateaus"] == 0


def test_noisy_plateau_is_stable():
    t = make_t(seconds=3.0)
    g = 0.2 + 0.001 * np.sin(np.linspace(0, 30 * np.pi, len(t)))
    result = segment_gripper_trajectory(t, g, cfg())
    assert states(result) == ["PLATEAU"]
    assert len(result.boundaries) == 0


def test_slow_continuous_motion_is_not_plateau_only():
    t = make_t(seconds=5.0)
    g = np.linspace(0.8, 0.5, len(t))
    result = segment_gripper_trajectory(t, g, cfg())
    assert states(result) == ["MOTION"]
    assert len(result.boundaries) == 0


def test_auto_thresholds_handle_different_scales():
    for scale in [1.0, 255.0, 0.08]:
        t = make_t(seconds=4.0)
        g = np.interp(t, [0.0, 1.0, 2.2, 4.0], [0.8 * scale, 0.8 * scale, 0.2 * scale, 0.2 * scale])
        result = segment_gripper_trajectory(t, g, cfg())
        assert states(result) == ["PLATEAU", "MOTION", "PLATEAU"]
        assert len(result.boundaries) == 2


def test_single_point_spike_does_not_create_phase():
    t = make_t(seconds=3.0)
    g = np.ones_like(t) * 0.5
    g[len(g) // 2] = 5.0
    result = segment_gripper_trajectory(t, g, cfg())
    assert len(result.boundaries) == 0
    assert states(result) == ["PLATEAU"]


def test_short_nan_gap_is_interpolated_and_long_gap_recorded():
    t = make_t(seconds=4.0)
    g = np.ones_like(t) * 0.7
    g[5:7] = np.nan
    g[60:75] = np.nan
    result = segment_gripper_trajectory(t, g, cfg(signal__max_interpolation_gap_sec=0.2))
    assert np.isfinite(result.gripper_clean).all()
    assert result.invalid_intervals


def test_irregular_timestamp_still_segments_plateau_motion_plateau():
    rng = np.random.default_rng(1)
    dt = rng.uniform(0.02, 0.05, size=120)
    t = np.cumsum(dt)
    g = np.interp(t, [t[0], t[30], t[80], t[-1]], [0.9, 0.9, 0.2, 0.2])
    result = segment_gripper_trajectory(t, g, cfg())
    assert states(result) == ["PLATEAU", "MOTION", "PLATEAU"]
