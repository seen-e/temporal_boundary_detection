# Gripper Phase Segmentation

This module segments a single gripper trajectory into low-level motion phases:
`STABLE`, `POSITIVE`, and `NEGATIVE`. These are signal states, not task semantics.

Minimal Python API:

```python
from keyframe_pipeline.traditional.phase_segment.gripper_phase_segment import segment_gripper_trajectory

result = segment_gripper_trajectory(timestamps, gripper_values)
print(result.boundaries)
print(result.segments)
```

Dual-gripper API:

```python
from keyframe_pipeline.traditional.phase_segment.gripper_phase_segment import segment_dual_gripper_trajectory

result = segment_dual_gripper_trajectory(
    timestamps=timestamps,
    left_gripper_values=left_gripper,
    right_gripper_values=right_gripper,
)
print(result.global_boundaries)
print(result.global_segments)
```

CLI example for an exported ABC trajectory:

```bash
python -m keyframe_pipeline.traditional.phase_segment.gripper_phase_segment.cli \
  --input /mnt/data/chachaxu/dataset/abc_40task/task_01_fold_and_stack_the_t_shirts/episode_01_abc_130k_v3_train__episode_000010/trajectory.parquet \
  --side left \
  --output-dir /mnt/workspace/temporal_boundary_detection/keyframe_pipeline/traditional/phase_segment/gripper_phase_segment/examples/episode_000010_left \
  --plot
```

Dual-gripper CLI:

```bash
python -m keyframe_pipeline.traditional.phase_segment.gripper_phase_segment.cli \
  --input /mnt/data/chachaxu/dataset/abc_40task/task_01_fold_and_stack_the_t_shirts/episode_01_abc_130k_v3_train__episode_000010/trajectory.parquet \
  --dual \
  --output-dir /mnt/workspace/temporal_boundary_detection/keyframe_pipeline/traditional/phase_segment/gripper_phase_segment/examples/episode_000010_dual \
  --plot
```

Outputs:

- `boundaries.json`
- `segments.json`
- `diagnostics.json`
- `gripper_phase_debug.png`

Dual mode additionally writes:

- `global_boundaries.json`
- `global_segments.json`
- `fusion_diagnostics.json`
- `left_boundaries.json`
- `right_boundaries.json`
- `dual_gripper_phase_debug.png`

Fusion is a union, not an intersection. Left and right grippers are segmented
independently first, then opposite-side nearby boundaries are matched one-to-one
within `dual_gripper.fusion.merge_window_sec`. Merged times use
`weighted_mean` by default, weighted by boundary confidence.

Batch processing for `abc_40task`:

```bash
python -m keyframe_pipeline.traditional.phase_segment.gripper_phase_segment.batch_abc40 \
  --dataset-root /mnt/data/chachaxu/dataset/abc_40task
```

For each episode directory, the batch command saves PNGs directly under the
episode directory:

- `left_gripper_phase.png`
- `right_gripper_phase.png`
- `dual_gripper_phase.png`

JSON results are saved under each episode's `gripper_phase_segment/` directory.
