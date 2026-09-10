"""Trajectory-only VLM review framework for gripper keyframe candidates.

Package layout:
- core: shared config and dataclasses.
- data: keyframe JSON and gripper trajectory loading.
- slicing: event-count based review slicing.
- visualization: slice plots and before/after plots.
- prompts: plain text VLM prompt templates.
- vlm: request building, response parsing, and client calls.
- validation: strict review schema and ownership checks.
- refinement: deterministic application of VLM decisions.
- reporting: aggregate counters.
- runner: orchestration and CLI implementation.
"""

__all__ = [
    "core",
    "data",
    "slicing",
    "visualization",
    "prompts",
    "vlm",
    "validation",
    "refinement",
    "reporting",
    "runner",
]
