"""Debug visualization for gripper phase segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import GripperPhaseSegmentationResult, MotionState, TrendState
from .types import DualGripperSegmentationResult


def plot_gripper_phase_result(
    result: GripperPhaseSegmentationResult,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = result.timestamps
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    if title:
        fig.suptitle(f"{title} | boundaries={len(result.boundaries)} segments={len(result.segments)}")

    axes[0].plot(timestamps, result.gripper_raw, label="raw gripper", alpha=0.45)
    axes[0].plot(timestamps, result.gripper_clean, label="clean gripper", linewidth=1.0)
    axes[0].plot(timestamps, result.gripper_smooth, label="smoothed gripper", linewidth=1.5)
    for interval in result.diagnostics.get("plateau_intervals", []):
        axes[0].axvspan(interval["start_time"], interval["end_time"], color="#cfcfcf", alpha=0.25)
    _plot_extrema(axes[0], result)
    axes[0].set_ylabel("gripper")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.2)

    _plot_plateau_diagnostics(axes[1], result)
    axes[1].grid(True, alpha=0.2)

    _shade_states(axes[2], result)
    for boundary in result.boundaries:
        axes[0].axvline(boundary.time, color="black", alpha=0.18, linewidth=0.7)
        axes[1].axvline(boundary.time, color="black", alpha=0.18, linewidth=0.7)
        axes[2].axvline(boundary.time, color="black", alpha=0.35, linewidth=0.7)
    axes[2].set_ylabel("main segment")
    axes[2].set_xlabel("time (s)")
    axes[2].set_yticks([])
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_dual_gripper_phase_result(
    result: DualGripperSegmentationResult,
    output_path: str | Path,
    title: str | None = "Dual Gripper Phase Segmentation",
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    side_results = [item for item in [("Left", result.left_result), ("Right", result.right_result)] if item[1] is not None]
    if not side_results:
        raise ValueError("dual visualization requires at least one side result")

    timestamps = side_results[0][1].timestamps
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    if title:
        fig.suptitle(title)

    for label, side_result in side_results:
        axes[0].plot(timestamps, side_result.gripper_smooth, label=f"{label} smoothed")
        axes[1].plot(timestamps, side_result.velocity_smooth, label=f"{label} velocity")
        for boundary in side_result.boundaries:
            axes[0].axvline(boundary.time, alpha=0.25, linewidth=0.8)
            axes[1].axvline(boundary.time, alpha=0.25, linewidth=0.8)

    axes[0].set_ylabel("gripper")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.2)
    axes[1].set_ylabel("velocity")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.2)

    _shade_side_timeline(axes[2], result)
    _shade_global_timeline(axes[3], result)
    for boundary in result.global_boundaries:
        axes[3].axvline(boundary.time, color="black", alpha=0.75, linewidth=1.0)
        if len(result.global_boundaries) <= 60:
            axes[3].text(boundary.time, 0.5, "+".join(boundary.source_sides), rotation=90, va="center", fontsize=7)

    axes[2].set_ylabel("side states")
    axes[3].set_ylabel("global")
    axes[3].set_xlabel("time (s)")
    for ax in axes[2:]:
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _shade_states(ax, result: GripperPhaseSegmentationResult) -> None:
    colors = {
        MotionState.STABLE.value: "#d9d9d9",
        MotionState.POSITIVE.value: "#8fd19e",
        MotionState.NEGATIVE.value: "#8db8e8",
        MotionState.INVALID.value: "#f4a6a6",
        TrendState.FLAT.value: "#d9d9d9",
        TrendState.UP.value: "#8fd19e",
        TrendState.DOWN.value: "#8db8e8",
        "PLATEAU": "#cfcfcf",
        "MOTION": "#93c5fd",
    }
    y = np.zeros_like(result.timestamps)
    ax.plot(result.timestamps, y, color="none")
    duration = float(result.timestamps[-1] - result.timestamps[0]) if len(result.timestamps) else 0.0
    label_segments = len(result.segments) <= 40
    for segment in result.segments:
        color = colors.get(segment.motion_state, "#eeeeee")
        ax.axvspan(segment.start_time, segment.end_time, color=color, alpha=0.55)
        if segment.possible_jitter:
            ax.axvspan(
                segment.start_time,
                segment.end_time,
                facecolor="none",
                edgecolor="tab:red",
                hatch="////",
                linewidth=0.0,
                alpha=0.65,
            )
        if label_segments and segment.duration_sec >= max(0.4, duration * 0.03):
            center = (segment.start_time + segment.end_time) / 2.0
            ax.text(center, 0.0, segment.motion_state, ha="center", va="center", fontsize=8)


def _plot_extrema(ax, result: GripperPhaseSegmentationResult) -> None:
    if not result.diagnostics.get("show_extrema", False):
        return
    accepted_max = [p for p in result.extrema if p.type == "local_max"]
    accepted_min = [p for p in result.extrema if p.type == "local_min"]
    rejected = result.rejected_extrema
    if accepted_max:
        ax.scatter([p.time for p in accepted_max], [p.value for p in accepted_max], marker="^", color="tab:red", s=28, label="accepted max", zorder=5)
    if accepted_min:
        ax.scatter([p.time for p in accepted_min], [p.value for p in accepted_min], marker="v", color="tab:blue", s=28, label="accepted min", zorder=5)
    if rejected:
        ax.scatter([p.time for p in rejected], [p.value for p in rejected], marker="x", color="0.6", s=16, alpha=0.45, label="rejected extrema", zorder=4)


def _plot_plateau_diagnostics(ax, result: GripperPhaseSegmentationResult) -> None:
    thresholds = result.diagnostics.get("plateau_thresholds", {})
    if result.plateau_local_range.size:
        ax.plot(result.timestamps, result.plateau_local_range, label="local value range", color="tab:purple", linewidth=1.1)
        if "range_threshold" in thresholds:
            ax.axhline(float(thresholds["range_threshold"]), color="tab:purple", linestyle="--", linewidth=1, label="range threshold")
        ax.set_ylabel("local range")
        twin = ax.twinx()
        twin.plot(result.timestamps, result.plateau_normalized_slope, label="normalized slope", color="tab:brown", linewidth=1.0, alpha=0.85)
        if "normalized_slope_threshold" in thresholds:
            twin.axhline(float(thresholds["normalized_slope_threshold"]), color="tab:brown", linestyle=":", linewidth=1, label="slope threshold")
        twin.set_ylabel("norm slope")
        lines, labels = ax.get_legend_handles_labels()
        twin_lines, twin_labels = twin.get_legend_handles_labels()
        ax.legend(lines + twin_lines, labels + twin_labels, loc="upper right")
        return

    ax.plot(result.timestamps, result.velocity_smooth, label="smoothed velocity", linewidth=1.3)
    ax.set_ylabel("velocity")
    ax.legend(loc="upper right")


def _shade_side_timeline(ax, result: DualGripperSegmentationResult) -> None:
    colors = {
        MotionState.STABLE.value: "#d9d9d9",
        MotionState.POSITIVE.value: "#8fd19e",
        MotionState.NEGATIVE.value: "#8db8e8",
        MotionState.INVALID.value: "#f4a6a6",
        TrendState.FLAT.value: "#d9d9d9",
        TrendState.UP.value: "#8fd19e",
        TrendState.DOWN.value: "#8db8e8",
        "PLATEAU": "#cfcfcf",
        "MOTION": "#93c5fd",
    }
    lanes = [("left", result.left_result, 0.65), ("right", result.right_result, 0.15)]
    for label, side_result, y in lanes:
        if side_result is None:
            continue
        for segment in side_result.segments:
            ax.axvspan(segment.start_time, segment.end_time, ymin=y, ymax=y + 0.25, color=colors.get(segment.motion_state, "#eeeeee"), alpha=0.7)
            if segment.possible_jitter:
                ax.axvspan(segment.start_time, segment.end_time, ymin=y, ymax=y + 0.25, facecolor="none", edgecolor="tab:red", hatch="////", linewidth=0.0, alpha=0.7)
        for boundary in side_result.boundaries:
            ax.axvline(boundary.time, ymin=y, ymax=y + 0.25, alpha=0.45, linewidth=0.8)
        ax.text(side_result.timestamps[0], y + 0.12, label, ha="right", va="center", fontsize=8)


def _shade_global_timeline(ax, result: DualGripperSegmentationResult) -> None:
    for idx, segment in enumerate(result.global_segments):
        color = "#eeeeee" if idx % 2 == 0 else "#d7e8f7"
        ax.axvspan(segment.start_time, segment.end_time, color=color, alpha=0.8)
        center = (segment.start_time + segment.end_time) / 2.0
        ax.text(center, 0.0, f"S{segment.segment_id}", ha="center", va="center", fontsize=8)
