from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.config import AppConfig
from ..core.models import Event, ReviewSlice
from ..data.io_utils import write_json
from ..vlm import build_pass3_payload, call_vlm, parse_vlm_response, summarize_messages


EVENT_COLORS = {"MAX": "#d62728", "MIN": "#1f77b4", "PL": "#2ca02c", "PR": "#9467bd"}
SEGMENT_TYPES = {
    "稳定打开",
    "稳定关闭",
    "打开中",
    "关闭中",
    "稳定打开-带波动",
    "稳定关闭-带波动",
    "打开中-带波动",
    "关闭中-带波动",
}


def run_pass3_completion(
    cfg: AppConfig,
    episode_id: str,
    episode_index: Optional[int],
    arm: str,
    original_events: List[Event],
    slices: List[ReviewSlice],
    slice_results: List[Dict[str, Any]],
    refinement: Dict[str, Any],
    time: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
) -> Dict[str, Any]:
    """Run Pass 3 over Pass 1 frozen internal boundaries, one VLM call per slice."""

    mode = str(getattr(cfg.vlm_trajectory_review, "pass3_mode", "frozen_boundary") or "frozen_boundary")
    if mode != "frozen_boundary":
        raise ValueError(f"unsupported pass3_mode={mode!r}; only frozen_boundary is active")

    out = deepcopy(refinement)
    pass3_root = Path(cfg.paths.output_root) / episode_id / arm / "pass3"
    pass3_root.mkdir(parents=True, exist_ok=True)
    max_added_per_slice = max(0, int(getattr(cfg.vlm_trajectory_review, "pass3_max_added_points_per_slice", 8) or 8))
    min_gap_sec = float(getattr(cfg.vlm_trajectory_review, "pass3_min_gap_duration_sec", 0.3) or 0.0)
    workers = max(1, int(getattr(cfg.run, "workers", 1) or 1))

    kept_events = _kept_events(out.get("final_events", []))
    kept_by_id = {str(event.get("event_id") or event.get("keyframe_id")): event for event in kept_events}
    slice_by_id = {slice_.slice_id: slice_ for slice_ in slices}
    tasks = []
    skipped_slices = []
    for slice_result in slice_results:
        if slice_result.get("status") != "completed":
            continue
        review_slice = slice_by_id.get(str(slice_result.get("slice_id")))
        if review_slice is None:
            continue
        task = build_pass3_slice_task(review_slice, slice_result, kept_by_id, min_gap_sec=min_gap_sec)
        if task["boundaries_to_check"]:
            tasks.append((review_slice, slice_result, task))
        else:
            skipped_slices.append(_skipped_slice_summary(task))
    invalid_boundaries = [
        boundary
        for item in [task for _, _, task in tasks] + [
            {"internal_boundaries": slice_summary.get("invalid_boundaries", [])}
            for slice_summary in skipped_slices
        ]
        for boundary in item.get("internal_boundaries", [])
        if boundary.get("ownership") == "target" and boundary.get("suggested_type") == "UNKNOWN"
    ]

    print(f"  {arm} pass3 frozen-boundary: running {len(tasks)} slices, skipped {len(skipped_slices)}")
    results: List[Dict[str, Any]] = []
    max_workers = min(workers, len(tasks)) if tasks else 1
    if max_workers <= 1:
        for task in tasks:
            results.append(_run_pass3_slice_task(cfg, task, time, raw, smooth, max_added_per_slice))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_pass3_slice_task, cfg, task, time, raw, smooth, max_added_per_slice) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item.get("slice_id", ""))

    added_events = [event for result in results for event in result.get("added_events", [])]
    out.setdefault("final_events", []).extend(added_events)
    out["final_events"] = sorted(out.get("final_events", []), key=lambda item: (float(item.get("time_sec", 0.0)), str(item.get("event_id", ""))))
    summary = {
        "enabled": True,
        "mode": "frozen_boundary",
        "status": "completed",
        "pass2_kept_count": len(kept_events),
        "num_slices_checked_by_vlm": len(results),
        "num_slices_skipped": len(skipped_slices),
        "num_internal_boundaries": sum(len(result.get("internal_boundaries", [])) for result in results) + sum(item.get("num_internal_boundaries", 0) for item in skipped_slices),
        "num_boundaries_to_check": sum(len(result.get("boundaries_to_check", [])) for result in results),
        "num_added_events": len(added_events),
        "added_by_type": _count_by_type(added_events),
        "needs_review": bool(invalid_boundaries) or any(result.get("needs_review") for result in results),
        "warnings": [
            _warning(
                "pass3_same_base_state_boundary",
                f"same-base or invalid Pass1 boundary {boundary['boundary_id']}",
                f"{boundary.get('boundary_id')}",
            )
            for boundary in invalid_boundaries
        ],
        "slice_results": [_compact_slice_result(result) for result in results],
        "skipped_slices": skipped_slices,
    }
    out["pass3"] = summary
    write_json(pass3_root / "summary.json", summary)
    write_json(pass3_root / "final_events.json", {"episode_id": episode_id, "arm": arm, "final_events": out["final_events"], "pass3": summary})
    _plot_pass3_abc(pass3_root / "abc_overview.png", episode_id, arm, original_events, refinement.get("final_events", []), out["final_events"], time, smooth)
    return out


def build_pass3_slice_task(
    review_slice: ReviewSlice,
    slice_result: Dict[str, Any],
    kept_by_id: Dict[str, Dict[str, Any]],
    min_gap_sec: float = 0.3,
) -> Dict[str, Any]:
    review = slice_result.get("review") or {}
    frozen_segments = review.get("frozen_segments") or (review.get("pass1") or {}).get("frozen_segments") or []
    if not frozen_segments:
        frozen_segments = (((slice_result.get("pass1") or {}).get("review") or {}).get("region") or {}).get("segments") or []
    pass2_segments = (review.get("raw_pass2_review") or {}).get("segments") or ((review.get("region") or {}).get("segments") or [])
    target_start, target_end = _target_time_range(review_slice)
    cleaned_keyframes = []
    for event, scope in [
        *[(item, "left_overlap") for item in review_slice.context_left],
        *[(item, "target") for item in review_slice.targets],
        *[(item, "right_overlap") for item in review_slice.context_right],
    ]:
        if event.event_id in kept_by_id:
            cleaned_keyframes.append(_cleaned_metadata(kept_by_id[event.event_id], event.display_id, scope))
    cleaned_display_ids = {item["display_id"] for item in cleaned_keyframes}
    internal_boundaries = build_internal_boundaries(
        frozen_segments=frozen_segments,
        pass2_segments=pass2_segments,
        target_time_range=(target_start, target_end),
        final_cleaned_display_ids=cleaned_display_ids,
        min_gap_sec=min_gap_sec,
    )
    boundaries_to_check = [boundary for boundary in internal_boundaries if boundary["ownership"] == "target" and not boundary["covered_by_shared_event"] and boundary["suggested_type"] != "UNKNOWN"]
    return {
        "episode_id": review_slice.episode_id,
        "episode_index": review_slice.episode_index,
        "arm": review_slice.arm,
        "slice_id": review_slice.slice_id,
        "time_range_sec": [float(review_slice.time_range[0]), float(review_slice.time_range[1])],
        "target_time_range_sec": [target_start, target_end],
        "left_overlap_ids": [event.display_id for event in review_slice.context_left],
        "target_ids": [event.display_id for event in review_slice.targets],
        "right_overlap_ids": [event.display_id for event in review_slice.context_right],
        "frozen_segments": frozen_segments,
        "pass2_segments": pass2_segments,
        "cleaned_keyframes": cleaned_keyframes,
        "internal_boundaries": internal_boundaries,
        "boundaries_to_check": boundaries_to_check,
    }


def build_internal_boundaries(
    frozen_segments: List[Dict[str, Any]],
    pass2_segments: List[Dict[str, Any]],
    target_time_range: Tuple[float, float],
    final_cleaned_display_ids: Optional[Iterable[str]] = None,
    min_gap_sec: float = 0.3,
) -> List[Dict[str, Any]]:
    pass2_by_id = {str(segment.get("id")): segment for segment in pass2_segments if isinstance(segment, dict)}
    final_cleaned_filter = None if final_cleaned_display_ids is None else {str(item) for item in final_cleaned_display_ids}
    boundaries: List[Dict[str, Any]] = []
    target_start, target_end = target_time_range
    for idx, (left, right) in enumerate(zip(frozen_segments, frozen_segments[1:])):
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        left_id = str(left.get("id") or f"{idx + 1:03d}")
        right_id = str(right.get("id") or f"{idx + 2:03d}")
        left_end = _maybe_float(left.get("end_time_sec"), _maybe_float(right.get("start_time_sec"), target_start))
        right_start = _maybe_float(right.get("start_time_sec"), left_end)
        approx = (left_end + right_start) / 2.0
        left_events = set(_as_list((pass2_by_id.get(left_id) or {}).get("events")))
        right_events = set(_as_list((pass2_by_id.get(right_id) or {}).get("events")))
        shared_candidates = left_events & right_events
        if final_cleaned_filter is not None:
            shared_candidates &= final_cleaned_filter
        shared = sorted(shared_candidates)
        duration = abs(_maybe_float(right.get("end_time_sec"), approx) - _maybe_float(left.get("start_time_sec"), approx))
        suggested_type = infer_boundary_type(left.get("type"), right.get("type")) or "UNKNOWN"
        ownership = "target" if target_start < approx < target_end and duration >= min_gap_sec else "overlap_or_outside_target"
        boundaries.append(
            {
                "boundary_id": f"{left_id}_{right_id}",
                "between": [left_id, right_id],
                "left_segment": deepcopy(left),
                "right_segment": deepcopy(right),
                "approx_time_sec_from_pass1": approx,
                "left_end_time_sec": left_end,
                "right_start_time_sec": right_start,
                "suggested_type": suggested_type,
                "base_transition": [base_state(left.get("type")), base_state(right.get("type"))],
                "covered_by_shared_event": bool(shared),
                "shared_cleaned_keyframe_ids": shared,
                "ownership": ownership,
                "needs_review": suggested_type == "UNKNOWN",
            }
        )
    return boundaries


def _run_pass3_slice_task(
    cfg: AppConfig,
    task_tuple: Tuple[ReviewSlice, Dict[str, Any], Dict[str, Any]],
    time: np.ndarray,
    raw: np.ndarray,
    smooth: np.ndarray,
    max_added_per_slice: int,
) -> Dict[str, Any]:
    review_slice, _slice_result, task = task_tuple
    slice_dir = Path(cfg.paths.output_root) / review_slice.episode_id / review_slice.arm / "pass3" / review_slice.slice_id
    slice_dir.mkdir(parents=True, exist_ok=True)
    local_path = slice_dir / "local_cleaned.png"
    global_path = slice_dir / "global.png"
    _render_pass3_local(local_path, review_slice, task, time, raw, smooth, cfg)
    _render_pass3_global(global_path, review_slice, task, time, raw, smooth, cfg)
    metadata = deepcopy(task)
    metadata["global_image_path"] = str(global_path)
    metadata["local_image_path"] = str(local_path)
    write_json(slice_dir / "metadata.json", metadata)

    if cfg.run.dry_run:
        raw_response: Any = {"additions": [], "needs_review": False}
        parsed = raw_response
    else:
        payload = build_pass3_payload(cfg.model, metadata)
        if cfg.vlm_trajectory_review.save_request:
            write_json(slice_dir / "request.json", _redact_payload(payload))
            write_json(slice_dir / "message_summary.json", summarize_messages(payload.get("messages", [])))
        raw_response = call_vlm(cfg.model, payload)
        parsed = parse_vlm_response(raw_response)

    review = normalize_pass3_completion(parsed, metadata)
    if cfg.vlm_trajectory_review.save_response:
        (slice_dir / "response.json").write_text(_extract_response_text(raw_response), encoding="utf-8")
    added_events = refine_pass3_additions(review, metadata, time, smooth, max_added_per_slice=max_added_per_slice)
    write_json(slice_dir / "review.json", review)
    write_json(slice_dir / "refined_boundaries.json", {"slice_id": review_slice.slice_id, "added_events": added_events})
    _render_pass3_result(slice_dir / "result.png", review_slice, task, time, smooth, added_events)
    return {
        "slice_id": review_slice.slice_id,
        "status": "completed",
        "internal_boundaries": task["internal_boundaries"],
        "boundaries_to_check": task["boundaries_to_check"],
        "additions": review.get("additions", []),
        "added_events": added_events,
        "needs_review": review.get("needs_review", False),
        "warnings": review.get("warnings", []),
    }


def normalize_pass3_completion(review: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    raw = deepcopy(review) if isinstance(review, dict) else {}
    warnings: List[Dict[str, str]] = []
    boundaries_by_id = {boundary["boundary_id"]: boundary for boundary in metadata.get("internal_boundaries", [])}
    to_check = {boundary["boundary_id"] for boundary in metadata.get("boundaries_to_check", [])}
    additions: List[Dict[str, Any]] = []
    for idx, item in enumerate(_as_list(raw.get("additions"))):
        path = f"additions[{idx}]"
        if not isinstance(item, dict):
            warnings.append(_warning("pass3_addition_not_object", "addition must be an object", path))
            continue
        between = item.get("between")
        if not isinstance(between, list) or len(between) != 2:
            warnings.append(_warning("pass3_bad_between", "addition.between must contain two adjacent frozen segment ids", f"{path}.between"))
            continue
        boundary_id = f"{between[0]}_{between[1]}"
        boundary = boundaries_by_id.get(boundary_id)
        if boundary is None:
            warnings.append(_warning("pass3_unknown_boundary", f"unknown frozen boundary {boundary_id}", f"{path}.between"))
            continue
        if boundary_id not in to_check:
            warnings.append(_warning("pass3_boundary_not_owned_or_already_covered", f"boundary {boundary_id} is not target-owned, already covered, or invalid", f"{path}.between"))
            continue
        approx, ok = _coerce_float(item.get("approx_time_sec"))
        if not ok:
            warnings.append(_warning("pass3_bad_approx_time", f"bad approx_time_sec {item.get('approx_time_sec')!r}", f"{path}.approx_time_sec"))
            continue
        target_start, target_end = metadata.get("target_time_range_sec", [0.0, 0.0])
        if not (float(target_start) < approx < float(target_end)):
            warnings.append(_warning("pass3_addition_outside_target", f"addition time {approx} outside target ownership", f"{path}.approx_time_sec"))
            continue
        out = deepcopy(item)
        out["between"] = [str(between[0]), str(between[1])]
        out["boundary_id"] = boundary_id
        out["approx_time_sec"] = approx
        out["suggested_type"] = boundary["suggested_type"]
        out["left_segment"] = boundary["left_segment"]
        out["right_segment"] = boundary["right_segment"]
        out["base_transition"] = boundary["base_transition"]
        additions.append(out)
    missing_invalid = [boundary for boundary in metadata.get("internal_boundaries", []) if boundary.get("ownership") == "target" and boundary.get("suggested_type") == "UNKNOWN"]
    for boundary in missing_invalid:
        warnings.append(_warning("pass3_same_base_state_boundary", f"same-base or invalid Pass1 boundary {boundary['boundary_id']}", "internal_boundaries"))
    return {
        "schema_version": "pass3_frozen_boundary_v1",
        "additions": additions,
        "needs_review": bool(raw.get("needs_review")) or bool(warnings),
        "warnings": warnings,
        "raw_vlm_review": raw,
    }


def refine_pass3_additions(
    review: Dict[str, Any],
    metadata: Dict[str, Any],
    time: np.ndarray,
    smooth: np.ndarray,
    max_added_per_slice: int = 8,
) -> List[Dict[str, Any]]:
    if max_added_per_slice <= 0:
        return []
    existing_times = [float(item.get("time_sec", -1e9)) for item in metadata.get("cleaned_keyframes", [])]
    window_sec = float(metadata.get("refine_window_sec", 0.8) or 0.8)
    added: List[Dict[str, Any]] = []
    for idx, addition in enumerate(review.get("additions", [])):
        if len(added) >= max_added_per_slice:
            break
        event_type = addition.get("suggested_type")
        if event_type not in EVENT_COLORS:
            continue
        approx = float(addition.get("approx_time_sec"))
        left = addition.get("left_segment") or {}
        right = addition.get("right_segment") or {}
        search_lo = max(
            float(metadata["target_time_range_sec"][0]),
            approx - window_sec,
            _maybe_float(left.get("end_time_sec"), approx - window_sec) - window_sec,
        )
        search_hi = min(
            float(metadata["target_time_range_sec"][1]),
            approx + window_sec,
            _maybe_float(right.get("start_time_sec"), approx + window_sec) + window_sec,
        )
        frame_idx, refined_time, value = _refine_time(event_type, approx, search_lo, search_hi, time, smooth)
        if any(abs(refined_time - old) <= 0.05 for old in existing_times):
            continue
        event_id = f"P3_{metadata['slice_id']}_{idx + 1:03d}"
        added.append(
            {
                "keyframe_id": event_id,
                "event_id": event_id,
                "source_index": None,
                "type": event_type,
                "original_type": event_type,
                "time_sec": refined_time,
                "frame_index": frame_idx,
                "value_smooth": value,
                "vlm_action": "PASS3_ADD",
                "final_status": "kept",
                "source": "pass3",
                "originating_slice": metadata["slice_id"],
                "between_segments": addition.get("between"),
                "pass3_boundary": {
                    "boundary_id": addition.get("boundary_id"),
                    "vlm_approx_time_sec": approx,
                    "pass1_approx_time_sec": _maybe_float((addition.get("left_segment") or {}).get("end_time_sec"), approx),
                    "search_range_sec": [search_lo, search_hi],
                    "left_segment_type": left.get("type"),
                    "right_segment_type": right.get("type"),
                    "description": addition.get("description", ""),
                },
            }
        )
    return added


def infer_boundary_type(left_segment_type: Any, right_segment_type: Any) -> Optional[str]:
    left = base_state(left_segment_type)
    right = base_state(right_segment_type)
    if not left or not right or left == right:
        return None
    if left == "打开" and right == "关闭":
        return "MAX"
    if left == "关闭" and right == "打开":
        return "MIN"
    if left in {"打开", "关闭"} and right == "稳定":
        return "PL"
    if left == "稳定" and right in {"打开", "关闭"}:
        return "PR"
    return None


def base_state(segment_type: Any) -> Optional[str]:
    text = str(segment_type or "")
    if text.startswith("稳定"):
        return "稳定"
    if text.startswith("打开中"):
        return "打开"
    if text.startswith("关闭中"):
        return "关闭"
    return None


def _render_pass3_local(out_path: Path, review_slice: ReviewSlice, metadata: Dict[str, Any], time: np.ndarray, raw: np.ndarray, smooth: np.ndarray, cfg: AppConfig) -> None:
    t0, t1 = _expanded_window(review_slice, cfg)
    mask = (time >= t0) & (time <= t1) if len(time) else np.asarray([], dtype=bool)
    fig, ax = plt.subplots(figsize=(14.0, 5.3), dpi=170)
    if len(time):
        _plot_configured_trajectory(ax, time[mask], raw[mask], smooth[mask], cfg)
    _draw_regions(ax, review_slice)
    for boundary in metadata.get("boundaries_to_check", []):
        x = float(boundary["approx_time_sec_from_pass1"])
        ax.axvline(x, color="#7f7f7f", lw=1.0, ls=":", alpha=0.65, zorder=2)
    _draw_cleaned_keyframes(ax, metadata.get("cleaned_keyframes", []), time, smooth)
    ax.set_xlim(t0, max(t1, t0 + 0.5))
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("gripper state")
    ax.set_title(f"{review_slice.episode_id} | {review_slice.arm} | {review_slice.slice_id} | Pass 3 cleaned-only LOCAL")
    ax.grid(True, alpha=0.24, lw=0.45)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_pass3_global(out_path: Path, review_slice: ReviewSlice, metadata: Dict[str, Any], time: np.ndarray, raw: np.ndarray, smooth: np.ndarray, cfg: AppConfig) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 4.2), dpi=170)
    if len(time):
        _plot_configured_trajectory(ax, time, raw, smooth, cfg, global_view=True)
        ax.set_xlim(float(time[0]), float(time[-1]))
    _draw_regions(ax, review_slice)
    for boundary in metadata.get("boundaries_to_check", []):
        ax.axvline(float(boundary["approx_time_sec_from_pass1"]), color="#7f7f7f", lw=0.9, ls=":", alpha=0.6)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("gripper state")
    ax.set_title(f"{review_slice.episode_id} | {review_slice.arm} | {review_slice.slice_id} | Pass 3 GLOBAL")
    ax.grid(True, alpha=0.22, lw=0.45)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_configured_trajectory(ax, time_values: np.ndarray, raw_values: np.ndarray, smooth_values: np.ndarray, cfg: AppConfig, global_view: bool = False) -> None:
    show_raw = bool(getattr(cfg.vlm_trajectory_review, "show_raw_trajectory", False))
    show_smooth = bool(getattr(cfg.vlm_trajectory_review, "show_smoothed_trajectory", True))
    if not show_raw and not show_smooth:
        show_smooth = True
    if show_raw:
        ax.plot(
            time_values,
            raw_values,
            color="#111111" if not show_smooth else "#888888",
            lw=1.05 if not global_view else 0.9,
            alpha=0.98 if not show_smooth else 0.42,
            label="raw",
        )
    if show_smooth:
        ax.plot(
            time_values,
            smooth_values,
            color="#111111",
            lw=1.8 if not global_view else 1.25,
            alpha=0.98,
            label="smoothed",
        )


def _render_pass3_result(out_path: Path, review_slice: ReviewSlice, metadata: Dict[str, Any], time: np.ndarray, smooth: np.ndarray, additions: List[Dict[str, Any]]) -> None:
    t0, t1 = _expanded_window(review_slice, None)
    mask = (time >= t0) & (time <= t1) if len(time) else np.asarray([], dtype=bool)
    fig, ax = plt.subplots(figsize=(14.0, 5.0), dpi=170)
    if len(time):
        ax.plot(time[mask], smooth[mask], color="#111111", lw=1.8, label="smoothed")
    _draw_regions(ax, review_slice)
    _draw_cleaned_keyframes(ax, metadata.get("cleaned_keyframes", []), time, smooth)
    for event in additions:
        x = float(event["time_sec"])
        y = _value_at(time, smooth, x, event.get("value_smooth"))
        color = EVENT_COLORS.get(event.get("type"), "#000000")
        ax.scatter([x], [y], s=96, marker="*", color=color, edgecolors="white", linewidths=0.8, zorder=7, label=f"Pass3 {event.get('type')}")
        ax.text(x, y + 0.06, event.get("type", "P3"), fontsize=8, ha="center", color=color)
    ax.set_xlim(t0, max(t1, t0 + 0.5))
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("gripper state")
    ax.set_title(f"{review_slice.episode_id} | {review_slice.arm} | {review_slice.slice_id} | Pass 3 additions={len(additions)}")
    ax.grid(True, alpha=0.24, lw=0.45)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pass3_abc(
    out_path: Path,
    episode_id: str,
    arm: str,
    original_events: List[Event],
    pass2_events: List[Dict[str, Any]],
    pass3_events: List[Dict[str, Any]],
    time: np.ndarray,
    smooth: np.ndarray,
) -> None:
    rows = [
        ("A original candidates", [_event_to_dict(event) for event in original_events]),
        ("B Pass 2 cleaned", _kept_events(pass2_events)),
        ("C Pass 3 completed", _kept_events(pass3_events)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(25, 10.5), dpi=170, sharex=True)
    for ax, (title, events) in zip(axes, rows):
        if len(time):
            ax.plot(time, smooth, color="#111111", lw=1.25, alpha=0.98)
        for event_type, color in EVENT_COLORS.items():
            selected = [event for event in events if event.get("type") == event_type]
            if not selected:
                continue
            xs = [float(event["time_sec"]) for event in selected]
            ys = [_value_at(time, smooth, x, event.get("value_smooth")) for x, event in zip(xs, selected)]
            marker = "*" if any(event.get("source") == "pass3" for event in selected) else "o"
            ax.scatter(xs, ys, s=24 if marker == "o" else 80, color=color, marker=marker, alpha=0.9, label=f"{event_type} {len(selected)}", zorder=4)
        ax.set_ylabel(title)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.22, lw=0.45)
        ax.legend(loc="upper right", ncol=7, fontsize=8)
    if len(time):
        axes[-1].set_xlim(float(time[0]), float(time[-1]))
    axes[-1].set_xlabel("time (sec)")
    fig.suptitle(f"{episode_id} | {arm} | original -> Pass2 -> Pass3", y=0.996, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_path)
    plt.close(fig)


def _draw_regions(ax, review_slice: ReviewSlice) -> None:
    visible_t0, visible_t1 = review_slice.time_range
    target_t0, target_t1 = _target_time_range(review_slice)
    if visible_t0 < target_t0:
        ax.axvspan(visible_t0, target_t0, color="#6baed6", alpha=0.13, zorder=0)
    if target_t1 < visible_t1:
        ax.axvspan(target_t1, visible_t1, color="#6baed6", alpha=0.13, zorder=0)
    ax.axvspan(target_t0, target_t1, color="#ffcc00", alpha=0.14, zorder=0)
    ax.axvline(target_t0, color="#b00020", lw=1.2, ls="--", alpha=0.85)
    ax.axvline(target_t1, color="#b00020", lw=1.2, ls="--", alpha=0.85)


def _draw_cleaned_keyframes(ax, events: List[Dict[str, Any]], time: np.ndarray, smooth: np.ndarray) -> None:
    for event in events:
        x = float(event.get("time_sec", 0.0))
        y = _value_at(time, smooth, x, event.get("value_smooth"))
        color = EVENT_COLORS.get(event.get("type"), "#111111")
        ax.scatter([x], [y], s=70, color=color, marker="D", edgecolors="white", linewidths=0.8, zorder=6)
        ax.text(x, y + 0.055, str(event.get("display_id") or event.get("event_id")), fontsize=8, ha="center", color=color)


def _expanded_window(review_slice: ReviewSlice, cfg: Optional[AppConfig]) -> Tuple[float, float]:
    padding = float(getattr(getattr(cfg, "visualization", None), "padding_sec", 1.0) if cfg is not None else 1.0)
    t0, t1 = review_slice.time_range
    return max(0.0, float(t0) - padding), max(float(t1) + padding, float(t0) + 0.5)


def _target_time_range(review_slice: ReviewSlice) -> Tuple[float, float]:
    if review_slice.targets:
        return float(review_slice.targets[0].time), float(review_slice.targets[-1].time)
    return float(review_slice.time_range[0]), float(review_slice.time_range[1])


def _cleaned_metadata(event: Dict[str, Any], display_id: str, scope: str) -> Dict[str, Any]:
    return {
        "display_id": display_id,
        "event_id": event.get("event_id") or event.get("keyframe_id"),
        "keyframe_id": event.get("event_id") or event.get("keyframe_id"),
        "scope": scope,
        "type": event.get("type"),
        "time_sec": event.get("time_sec"),
        "frame_index": event.get("frame_index"),
        "value_smooth": event.get("value_smooth"),
        "source": event.get("source", "pass2_kept"),
    }


def _kept_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [event for event in events if event.get("final_status") not in {"removed", "merged_removed"}]


def _count_by_type(events: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for event in events:
        key = str(event.get("type", "UNKNOWN"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _compact_slice_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "slice_id": result.get("slice_id"),
        "num_internal_boundaries": len(result.get("internal_boundaries", [])),
        "num_boundaries_to_check": len(result.get("boundaries_to_check", [])),
        "num_additions": len(result.get("additions", [])),
        "num_added_events": len(result.get("added_events", [])),
        "needs_review": result.get("needs_review", False),
        "warnings": result.get("warnings", []),
    }


def _skipped_slice_summary(task: Dict[str, Any]) -> Dict[str, Any]:
    invalid = [
        boundary
        for boundary in task.get("internal_boundaries", [])
        if boundary.get("ownership") == "target" and boundary.get("suggested_type") == "UNKNOWN"
    ]
    return {
        "slice_id": task.get("slice_id"),
        "num_internal_boundaries": len(task.get("internal_boundaries", [])),
        "invalid_boundaries": invalid,
        "reason": "no target-owned uncovered valid frozen boundary",
    }


def _event_to_dict(event: Event) -> Dict[str, Any]:
    return {"event_id": event.event_id, "type": event.event_type, "time_sec": event.time, "value_smooth": event.value_smooth, "final_status": "candidate"}


def _refine_time(event_type: str, approx: float, search_lo: float, search_hi: float, time: np.ndarray, smooth: np.ndarray) -> Tuple[int, float, float]:
    if len(time) == 0 or len(smooth) == 0:
        return 0, float(approx), 0.5
    lo, hi = sorted((search_lo, search_hi))
    mask = (time >= lo) & (time <= hi)
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        idx = int(np.argmin(np.abs(time - approx)))
        return idx, float(time[idx]), float(smooth[idx])
    values = smooth[idxs]
    if event_type == "MAX":
        idx = int(idxs[int(np.argmax(values))])
    elif event_type == "MIN":
        idx = int(idxs[int(np.argmin(values))])
    else:
        idx = _refine_plateau_edge(idxs, approx, time, smooth, event_type)
    return idx, float(time[idx]), float(smooth[idx])


def _refine_plateau_edge(idxs: np.ndarray, approx: float, time: np.ndarray, smooth: np.ndarray, event_type: str) -> int:
    if len(idxs) < 5:
        return int(idxs[int(np.argmin(np.abs(time[idxs] - approx)))])
    grad = np.gradient(smooth, time, edge_order=1)
    abs_grad = np.abs(grad[idxs])
    threshold = max(0.003, float(np.percentile(abs_grad, 35)))
    center_pos = int(np.argmin(np.abs(time[idxs] - approx)))
    radius = max(2, min(5, len(idxs) // 5))
    if event_type == "PL":
        for pos in range(max(0, center_pos - radius), min(len(idxs), center_pos + radius + 1)):
            tail = abs_grad[pos : min(len(idxs), pos + radius)]
            if len(tail) and float(np.median(tail)) <= threshold:
                return int(idxs[pos])
    if event_type == "PR":
        for pos in range(min(len(idxs) - 1, center_pos + radius), max(-1, center_pos - radius - 1), -1):
            tail = abs_grad[pos : min(len(idxs), pos + radius)]
            if len(tail) and float(np.median(tail)) >= threshold:
                return int(idxs[pos])
    return int(idxs[center_pos])


def _value_at(time: np.ndarray, smooth: np.ndarray, x: float, fallback: Any = None) -> float:
    if fallback is not None:
        return float(fallback)
    if len(time) and len(smooth):
        return float(smooth[int(np.argmin(np.abs(time - x)))])
    return 0.5


def _extract_response_text(response: Any) -> str:
    if isinstance(response, dict) and "choices" in response:
        content = response.get("choices", [{}])[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content.strip() + "\n"
        if content is not None:
            return json.dumps(content, ensure_ascii=False, indent=2) + "\n"
    if isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False, indent=2) + "\n"
    return str(response).strip() + "\n"


def _redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = json.loads(json.dumps(payload))
    for message in redacted.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    part["image_url"] = {"url": "<base64 image omitted>"}
    return redacted


def _maybe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_float(value: Any) -> Tuple[float, bool]:
    try:
        return float(value), True
    except (TypeError, ValueError):
        return 0.0, False


def _warning(code: str, message: str, path: str) -> Dict[str, str]:
    return {"code": code, "message": message, "path": path}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []
