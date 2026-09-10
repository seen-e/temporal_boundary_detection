from __future__ import annotations

from typing import Any, Dict, List, Set

from ..core.models import (
    CONFIDENCE_LEVELS,
    EVENT_ACTIONS,
    EVENT_TYPES,
    GRIPPER_STATES,
    INTERVAL_ACTIONS,
    RELOCALIZE_DIRECTIONS,
    ReviewSlice,
    ValidationIssue,
    ValidationResult,
    ensure_sequence,
)


def validate_review(review: Dict[str, Any], review_slice: ReviewSlice) -> ValidationResult:
    issues: List[ValidationIssue] = []
    target_ids = set(review_slice.target_ids)
    context_ids = set(review_slice.context_ids)
    all_ids = target_ids | context_ids
    events_by_id = {event.event_id: event for event in review_slice.all_events}
    owned_interval_ids = {interval.interval_id for interval in review_slice.owned_intervals}
    intervals_by_id = {interval.interval_id: interval for interval in review_slice.owned_intervals}

    if not isinstance(review, dict):
        return ValidationResult(False, [ValidationIssue("not_object", "review must be a JSON object")])

    _validate_region_analysis(review, all_ids, issues)

    event_reviews = list(ensure_sequence(review.get("event_reviews")))
    seen: Set[str] = set()
    merge_edges: Dict[str, str] = {}
    for idx, item in enumerate(event_reviews):
        path = f"event_reviews[{idx}]"
        if not isinstance(item, dict):
            issues.append(ValidationIssue("event_not_object", "event review must be an object", path))
            continue
        event_id = item.get("event_id")
        if event_id not in all_ids:
            issues.append(ValidationIssue("unknown_event_id", f"unknown event_id {event_id}", path))
            continue
        if event_id in context_ids:
            issues.append(ValidationIssue("context_modified", f"context event {event_id} cannot be reviewed as target", path))
        if event_id in seen:
            issues.append(ValidationIssue("duplicate_event_review", f"duplicate review for {event_id}", path))
        seen.add(event_id)
        action = item.get("action")
        if action not in EVENT_ACTIONS:
            issues.append(ValidationIssue("illegal_event_action", f"illegal event action {action}", path))
        if item.get("state_before") not in GRIPPER_STATES:
            issues.append(ValidationIssue("illegal_state_before", "state_before is invalid", path))
        if item.get("state_after") not in GRIPPER_STATES:
            issues.append(ValidationIssue("illegal_state_after", "state_after is invalid", path))
        if item.get("confidence") not in CONFIDENCE_LEVELS:
            issues.append(ValidationIssue("illegal_confidence", "confidence is invalid", path))
        if action == "RELABEL":
            _validate_relabel(item, events_by_id.get(event_id), issues, path)
        if action == "MERGE":
            merge_into = item.get("merge_into")
            merge_with = list(ensure_sequence(item.get("merge_with")))
            if not merge_into and merge_with:
                merge_into = merge_with[0]
            if not merge_into:
                issues.append(ValidationIssue("merge_missing_ref", "MERGE requires operation.merge_into", path))
            elif merge_into not in all_ids:
                issues.append(ValidationIssue("merge_ref_not_visible", f"MERGE ref {merge_into} is not visible in this slice", path))
            elif merge_into == event_id:
                issues.append(ValidationIssue("merge_self", "MERGE cannot point to itself", path))
            else:
                merge_edges[str(event_id)] = str(merge_into)
        if action == "RELOCALIZE":
            relocalize = item.get("relocalize")
            if not isinstance(relocalize, dict):
                issues.append(ValidationIssue("relocalize_missing", "RELOCALIZE requires relocalize object", path))
            else:
                if relocalize.get("direction") not in RELOCALIZE_DIRECTIONS:
                    issues.append(ValidationIssue("illegal_relocalize_direction", "bad relocalize direction", path))
                _validate_time_hint(relocalize.get("approx_time_sec"), review_slice.time_range, issues, path, required=True)

    _validate_no_merge_cycles(merge_edges, issues)

    missing = target_ids - seen
    extra = seen - target_ids
    for event_id in sorted(missing):
        issues.append(ValidationIssue("missing_target_review", f"missing review for target {event_id}", "event_reviews"))
    for event_id in sorted(extra):
        if event_id in context_ids:
            continue
        issues.append(ValidationIssue("extra_event_review", f"event review outside target set {event_id}", "event_reviews"))

    interval_reviews = list(ensure_sequence(review.get("interval_reviews")))
    seen_intervals: Set[str] = set()
    for idx, item in enumerate(interval_reviews):
        path = f"interval_reviews[{idx}]"
        if not isinstance(item, dict):
            issues.append(ValidationIssue("interval_not_object", "interval review must be an object", path))
            continue
        interval_id = item.get("interval_id")
        if interval_id not in owned_interval_ids:
            issues.append(ValidationIssue("interval_not_owned", f"interval {interval_id} is not owned by this slice", path))
            continue
        if interval_id in seen_intervals:
            issues.append(ValidationIssue("duplicate_interval_review", f"duplicate review for {interval_id}", path))
        seen_intervals.add(interval_id)
        action = item.get("action")
        if action not in INTERVAL_ACTIONS:
            issues.append(ValidationIssue("illegal_interval_action", f"illegal interval action {action}", path))
        if item.get("confidence") not in CONFIDENCE_LEVELS:
            issues.append(ValidationIssue("illegal_interval_confidence", "confidence is invalid", path))
        if action == "ADD":
            suggested_type = item.get("suggested_type", "UNKNOWN")
            if suggested_type != "UNKNOWN" and suggested_type not in EVENT_TYPES:
                issues.append(ValidationIssue("illegal_suggested_type", f"illegal suggested type {suggested_type}", path))
            interval = intervals_by_id[interval_id]
            _validate_time_hint(item.get("approx_time_sec"), (interval.start_time, interval.end_time), issues, path, required=False)

    for interval_id in sorted(owned_interval_ids - seen_intervals):
        issues.append(ValidationIssue("missing_interval_review", f"missing review for interval {interval_id}", "interval_reviews"))

    return ValidationResult(ok=not issues, issues=issues)


def _validate_region_analysis(review: Dict[str, Any], all_ids: Set[str], issues: List[ValidationIssue]) -> None:
    region = review.get("region_analysis")
    if region is None:
        return
    if not isinstance(region, dict):
        issues.append(ValidationIssue("region_analysis_not_object", "region_analysis must be an object", "region_analysis"))
        return
    for idx, part in enumerate(ensure_sequence(region.get("parts"))):
        path = f"region_analysis.parts[{idx}]"
        if not isinstance(part, dict):
            issues.append(ValidationIssue("part_not_object", "part must be an object", path))
            continue
        for key in ("start_event", "end_event"):
            value = part.get(key)
            if value is not None and value not in all_ids:
                issues.append(ValidationIssue("part_unknown_event", f"{key} {value} is not visible", path))
        for value in ensure_sequence(part.get("events")):
            if value not in all_ids:
                issues.append(ValidationIssue("part_unknown_event", f"part event {value} is not visible", path))


def _validate_relabel(item: Dict[str, Any], event: Any, issues: List[ValidationIssue], path: str) -> None:
    new_type = item.get("new_type")
    if new_type not in {"MAX", "MIN"}:
        issues.append(ValidationIssue("illegal_relabel_type", f"RELABEL new_type must be MAX or MIN, got {new_type}", path))
        return
    old_type = getattr(event, "event_type", None)
    if old_type not in {"MAX", "MIN"}:
        issues.append(ValidationIssue("relabel_non_extremum", f"{old_type} cannot be relabeled", path))
    elif new_type == old_type:
        issues.append(ValidationIssue("relabel_same_type", "RELABEL new_type must differ from original type", path))


def _validate_no_merge_cycles(edges: Dict[str, str], issues: List[ValidationIssue]) -> None:
    for start in edges:
        seen: Set[str] = set()
        current = start
        while current in edges:
            if current in seen:
                issues.append(ValidationIssue("merge_cycle", f"MERGE cycle detected at {current}", "event_reviews"))
                break
            seen.add(current)
            current = edges[current]


def _validate_time_hint(value: Any, time_range: tuple[float, float], issues: List[ValidationIssue], path: str, required: bool) -> None:
    if value is None and not required:
        return
    try:
        t = float(value)
    except (TypeError, ValueError):
        issues.append(ValidationIssue("bad_time_hint", "approx_time_sec must be numeric", path))
        return
    lo, hi = time_range
    if t < min(lo, hi) or t > max(lo, hi):
        issues.append(ValidationIssue("time_hint_out_of_range", f"time hint {t} outside [{lo}, {hi}]", path))
