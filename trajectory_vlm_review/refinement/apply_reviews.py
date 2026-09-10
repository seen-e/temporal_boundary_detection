from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from ..core.models import Event, new_event_id


def apply_reviews(events: List[Event], slice_reviews: List[Dict[str, Any]], protect_significant_extrema: bool = False) -> Dict[str, Any]:
    event_ids = {event.event_id for event in events}
    review_by_event: Dict[str, Dict[str, Any]] = {}
    interval_reviews: List[Dict[str, Any]] = []
    for item in slice_reviews:
        if item.get("status") != "completed":
            continue
        review = item.get("review", {})
        for event_review in review.get("event_reviews", []):
            review_by_event[event_review["event_id"]] = event_review
        interval_reviews.extend(review.get("interval_reviews", []))

    merge_parent = {event.event_id: event.event_id for event in events}
    for event_id, review in review_by_event.items():
        if review.get("action") != "MERGE":
            continue
        merge_into = review.get("merge_into")
        if not merge_into:
            refs = review.get("merge_with") or []
            merge_into = refs[0] if refs else None
        if merge_into in event_ids and merge_into != event_id:
            merge_parent[event_id] = merge_into

    merge_parent = {event_id: _resolve_parent(event_id, merge_parent) for event_id in merge_parent}

    groups: Dict[str, List[str]] = defaultdict(list)
    for event_id, parent in merge_parent.items():
        groups[parent].append(event_id)

    final_events: List[Dict[str, Any]] = []
    relocation_requests: List[Dict[str, Any]] = []
    for event in events:
        review = review_by_event.get(event.event_id, {})
        action = review.get("action", "UNCERTAIN")
        parent = merge_parent[event.event_id]
        final_status = _final_status(action, parent, event.event_id)
        event_type = _final_type(event, review)
        item = {
            "keyframe_id": event.event_id,
            "event_id": event.event_id,
            "source_index": event.source_index,
            "type": event_type,
            "original_type": event.event_type,
            "time_sec": event.time,
            "frame_index": event.frame_index,
            "value_smooth": event.value_smooth,
            "plateau_pair_id": event.plateau_pair_id,
            "vlm_action": action,
            "final_status": final_status,
            "review": review,
            "source_keyframe": event.source_keyframe,
        }
        if action == "RELABEL":
            item["relabeled_from"] = event.event_type
            item["relabeled_to"] = event_type
        if parent != event.event_id:
            item["merge_representative_event_id"] = parent
        final_events.append(item)
        if action == "RELOCALIZE":
            relocation_requests.append(
                {
                    "keyframe_id": event.event_id,
                    "event_id": event.event_id,
                    "status": "pending",
                    "original_time_sec": event.time,
                    "type": event_type,
                    "original_type": event.event_type,
                    "relocalize": review.get("relocalize", {}),
                }
            )

    protection_overrides = _protect_significant_extrema(final_events) if protect_significant_extrema else []

    localization_requests: List[Dict[str, Any]] = []
    add_idx = 0
    for review in interval_reviews:
        if review.get("action") != "ADD":
            continue
        left_event_id, right_event_id = _interval_refs(review.get("interval_id", ""))
        localization_requests.append(
            {
                "new_keyframe_id": new_event_id(add_idx),
                "new_event_id": new_event_id(add_idx),
                "status": "pending",
                "interval_id": review.get("interval_id"),
                "between_keyframes": [left_event_id, right_event_id],
                "between": [left_event_id, right_event_id],
                "suggested_type": review.get("suggested_type", "UNKNOWN"),
                "approx_time_sec": review.get("approx_time_sec"),
                "confidence": review.get("confidence"),
                "reason": review.get("reason", ""),
                "operation": review.get("operation"),
                "display_interval": review.get("display_interval"),
                "source_addition": review.get("source_addition"),
            }
        )
        add_idx += 1

    merge_groups = [
        {"representative_event_id": parent, "member_event_ids": sorted(members), "status": "merged"}
        for parent, members in sorted(groups.items())
        if len(members) > 1
    ]
    return {
        "final_events": final_events,
        "localization_requests": localization_requests,
        "relocation_requests": relocation_requests,
        "merge_groups": merge_groups,
        "protection_overrides": protection_overrides,
    }


def _resolve_parent(event_id: str, merge_parent: Dict[str, str]) -> str:
    seen = set()
    current = event_id
    while merge_parent.get(current, current) != current and current not in seen:
        seen.add(current)
        current = merge_parent[current]
    return current


def _final_type(event: Event, review: Dict[str, Any]) -> str:
    if review.get("action") == "RELABEL" and review.get("new_type") in {"MAX", "MIN"}:
        return str(review["new_type"])
    return event.event_type


def _protect_significant_extrema(final_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    overrides: List[Dict[str, Any]] = []
    for idx, event in enumerate(final_events):
        if event.get("final_status") not in {"removed", "merged_removed"}:
            continue
        if event.get("type") not in {"MAX", "MIN"}:
            continue
        value = event.get("value_smooth")
        if value is None:
            continue
        left = _nearest_value(final_events, idx, -1)
        right = _nearest_value(final_events, idx, 1)
        if left is None or right is None:
            continue
        prominence = min(abs(float(value) - left), abs(float(value) - right))
        if prominence < 0.12:
            continue
        old_action = event.get("vlm_action")
        old_status = event.get("final_status")
        event["vlm_action"] = "KEEP"
        event["final_status"] = "kept"
        review = event.get("review")
        if isinstance(review, dict):
            review["action_before_protection"] = old_action
            review["action"] = "KEEP"
            review["protected_by"] = "significant_extremum_prominence"
            review["protection_reason"] = (
                "VLM marked this extremum as quasi-plateau, but its smoothed-value prominence "
                f"against both neighboring events is {prominence:.4f}, so it is preserved as a significant reversal."
            )
        overrides.append(
            {
                "keyframe_id": event.get("keyframe_id") or event.get("event_id"),
                "event_id": event.get("event_id"),
                "type": event.get("type"),
                "time_sec": event.get("time_sec"),
                "value_smooth": value,
                "old_action": old_action,
                "old_status": old_status,
                "new_action": "KEEP",
                "new_status": "kept",
                "prominence": prominence,
                "rule": "significant_extremum_prominence>=0.12",
            }
        )
    return overrides


def _nearest_value(final_events: List[Dict[str, Any]], start_idx: int, step: int) -> float | None:
    idx = start_idx + step
    while 0 <= idx < len(final_events):
        value = final_events[idx].get("value_smooth")
        if value is not None:
            return float(value)
        idx += step
    return None


def _final_status(action: str, parent: str, event_id: str) -> str:
    if action == "REMOVE":
        return "removed"
    if action == "RELOCALIZE":
        return "pending_relocalize"
    if action == "MERGE" or parent != event_id:
        return "merged_representative" if parent == event_id else "merged_removed"
    if action == "UNCERTAIN":
        return "manual_review"
    return "kept"


def _interval_refs(interval_id: str) -> Tuple[str | None, str | None]:
    parts = interval_id.split("_")
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None
