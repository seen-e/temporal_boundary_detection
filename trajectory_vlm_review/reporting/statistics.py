from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

from ..core.models import Event


def count_events(events: Iterable[Event]) -> Dict[str, int]:
    return dict(Counter(event.event_type for event in events))


def summarize_slice_reviews(slice_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    event_actions: Counter[str] = Counter()
    interval_actions: Counter[str] = Counter()
    invalid = 0
    for item in slice_results:
        if item.get("status") != "completed":
            invalid += 1
            continue
        review = item.get("review", {})
        for event_review in review.get("event_reviews", []):
            event_actions[event_review.get("action", "UNKNOWN")] += 1
        for interval_review in review.get("interval_reviews", []):
            interval_actions[interval_review.get("action", "UNKNOWN")] += 1
    return {
        "event_actions": dict(event_actions),
        "interval_actions": dict(interval_actions),
        "invalid_or_skipped_slices": invalid,
    }
