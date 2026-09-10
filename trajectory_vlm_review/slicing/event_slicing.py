from __future__ import annotations

from typing import List, Optional

from ..core.models import Event, OwnedInterval, ReviewSlice


def build_review_slices(
    episode_id: str,
    episode_index: Optional[int],
    arm: str,
    events: List[Event],
    left_context_events: int = 2,
    target_events_per_slice: int = 6,
    right_context_events: int = 2,
) -> List[ReviewSlice]:
    if target_events_per_slice <= 0:
        raise ValueError("target_events_per_slice must be positive")
    slices: List[ReviewSlice] = []
    for slice_idx, start in enumerate(range(0, len(events), target_events_per_slice)):
        end = min(start + target_events_per_slice, len(events))
        context_left = events[max(0, start - left_context_events) : start]
        targets = events[start:end]
        context_right = events[end : min(len(events), end + right_context_events)]
        owned_intervals = []
        for left, right in zip(targets, targets[1:]):
            owned_intervals.append(
                OwnedInterval(
                    interval_id=f"{left.event_id}_{right.event_id}",
                    left_event_id=left.event_id,
                    right_event_id=right.event_id,
                    start_time=left.time,
                    end_time=right.time,
                )
            )
        visible_events = [*context_left, *targets, *context_right]
        if visible_events:
            t0 = min(event.time for event in visible_events)
            t1 = max(event.time for event in visible_events)
        else:
            t0 = t1 = 0.0
        slices.append(
            ReviewSlice(
                episode_id=episode_id,
                episode_index=episode_index,
                arm=arm,
                slice_id=f"slice_{slice_idx:03d}",
                context_left=context_left,
                targets=targets,
                context_right=context_right,
                owned_intervals=owned_intervals,
                time_range=(t0, t1),
                episode_events=list(events),
            )
        )
    return slices
