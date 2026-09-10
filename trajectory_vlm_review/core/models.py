from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


EVENT_TYPES = {"MAX", "MIN", "PL", "PR"}
EVENT_ACTIONS = {"KEEP", "REMOVE", "RELABEL", "MERGE", "RELOCALIZE", "UNCERTAIN"}
INTERVAL_ACTIONS = {"NO_MISSING_EVENT", "ADD", "UNCERTAIN"}
GRIPPER_STATES = {"OPENING", "CLOSING", "STABLE", "UNCERTAIN"}
CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}
RELOCALIZE_DIRECTIONS = {"LEFT", "RIGHT", "NEARBY", "UNKNOWN"}


@dataclass(frozen=True)
class Event:
    event_id: str
    source_index: int
    event_type: str
    original_type: str
    kind: str
    arm: str
    time: float
    frame_index: Optional[int] = None
    sample_index: Optional[int] = None
    value_raw: Optional[float] = None
    value_smooth: Optional[float] = None
    plateau_pair_id: Optional[str] = None
    plateau_id: Optional[int] = None
    plateau_start_time: Optional[float] = None
    plateau_end_time: Optional[float] = None
    plateau_duration_sec: Optional[float] = None
    source_keyframe: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_id(self) -> str:
        return display_id_for_index(self.source_index)

    def to_prompt_dict(self, role: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.display_id,
            "keyframe_id": self.display_id,
            "internal_event_id": self.event_id,
            "role": role,
            "type": self.event_type,
            "time_sec": round(self.time, 4),
            "frame_index": self.frame_index,
            "value_smooth": self.value_smooth,
        }
        if self.plateau_pair_id:
            data["plateau_pair_id"] = self.plateau_pair_id
            data["plateau_start_time_sec"] = self.plateau_start_time
            data["plateau_end_time_sec"] = self.plateau_end_time
        return data


@dataclass(frozen=True)
class OwnedInterval:
    interval_id: str
    left_event_id: str
    right_event_id: str
    start_time: float
    end_time: float

    @property
    def display_interval_id(self) -> str:
        return f"{display_id_from_event_id(self.left_event_id)}_{display_id_from_event_id(self.right_event_id)}"

    def to_prompt_dict(self) -> Dict[str, Any]:
        left_display_id = display_id_from_event_id(self.left_event_id)
        right_display_id = display_id_from_event_id(self.right_event_id)
        return {
            "interval": f"{left_display_id}_{right_display_id}",
            "interval_id": f"{left_display_id}_{right_display_id}",
            "internal_interval_id": self.interval_id,
            "left_keyframe_id": left_display_id,
            "right_keyframe_id": right_display_id,
            "internal_left_event_id": self.left_event_id,
            "internal_right_event_id": self.right_event_id,
            "start_time_sec": round(self.start_time, 4),
            "end_time_sec": round(self.end_time, 4),
        }


@dataclass
class ReviewSlice:
    episode_id: str
    episode_index: Optional[int]
    arm: str
    slice_id: str
    context_left: List[Event]
    targets: List[Event]
    context_right: List[Event]
    owned_intervals: List[OwnedInterval]
    time_range: Tuple[float, float]
    episode_events: List[Event] = field(default_factory=list)
    image_path: Optional[str] = None
    global_image_path: Optional[str] = None
    metadata_path: Optional[str] = None

    @property
    def all_events(self) -> List[Event]:
        return [*self.context_left, *self.targets, *self.context_right]

    @property
    def target_ids(self) -> List[str]:
        return [event.event_id for event in self.targets]

    @property
    def target_display_ids(self) -> List[str]:
        return [event.display_id for event in self.targets]

    @property
    def context_ids(self) -> List[str]:
        return [event.event_id for event in [*self.context_left, *self.context_right]]

    @property
    def context_display_ids(self) -> List[str]:
        return [event.display_id for event in [*self.context_left, *self.context_right]]

    def metadata(self) -> Dict[str, Any]:
        def scoped_event(event: Event, scope: str) -> Dict[str, Any]:
            item = event.to_prompt_dict(scope)
            item["scope"] = scope
            return item

        return {
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "arm": self.arm,
            "slice_id": self.slice_id,
            "time_range_sec": [round(self.time_range[0], 4), round(self.time_range[1], 4)],
            "local_image_path": self.image_path,
            "global_image_path": self.global_image_path,
            "left_overlap_ids": [event.display_id for event in self.context_left],
            "target_keyframe_ids": self.target_display_ids,
            "right_overlap_ids": [event.display_id for event in self.context_right],
            "local_keyframe_ids": [event.display_id for event in self.all_events],
            "target_internal_event_ids": self.target_ids,
            "left_context_keyframe_ids": [event.display_id for event in self.context_left],
            "right_context_keyframe_ids": [event.display_id for event in self.context_right],
            "left_context_internal_event_ids": [event.event_id for event in self.context_left],
            "right_context_internal_event_ids": [event.event_id for event in self.context_right],
            "context_keyframe_ids": self.context_display_ids,
            "context_internal_event_ids": self.context_ids,
            "has_left_context": bool(self.context_left),
            "has_right_context": bool(self.context_right),
            "id_mapping": {
                event.display_id: event.event_id
                for event in self.all_events
            },
            "scope_by_id": {
                **{event.display_id: "left_overlap" for event in self.context_left},
                **{event.display_id: "target" for event in self.targets},
                **{event.display_id: "right_overlap" for event in self.context_right},
            },
            "events": [
                *[scoped_event(event, "left_overlap") for event in self.context_left],
                *[scoped_event(event, "target") for event in self.targets],
                *[scoped_event(event, "right_overlap") for event in self.context_right],
            ],
            "owned_intervals": [interval.to_prompt_dict() for interval in self.owned_intervals],
        }


@dataclass
class ValidationIssue:
    code: str
    message: str
    path: str = ""


@dataclass
class ValidationResult:
    ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class ArmReviewResult:
    episode_id: str
    arm: str
    original_events: List[Event]
    slice_results: List[Dict[str, Any]]
    final_events: List[Dict[str, Any]]
    localization_requests: List[Dict[str, Any]]
    relocation_requests: List[Dict[str, Any]]
    merge_groups: List[Dict[str, Any]]
    stats: Dict[str, Any]


def event_id_for_index(index: int) -> str:
    return f"K{index + 1:03d}"


def display_id_for_index(index: int) -> str:
    return f"{index + 1:03d}"


def display_id_from_event_id(event_id: Any) -> str:
    text = str(event_id)
    match = re.search(r"(\d+)$", text)
    if match:
        return f"{int(match.group(1)):03d}"
    return text


def new_event_id(index: int) -> str:
    return f"NEW_{index + 1:03d}"


def ensure_sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else []
