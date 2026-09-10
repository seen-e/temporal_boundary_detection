from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 1200
    max_retries: int = 2


@dataclass
class PathConfig:
    dataset_root: str = "/mnt/data/yuluo/data/abc_130k_v3_train"
    keyframe_root: str = "/mnt/data/chachaxu/dataset/abc_130k_train_key_frame"
    output_root: str = "/mnt/workspace/wrist_label_exp/outputs"
    phase_module_root: str = "/mnt/workspace/temporal_boundary_detection"


@dataclass
class ReviewConfig:
    enabled: bool = True
    review_mode: str = "single_pass"
    left_context_events: int = 2
    target_events_per_slice: int = 6
    right_context_events: int = 2
    pass1_left_context_events: Optional[int] = None
    pass1_target_events_per_slice: Optional[int] = None
    pass1_right_context_events: Optional[int] = None
    pass2_left_context_events: Optional[int] = None
    pass2_target_events_per_slice: Optional[int] = None
    pass2_right_context_events: Optional[int] = None
    pass3_left_context_events: Optional[int] = None
    pass3_target_events_per_slice: Optional[int] = None
    pass3_right_context_events: Optional[int] = None
    review_left_gripper: bool = True
    review_right_gripper: bool = True
    show_raw_trajectory: bool = True
    show_smoothed_trajectory: bool = True
    show_event_time: bool = True
    show_event_type: bool = True
    show_event_id: bool = True
    enable_add: bool = True
    enable_merge: bool = True
    enable_relocalize: bool = True
    enable_significant_extrema_protection: bool = False
    save_visualization: bool = True
    save_request: bool = True
    save_response: bool = True
    resume: bool = True
    force: bool = False
    max_retries: int = 3
    enable_pass3: bool = True
    pass3_mode: str = "frozen_boundary"
    pass3_max_iterations: int = 1
    pass3_max_added_points_per_gap: int = 4
    pass3_max_added_points_per_slice: int = 8
    pass3_min_gap_duration_sec: float = 0.3
    pass3_context_sec: float = 1.5
    pass3_refine_window_sec: float = 0.8


@dataclass
class VisualizationConfig:
    dpi: int = 170
    width: float = 14.0
    height: float = 5.5
    padding_sec: float = 1.0
    min_window_sec: float = 4.0
    dense_ticks: bool = True
    pass1_global_major_xticks: int = 10
    pass1_global_minor_xticks: int = 30
    pass1_local_major_xticks: int = 10
    pass1_local_minor_xticks: int = 24


@dataclass
class RunConfig:
    dry_run: bool = False
    arms: List[str] = field(default_factory=lambda: ["left", "right"])
    workers: int = 1
    episode_index: Optional[int] = None
    episode_start: Optional[int] = None
    episode_end: Optional[int] = None
    limit_episodes: Optional[int] = None
    limit_slices: Optional[int] = None


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    vlm_trajectory_review: ReviewConfig = field(default_factory=ReviewConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _merge_dataclass(cls, data: Optional[Dict[str, Any]]):
    if not isinstance(data, dict):
        return cls()
    defaults = cls()
    values = defaults.__dict__.copy()
    values.update({k: v for k, v in data.items() if k in values})
    return cls(**values)


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig(
        model=_merge_dataclass(ModelConfig, data.get("model")),
        paths=_merge_dataclass(PathConfig, data.get("paths")),
        vlm_trajectory_review=_merge_dataclass(ReviewConfig, data.get("vlm_trajectory_review")),
        visualization=_merge_dataclass(VisualizationConfig, data.get("visualization")),
        run=_merge_dataclass(RunConfig, data.get("run")),
    )


def default_config_dict() -> Dict[str, Any]:
    cfg = AppConfig()
    return {
        "paths": cfg.paths.__dict__,
        "vlm_trajectory_review": cfg.vlm_trajectory_review.__dict__,
        "visualization": cfg.visualization.__dict__,
        "run": cfg.run.__dict__,
    }


def ensure_config_defaults(path: str) -> None:
    p = Path(path)
    data: Dict[str, Any] = {}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    defaults = default_config_dict()
    changed = False
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
            changed = True
        elif isinstance(value, dict) and isinstance(data[key], dict):
            for child_key, child_value in value.items():
                if child_key not in data[key]:
                    data[key][child_key] = child_value
                    changed = True
    if changed:
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
