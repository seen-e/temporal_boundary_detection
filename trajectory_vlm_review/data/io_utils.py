from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..core.models import Event, event_id_for_index


TYPE_MAP = {
    "local_maximum": "MAX",
    "maximum": "MAX",
    "max": "MAX",
    "local_minimum": "MIN",
    "minimum": "MIN",
    "min": "MIN",
    "plateau_left_endpoint": "PL",
    "plateau_left": "PL",
    "pl": "PL",
    "plateau_right_endpoint": "PR",
    "plateau_right": "PR",
    "pr": "PR",
}


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def episode_id_from_index(index: int) -> str:
    return f"episode_{index:06d}"


def find_keyframe_json(keyframe_root: str | Path, episode_index: int) -> Optional[Path]:
    root = Path(keyframe_root)
    episode_id = episode_id_from_index(episode_index)
    direct = root / f"{episode_id}.json"
    if direct.exists():
        return direct
    matches = list(root.rglob(f"{episode_id}.json"))
    return matches[0] if matches else None


def list_keyframe_jsons(keyframe_root: str | Path) -> List[Path]:
    root = Path(keyframe_root)
    return sorted(root.rglob("episode_*.json"))


def parse_episode_index(path_or_id: str) -> Optional[int]:
    stem = Path(path_or_id).stem
    if "episode_" not in stem:
        return None
    try:
        return int(stem.split("episode_")[-1][:6])
    except ValueError:
        return None


def _short_type(raw_type: str) -> str:
    key = str(raw_type or "").strip().lower()
    return TYPE_MAP.get(key, key.upper())


def normalize_events(keyframe_data: Dict[str, Any], arm: str) -> List[Event]:
    gripper_key = f"{arm}_gripper"
    section = keyframe_data.get(gripper_key, {})
    raw_events = section.get("cleaned_keyframes") or section.get("keyframes") or []
    sorted_events = sorted(
        raw_events,
        key=lambda e: (
            float(e.get("time", e.get("time_sec", 0.0))),
            int(e.get("frame_index", e.get("sample_index", 0)) or 0),
        ),
    )
    events: List[Event] = []
    for idx, item in enumerate(sorted_events):
        raw_type = item.get("type") or item.get("kind") or item.get("event_type")
        event_type = _short_type(raw_type)
        plateau_id = item.get("plateau_id")
        plateau_pair_id = None
        if plateau_id is not None:
            try:
                plateau_pair_id = f"P{int(plateau_id):03d}"
            except (TypeError, ValueError):
                plateau_pair_id = f"P{plateau_id}"
        events.append(
            Event(
                event_id=event_id_for_index(idx),
                source_index=idx,
                event_type=event_type,
                original_type=str(raw_type),
                kind=str(item.get("kind", "")),
                arm=arm,
                time=float(item.get("time", item.get("time_sec", 0.0))),
                frame_index=_maybe_int(item.get("frame_index")),
                sample_index=_maybe_int(item.get("sample_index")),
                value_raw=_maybe_float(item.get("value_raw")),
                value_smooth=_maybe_float(item.get("value_smooth")),
                plateau_pair_id=plateau_pair_id,
                plateau_id=_maybe_int(plateau_id),
                plateau_start_time=_maybe_float(item.get("plateau_start_time")),
                plateau_end_time=_maybe_float(item.get("plateau_end_time")),
                plateau_duration_sec=_maybe_float(item.get("plateau_duration_sec")),
                source_keyframe=item,
            )
        )
    return events


def load_gripper_trajectory(
    keyframe_data: Dict[str, Any],
    arm: str,
    phase_module_root: str,
    dataset_root: Optional[str] = None,
    episode_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, raw trajectory, smoothed trajectory for one gripper.

    The keyframe extractor records the source parquet path in normal runs. When
    that path is unavailable, this falls back to interpolating over event values
    so the review framework can still render diagnostic slices.
    """

    data_file = keyframe_data.get("data_file") or keyframe_data.get("source_data_file")
    source = keyframe_data.get("source") or {}
    if not data_file and isinstance(source, dict):
        data_file = source.get("data_file")
    if not data_file and dataset_root:
        inferred = infer_lerobot_parquet(dataset_root, episode_index if episode_index is not None else keyframe_data.get("episode_index"))
        if inferred:
            data_file = str(inferred)
    if data_file and Path(data_file).exists():
        return _load_from_parquet(Path(data_file), arm, phase_module_root, episode_index, keyframe_data)
    events = normalize_events(keyframe_data, arm)
    if not events:
        return np.asarray([]), np.asarray([]), np.asarray([])
    times = np.asarray([event.time for event in events], dtype=float)
    values = np.asarray(
        [
            event.value_smooth
            if event.value_smooth is not None
            else event.value_raw
            if event.value_raw is not None
            else 0.0
            for event in events
        ],
        dtype=float,
    )
    dense_t = np.linspace(times.min(), times.max(), max(50, len(times) * 8))
    dense_v = np.interp(dense_t, times, values)
    return dense_t, dense_v.copy(), dense_v


def infer_lerobot_parquet(dataset_root: str | Path, episode_index: Optional[int]) -> Optional[Path]:
    if episode_index is None:
        return None
    try:
        episode_index = int(episode_index)
    except (TypeError, ValueError):
        return None
    root = Path(dataset_root)
    data_root = root / "data"
    if not data_root.exists():
        return None
    for candidate in sorted(data_root.rglob("*.parquet")):
        try:
            import pandas as pd

            episode_col = pd.read_parquet(candidate, columns=["episode_index"])["episode_index"]
        except Exception:
            continue
        if int(episode_col.min()) <= episode_index <= int(episode_col.max()):
            if bool((episode_col == episode_index).any()):
                return candidate
    return None


def _load_from_parquet(
    data_file: Path,
    arm: str,
    phase_module_root: str,
    episode_index: Optional[int] = None,
    keyframe_data: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    if phase_module_root:
        sys.path.insert(0, phase_module_root)
    try:
        from phase_segment.gripper_phase_segment.config import GripperPhaseSegmentationConfig
        from phase_segment.gripper_phase_segment.segmenter import segment_gripper_trajectory
    except Exception:
        segment_gripper_trajectory = None
        GripperPhaseSegmentationConfig = None

    df = pd.read_parquet(data_file)
    if episode_index is not None and "episode_index" in df.columns:
        df = df[df["episode_index"] == int(episode_index)].copy()
    if df.empty:
        raise ValueError(f"no rows for episode_index={episode_index} in {data_file}")
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    states = np.stack(df["observation.state"].to_numpy()).astype(float)
    index = 6 if arm == "left" else 13
    raw = states[:, index]
    if "timestamp" in df.columns:
        time = df["timestamp"].to_numpy(dtype=float)
    else:
        fps = float((df.get("fps") or [10])[0]) if "fps" in df.columns else 10.0
        time = np.arange(len(raw), dtype=float) / fps
    if segment_gripper_trajectory is None:
        smooth = _moving_average(raw, window=9)
    else:
        phase_config = None
        if keyframe_data and isinstance(keyframe_data.get("phase_config"), dict) and GripperPhaseSegmentationConfig is not None:
            phase_config = GripperPhaseSegmentationConfig.from_dict(keyframe_data["phase_config"])
        frame_indices = df["frame_index"].to_numpy(dtype=int) if "frame_index" in df.columns else None
        result = segment_gripper_trajectory(time, raw, config=phase_config, frame_indices=frame_indices, side=arm)
        smooth = np.asarray(result.gripper_smooth, dtype=float)
    return time, raw, smooth


def _infer_fps(time: np.ndarray) -> float:
    if len(time) < 2:
        return 10.0
    dt = np.diff(time)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 10.0
    return float(1.0 / np.median(dt))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    window = max(3, min(window, len(values) // 2 * 2 + 1))
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
