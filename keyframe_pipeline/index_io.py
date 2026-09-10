from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Dict[str, Any], pretty: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(json_safe(data), f, ensure_ascii=False, indent=2)
        else:
            json.dump(json_safe(data), f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def load_index(path: str | Path) -> Dict[str, Any]:
    index = read_json(path)
    if "dataset_root" not in index or "episodes" not in index:
        raise ValueError(f"not a supported gripper trajectory index: {path}")
    return index


def select_episode_ids(index: Dict[str, Any], run_cfg: Dict[str, Any]) -> List[int]:
    ids = sorted(int(k) for k in index["episodes"].keys())
    if run_cfg.get("episode_index") is not None:
        wanted = int(run_cfg["episode_index"])
        return [wanted] if str(wanted) in index["episodes"] else []
    if run_cfg.get("episode_start") is not None:
        ids = [i for i in ids if i >= int(run_cfg["episode_start"])]
    if run_cfg.get("episode_end") is not None:
        ids = [i for i in ids if i < int(run_cfg["episode_end"])]
    if run_cfg.get("limit_episodes") is not None:
        ids = ids[: int(run_cfg["limit_episodes"])]
    return ids


def group_index_episodes_by_data_file(index: Dict[str, Any], episode_ids: Iterable[int]) -> List[Dict[str, Any]]:
    root = Path(index["dataset_root"])
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for episode_index in episode_ids:
        episode = index["episodes"][str(int(episode_index))]
        rel_path = episode["left_gripper"]["path"]
        groups[rel_path].append(episode)
    tasks = []
    for rel_path, episodes in sorted(groups.items()):
        tasks.append({"dataset_root": str(root), "data_file": str(root / rel_path), "episodes": episodes})
    return tasks


def read_episode_table(data_file: str | Path, episodes: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    for episode in episodes:
        columns.add(episode["left_gripper"]["column"])
        columns.add(episode["right_gripper"]["column"])
    available = set(pq.ParquetFile(data_file).schema_arrow.names)
    columns = [col for col in columns if col in available]
    return pd.read_parquet(data_file, columns=columns)


def extract_gripper_values(df: pd.DataFrame, segment: Dict[str, Any]) -> np.ndarray:
    values = df.iloc[int(segment["row_start"]) : int(segment["row_end"])][segment["column"]]
    indices = segment.get("indices")
    if indices is None:
        arr = values.to_numpy()
        return np.asarray(arr, dtype=float).reshape(-1)
    first = values.iloc[0] if len(values) else None
    if isinstance(first, (list, tuple, np.ndarray)):
        stacked = np.stack(values.to_numpy()).astype(float, copy=False)
    else:
        stacked = values.to_numpy(dtype=float).reshape(-1, 1)
    selected = stacked[:, [int(i) for i in indices]]
    if selected.shape[1] != 1:
        raise ValueError(f"expected one gripper value, got shape={selected.shape}")
    return selected[:, 0]


def keyframe_output_path(root: str | Path, episode_index: int, chunk_size: int) -> Path:
    chunk_index = int(episode_index) // int(chunk_size)
    return Path(root) / f"chunk-{chunk_index:03d}" / f"episode_{int(episode_index):06d}.json"


def scan_completed_keyframes(root: str | Path) -> set[int]:
    p = Path(root)
    if not p.exists():
        return set()
    out: set[int] = set()
    for path in p.rglob("episode_*.json"):
        try:
            stem = path.stem
            episode_index = int(stem.split("episode_")[-1][:6])
            data = read_json(path)
            if data.get("status") == "complete":
                out.add(episode_index)
        except Exception:
            continue
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
