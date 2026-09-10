from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .index_io import (
    group_index_episodes_by_data_file,
    keyframe_output_path,
    load_index,
    read_episode_table,
    scan_completed_keyframes,
    select_episode_ids,
    write_json,
)
from .traditional_extractor import build_traditional_configs, extract_episode_from_index


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    apply_cli_overrides(config, args)
    ensure_paths(config, args.config)

    summaries: Dict[str, Any] = {"config": str(Path(args.config).resolve()), "stages": {}}
    stages = set(config.get("pipeline", {}).get("stages") or ["traditional", "vlm"])
    if "all" in stages:
        stages = {"traditional", "vlm", "visualize"}

    if "traditional" in stages:
        summaries["stages"]["traditional"] = run_traditional_stage(config)
    if "vlm" in stages:
        summaries["stages"]["vlm"] = run_vlm_stage(config)
    if "visualize" in stages:
        summaries["stages"]["visualize"] = run_visualization_stage(config)

    write_json(Path(config["paths"]["output_root"]) / "pipeline_summary.json", summaries, pretty=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2)[:6000])
    return 0


def run_traditional_stage(config: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    index = load_index(config["paths"]["index_json"])
    run_cfg = dict(config.get("run") or {})
    selected = select_episode_ids(index, run_cfg)
    traditional_cfg = dict(config.get("traditional") or {})
    keyframe_root = Path(config["paths"]["keyframe_root"])
    overwrite = bool(traditional_cfg.get("overwrite", False))
    completed = set() if overwrite else scan_completed_keyframes(keyframe_root)
    pending = [ep for ep in selected if ep not in completed]
    if traditional_cfg.get("limit_files") is not None:
        grouped = group_index_episodes_by_data_file(index, pending)[: int(traditional_cfg["limit_files"])]
    else:
        grouped = group_index_episodes_by_data_file(index, pending)
    workers = max(1, int(run_cfg.get("workers") or 1))
    chunk_size = int(traditional_cfg.get("output_chunk_size") or 1000)
    print(
        "traditional_plan:",
        f"selected={len(selected)}",
        f"skipped={len(selected) - len(pending)}",
        f"pending={len(pending)}",
        f"files={len(grouped)}",
        f"workers={workers}",
        flush=True,
    )

    tasks = [
        {
            "config": config,
            "data_file": item["data_file"],
            "episodes": item["episodes"],
            "keyframe_root": str(keyframe_root),
            "chunk_size": chunk_size,
            "pretty": bool(traditional_cfg.get("pretty", False)),
        }
        for item in grouped
    ]
    summary = {
        "selected": len(selected),
        "skipped_existing": len(selected) - len(pending),
        "scheduled": sum(len(t["episodes"]) for t in tasks),
        "files": len(tasks),
        "succeeded": 0,
        "failed": 0,
        "errors": [],
    }
    done_files = 0
    if workers == 1 or len(tasks) <= 1:
        for task in tasks:
            result = process_file_task(task)
            done_files += 1
            merge_traditional_result(summary, result)
            print_progress("traditional_progress", done_files, len(tasks), summary, started)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = [pool.submit(process_file_task, task) for task in tasks]
            for future in as_completed(futures):
                done_files += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"succeeded": 0, "failed": 1, "errors": [{"error": repr(exc)}]}
                merge_traditional_result(summary, result)
                print_progress("traditional_progress", done_files, len(tasks), summary, started)
    summary["elapsed_sec"] = round(time.time() - started, 3)
    summary["episodes_per_sec"] = round((summary["succeeded"] + summary["failed"]) / max(summary["elapsed_sec"], 1e-9), 3)
    write_json(Path(config["paths"]["output_root"]) / "traditional_summary.json", summary, pretty=True)
    return summary


def process_file_task(task: Dict[str, Any]) -> Dict[str, Any]:
    config = task["config"]
    phase_cfg, filter_cfg, base_cfg, post_base_cfg = build_traditional_configs(config)
    df = read_episode_table(task["data_file"], task["episodes"])
    result = {"data_file": task["data_file"], "succeeded": 0, "failed": 0, "errors": []}
    for episode in task["episodes"]:
        try:
            payload = extract_episode_from_index(df, episode, task["data_file"], phase_cfg, filter_cfg, base_cfg, post_base_cfg, config)
            out_path = keyframe_output_path(task["keyframe_root"], int(episode["episode_index"]), int(task["chunk_size"]))
            write_json(out_path, payload, pretty=bool(task["pretty"]))
            result["succeeded"] += 1
        except Exception as exc:
            result["failed"] += 1
            result["errors"].append({"episode_index": episode.get("episode_index"), "error": repr(exc)})
    return result


def run_vlm_stage(config: Dict[str, Any]) -> Dict[str, Any]:
    from trajectory_vlm_review.core.config import load_config as load_vlm_config
    from trajectory_vlm_review.runner.review_runner import ReviewRunner

    started = time.time()
    cfg = load_vlm_config(config["_config_path"])
    run_cfg = dict(config.get("run") or {})
    review_cfg = dict(config.get("vlm_trajectory_review") or {})
    vlm_cfg = dict(config.get("vlm") or {})
    if "stages" in vlm_cfg:
        review_cfg["stages"] = vlm_cfg["stages"]
    cfg.paths.keyframe_root = config["paths"]["keyframe_root"]
    cfg.paths.output_root = config["paths"]["vlm_output_root"]
    cfg.paths.dataset_root = config.get("paths", {}).get("dataset_root") or load_index(config["paths"]["index_json"])["dataset_root"]
    cfg.paths.phase_module_root = config["paths"]["phase_module_root"]
    for name in ["dry_run", "arms", "workers", "episode_index", "episode_start", "episode_end", "limit_episodes", "limit_slices"]:
        if name in run_cfg:
            setattr(cfg.run, name, run_cfg[name])
    for name in ["force", "resume", "review_mode", "enable_pass3", "stages"]:
        if name in review_cfg:
            setattr(cfg.vlm_trajectory_review, name, review_cfg[name])
    runner = ReviewRunner(cfg)
    summaries = runner.run()
    summary = {
        "episodes": len(summaries),
        "elapsed_sec": round(time.time() - started, 3),
        "vlm_stages": list(getattr(cfg.vlm_trajectory_review, "stages", []) or []),
        "summaries": summaries[:20],
    }
    write_json(Path(config["paths"]["output_root"]) / "vlm_summary.json", summary, pretty=True)
    return summary


def run_visualization_stage(config: Dict[str, Any]) -> Dict[str, Any]:
    from trajectory_vlm_review.visualization.three_stage_comparison import plot_episode_three_stage

    index = load_index(config["paths"]["index_json"])
    selected = select_episode_ids(index, dict(config.get("run") or {}))
    vis_cfg = dict(config.get("pipeline", {}).get("visualization") or {})
    max_episodes = vis_cfg.get("max_episodes")
    if max_episodes is not None:
        selected = selected[: int(max_episodes)]
    results = []
    for episode_index in selected:
        try:
            results.append(
                plot_episode_three_stage(
                    episode_index=episode_index,
                    keyframe_root=config["paths"]["keyframe_root"],
                    output_root=config["paths"]["vlm_output_root"],
                    dataset_root=config["paths"]["dataset_root"],
                    phase_module_root=config["paths"]["phase_module_root"],
                )
            )
        except Exception as exc:
            results.append({"episode_index": episode_index, "status": "failed", "error": repr(exc)})
    summary = {"episodes": len(results), "results": results}
    write_json(Path(config["paths"]["output_root"]) / "visualization_summary.json", summary, pretty=True)
    return summary


def merge_traditional_result(summary: Dict[str, Any], result: Dict[str, Any]) -> None:
    summary["succeeded"] += int(result.get("succeeded", 0))
    summary["failed"] += int(result.get("failed", 0))
    summary["errors"].extend(result.get("errors", []))
    summary["errors"] = summary["errors"][:100]


def print_progress(prefix: str, done_files: int, total_files: int, summary: Dict[str, Any], started: float) -> None:
    elapsed = max(time.time() - started, 1e-9)
    done_eps = int(summary["succeeded"]) + int(summary["failed"])
    print(
        f"{prefix}:",
        f"files={done_files}/{total_files}",
        f"episodes={done_eps}/{summary['scheduled']}",
        f"success={summary['succeeded']}",
        f"failed={summary['failed']}",
        f"avg={done_eps / elapsed:.2f} eps/s",
        flush=True,
    )


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_config_path"] = str(Path(path).resolve())
    return data


def ensure_paths(config: Dict[str, Any], config_path: str | Path) -> None:
    root = Path(config_path).resolve().parent
    paths = config.setdefault("paths", {})
    paths.setdefault("phase_module_root", str(root))
    paths.setdefault("output_root", str(root / "outputs" / "abc_130k_v3_pipeline"))
    paths.setdefault("keyframe_root", str(Path(paths["output_root"]) / "traditional_keyframes"))
    paths.setdefault("vlm_output_root", str(Path(paths["output_root"]) / "vlm_review"))
    if not paths.get("dataset_root") and paths.get("index_json"):
        paths["dataset_root"] = load_index(paths["index_json"])["dataset_root"]
    Path(paths["output_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["keyframe_root"]).mkdir(parents=True, exist_ok=True)
    Path(paths["vlm_output_root"]).mkdir(parents=True, exist_ok=True)


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    paths = config.setdefault("paths", {})
    run = config.setdefault("run", {})
    pipeline = config.setdefault("pipeline", {})
    if args.index_json:
        paths["index_json"] = args.index_json
    if args.output_root:
        paths["output_root"] = args.output_root
        paths["keyframe_root"] = str(Path(args.output_root) / "traditional_keyframes")
        paths["vlm_output_root"] = str(Path(args.output_root) / "vlm_review")
    if args.stage:
        pipeline["stages"] = args.stage
    if args.vlm_stage:
        config.setdefault("vlm", {})["stages"] = args.vlm_stage
    if args.episode_index is not None:
        run["episode_index"] = args.episode_index
    if args.episode_start is not None:
        run["episode_start"] = args.episode_start
    if args.episode_end is not None:
        run["episode_end"] = args.episode_end
    if args.limit_episodes is not None:
        run["limit_episodes"] = args.limit_episodes
    if args.limit_slices is not None:
        run["limit_slices"] = args.limit_slices
    if args.workers is not None:
        run["workers"] = args.workers
    if args.dry_run:
        run["dry_run"] = True
    if args.force:
        config.setdefault("vlm_trajectory_review", {})["force"] = True
        config.setdefault("traditional", {})["overwrite"] = True


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional gripper keyframe extraction plus three-stage VLM filtering from an ABC-130K index JSON.")
    parser.add_argument("--config", default="/mnt/workspace/temporal_boundary_detection/config.yaml")
    parser.add_argument("--index-json")
    parser.add_argument("--output-root")
    parser.add_argument("--stage", nargs="+", choices=["traditional", "vlm", "visualize", "all"])
    parser.add_argument("--vlm-stage", nargs="+", choices=["pass1", "pass2", "pass3", "stage1", "stage2", "stage3"])
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--limit-slices", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
