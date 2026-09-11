from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import AppConfig, ensure_config_defaults, load_config
from ..core.models import ReviewSlice, display_id_from_event_id
from ..data.io_utils import (
    episode_id_from_index,
    find_keyframe_json,
    list_keyframe_jsons,
    load_gripper_trajectory,
    normalize_events,
    parse_episode_index,
    read_json,
    write_json,
)
from ..refinement import apply_reviews, run_pass3_completion
from ..reporting import count_events, summarize_slice_reviews
from ..slicing import build_review_slices
from ..validation import normalize_pass1_structure, normalize_review_actions, validate_review
from ..visualization import render_slice
from ..vlm import build_pass1_payload, build_pass2_payload, build_payload, call_vlm, parse_vlm_response, summarize_messages


def run_from_args(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    ensure_config_defaults(args.config)
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    runner = ReviewRunner(cfg)
    summaries = runner.run()
    print(json.dumps({"num_episode_summaries": len(summaries), "summaries": summaries[:5]}, ensure_ascii=False, indent=2))
    return 0


class ReviewRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def run(self) -> List[Dict[str, Any]]:
        episode_paths = self._episode_paths()
        summaries: List[Dict[str, Any]] = []
        for idx, path in enumerate(episode_paths, start=1):
            episode_index = parse_episode_index(str(path))
            print(f"[{idx}/{len(episode_paths)}] review {path.name}")
            try:
                summaries.append(self.run_episode(path, episode_index))
            except Exception as exc:
                episode_id = episode_id_from_index(episode_index) if episode_index is not None else path.stem
                summary = {"episode_id": episode_id, "status": "failed", "error": str(exc)}
                summaries.append(summary)
                out = Path(self.cfg.paths.output_root) / episode_id / "summary.json"
                write_json(out, summary)
                print(f"  failed: {exc}")
        aggregate_path = Path(self.cfg.paths.output_root) / "aggregate_summary.json"
        write_json(aggregate_path, {"summaries": summaries})
        return summaries

    def run_episode(self, keyframe_path: Path, episode_index: Optional[int]) -> Dict[str, Any]:
        data = read_json(keyframe_path)
        episode_id = data.get("episode_id") or (episode_id_from_index(episode_index) if episode_index is not None else keyframe_path.stem)
        arms = self._arms()
        arm_summaries = []
        for arm in arms:
            arm_summaries.append(self.run_arm(data, episode_id, episode_index, arm))
        summary = {"episode_id": episode_id, "episode_index": episode_index, "status": "completed", "arms": arm_summaries}
        write_json(Path(self.cfg.paths.output_root) / episode_id / "summary.json", summary)
        return summary

    def run_arm(self, data: Dict[str, Any], episode_id: str, episode_index: Optional[int], arm: str) -> Dict[str, Any]:
        events = normalize_events(data, arm)
        time, raw, smooth = load_gripper_trajectory(
            data,
            arm,
            self.cfg.paths.phase_module_root,
            dataset_root=self.cfg.paths.dataset_root,
            episode_index=episode_index,
        )
        pass1_params = _stage_slice_params(self.cfg, "pass1")
        pass2_params = _stage_slice_params(self.cfg, "pass2")
        pass3_params = _stage_slice_params(self.cfg, "pass3")
        vlm_stages = _vlm_stages(self.cfg)
        _validate_stage_target_alignment(pass1_params, pass2_params, pass3_params, vlm_stages)
        slice_params = pass2_params if "pass2" in vlm_stages else pass1_params
        slices = build_review_slices(
            episode_id=episode_id,
            episode_index=episode_index,
            arm=arm,
            events=events,
            left_context_events=slice_params[0],
            target_events_per_slice=slice_params[1],
            right_context_events=slice_params[2],
        )
        if self.cfg.run.limit_slices is not None:
            slices = slices[: self.cfg.run.limit_slices]

        slice_results: List[Dict[str, Any]] = []
        pending_slices = []
        for review_slice in slices:
            result_path = self._slice_result_path(episode_id, arm, review_slice.slice_id)
            if self._can_resume(result_path):
                slice_results.append(read_json(result_path))
                continue
            pending_slices.append(review_slice)

        workers = max(1, int(self.cfg.run.workers or 1))
        if workers == 1 or len(pending_slices) <= 1:
            for review_slice in pending_slices:
                result = _run_slice_task(self.cfg, review_slice, time, raw, smooth)
                slice_results.append(result)
        else:
            max_workers = min(workers, len(pending_slices))
            print(f"  {arm}: running {len(pending_slices)} slices with {max_workers} workers")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_slice_task, self.cfg, review_slice, time, raw, smooth): review_slice.slice_id
                    for review_slice in pending_slices
                }
                done = 0
                for future in as_completed(futures):
                    done += 1
                    slice_id = futures[future]
                    result = future.result()
                    slice_results.append(result)
                    print(f"    {arm} {done}/{len(pending_slices)} {slice_id}: {result.get('status')}")
        slice_results = sorted(slice_results, key=lambda item: item.get("slice_id", ""))

        pass2_refinement = apply_reviews(
            events,
            slice_results,
            protect_significant_extrema=bool(
                getattr(self.cfg.vlm_trajectory_review, "enable_significant_extrema_protection", False)
            ),
        )
        refinement = pass2_refinement
        if "pass3" in vlm_stages:
            pass3_slices = [_slice_with_context(slice_, pass3_params[0], pass3_params[2]) for slice_ in slices]
            refinement = run_pass3_completion(
                self.cfg,
                episode_id,
                episode_index,
                arm,
                events,
                pass3_slices,
                slice_results,
                refinement,
                time,
                raw,
                smooth,
            )
        stats = {
            "input_event_counts": count_events(events),
            "num_input_events": len(events),
            "num_slices": len(slices),
            **summarize_slice_reviews(slice_results),
            "num_final_events": len(refinement["final_events"]),
            "num_add_requests": len(refinement["localization_requests"]),
            "num_relocalize_requests": len(refinement["relocation_requests"]),
            "num_merge_groups": len(refinement["merge_groups"]),
        }
        if "pass3" in refinement:
            stats["pass3"] = refinement["pass3"]
        final = {
            "episode_id": episode_id,
            "arm": arm,
            "status": "completed",
            "source": "vlm_trajectory_review",
            "vlm_stages": vlm_stages,
            "original_event_ids": [event.event_id for event in events],
            **refinement,
            "stats": stats,
        }
        write_json(Path(self.cfg.paths.output_root) / episode_id / arm / "final_review.json", final)
        _write_stage_point_results(
            self.cfg,
            episode_id=episode_id,
            episode_index=episode_index,
            arm=arm,
            events=events,
            slice_results=slice_results,
            pass2_refinement=pass2_refinement,
            pass3_refinement=refinement,
            vlm_stages=vlm_stages,
        )
        print(f"  {arm}: {len(events)} events, {len(slices)} slices")
        return {"arm": arm, "status": "completed", "stats": stats}

    def _episode_paths(self) -> List[Path]:
        if self.cfg.run.episode_index is not None:
            path = find_keyframe_json(self.cfg.paths.keyframe_root, self.cfg.run.episode_index)
            if not path:
                raise FileNotFoundError(f"keyframe json not found for episode {self.cfg.run.episode_index}")
            return [path]
        paths = list_keyframe_jsons(self.cfg.paths.keyframe_root)
        if self.cfg.run.episode_start is not None:
            paths = [p for p in paths if (parse_episode_index(str(p)) or -1) >= self.cfg.run.episode_start]
        if self.cfg.run.episode_end is not None:
            paths = [p for p in paths if (parse_episode_index(str(p)) or 10**12) <= self.cfg.run.episode_end]
        if self.cfg.run.limit_episodes is not None:
            paths = paths[: self.cfg.run.limit_episodes]
        return paths

    def _arms(self) -> List[str]:
        arms = list(self.cfg.run.arms)
        if not self.cfg.vlm_trajectory_review.review_left_gripper:
            arms = [arm for arm in arms if arm != "left"]
        if not self.cfg.vlm_trajectory_review.review_right_gripper:
            arms = [arm for arm in arms if arm != "right"]
        return arms

    def _slice_result_path(self, episode_id: str, arm: str, slice_id: str) -> Path:
        return Path(self.cfg.paths.output_root) / episode_id / arm / f"{slice_id}.review.json"

    def _can_resume(self, path: Path) -> bool:
        if self.cfg.vlm_trajectory_review.force or self.cfg.run.dry_run:
            return False
        if not self.cfg.vlm_trajectory_review.resume or not path.exists():
            return False
        try:
            return read_json(path).get("status") == "completed"
        except Exception:
            return False

def _default_keep_review(review_slice, parse_error: str) -> Dict[str, Any]:
    return {
        "region_analysis": {
            "summary": "VLM response could not be parsed after retries; target events are kept for manual review.",
        },
        "event_reviews": [
            {
                "event_id": event.event_id,
                "action": "KEEP",
                "state_before": "UNCERTAIN",
                "state_after": "UNCERTAIN",
                "confidence": "LOW",
                "reason": f"Fallback KEEP because VLM response JSON parse failed: {parse_error}",
            }
            for event in review_slice.targets
        ],
        "interval_reviews": [],
        "manual_review": True,
        "parse_error": parse_error,
    }


def _write_stage_point_results(
    cfg: AppConfig,
    episode_id: str,
    episode_index: Optional[int],
    arm: str,
    events,
    slice_results: List[Dict[str, Any]],
    pass2_refinement: Dict[str, Any],
    pass3_refinement: Dict[str, Any],
    vlm_stages: List[str],
) -> None:
    out_dir = Path(cfg.paths.output_root) / episode_id / arm / "stage_points"
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "episode_id": episode_id,
        "episode_index": episode_index,
        "arm": arm,
        "source": "episode_arm_stage_points",
        "vlm_stages": vlm_stages,
    }
    if "pass1" in vlm_stages:
        pass1_segments = []
        for item in slice_results:
            review = (((item.get("pass1") or {}).get("review") or {}).get("region") or {})
            pass1_segments.append(
                {
                    "slice_id": item.get("slice_id"),
                    "segments": review.get("segments", []),
                    "coarse_segments": review.get("coarse_segments", []),
                    "needs_review": ((item.get("pass1") or {}).get("review") or {}).get("needs_review", False),
                }
            )
        write_json(
            out_dir / "pass1_points.json",
            {
                **common,
                "stage": "pass1",
                "description": "Pass 1 only freezes trajectory segments; it does not remove, merge, relabel, or add keyframes.",
                "num_points": len(events),
                "points": [_event_point(event, "pass1_input") for event in events],
                "slice_segments": pass1_segments,
            },
        )
    if "pass2" in vlm_stages:
        pass2_points = [_reviewed_point(item, "pass2") for item in pass2_refinement.get("final_events", [])]
        write_json(
            out_dir / "pass2_points.json",
            {
                **common,
                "stage": "pass2",
                "description": "Pass 2 applies VLM keep/remove/merge/relabel decisions to the traditional filtered keyframes.",
                "num_points": len(pass2_points),
                "num_kept_points": sum(1 for item in pass2_points if item.get("final_status") not in {"removed", "merged_removed"}),
                "points": pass2_points,
                "merge_groups": pass2_refinement.get("merge_groups", []),
                "localization_requests": pass2_refinement.get("localization_requests", []),
                "relocation_requests": pass2_refinement.get("relocation_requests", []),
            },
        )
    if "pass3" in vlm_stages:
        pass3_points = [_reviewed_point(item, "pass3") for item in pass3_refinement.get("final_events", [])]
        write_json(
            out_dir / "pass3_points.json",
            {
                **common,
                "stage": "pass3",
                "description": "Pass 3 starts from Pass 2 cleaned keyframes and adds missing keyframes for uncovered frozen segment boundaries.",
                "num_points": len(pass3_points),
                "num_kept_points": sum(1 for item in pass3_points if item.get("final_status") not in {"removed", "merged_removed"}),
                "points": pass3_points,
                "pass3": pass3_refinement.get("pass3", {}),
                "merge_groups": pass3_refinement.get("merge_groups", []),
                "localization_requests": pass3_refinement.get("localization_requests", []),
                "relocation_requests": pass3_refinement.get("relocation_requests", []),
            },
        )


def _event_point(event, stage: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "keyframe_id": event.display_id,
        "internal_event_id": event.event_id,
        "source_index": event.source_index,
        "type": event.event_type,
        "original_type": event.original_type,
        "kind": event.kind,
        "time_sec": event.time,
        "frame_index": event.frame_index,
        "sample_index": event.sample_index,
        "value_raw": event.value_raw,
        "value_smooth": event.value_smooth,
        "plateau_pair_id": event.plateau_pair_id,
        "source_keyframe": event.source_keyframe,
    }


def _reviewed_point(item: Dict[str, Any], stage: str) -> Dict[str, Any]:
    event_id = str(item.get("event_id") or item.get("keyframe_id") or "")
    keyframe_id = str(item.get("keyframe_id") or (display_id_from_event_id(event_id) if event_id else ""))
    return {
        "stage": stage,
        "keyframe_id": keyframe_id,
        "internal_event_id": event_id,
        "source_index": item.get("source_index"),
        "type": item.get("type"),
        "original_type": item.get("original_type"),
        "time_sec": item.get("time_sec"),
        "frame_index": item.get("frame_index"),
        "value_smooth": item.get("value_smooth"),
        "plateau_pair_id": item.get("plateau_pair_id"),
        "vlm_action": item.get("vlm_action"),
        "final_status": item.get("final_status"),
        "merge_representative_event_id": item.get("merge_representative_event_id"),
        "review": item.get("review", {}),
        "source_keyframe": item.get("source_keyframe", {}),
    }


def _fill_missing_interval_reviews(review: Dict[str, Any], review_slice) -> Dict[str, Any]:
    """Default missing interval reviews to no-add for prompts that only audit events."""
    interval_reviews = review.setdefault("interval_reviews", [])
    if not isinstance(interval_reviews, list):
        interval_reviews = []
        review["interval_reviews"] = interval_reviews
    seen = {item.get("interval_id") for item in interval_reviews if isinstance(item, dict)}
    for interval in review_slice.owned_intervals:
        if interval.interval_id in seen:
            continue
        interval_reviews.append(
            {
                "interval_id": interval.interval_id,
                "action": "NO_MISSING_EVENT",
                "suggested_type": "UNKNOWN",
                "approx_time_sec": None,
                "confidence": "MEDIUM",
                "reason": "Default no-add interval review inserted because the active prompt returned no interval review.",
            }
        )
    return review


def _dry_run_response(review_slice) -> Dict[str, Any]:
    return {
        "status": "completed",
        "episode_id": review_slice.episode_id,
        "arm": review_slice.arm,
        "slice_id": review_slice.slice_id,
        "event_reviews": [
            {
                "event_id": event.event_id,
                "action": "KEEP",
                "state_before": "UNCERTAIN",
                "state_after": "UNCERTAIN",
                "confidence": "HIGH",
                "reason": "dry-run placeholder keeps target event",
            }
            for event in review_slice.targets
        ],
        "interval_reviews": [
            {
                "interval_id": interval.interval_id,
                "action": "NO_MISSING_EVENT",
                "suggested_type": "UNKNOWN",
                "approx_time_sec": (interval.start_time + interval.end_time) / 2.0,
                "confidence": "HIGH",
                "reason": "dry-run placeholder keeps interval",
            }
            for interval in review_slice.owned_intervals
        ],
        "manual_review": False,
    }


def _redact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = json.loads(json.dumps(payload))
    for message in redacted.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    part["image_url"] = {"url": "<base64 image omitted>"}
    return redacted


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory-only VLM review for gripper keyframe candidates.")
    parser.add_argument("--config", default="/mnt/workspace/wrist_label_exp/config.yaml")
    parser.add_argument("--keyframe-root")
    parser.add_argument("--output-root")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--limit-slices", type=int)
    parser.add_argument("--arms", nargs="+", choices=["left", "right"])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--review-mode", choices=["single_pass", "two_pass"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    if args.keyframe_root:
        cfg.paths.keyframe_root = args.keyframe_root
    if args.output_root:
        cfg.paths.output_root = args.output_root
    if args.episode_index is not None:
        cfg.run.episode_index = args.episode_index
    if args.episode_start is not None:
        cfg.run.episode_start = args.episode_start
    if args.episode_end is not None:
        cfg.run.episode_end = args.episode_end
    if args.limit_episodes is not None:
        cfg.run.limit_episodes = args.limit_episodes
    if args.limit_slices is not None:
        cfg.run.limit_slices = args.limit_slices
    if args.arms:
        cfg.run.arms = args.arms
    if args.workers is not None:
        cfg.run.workers = args.workers
    if args.review_mode:
        cfg.vlm_trajectory_review.review_mode = args.review_mode
    if args.dry_run:
        cfg.run.dry_run = True
    if args.force:
        cfg.vlm_trajectory_review.force = True


def _stage_slice_params(cfg: AppConfig, stage: str) -> Tuple[int, int, int]:
    review = cfg.vlm_trajectory_review
    left = getattr(review, f"{stage}_left_context_events", None)
    target = getattr(review, f"{stage}_target_events_per_slice", None)
    right = getattr(review, f"{stage}_right_context_events", None)
    return (
        max(0, int(review.left_context_events if left is None else left)),
        max(1, int(review.target_events_per_slice if target is None else target)),
        max(0, int(review.right_context_events if right is None else right)),
    )


def _vlm_stages(cfg: AppConfig) -> List[str]:
    configured = list(getattr(cfg.vlm_trajectory_review, "stages", []) or [])
    if configured:
        return _normalize_vlm_stages(configured)
    mode = str(getattr(cfg.vlm_trajectory_review, "review_mode", "single_pass") or "single_pass")
    if mode == "two_pass":
        stages = ["pass1", "pass2"]
        if bool(getattr(cfg.vlm_trajectory_review, "enable_pass3", False)):
            stages.append("pass3")
        return stages
    return ["single_pass"]


def _normalize_vlm_stages(stages: List[str]) -> List[str]:
    aliases = {
        "1": "pass1",
        "2": "pass2",
        "3": "pass3",
        "stage1": "pass1",
        "stage2": "pass2",
        "stage3": "pass3",
    }
    normalized = []
    for item in stages:
        value = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if value not in {"pass1", "pass2", "pass3"}:
            raise ValueError(f"unsupported vlm stage {item!r}; use pass1/pass2/pass3")
        if value not in normalized:
            normalized.append(value)
    valid_prefixes = [["pass1"], ["pass1", "pass2"], ["pass1", "pass2", "pass3"]]
    if normalized not in valid_prefixes:
        raise ValueError(f"vlm stages must be a prefix flow: {valid_prefixes}; got {normalized}")
    return normalized


def _validate_stage_target_alignment(
    pass1_params: Tuple[int, int, int],
    pass2_params: Tuple[int, int, int],
    pass3_params: Tuple[int, int, int],
    vlm_stages: List[str] | None = None,
) -> None:
    vlm_stages = vlm_stages or ["pass1", "pass2", "pass3"]
    all_targets = {"pass1": pass1_params[1], "pass2": pass2_params[1], "pass3": pass3_params[1]}
    targets = {stage: all_targets[stage] for stage in vlm_stages if stage in all_targets}
    if len(set(targets.values())) == 1:
        return
    raise ValueError(
        "Active pass1/pass2/pass3 target_events_per_slice must match because target is the owner write-back partition; "
        f"got {targets}. You can tune pass*_left_context_events and pass*_right_context_events independently."
    )


def _slice_with_context(review_slice: ReviewSlice, left_context_events: int, right_context_events: int) -> ReviewSlice:
    events = list(review_slice.episode_events or review_slice.all_events)
    if not review_slice.targets or not events:
        return deepcopy(review_slice)
    by_id = {event.event_id: idx for idx, event in enumerate(events)}
    target_positions = [by_id[event.event_id] for event in review_slice.targets if event.event_id in by_id]
    if not target_positions:
        return deepcopy(review_slice)
    start = min(target_positions)
    end = max(target_positions) + 1
    context_left = events[max(0, start - left_context_events) : start]
    targets = events[start:end]
    context_right = events[end : min(len(events), end + right_context_events)]
    visible_events = [*context_left, *targets, *context_right]
    if visible_events:
        time_range = (min(event.time for event in visible_events), max(event.time for event in visible_events))
    else:
        time_range = review_slice.time_range
    return ReviewSlice(
        episode_id=review_slice.episode_id,
        episode_index=review_slice.episode_index,
        arm=review_slice.arm,
        slice_id=review_slice.slice_id,
        context_left=context_left,
        targets=targets,
        context_right=context_right,
        owned_intervals=review_slice.owned_intervals,
        time_range=time_range,
        episode_events=events,
    )


def _run_slice_task(cfg: AppConfig, review_slice, time, raw, smooth) -> Dict[str, Any]:
    vlm_stages = _vlm_stages(cfg)
    if vlm_stages == ["pass1"]:
        return _run_pass1_only_slice_task(cfg, review_slice, time, raw, smooth)
    if vlm_stages in (["pass1", "pass2"], ["pass1", "pass2", "pass3"]):
        return _run_two_pass_slice_task(cfg, review_slice, time, raw, smooth)
    mode = str(getattr(cfg.vlm_trajectory_review, "review_mode", "single_pass") or "single_pass")
    if mode == "two_pass":
        return _run_two_pass_slice_task(cfg, review_slice, time, raw, smooth)
    rendered = render_slice(
        review_slice,
        time,
        raw,
        smooth,
        cfg.paths.output_root,
        cfg.visualization,
        show_raw=cfg.vlm_trajectory_review.show_raw_trajectory,
        show_smooth=cfg.vlm_trajectory_review.show_smoothed_trajectory,
        show_time=cfg.vlm_trajectory_review.show_event_time,
        show_type=cfg.vlm_trajectory_review.show_event_type,
        show_id=cfg.vlm_trajectory_review.show_event_id,
        show_candidates=True,
        show_global_points=True,
    )
    result = _review_slice_with_cfg(cfg, rendered)
    write_json(
        Path(cfg.paths.output_root) / rendered.episode_id / rendered.arm / f"{rendered.slice_id}.review.json",
        result,
    )
    return result


def _run_pass1_only_slice_task(cfg: AppConfig, review_slice, time, raw, smooth) -> Dict[str, Any]:
    pass1_params = _stage_slice_params(cfg, "pass1")
    pass1_slice = _slice_with_context(review_slice, pass1_params[0], pass1_params[2])
    pass1_rendered = render_slice(
        pass1_slice,
        time,
        raw,
        smooth,
        cfg.paths.output_root,
        cfg.visualization,
        show_raw=cfg.vlm_trajectory_review.show_raw_trajectory,
        show_smooth=cfg.vlm_trajectory_review.show_smoothed_trajectory,
        show_time=False,
        show_type=False,
        show_id=False,
        output_subdir=f"{review_slice.slice_id}/pass1",
        local_filename="local.png",
        global_filename="global.png",
        show_candidates=False,
        show_global_points=False,
        render_global=True,
        pass1_ticks=True,
    )
    _, pass1_review = _call_pass1(cfg, pass1_rendered)
    pass1_review = _ensure_pass1_segments(pass1_review, pass1_slice)
    if cfg.vlm_trajectory_review.save_visualization:
        _write_pass1_segment_visualizations(cfg, pass1_slice, time, smooth, pass1_review)
    review = _keep_target_events_review(review_slice, "Pass 1 only mode keeps target keyframes because Pass 2 event filtering is disabled.")
    review = _fill_missing_interval_reviews(review, review_slice)
    validation = validate_review(review, review_slice)
    result = {
        "status": "completed" if validation.ok else "invalid",
        "review_mode": "pass1_only",
        "vlm_stages": ["pass1"],
        "episode_id": review_slice.episode_id,
        "arm": review_slice.arm,
        "slice_id": review_slice.slice_id,
        "image_path": pass1_rendered.image_path,
        "metadata_path": pass1_rendered.metadata_path,
        "pass1": {
            "global_image_path": pass1_rendered.global_image_path,
            "local_image_path": pass1_rendered.image_path,
            "metadata_path": pass1_rendered.metadata_path,
            "review": pass1_review,
        },
        "review": review,
        "validation": {
            "ok": validation.ok,
            "issues": [issue.__dict__ for issue in validation.issues],
        },
    }
    base = Path(cfg.paths.output_root) / review_slice.episode_id / review_slice.arm
    write_json(base / f"{review_slice.slice_id}.review.json", result)
    _cleanup_rendered_pngs(cfg, pass1_rendered)
    return result


def _run_two_pass_slice_task(cfg: AppConfig, review_slice, time, raw, smooth) -> Dict[str, Any]:
    pass1_params = _stage_slice_params(cfg, "pass1")
    pass2_params = _stage_slice_params(cfg, "pass2")
    _validate_stage_target_alignment(pass1_params, pass2_params, _stage_slice_params(cfg, "pass3"), _vlm_stages(cfg))
    pass1_slice = _slice_with_context(review_slice, pass1_params[0], pass1_params[2])
    pass1_rendered = render_slice(
        pass1_slice,
        time,
        raw,
        smooth,
        cfg.paths.output_root,
        cfg.visualization,
        show_raw=cfg.vlm_trajectory_review.show_raw_trajectory,
        show_smooth=cfg.vlm_trajectory_review.show_smoothed_trajectory,
        show_time=False,
        show_type=False,
        show_id=False,
        output_subdir=f"{review_slice.slice_id}/pass1",
        local_filename="local.png",
        global_filename="global.png",
        show_candidates=False,
        show_global_points=False,
        render_global=True,
        pass1_ticks=True,
    )
    pass1_raw, pass1_review = _call_pass1(cfg, pass1_rendered)
    pass1_review = _ensure_pass1_segments(pass1_review, pass1_slice)
    if cfg.vlm_trajectory_review.save_visualization:
        _write_pass1_segment_visualizations(cfg, pass1_slice, time, smooth, pass1_review)

    pass2_slice = _slice_with_context(review_slice, pass2_params[0], pass2_params[2])
    pass2_rendered = render_slice(
        pass2_slice,
        time,
        raw,
        smooth,
        cfg.paths.output_root,
        cfg.visualization,
        show_raw=cfg.vlm_trajectory_review.show_raw_trajectory,
        show_smooth=cfg.vlm_trajectory_review.show_smoothed_trajectory,
        show_time=cfg.vlm_trajectory_review.show_event_time,
        show_type=cfg.vlm_trajectory_review.show_event_type,
        show_id=cfg.vlm_trajectory_review.show_event_id,
        output_subdir=f"{review_slice.slice_id}/pass2",
        local_filename="local_candidates.png",
        show_candidates=True,
        show_global_points=True,
        render_global=False,
        pass1_ticks=False,
    )
    pass2_raw, review, validation = _call_pass2(cfg, pass2_rendered, pass1_review)

    result = {
        "status": "completed" if validation.ok else "invalid",
        "review_mode": "two_pass",
        "vlm_stages": _vlm_stages(cfg),
        "episode_id": review_slice.episode_id,
        "arm": review_slice.arm,
        "slice_id": review_slice.slice_id,
        "image_path": pass2_rendered.image_path,
        "metadata_path": pass2_rendered.metadata_path,
        "pass1": {
            "global_image_path": pass1_rendered.global_image_path,
            "local_image_path": pass1_rendered.image_path,
            "metadata_path": pass1_rendered.metadata_path,
            "review": pass1_review,
        },
        "pass2": {
            "local_candidates_path": pass2_rendered.image_path,
            "metadata_path": pass2_rendered.metadata_path,
            "raw_response_saved": str(_stage_path(cfg, review_slice, "pass2") / "response.json"),
        },
        "review": review,
        "validation": {
            "ok": validation.ok,
            "issues": [issue.__dict__ for issue in validation.issues],
        },
    }
    base = Path(cfg.paths.output_root) / review_slice.episode_id / review_slice.arm
    write_json(base / f"{review_slice.slice_id}.review.json", result)
    _write_two_pass_slice_result(cfg, pass2_slice, time, smooth, review, result)
    _cleanup_rendered_pngs(cfg, pass1_rendered, pass2_rendered)
    return result


def _keep_target_events_review(review_slice, reason: str) -> Dict[str, Any]:
    return {
        "region_analysis": {
            "summary": reason,
        },
        "event_reviews": [
            {
                "event_id": event.event_id,
                "action": "KEEP",
                "state_before": "UNCERTAIN",
                "state_after": "UNCERTAIN",
                "confidence": "HIGH",
                "reason": reason,
            }
            for event in review_slice.targets
        ],
        "interval_reviews": [],
        "manual_review": False,
    }


def _call_pass1(cfg: AppConfig, review_slice) -> tuple[Any, Dict[str, Any]]:
    stage_dir = _stage_path(cfg, review_slice, "pass1")
    payload = build_pass1_payload(cfg.model, review_slice)
    if cfg.vlm_trajectory_review.save_request:
        write_json(stage_dir / "request.json", _redact_payload(payload))
        write_json(stage_dir / "message_summary.json", summarize_messages(payload.get("messages", [])))
    if cfg.run.dry_run:
        raw_response = _dry_run_pass1_response(review_slice)
        parsed = raw_response
    else:
        raw_response = call_vlm(cfg.model, payload)
        parsed = parse_vlm_response(raw_response)
    pass1_review = normalize_pass1_structure(parsed)
    if cfg.vlm_trajectory_review.save_response:
        (stage_dir / "response.json").write_text(_extract_response_text(raw_response), encoding="utf-8")
    write_json(stage_dir / "review.json", pass1_review)
    return raw_response, pass1_review


def _call_pass2(cfg: AppConfig, review_slice, pass1_review: Dict[str, Any]) -> tuple[Any, Dict[str, Any], Any]:
    stage_dir = _stage_path(cfg, review_slice, "pass2")
    payload = build_pass2_payload(cfg.model, review_slice, pass1_review)
    if cfg.vlm_trajectory_review.save_request:
        write_json(stage_dir / "request.json", _redact_payload(payload))
        write_json(stage_dir / "message_summary.json", summarize_messages(payload.get("messages", [])))
    parse_errors: List[str] = []
    last_raw_response: Any = {}
    last_review: Dict[str, Any] = {}
    last_validation = None
    attempts = 1 if cfg.run.dry_run else max(1, cfg.vlm_trajectory_review.max_retries)
    frozen_segments = (pass1_review.get("region") or {}).get("segments") or []
    for _attempt in range(attempts):
        if cfg.run.dry_run:
            raw_response = _dry_run_pass2_response(review_slice, frozen_segments)
            parsed = raw_response
        else:
            raw_response = call_vlm(cfg.model, payload)
            try:
                parsed = parse_vlm_response(raw_response)
            except Exception as exc:
                last_raw_response = raw_response if isinstance(raw_response, dict) else {"response": raw_response}
                parse_errors.append(str(exc))
                continue
        try:
            review = normalize_review_actions(parsed, review_slice, frozen_segments=frozen_segments)
            review = _fill_missing_interval_reviews(review, review_slice)
            validation = validate_review(review, review_slice)
        except Exception as exc:
            last_raw_response = raw_response if isinstance(raw_response, dict) else {"response": raw_response}
            parse_errors.append(str(exc))
            continue
        last_raw_response = raw_response if isinstance(raw_response, dict) else {"response": raw_response}
        last_review = review
        last_validation = validation
        if validation.ok:
            break
    if last_validation is None:
        review = _default_keep_review(review_slice, parse_errors[-1] if parse_errors else "unknown pass2 parse error")
        review = _fill_missing_interval_reviews(review, review_slice)
        validation = validate_review(review, review_slice)
        last_review = review
        last_validation = validation
    if cfg.vlm_trajectory_review.save_response:
        (stage_dir / "response.json").write_text(_extract_response_text(last_raw_response), encoding="utf-8")
    write_json(stage_dir / "review.json", last_review)
    return last_raw_response, last_review, last_validation


def _ensure_pass1_segments(pass1_review: Dict[str, Any], review_slice) -> Dict[str, Any]:
    region = pass1_review.setdefault("region", {})
    segments = region.get("segments") if isinstance(region.get("segments"), list) else []
    if segments:
        return pass1_review
    region["segments"] = [
        {
            "id": "001",
            "type": "稳定关闭-带波动",
            "start_time_sec": float(review_slice.time_range[0]),
            "end_time_sec": float(review_slice.time_range[1]),
            "description": "Fallback single frozen segment because Pass 1 returned no valid segments.",
        }
    ]
    pass1_review.setdefault("warnings", []).append(
        {"code": "pass1_empty_segments_fallback", "message": "Pass 1 returned no valid segments; inserted one fallback segment.", "path": "region.segments"}
    )
    pass1_review["needs_review"] = True
    pass1_review["manual_review"] = True
    return pass1_review


def _stage_path(cfg: AppConfig, review_slice, stage: str) -> Path:
    path = Path(cfg.paths.output_root) / review_slice.episode_id / review_slice.arm / review_slice.slice_id / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_rendered_pngs(cfg: AppConfig, *rendered_slices) -> None:
    if cfg.vlm_trajectory_review.save_visualization:
        return
    for rendered in rendered_slices:
        for attr in ("image_path", "global_image_path"):
            value = getattr(rendered, attr, None)
            if not value:
                continue
            try:
                Path(value).unlink(missing_ok=True)
            except OSError:
                pass


def _write_pass1_segment_visualizations(cfg: AppConfig, review_slice, time, smooth, pass1_review: Dict[str, Any]) -> None:
    pass1_dir = Path(cfg.paths.output_root) / review_slice.episode_id / review_slice.arm / review_slice.slice_id / "pass1"
    pass1_dir.mkdir(parents=True, exist_ok=True)
    segments = (pass1_review.get("region") or {}).get("segments") or []
    _plot_pass1_segments(
        out_path=pass1_dir / "segments_local.png",
        review_slice=review_slice,
        time=time,
        smooth=smooth,
        segments=segments,
        local=True,
    )
    _plot_pass1_segments(
        out_path=pass1_dir / "segments_global.png",
        review_slice=review_slice,
        time=time,
        smooth=smooth,
        segments=segments,
        local=False,
    )


def _plot_pass1_segments(out_path: Path, review_slice, time, smooth, segments: List[Dict[str, Any]], local: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    colors = ["#f6c85f", "#6baed6", "#8dd3c7", "#fb8072", "#bebada", "#80b1d3"]
    if local:
        t0 = max(0.0, float(review_slice.time_range[0]) - 1.0)
        t1 = float(review_slice.time_range[1]) + 1.0
        title = f"{review_slice.episode_id} | {review_slice.arm} | {review_slice.slice_id} | Pass 1 local segments"
        figsize = (14.0, 5.0)
    else:
        t0 = float(time[0]) if len(time) else 0.0
        t1 = float(time[-1]) if len(time) else 1.0
        title = f"{review_slice.episode_id} | {review_slice.arm} | {review_slice.slice_id} | Pass 1 global segments"
        figsize = (15.5, 4.2)
    mask = (time >= t0) & (time <= t1) if len(time) else np.asarray([], dtype=bool)
    fig, ax = plt.subplots(figsize=figsize, dpi=170)
    target_start = float(review_slice.targets[0].time) if review_slice.targets else float(review_slice.time_range[0])
    target_end = float(review_slice.targets[-1].time) if review_slice.targets else float(review_slice.time_range[1])
    visible_t0, visible_t1 = review_slice.time_range
    if not local:
        if visible_t0 < target_start:
            ax.axvspan(visible_t0, target_start, color="#6baed6", alpha=0.12, zorder=0)
        if target_end < visible_t1:
            ax.axvspan(target_end, visible_t1, color="#6baed6", alpha=0.12, zorder=0)
    ax.axvspan(target_start, target_end, color="#ffcc00", alpha=0.09, zorder=0)
    for idx, seg in enumerate(segments):
        start = _segment_time(seg.get("start_time_sec"), target_start)
        end = _segment_time(seg.get("end_time_sec"), target_end)
        lo, hi = sorted((max(t0, start), min(t1, end)))
        if hi <= lo:
            continue
        color = colors[idx % len(colors)]
        ax.axvspan(lo, hi, color=color, alpha=0.24, zorder=1)
        ax.axvline(lo, color=color, linewidth=1.4, linestyle="-", alpha=0.8, zorder=3)
        ax.axvline(hi, color=color, linewidth=1.4, linestyle="-", alpha=0.8, zorder=3)
        ax.text((lo + hi) / 2.0, 1.045, f"S{seg.get('id', idx + 1)} {seg.get('type', '')}", ha="center", va="bottom", fontsize=8, color="#222222")
    if len(time):
        ax.plot(time[mask], smooth[mask], color="#111111", lw=1.45 if local else 1.25, label="smoothed", zorder=4)
    ax.axvline(target_start, color="#b00020", linewidth=1.3, linestyle="--", alpha=0.9, zorder=5)
    ax.axvline(target_end, color="#b00020", linewidth=1.3, linestyle="--", alpha=0.9, zorder=5)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("gripper state")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.grid(True, axis="x", which="major", linewidth=0.5, alpha=0.25)
    ax.grid(True, axis="x", which="minor", linewidth=0.35, alpha=0.14)
    ax.grid(True, axis="y", which="major", linewidth=0.4, alpha=0.16)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _segment_time(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _dry_run_pass1_response(review_slice) -> Dict[str, Any]:
    start = float(review_slice.time_range[0])
    end = float(review_slice.time_range[1])
    return {
        "region": {
            "adjacent_regions": {"left": None, "right": None},
            "description": "dry-run frozen whole LOCAL region as one stable segment",
            "coarse_segments": [{"id": "001", "state": "稳定", "start_time_sec": start, "end_time_sec": end, "description": "dry-run"}],
            "segments": [{"id": "001", "type": "稳定关闭-带波动", "start_time_sec": start, "end_time_sec": end, "description": "dry-run"}],
        },
        "needs_review": False,
    }


def _dry_run_pass2_response(review_slice, frozen_segments) -> Dict[str, Any]:
    local_ids = [event.display_id for event in review_slice.all_events]
    events = local_ids[:1] + local_ids[-1:] if len(local_ids) > 1 else local_ids
    return {
        "keyframes": [
            {
                "id": event.display_id,
                "scope": "target" if event in review_slice.targets else "overlap",
                "before": "dry-run",
                "point": "dry-run",
                "after": "dry-run",
                "action": {"motion": "稳定", "magnitude": "小幅", "duration": "短时"},
                "relation": {"previous": "dry-run", "next": "dry-run"},
            }
            for event in review_slice.all_events
        ],
        "segments": [{"id": str(seg.get("id", "001")), "events": events} for seg in frozen_segments],
        "needs_review": False,
    }


def _write_two_pass_slice_result(cfg: AppConfig, review_slice, time, smooth, review: Dict[str, Any], result: Dict[str, Any]) -> None:
    result_dir = Path(cfg.paths.output_root) / review_slice.episode_id / review_slice.arm / review_slice.slice_id / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    cleaned = {
        "episode_id": review_slice.episode_id,
        "arm": review_slice.arm,
        "slice_id": review_slice.slice_id,
        "left_overlap_ids": review.get("left_overlap_display_ids", []),
        "target_keyframe_ids": review_slice.target_display_ids,
        "right_overlap_ids": review.get("right_overlap_display_ids", []),
        "pass1_segments": review.get("frozen_segments", []),
        "pass2_selected_local_ids": review.get("selected_local_display_ids", []),
        "pass2_kept_target_ids": review.get("kept_display_ids", []),
        "pass2_deleted_target_ids": review.get("deleted_display_ids", []),
        "kept_display_ids": review.get("kept_display_ids", []),
        "deleted_display_ids": review.get("deleted_display_ids", []),
        "kept_event_ids": review.get("kept_event_ids", []),
        "deleted_event_ids": review.get("deleted_event_ids", []),
        "frozen_segments": review.get("frozen_segments", []),
        "pass2_segments": (review.get("raw_pass2_review") or {}).get("segments", []),
        "needs_review": review.get("needs_review", False),
        "warnings": review.get("warnings", []),
    }
    write_json(result_dir / "cleaned_events.json", cleaned)
    if cfg.vlm_trajectory_review.save_visualization:
        _plot_slice_before_after(result_dir / "before_after.png", review_slice, time, smooth, review)


def _plot_slice_before_after(out_path: Path, review_slice, time, smooth, review: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    kept = set(review.get("kept_event_ids", []))
    selected_local = set(review.get("selected_local_ids", []))
    deleted_target = set(review.get("deleted_event_ids", []))
    t0 = max(0.0, review_slice.time_range[0] - 1.0)
    t1 = review_slice.time_range[1] + 1.0
    mask = (time >= t0) & (time <= t1) if len(time) else np.asarray([], dtype=bool)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), dpi=170, sharex=True)
    for ax, title, selected_events in [
        (axes[0], "before: all LOCAL candidates", review_slice.all_events),
        (axes[1], "after: selected LOCAL + target owner decisions", review_slice.all_events),
    ]:
        if len(time):
            ax.plot(time[mask], smooth[mask], color="#111111", lw=1.6)
        for event in selected_events:
            if title.startswith("after"):
                if event.event_id in kept:
                    marker, color, alpha, size = "o", "#2ca02c", 0.92, 58
                elif event.event_id in deleted_target:
                    marker, color, alpha, size = "x", "#d62728", 0.9, 56
                elif event.event_id in selected_local:
                    marker, color, alpha, size = "o", "#ff7f0e", 0.72, 52
                else:
                    marker, color, alpha, size = ".", "#8a8a8a", 0.36, 34
            else:
                marker, color, alpha, size = "o", "#444444", 0.68, 42
            ax.scatter([event.time], [_event_value(event, time, smooth)], s=size, marker=marker, color=color, alpha=alpha, zorder=5)
            scope = "T" if event in review_slice.targets else ("L" if event in review_slice.context_left else "R")
            ax.text(event.time, _event_value(event, time, smooth), f"{event.display_id}/{scope}", fontsize=7, ha="center", va="bottom")
        ax.axvspan(review_slice.targets[0].time, review_slice.targets[-1].time, color="#ffcc00", alpha=0.12, zorder=0)
        ax.set_ylabel(title)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.22, lw=0.45)
    axes[-1].set_xlabel("time (sec)")
    axes[-1].set_xlim(t0, t1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _event_value(event, time, smooth) -> float:
    if event.value_smooth is not None:
        return float(event.value_smooth)
    if len(time) and len(smooth):
        return float(smooth[int(np.argmin(np.abs(time - float(event.time))))])
    return 0.5


def _review_slice_with_cfg(cfg: AppConfig, review_slice) -> Dict[str, Any]:
    payload = build_payload(cfg.model, review_slice)
    base = Path(cfg.paths.output_root) / review_slice.episode_id / review_slice.arm
    if cfg.vlm_trajectory_review.save_request:
        write_json(base / f"{review_slice.slice_id}.request.json", _redact_payload(payload))

    last_raw_response: Any = {}
    last_review: Dict[str, Any] = {}
    last_validation = None
    parse_errors: List[str] = []
    attempts = 1 if cfg.run.dry_run else max(1, cfg.vlm_trajectory_review.max_retries)
    for _attempt in range(attempts):
        if cfg.run.dry_run:
            raw_response = _dry_run_response(review_slice)
            review = raw_response
        else:
            raw_response = call_vlm(cfg.model, payload)
            try:
                review = normalize_review_actions(parse_vlm_response(raw_response), review_slice)
            except Exception as exc:
                last_raw_response = raw_response if isinstance(raw_response, dict) else {"response": raw_response}
                parse_errors.append(str(exc))
                continue
            review = _fill_missing_interval_reviews(review, review_slice)
        validation = validate_review(review, review_slice)
        last_raw_response = raw_response if isinstance(raw_response, dict) else {"response": raw_response}
        last_review = review
        last_validation = validation
        if validation.ok:
            break
    if last_validation is None:
        review = _default_keep_review(review_slice, parse_errors[-1] if parse_errors else "unknown parse error")
        review = _fill_missing_interval_reviews(review, review_slice)
        validation = validate_review(review, review_slice)
        last_review = review
        last_validation = validation

    if cfg.vlm_trajectory_review.save_response:
        response_path = base / f"{review_slice.slice_id}.response.json"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(_extract_response_text(last_raw_response), encoding="utf-8")

    validation = last_validation
    review = last_review
    return {
        "status": "completed" if validation.ok else "invalid",
        "episode_id": review_slice.episode_id,
        "arm": review_slice.arm,
        "slice_id": review_slice.slice_id,
        "image_path": review_slice.image_path,
        "metadata_path": review_slice.metadata_path,
        "review": review,
        "validation": {
            "ok": validation.ok,
            "issues": [issue.__dict__ for issue in validation.issues],
        },
    }


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
