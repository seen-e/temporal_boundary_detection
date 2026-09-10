from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

EVENT_TYPES = ["MAX", "MIN", "PL", "PR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare single-pass and two-pass VLM review outputs.")
    parser.add_argument("--single-root", required=True, type=Path)
    parser.add_argument("--two-root", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = compare_roots(args.single_root, args.two_root)
    summary = summarize(rows)
    payload = {"single_root": str(args.single_root), "two_root": str(args.two_root), "rows": rows, "summary": summary}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        write_csv(args.out_csv, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def compare_roots(single_root: Path, two_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for two_final in sorted(two_root.glob("episode_*/**/final_review.json")):
        arm = two_final.parent.name
        episode_id = two_final.parent.parent.name
        single_final = single_root / episode_id / arm / "final_review.json"
        if not single_final.exists():
            continue
        rows.append(compare_final(episode_id, arm, load(single_final), load(two_final)))
    return rows


def compare_final(episode_id: str, arm: str, single: Dict[str, Any], two: Dict[str, Any]) -> Dict[str, Any]:
    single_counts = counts(single)
    two_counts = counts(two)
    input_counts = single.get("stats", {}).get("input_event_counts") or two.get("stats", {}).get("input_event_counts") or {}
    input_total = int(single.get("stats", {}).get("num_input_events") or two.get("stats", {}).get("num_input_events") or 0)
    row: Dict[str, Any] = {
        "episode_id": episode_id,
        "arm": arm,
        "candidate_total": input_total,
        "single_kept_total": single_counts["kept_total"],
        "two_kept_total": two_counts["kept_total"],
        "single_removed_total": input_total - single_counts["kept_total"],
        "two_removed_total": input_total - two_counts["kept_total"],
        "single_delete_ratio": ratio(input_total - single_counts["kept_total"], input_total),
        "two_delete_ratio": ratio(input_total - two_counts["kept_total"], input_total),
    }
    for event_type in EVENT_TYPES:
        total = int(input_counts.get(event_type, 0))
        single_kept = single_counts["kept_by_type"].get(event_type, 0)
        two_kept = two_counts["kept_by_type"].get(event_type, 0)
        row[f"{event_type}_candidate"] = total
        row[f"single_{event_type}_removed"] = total - single_kept
        row[f"two_{event_type}_removed"] = total - two_kept
        row[f"single_{event_type}_delete_ratio"] = ratio(total - single_kept, total)
        row[f"two_{event_type}_delete_ratio"] = ratio(total - two_kept, total)
    return row


def counts(final: Dict[str, Any]) -> Dict[str, Any]:
    kept_by_type = {event_type: 0 for event_type in EVENT_TYPES}
    kept_total = 0
    for event in final.get("final_events", []):
        if event.get("final_status") in {"removed", "merged_removed"}:
            continue
        kept_total += 1
        event_type = event.get("type")
        if event_type in kept_by_type:
            kept_by_type[event_type] += 1
    return {"kept_total": kept_total, "kept_by_type": kept_by_type}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"num_arm_results": len(rows)}
    for key in ["candidate_total", "single_kept_total", "two_kept_total", "single_removed_total", "two_removed_total"]:
        out[key] = sum(int(row.get(key, 0)) for row in rows)
    out["single_delete_ratio"] = ratio(out.get("single_removed_total", 0), out.get("candidate_total", 0))
    out["two_delete_ratio"] = ratio(out.get("two_removed_total", 0), out.get("candidate_total", 0))
    for event_type in EVENT_TYPES:
        total = sum(int(row.get(f"{event_type}_candidate", 0)) for row in rows)
        single_removed = sum(int(row.get(f"single_{event_type}_removed", 0)) for row in rows)
        two_removed = sum(int(row.get(f"two_{event_type}_removed", 0)) for row in rows)
        out[f"{event_type}_candidate"] = total
        out[f"single_{event_type}_removed"] = single_removed
        out[f"two_{event_type}_removed"] = two_removed
        out[f"single_{event_type}_delete_ratio"] = ratio(single_removed, total)
        out[f"two_{event_type}_delete_ratio"] = ratio(two_removed, total)
    return out


def ratio(value: int | float, total: int | float) -> float:
    total = float(total)
    return 0.0 if total <= 0 else float(value) / total


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
