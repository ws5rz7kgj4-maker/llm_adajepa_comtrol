"""Offline verification of archived evidence; no simulation or network calls."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from experiment.core import parse_proposal
from experiment.spec import MODEL_SEEDS, fingerprint, scenes

METHODS = ("adaptive_only", "error", "joint", "joint_async", "joint_rule")
DATA = ROOT / "data" / "audit"


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank record: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object: {path}:{line_number}")
            rows.append(row)
    return rows


def verify_hashes(root=ROOT):
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Manifest escapes repository: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {name}")
    return len(manifest["sha256"])


def validate(episodes, calls):
    """Reject missing, duplicate, mismatched, or parser-inconsistent evidence."""
    expected = {
        f"{scene['id']}__seed{seed}__{method}": (scene, seed, method)
        for scene in scenes() for seed in MODEL_SEEDS for method in METHODS
    }
    indexed = {}
    for row in episodes:
        ident = row["id"]
        if ident in indexed:
            raise ValueError(f"Duplicate episode: {ident}")
        if ident not in expected:
            raise ValueError(f"Unexpected episode: {ident}")
        scene, seed, method = expected[ident]
        if (row["task"], row["condition"], row["method"]) != (
            scene["task"], scene["condition"], method
        ) or type(row["success"]) is not bool:
            raise ValueError(f"Episode metadata mismatch: {ident}")
        indexed[ident] = dict(row, model_seed=seed, scene_id=scene["id"])
    if set(indexed) != set(expected):
        raise ValueError(f"Incomplete audit: {len(indexed)} / {len(expected)} episodes")
    by_episode = defaultdict(list)
    seen_calls = set()
    vlm_replays = 0
    for call in calls:
        ident = call["episode"]
        if ident not in indexed:
            raise ValueError(f"Call has unknown episode: {ident}")
        key = (ident, call["call"])
        if key in seen_calls:
            raise ValueError(f"Duplicate call: {key}")
        seen_calls.add(key)
        row = indexed[ident]
        if call["method"] != row["method"] or call["success"] != row["success"]:
            raise ValueError(f"Call metadata mismatch: {key}")
        if row["method"] == "joint_rule":
            if call["raw"] != "rule" or call["recorded_invalid"] is not None:
                raise ValueError(f"Unexpected rule response: {key}")
            parse_proposal(json.dumps(call["proposal"]))
        else:
            vlm_replays += 1
            error, proposal = None, None
            try:
                proposal = parse_proposal(call["raw"])
            except (ValueError, TypeError) as exc:
                error = str(exc)
            if error != call["recorded_invalid"] or error != call["reparsed_invalid"]:
                raise ValueError(f"Parser mismatch: {key}")
            if proposal != call["proposal"]:
                raise ValueError(f"Parsed proposal mismatch: {key}")
        if call["applied"] and (
            call["stale"] or call["episode_ended"] or call["proposal"] is None
        ):
            raise ValueError(f"Invalid application record: {key}")
        by_episode[ident].append(call)
    for ident, row in indexed.items():
        records = by_episode[ident]
        if {r["call"] for r in records} != set(range(row["calls"])):
            raise ValueError(f"Call coverage mismatch: {ident}")
        invalid = sum(r["recorded_invalid"] is not None for r in records)
        if invalid != row["invalid"] or row["parser_replay_mismatches"] != 0:
            raise ValueError(f"Invalid-response count mismatch: {ident}")
    return list(indexed.values()), vlm_replays


def aggregate(episodes, calls):
    output = []
    for method in METHODS:
        rows = [r for r in episodes if r["method"] == method]
        replies = [r for r in calls if r["method"] == method]
        count = len(rows)
        successes = sum(r["success"] for r in rows)
        invalid = sum(r["recorded_invalid"] is not None for r in replies)
        output.append({
            "method": method, "episodes": count, "successes": successes,
            "success_rate": successes / count, "response_records": len(replies),
            "vlm_responses": len(replies) if method != "joint_rule" else 0,
            "invalid_responses": invalid,
            "invalid_fraction": invalid / len(replies) if replies else None,
        })
    return output


def grouped(episodes, field):
    output = []
    for method in METHODS:
        for group in sorted({r[field] for r in episodes}):
            rows = [r for r in episodes if r["method"] == method and r[field] == group]
            successes = sum(r["success"] for r in rows)
            output.append({
                "method": method, field: group, "episodes": len(rows),
                "successes": successes, "success_rate": successes / len(rows),
            })
    return output


def paired_counts(episodes):
    pairs = defaultdict(dict)
    for row in episodes:
        pairs[(row["scene_id"], row["model_seed"])][row["method"]] = row["success"]
    output = []
    for left, right in [("joint_async", "joint"), ("joint_async", "adaptive_only")]:
        counts = Counter((r[left], r[right]) for r in pairs.values())
        output.append({
            "left": left, "right": right, "paired_scene_seed_cells": len(pairs),
            "both_success": counts[(True, True)], "left_only": counts[(True, False)],
            "right_only": counts[(False, True)], "both_failure": counts[(False, False)],
            "difference_percentage_points":
                100 * (counts[(True, False)] - counts[(False, True)]) / len(pairs),
        })
    return output


def load_verified():
    file_count = verify_hashes()
    episodes, calls = read_jsonl(DATA / "episodes.jsonl"), read_jsonl(DATA / "calls.jsonl")
    episodes, replays = validate(episodes, calls)
    historical = json.loads((DATA / "report.json").read_text())
    provenance = json.loads((ROOT / "docs" / "provenance.json").read_text())
    if fingerprint() != historical["provenance"]["protocol"]:
        raise ValueError("Code configuration does not match archived protocol")
    if fingerprint() != provenance["protocol_fingerprint"]:
        raise ValueError("Packaging provenance protocol mismatch")
    methods = aggregate(episodes, calls)
    for method in methods:
        archived = historical["methods"][method["method"]]
        for key in ("episodes", "successes", "response_records", "invalid_responses"):
            if method[key] != archived[key]:
                raise ValueError(f"Historical summary mismatch: {method['method']}:{key}")
    receipts = json.loads((DATA / "file_receipts.json").read_text())
    if len(receipts) != len(episodes) or {r["episode"] for r in receipts} != {r["id"] for r in episodes}:
        raise ValueError("Inherited receipt coverage mismatch")
    return episodes, calls, {
        "evidence_scope": "Five-method historical audit subset; not the 21-arm full protocol",
        "protocol_fingerprint": fingerprint(),
        "episodes": len(episodes), "response_records": len(calls),
        "vlm_responses_reparsed": replays, "parser_mismatches": 0,
        "packaged_hashes_verified": file_count,
        "simulation_rerun": False, "methods": methods,
        "by_task": grouped(episodes, "task"),
        "by_condition": grouped(episodes, "condition"),
        "by_seed": grouped(episodes, "model_seed"),
        "paired_descriptive_counts": paired_counts(episodes),
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, zh=False):
    title = "| 方法 | 成功回合 | 成功率 | 无效回复 / 回复数 |" if zh else (
        "| Method | Successful episodes | Success rate | Invalid / responses |"
    )
    lines = [title, "|---|---:|---:|---:|"]
    for row in rows:
        denominator = row["response_records"]
        response = f"{row['invalid_responses']}/{denominator}" if denominator else "—"
        lines.append(
            f"| `{row['method']}` | {row['successes']}/{row['episodes']} | "
            f"{100 * row['success_rate']:.2f}% | {response} |"
        )
    return "\n".join(lines)
