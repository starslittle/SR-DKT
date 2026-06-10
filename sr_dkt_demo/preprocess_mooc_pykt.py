#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOOCCubeX → pyKT baseline preprocessing.

This script is intentionally separate from preprocess_mooc.py:
  - preprocess_mooc.py keeps the SR-DKT 7-tuple format unchanged.
  - preprocess_mooc_pykt.py exports standard KT fields for pyKT baselines.

Default output:
  data/mooc_pykt/train.csv
  data/mooc_pykt/val.csv
  data/mooc_pykt/test.csv
  data/mooc_pykt/interactions.csv
  data/mooc_pykt/meta_pykt.json

CSV fields:
  split,user_id,order_id,question_id,concept_id,response,timestamp,duration

The default split source is the existing SR-DKT pkl files. That keeps pyKT
baselines on the same train/val/test users as SR-DKT, without changing SR-DKT
data or training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MOOC_DIR = ROOT_DIR / "data" / "mooc"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "mooc_pykt"
SEED = 42


def parse_datetime_to_epoch(s: str) -> float:
    """Fast parser for 'YYYY-MM-DD HH:MM:SS'. Returns NaN on failure."""
    try:
        year = int(s[0:4])
        month = int(s[5:7])
        day = int(s[8:10])
        hour = int(s[11:13])
        minute = int(s[14:16])
        second = int(s[17:19])
        days_in_month = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
        total_days = (year - 1970) * 365 + (year - 1969) // 4
        total_days += days_in_month[month - 1] + day - 1
        if month > 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            total_days += 1
        return float(total_days * 86400 + hour * 3600 + minute * 60 + second)
    except (ValueError, IndexError):
        return float("nan")


def parse_problem_id(problem_id: str | None) -> int | None:
    if not problem_id:
        return None
    try:
        return int(problem_id[3:]) if problem_id.startswith("Pm_") else int(problem_id)
    except ValueError:
        return None


def extract_duration_seconds(obj: dict) -> str:
    """Best-effort answer duration extraction. Empty string means missing."""
    for key in (
        "duration",
        "answer_time",
        "time_cost",
        "timecost",
        "cost_time",
        "elapsed_time",
        "used_time",
    ):
        value = obj.get(key)
        if value is None or value == "":
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration >= 0:
            return str(duration)
    return ""


def load_problem_to_exercise(mooc_dir: Path) -> dict[int, str]:
    """Load problem_id(int) -> exercise_id(str)."""
    relations_dir = mooc_dir / "relations"
    entities_dir = mooc_dir / "entities"
    p2e: dict[int, str] = {}

    txt_path = relations_dir / "exercise-problem.txt"
    if txt_path.exists():
        with open(txt_path, "rb") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                sep = line.find(b"\t")
                if sep < 0:
                    continue
                ex_id = line[:sep].decode("utf-8", errors="ignore")
                pm_id = line[sep + 1 :].decode("utf-8", errors="ignore")
                pid = parse_problem_id(pm_id)
                if pid is not None and ex_id:
                    p2e[pid] = ex_id

    json_path = entities_dir / "problem.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = parse_problem_id(obj.get("id") or obj.get("problem_id"))
                exercise_id = obj.get("exercise_id")
                if pid is not None and exercise_id and pid not in p2e:
                    p2e[pid] = str(exercise_id)

    return p2e


def load_pickle_keys(path: Path) -> set[str]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return set(data.keys())


def load_split_map(mooc_dir: Path, split_source: str) -> dict[str, str] | None:
    """
    Return {user_id: split}, or None for hash split.

    sr-dkt source loads existing train/val/test pkl keys only. This can be
    memory-heavy but keeps pyKT and SR-DKT evaluation populations aligned.
    """
    if split_source == "hash":
        return None

    split_json = mooc_dir / "split_users.json"
    if split_json.exists():
        obj = json.loads(split_json.read_text(encoding="utf-8"))
        split_map: dict[str, str] = {}
        for split in ("train", "val", "test"):
            for user_id in obj.get(split, []):
                split_map[str(user_id)] = split
        if split_map:
            return split_map

    split_files = {
        "train": mooc_dir / "train.pkl",
        "val": mooc_dir / "val.pkl",
        "test": mooc_dir / "test.pkl",
    }
    if all(path.exists() for path in split_files.values()):
        split_map = {}
        for split, path in split_files.items():
            print(f"[pykt-preprocess] Loading SR-DKT split keys: {path}")
            for user_id in load_pickle_keys(path):
                split_map[str(user_id)] = split
        return split_map

    if split_source == "sr-dkt":
        raise FileNotFoundError(
            "split_source=sr-dkt requires split_users.json or train/val/test.pkl"
        )
    return None


def hash_split(user_id: str, train_ratio: float, val_ratio: float) -> str:
    rng = random.Random(f"{SEED}:{user_id}")
    value = rng.random()
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def first_pass_counts(
    path: Path,
    p2e: dict[int, str],
    split_map: dict[str, str] | None,
    chunk_size: int,
) -> tuple[dict[str, int], dict[str, int], int, int]:
    """Count exercise and user interactions for filtering/top-K."""
    exercise_counts: dict[str, int] = {}
    user_counts: dict[str, int] = {}
    total = 0
    kept = 0
    t0 = time.time()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if total % chunk_size == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    "[pykt-preprocess] Pass 1 {:,} lines ({:,.0f}/s), kept {:,}".format(
                        total, total / elapsed, kept
                    )
                )
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = obj.get("user_id")
            if not user_id:
                continue
            user_id = str(user_id)
            if split_map is not None and user_id not in split_map:
                continue
            pid = parse_problem_id(obj.get("problem_id"))
            if pid is None:
                continue
            exercise_id = p2e.get(pid)
            if not exercise_id:
                continue
            exercise_counts[exercise_id] = exercise_counts.get(exercise_id, 0) + 1
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
            kept += 1

    return exercise_counts, user_counts, total, kept


def open_writers(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "user_id",
        "order_id",
        "question_id",
        "concept_id",
        "response",
        "timestamp",
        "duration",
    ]
    files = {}
    writers = {}
    for split in ("train", "val", "test", "interactions"):
        f = open(output_dir / f"{split}.csv", "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        files[split] = f
        writers[split] = writer
    return files, writers


def second_pass_export(
    path: Path,
    output_dir: Path,
    p2e: dict[int, str],
    split_map: dict[str, str] | None,
    valid_users: set[str],
    top_concepts: set[str] | None,
    train_ratio: float,
    val_ratio: float,
    chunk_size: int,
) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0, "skipped": 0}
    order_by_user: dict[str, int] = {}
    files, writers = open_writers(output_dir)
    total = 0
    t0 = time.time()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                if total % chunk_size == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    exported = counts["train"] + counts["val"] + counts["test"]
                    print(
                        "[pykt-preprocess] Pass 2 {:,} lines ({:,.0f}/s), exported {:,}".format(
                            total, total / elapsed, exported
                        )
                    )
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    counts["skipped"] += 1
                    continue

                user_id_raw = obj.get("user_id")
                if not user_id_raw:
                    counts["skipped"] += 1
                    continue
                user_id = str(user_id_raw)
                if user_id not in valid_users:
                    counts["skipped"] += 1
                    continue

                split = (
                    split_map[user_id]
                    if split_map is not None
                    else hash_split(user_id, train_ratio, val_ratio)
                )
                pid_str = obj.get("problem_id")
                pid = parse_problem_id(pid_str)
                if pid is None or not pid_str:
                    counts["skipped"] += 1
                    continue
                concept_id = p2e.get(pid)
                if not concept_id:
                    counts["skipped"] += 1
                    continue
                if top_concepts is not None and concept_id not in top_concepts:
                    concept_id = "OTHER"

                timestamp = parse_datetime_to_epoch(obj.get("submit_time", ""))
                if timestamp != timestamp:
                    counts["skipped"] += 1
                    continue
                response = 1 if obj.get("is_correct", 0) else 0
                order_id = order_by_user.get(user_id, 0)
                order_by_user[user_id] = order_id + 1
                row = {
                    "split": split,
                    "user_id": user_id,
                    "order_id": order_id,
                    "question_id": pid_str,
                    "concept_id": concept_id,
                    "response": response,
                    "timestamp": int(timestamp),
                    "duration": extract_duration_seconds(obj),
                }
                writers[split].writerow(row)
                writers["interactions"].writerow(row)
                counts[split] += 1
    finally:
        for f in files.values():
            f.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="MOOCCubeX -> pyKT preprocessing")
    parser.add_argument("--mooc-dir", type=Path, default=DEFAULT_MOOC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-source",
        choices=["auto", "sr-dkt", "hash"],
        default="auto",
        help="Use existing SR-DKT split when available, otherwise hash split.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-interactions", type=int, default=3)
    parser.add_argument("--max-concepts", type=int, default=30000)
    parser.add_argument("--chunk-size", type=int, default=5_000_000)
    args = parser.parse_args()

    problem_path = args.mooc_dir / "relations" / "user-problem.json"
    if not problem_path.exists():
        print(f"[ERROR] Missing {problem_path}", file=sys.stderr)
        sys.exit(1)

    print("[pykt-preprocess] Loading problem -> exercise mapping...")
    p2e = load_problem_to_exercise(args.mooc_dir)
    print(f"[pykt-preprocess] Loaded {len(p2e):,} mappings")

    split_source = args.split_source
    if split_source == "auto":
        split_source = "sr-dkt"
        try:
            split_map = load_split_map(args.mooc_dir, split_source)
        except FileNotFoundError:
            print("[pykt-preprocess] No SR-DKT split found, falling back to hash split")
            split_map = None
    else:
        split_map = load_split_map(args.mooc_dir, split_source)

    print("[pykt-preprocess] Pass 1: counting users and concepts...")
    exercise_counts, user_counts, total_lines, kept_lines = first_pass_counts(
        problem_path, p2e, split_map, args.chunk_size
    )
    valid_users = {
        user_id
        for user_id, count in user_counts.items()
        if count >= args.min_interactions
    }
    if split_map is not None:
        valid_users &= set(split_map)

    top_concepts = None
    if args.max_concepts and len(exercise_counts) > args.max_concepts:
        ranked = sorted(exercise_counts.items(), key=lambda item: item[1], reverse=True)
        top_concepts = {concept for concept, _ in ranked[: args.max_concepts]}
        print(
            f"[pykt-preprocess] Concepts: {len(exercise_counts):,}; "
            f"keeping top {args.max_concepts:,} + OTHER"
        )
    else:
        print(f"[pykt-preprocess] Concepts: {len(exercise_counts):,}; no top-K filter")

    print(
        "[pykt-preprocess] Valid users after min_interactions={}: {:,}".format(
            args.min_interactions, len(valid_users)
        )
    )
    print("[pykt-preprocess] Pass 2: exporting CSV files...")
    counts = second_pass_export(
        problem_path,
        args.output_dir,
        p2e,
        split_map,
        valid_users,
        top_concepts,
        args.train_ratio,
        args.val_ratio,
        args.chunk_size,
    )

    meta = {
        "dataset": "MOOCCubeX",
        "format": "pyKT interaction CSV",
        "source_file": str(problem_path),
        "split_source": "sr-dkt" if split_map is not None else "hash",
        "total_lines_seen": total_lines,
        "valid_problem_interactions_pass1": kept_lines,
        "valid_users": len(valid_users),
        "num_concepts_raw": len(exercise_counts),
        "max_concepts": args.max_concepts,
        "export_counts": counts,
        "columns": [
            "split",
            "user_id",
            "order_id",
            "question_id",
            "concept_id",
            "response",
            "timestamp",
            "duration",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "meta_pykt.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[pykt-preprocess] Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
