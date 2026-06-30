#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export MOOCCubeX pyKT interaction CSVs to pyKT sequence CSVs.

Input comes from preprocess_mooc_pykt.py:
  data/mooc_pykt/train.csv
  data/mooc_pykt/val.csv
  data/mooc_pykt/test.csv

Outputs are directly compatible with pyKT's sequence dataloaders:
  data/mooc_pykt/pykt_ready/train_valid_sequences.csv
  data/mooc_pykt/pykt_ready/test_sequences.csv
  data/mooc_pykt/pykt_ready/test_window_sequences.csv
  data/mooc_pykt/pykt_ready/data_config_moocx.json
  data/mooc_pykt/pykt_ready/id_maps.json

Fold convention:
  train -> fold 0
  val   -> fold 1
  test  -> fold -1
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "mooc_pykt"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "pykt_ready"
FIELDNAMES = [
    "fold",
    "uid",
    "questions",
    "concepts",
    "responses",
    "timestamps",
    "usetimes",
    "selectmasks",
]


def get_or_add(mapping: dict[str, int], key: str) -> int:
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_duration_ms(value: str) -> int:
    if value is None or value == "":
        return 0
    try:
        duration = float(value)
    except ValueError:
        return 0
    return max(int(duration * 1000), 0)


def iter_interactions(path: Path, max_rows: int | None = None) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            yield row


def load_events(
    input_dir: Path,
    max_rows_per_split: int | None,
) -> tuple[
    dict[str, list[tuple[int, int, int, int, int, int, int]]],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    """Load split CSVs into grouped user sequences.

    Event tuple:
      (fold, order_id, qid, cid, response, timestamp_ms, duration_ms)
    """
    split_files = {
        "train": (0, input_dir / "train.csv"),
        "val": (1, input_dir / "val.csv"),
        "test": (-1, input_dir / "test.csv"),
    }
    missing = [str(path) for _, path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pyKT interaction CSVs: " + ", ".join(missing))

    q_map: dict[str, int] = {}
    c_map: dict[str, int] = {}
    uid_map: dict[str, int] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    events_by_uid: dict[str, list[tuple[int, int, int, int, int, int, int]]] = defaultdict(list)

    for split_name, (fold, path) in split_files.items():
        for row in iter_interactions(path, max_rows_per_split):
            user_id = row.get("user_id", "")
            question_id = row.get("question_id", "")
            concept_id = row.get("concept_id", "")
            if not user_id or not question_id or not concept_id:
                continue
            uid = get_or_add(uid_map, user_id)
            qid = get_or_add(q_map, question_id)
            cid = get_or_add(c_map, concept_id)
            response = 1 if parse_int(row.get("response", "0")) else 0
            order_id = parse_int(row.get("order_id", "0"))
            timestamp_ms = parse_int(row.get("timestamp", "0")) * 1000
            duration_ms = parse_duration_ms(row.get("duration", ""))
            events_by_uid[str(uid)].append(
                (fold, order_id, qid, cid, response, timestamp_ms, duration_ms)
            )
            split_counts[split_name] += 1

    return events_by_uid, q_map, c_map, uid_map, split_counts


def pad(values: list[int], maxlen: int, pad_val: int = -1) -> list[int]:
    return values + [pad_val] * (maxlen - len(values))


def write_sequence_row(
    writer: csv.DictWriter,
    fold: int,
    uid: str,
    events: list[tuple[int, int, int, int, int, int, int]],
    maxlen: int,
) -> None:
    questions = pad([event[2] for event in events], maxlen)
    concepts = pad([event[3] for event in events], maxlen)
    responses = pad([event[4] for event in events], maxlen)
    timestamps = pad([event[5] for event in events], maxlen)
    usetimes = pad([event[6] for event in events], maxlen)
    selectmasks = [1] * len(events) + [-1] * (maxlen - len(events))
    writer.writerow(
        {
            "fold": fold,
            "uid": uid,
            "questions": ",".join(map(str, questions)),
            "concepts": ",".join(map(str, concepts)),
            "responses": ",".join(map(str, responses)),
            "timestamps": ",".join(map(str, timestamps)),
            "usetimes": ",".join(map(str, usetimes)),
            "selectmasks": ",".join(map(str, selectmasks)),
        }
    )


def write_sequences(
    events_by_uid: dict[str, list[tuple[int, int, int, int, int, int, int]]],
    output_dir: Path,
    maxlen: int,
    min_seq_len: int,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_valid_path = output_dir / "train_valid_sequences.csv"
    test_path = output_dir / "test_sequences.csv"
    counts = {
        "train_valid_sequences": 0,
        "test_sequences": 0,
        "skipped_short_sequences": 0,
    }

    with train_valid_path.open("w", encoding="utf-8", newline="") as f_train, test_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_test:
        train_writer = csv.DictWriter(f_train, fieldnames=FIELDNAMES)
        test_writer = csv.DictWriter(f_test, fieldnames=FIELDNAMES)
        train_writer.writeheader()
        test_writer.writeheader()

        for uid, events in events_by_uid.items():
            by_fold: dict[int, list[tuple[int, int, int, int, int, int, int]]] = defaultdict(list)
            for event in events:
                by_fold[event[0]].append(event)

            for fold in (0, 1, -1):
                fold_events = by_fold.get(fold)
                if not fold_events:
                    continue
                # KT sequence order must be chronological. order_id is only a
                # stable tie-breaker when two submissions have the same timestamp.
                fold_events.sort(key=lambda event: (event[5], event[1]))
                for start in range(0, len(fold_events), maxlen):
                    chunk = fold_events[start : start + maxlen]
                    if len(chunk) < min_seq_len:
                        counts["skipped_short_sequences"] += 1
                        continue
                    if fold == -1:
                        write_sequence_row(test_writer, fold, uid, chunk, maxlen)
                        counts["test_sequences"] += 1
                    else:
                        write_sequence_row(train_writer, fold, uid, chunk, maxlen)
                        counts["train_valid_sequences"] += 1

    # pyKT configs usually expect a window test file. For this export path we
    # use the same fixed-window test sequences as the default window file.
    test_window_path = output_dir / "test_window_sequences.csv"
    shutil.copyfile(test_path, test_window_path)
    return counts


def write_metadata(
    output_dir: Path,
    q_map: dict[str, int],
    c_map: dict[str, int],
    uid_map: dict[str, int],
    split_counts: dict[str, int],
    sequence_counts: dict[str, int],
    maxlen: int,
    min_seq_len: int,
) -> None:
    id_maps = {
        "question_id_to_idx": q_map,
        "concept_id_to_idx": c_map,
        "user_id_to_idx": uid_map,
    }
    (output_dir / "id_maps.json").write_text(
        json.dumps(id_maps, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    data_config = {
        "moocx": {
            "dpath": str(output_dir),
            "num_q": len(q_map),
            "num_c": len(c_map),
            "input_type": ["questions", "concepts"],
            "max_concepts": 1,
            "min_seq_len": min_seq_len,
            "maxlen": maxlen,
            "emb_path": "",
            "train_valid_file": "train_valid_sequences.csv",
            "folds": [0, 1],
            "test_file": "test_sequences.csv",
            "test_window_file": "test_window_sequences.csv",
        }
    }
    (output_dir / "data_config_moocx.json").write_text(
        json.dumps(data_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "format": "pyKT sequence CSV",
        "folds": {"train": 0, "val": 1, "test": -1},
        "maxlen": maxlen,
        "min_seq_len": min_seq_len,
        "num_q": len(q_map),
        "num_c": len(c_map),
        "num_users": len(uid_map),
        "split_interaction_counts": split_counts,
        "sequence_counts": sequence_counts,
        "pykt_notes": [
            "Copy or merge data_config_moocx.json into pyKT configs/data_config.json.",
            "Run pyKT with dataset_name=moocx and fold=1 so fold 0 is train and fold 1 is validation.",
            "DKT-Forget requires timestamps; this exporter writes timestamps in milliseconds.",
            "Within each user and fold, events are sorted by timestamp_ms then order_id before windowing.",
        ],
    }
    (output_dir / "meta_sequences.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MOOCCubeX pyKT sequence files")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--min-seq-len", type=int, default=3)
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Debug/smoke-test limit. Omit for full export.",
    )
    args = parser.parse_args()

    events_by_uid, q_map, c_map, uid_map, split_counts = load_events(
        args.input_dir, args.max_rows_per_split
    )
    sequence_counts = write_sequences(
        events_by_uid,
        args.output_dir,
        args.maxlen,
        args.min_seq_len,
    )
    write_metadata(
        args.output_dir,
        q_map,
        c_map,
        uid_map,
        split_counts,
        sequence_counts,
        args.maxlen,
        args.min_seq_len,
    )
    print(f"[pykt-seq] Done. Output: {args.output_dir}")
    print(f"[pykt-seq] num_q={len(q_map):,}, num_c={len(c_map):,}, users={len(uid_map):,}")
    print(f"[pykt-seq] sequences={sequence_counts}")


if __name__ == "__main__":
    main()
