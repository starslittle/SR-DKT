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
import time
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


def write_event_chunks(
    writer: csv.DictWriter,
    fold: int,
    uid: str,
    events: list[tuple[int, int, int, int, int, int, int]],
    maxlen: int,
    min_seq_len: int,
) -> tuple[int, int]:
    """Sort one user's events and write fixed-length pyKT sequence rows."""
    if not events:
        return 0, 0
    events.sort(key=lambda event: (event[5], event[1]))
    written = 0
    skipped = 0
    for start in range(0, len(events), maxlen):
        chunk = events[start : start + maxlen]
        if len(chunk) < min_seq_len:
            skipped += 1
            continue
        write_sequence_row(writer, fold, uid, chunk, maxlen)
        written += 1
    return written, skipped


def process_split_stream(
    split_name: str,
    path: Path,
    fold: int,
    writer: csv.DictWriter,
    output_count_key: str,
    q_map: dict[str, int],
    c_map: dict[str, int],
    uid_map: dict[str, int],
    sequence_counts: dict[str, int],
    maxlen: int,
    min_seq_len: int,
    progress_rows: int,
    max_rows: int | None,
) -> tuple[int, int]:
    """Stream one interaction CSV and flush sequences per contiguous user block."""
    current_user_id: str | None = None
    current_uid = ""
    events: list[tuple[int, int, int, int, int, int, int]] = []
    closed_users: set[str] = set()
    reopened_users = 0
    split_count = 0
    skipped_rows = 0
    t0 = time.time()

    def flush_current() -> None:
        nonlocal events
        if current_user_id is None:
            return
        written, skipped = write_event_chunks(
            writer, fold, current_uid, events, maxlen, min_seq_len
        )
        sequence_counts[output_count_key] += written
        sequence_counts["skipped_short_sequences"] += skipped
        events = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_idx, row in enumerate(reader, start=1):
            if max_rows is not None and raw_idx > max_rows:
                break
            if progress_rows > 0 and raw_idx % progress_rows == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    "[pykt-seq] {} {:,} rows ({:,.0f}/s), sequences {:,}".format(
                        split_name,
                        raw_idx,
                        raw_idx / elapsed,
                        sequence_counts[output_count_key],
                    ),
                    flush=True,
                )

            user_id = row.get("user_id", "")
            question_id = row.get("question_id", "")
            concept_id = row.get("concept_id", "")
            if not user_id or not question_id or not concept_id:
                skipped_rows += 1
                continue

            if current_user_id is None:
                current_user_id = user_id
                current_uid = str(get_or_add(uid_map, user_id))
            elif user_id != current_user_id:
                flush_current()
                closed_users.add(current_user_id)
                if user_id in closed_users:
                    reopened_users += 1
                current_user_id = user_id
                current_uid = str(get_or_add(uid_map, user_id))

            qid = get_or_add(q_map, question_id)
            cid = get_or_add(c_map, concept_id)
            response = 1 if parse_int(row.get("response", "0")) else 0
            order_id = parse_int(row.get("order_id", "0"))
            timestamp_ms = parse_int(row.get("timestamp", "0")) * 1000
            duration_ms = parse_duration_ms(row.get("duration", ""))
            events.append((fold, order_id, qid, cid, response, timestamp_ms, duration_ms))
            split_count += 1

    flush_current()
    elapsed = max(time.time() - t0, 1e-6)
    print(
        "[pykt-seq] {} done: {:,} rows, {:,} sequences, skipped_rows {:,}, reopened_users {}, {:.1f}s".format(
            split_name,
            split_count,
            sequence_counts[output_count_key],
            skipped_rows,
            reopened_users,
            elapsed,
        ),
        flush=True,
    )
    return split_count, reopened_users


def write_sequences_streaming(
    input_dir: Path,
    output_dir: Path,
    maxlen: int,
    min_seq_len: int,
    max_rows_per_split: int | None,
    progress_rows: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    split_files = {
        "train": (0, input_dir / "train.csv"),
        "val": (1, input_dir / "val.csv"),
        "test": (-1, input_dir / "test.csv"),
    }
    missing = [str(path) for _, path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pyKT interaction CSVs: " + ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    q_map: dict[str, int] = {}
    c_map: dict[str, int] = {}
    uid_map: dict[str, int] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    sequence_counts = {
        "train_valid_sequences": 0,
        "test_sequences": 0,
        "skipped_short_sequences": 0,
        "reopened_users": 0,
    }

    train_valid_path = output_dir / "train_valid_sequences.csv"
    test_path = output_dir / "test_sequences.csv"
    with train_valid_path.open("w", encoding="utf-8", newline="") as f_train, test_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_test:
        train_writer = csv.DictWriter(f_train, fieldnames=FIELDNAMES)
        test_writer = csv.DictWriter(f_test, fieldnames=FIELDNAMES)
        train_writer.writeheader()
        test_writer.writeheader()

        for split_name in ("train", "val", "test"):
            fold, path = split_files[split_name]
            writer = test_writer if split_name == "test" else train_writer
            output_count_key = (
                "test_sequences" if split_name == "test" else "train_valid_sequences"
            )
            count, reopened = process_split_stream(
                split_name,
                path,
                fold,
                writer,
                output_count_key,
                q_map,
                c_map,
                uid_map,
                sequence_counts,
                maxlen,
                min_seq_len,
                progress_rows,
                max_rows_per_split,
            )
            split_counts[split_name] = count
            sequence_counts["reopened_users"] += reopened

    shutil.copyfile(test_path, output_dir / "test_window_sequences.csv")
    return q_map, c_map, uid_map, split_counts, sequence_counts


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
    parser.add_argument(
        "--progress-rows",
        type=int,
        default=5_000_000,
        help="Print progress every N input rows per split.",
    )
    args = parser.parse_args()

    q_map, c_map, uid_map, split_counts, sequence_counts = write_sequences_streaming(
        args.input_dir,
        args.output_dir,
        args.maxlen,
        args.min_seq_len,
        args.max_rows_per_split,
        args.progress_rows,
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
