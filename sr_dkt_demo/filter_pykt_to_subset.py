#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将官方 pyKT 的训练序列过滤到与 SR-DKT 相同的 10% 学生子集。

目的(方案②的对齐核心):
  方案②让标准 KT 基线在官方 pyKT 框架内运行、SR-DKT 在自研框架内运行。为
  保证两边可比,二者必须在【同一批学生】上训练。本脚本读取 subsample_mooc.py
  产出的 10% 学生 ID 列表,把 pyKT 的 train_valid_sequences.csv 中【fold==0
  (训练)】的行过滤到这 10% 学生;【fold==1(验证)】与 test 序列保持全量。
  同时对所有 pyKT sequence CSV 做严格整数校验/清洗,避免非整数 token 传给 pyKT。
  —— 与 SR-DKT "只采样 train、val/test 全量" 的策略完全一致。

输入:
  - subset_train_user_ids.json   (subsample_mooc.py 产出的 10% 学生 ID)
  - pyKT pykt_ready 目录          (train_valid_sequences.csv / test_*.csv / id_maps.json / data_config / meta)

输出:
  - 新的 pykt_ready 目录:train fold 已过滤到 10%,sequence CSV 已清洗,并重写 dpath。
  - filter_clean_report.json:各文件清洗/丢弃/替换 token 统计。

用法:
  python filter_pykt_to_subset.py \
      --ids data/mooc_10pct/subset_train_user_ids.json \
      --pykt-dir data/mooc_pykt/pykt_ready \
      --out-dir  data/mooc_pykt_10pct/pykt_ready

随后将 pyKT 的 dataset_name 指向该目录(或把新的 data_config 合并进 pyKT 的
configs/data_config.json),即可在 10% 训练子集 + 全量验证/测试上跑标准基线。

作者: SR-DKT 项目组
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path


# pyKT 序列 CSV 的字段可能非常长(maxlen=200 的逗号串),放开 csv 字段上限
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REQUIRED_COLUMNS = [
    "fold",
    "uid",
    "questions",
    "concepts",
    "responses",
    "timestamps",
    "usetimes",
    "selectmasks",
]
SEQ_COLUMNS = ["questions", "concepts", "responses", "timestamps", "usetimes", "selectmasks"]
INT_TOKEN_RE = re.compile(r"^-?\d+$")
BAD_EXAMPLE_LIMIT = 20


def is_valid_token_value(column: str, value: int) -> bool:
    """校验 pyKT 序列字段的整数取值范围。"""
    if column in {"questions", "concepts", "timestamps", "usetimes"}:
        return value == -1 or value >= 0
    if column == "responses":
        return value in {-1, 0, 1}
    if column == "selectmasks":
        return value in {-1, 0, 1}
    return True


def add_bad_example(
    stats: dict,
    line_no: int,
    column: str,
    token: str,
    reason: str,
) -> None:
    if len(stats["bad_examples"]) >= BAD_EXAMPLE_LIMIT:
        return
    stats["bad_examples"].append(
        {
            "line": line_no,
            "column": column,
            "token": token[:80],
            "reason": reason,
        }
    )


def invalid_token(
    policy: str,
    stats: dict,
    line_no: int,
    column: str,
    token: str,
    reason: str,
) -> str | None:
    add_bad_example(stats, line_no, column, token, reason)
    if policy == "error":
        raise ValueError(
            f"Invalid pyKT token at line {line_no}, column {column}: {token!r} ({reason})"
        )
    if policy == "drop":
        return None
    stats["replaced_tokens"] += 1
    return "-1"


def clean_sequence_row(
    row: dict[str, str],
    header: list[str],
    stats: dict,
    line_no: int,
    invalid_policy: str,
) -> dict[str, str] | None:
    """严格校验并清洗一行 pyKT sequence CSV。"""
    if None in row:
        add_bad_example(stats, line_no, "<row>", str(row[None][:3]), "extra_csv_fields")
        if invalid_policy == "error":
            raise ValueError(f"Malformed CSV row at line {line_no}: extra fields")
        stats["dropped_rows"] += 1
        return None

    for column in REQUIRED_COLUMNS:
        if column not in row or row[column] is None:
            add_bad_example(stats, line_no, column, "", "missing_required_column")
            if invalid_policy == "error":
                raise ValueError(f"Missing required column {column!r} at line {line_no}")
            stats["dropped_rows"] += 1
            return None

    cleaned = {column: row.get(column, "") for column in header}

    for column in ("fold", "uid"):
        token = cleaned[column].strip()
        if not INT_TOKEN_RE.match(token):
            add_bad_example(stats, line_no, column, token, "non_integer_id")
            if invalid_policy == "error":
                raise ValueError(f"Invalid {column} at line {line_no}: {token!r}")
            stats["dropped_rows"] += 1
            return None
        cleaned[column] = str(int(token))

    lengths: list[int] = []
    row_replaced = False
    for column in SEQ_COLUMNS:
        raw_tokens = cleaned[column].split(",") if cleaned[column] else []
        out_tokens: list[str] = []
        for raw_token in raw_tokens:
            token = raw_token.strip()
            reason = ""
            value: int | None = None
            if not INT_TOKEN_RE.match(token):
                reason = "non_integer_token"
            else:
                value = int(token)
                if not is_valid_token_value(column, value):
                    reason = "out_of_range"
            if reason:
                replacement = invalid_token(
                    invalid_policy, stats, line_no, column, token, reason
                )
                if replacement is None:
                    stats["dropped_rows"] += 1
                    return None
                out_tokens.append(replacement)
                row_replaced = True
            else:
                out_tokens.append(str(value))

        lengths.append(len(out_tokens))
        cleaned[column] = ",".join(out_tokens)

    if not lengths or len(set(lengths)) != 1 or lengths[0] == 0:
        add_bad_example(stats, line_no, "<row>", str(lengths), "inconsistent_sequence_lengths")
        if invalid_policy == "error":
            raise ValueError(f"Inconsistent sequence lengths at line {line_no}: {lengths}")
        stats["dropped_rows"] += 1
        return None

    if row_replaced:
        stats["cleaned_rows"] += 1
    return cleaned


def new_stats(filename: str) -> dict:
    return {
        "file": filename,
        "read_rows": 0,
        "written_rows": 0,
        "dropped_rows": 0,
        "cleaned_rows": 0,
        "replaced_tokens": 0,
        "kept_train_rows": 0,
        "dropped_train_rows": 0,
        "kept_non_train_rows": 0,
        "bad_examples": [],
    }


def print_stats(prefix: str, stats: dict) -> None:
    print(
        f"{prefix}: 读取 {stats['read_rows']:,}，写出 {stats['written_rows']:,}，"
        f"丢弃 {stats['dropped_rows']:,}，清洗行 {stats['cleaned_rows']:,}，"
        f"替换 token {stats['replaced_tokens']:,}"
    )
    if stats["bad_examples"]:
        print(f"{prefix}: bad examples {stats['bad_examples'][:5]}")


def load_subset_uids(ids_path: Path, id_maps_path: Path) -> set[str]:
    """读取 10% 学生原始 ID,经 user_id_to_idx 映射为 pyKT uid 字符串集合。"""
    payload = json.loads(ids_path.read_text(encoding="utf-8"))
    raw_ids = payload["user_ids"] if isinstance(payload, dict) else list(payload)

    id_maps = json.loads(id_maps_path.read_text(encoding="utf-8"))
    user_to_idx = id_maps["user_id_to_idx"]

    uids: set[str] = set()
    miss = 0
    for rid in raw_ids:
        rid = str(rid)
        # 直接匹配;否则尝试补 'U_' 前缀(SR-DKT 侧可能存的是裸 ID)
        if rid in user_to_idx:
            uids.add(str(user_to_idx[rid]))
        elif f"U_{rid}" in user_to_idx:
            uids.add(str(user_to_idx[f"U_{rid}"]))
        else:
            miss += 1

    print(f"[filter] 10% 学生 {len(raw_ids):,} 个 → 命中 pyKT uid {len(uids):,} 个，未命中 {miss:,}")
    if miss > len(raw_ids) * 0.05:
        print(
            f"[filter][WARN] 未命中比例偏高({miss/max(len(raw_ids),1):.1%})，"
            f"请确认 ID 列表与 id_maps 来自同一次预处理。",
            file=sys.stderr,
        )
    if not uids:
        print("[filter][ERROR] 无任何 uid 命中，终止。", file=sys.stderr)
        sys.exit(1)
    return uids


def clean_pykt_sequence_file(
    src: Path,
    dst: Path,
    invalid_policy: str,
    subset_uids: set[str] | None = None,
) -> dict:
    """清洗 pyKT sequence CSV;如传入 subset_uids,则仅过滤 fold==0 训练行。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    stats = new_stats(src.name)
    with src.open("r", encoding="utf-8-sig", errors="replace", newline="") as fin, \
         dst.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"{src} is empty or missing header")
        header = list(reader.fieldnames)
        missing = [column for column in REQUIRED_COLUMNS if column not in header]
        if missing:
            raise ValueError(f"{src} missing required columns: {missing}")

        writer = csv.DictWriter(fout, fieldnames=header)
        writer.writeheader()

        for line_no, row in enumerate(reader, start=2):
            stats["read_rows"] += 1
            cleaned = clean_sequence_row(row, header, stats, line_no, invalid_policy)
            if cleaned is None:
                continue

            fold = cleaned["fold"]
            uid = cleaned["uid"]
            if subset_uids is not None and fold == "0":
                if uid in subset_uids:
                    writer.writerow(cleaned)
                    stats["kept_train_rows"] += 1
                    stats["written_rows"] += 1
                else:
                    stats["dropped_train_rows"] += 1
            else:
                writer.writerow(cleaned)
                stats["kept_non_train_rows"] += 1
                stats["written_rows"] += 1
    return stats


def link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[filter][WARN] 缺少 {src}，跳过")
        return
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        import os
        os.link(src, dst)
        print(f"  硬链接 {src.name}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"  复制   {src.name}")


def rewrite_data_config(src_dir: Path, out_dir: Path) -> None:
    """复制 data_config 并把 dpath 指向新目录。"""
    cfgs = list(src_dir.glob("data_config*.json"))
    for cfg_path in cfgs:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for ds_name, ds_cfg in cfg.items():
            if isinstance(ds_cfg, dict) and "dpath" in ds_cfg:
                ds_cfg["dpath"] = str(out_dir.resolve())
        out_cfg = out_dir / cfg_path.name
        out_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  重写 {cfg_path.name}（dpath → {out_dir}）")


def main() -> None:
    parser = argparse.ArgumentParser(description="过滤 pyKT 训练 fold 到 10% 学生子集")
    parser.add_argument("--ids", type=Path, required=True,
                        help="subsample_mooc.py 产出的 subset_train_user_ids.json")
    parser.add_argument("--pykt-dir", type=Path, default=Path("data/mooc_pykt/pykt_ready"),
                        help="官方 pyKT pykt_ready 目录(全量)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/mooc_pykt_10pct/pykt_ready"),
                        help="输出目录")
    parser.add_argument("--train-valid-file", default="train_valid_sequences.csv")
    parser.add_argument("--test-file", default="test_sequences.csv")
    parser.add_argument("--test-window-file", default="test_window_sequences.csv")
    parser.add_argument(
        "--invalid-token-policy",
        choices=["replace", "drop", "error"],
        default="replace",
        help="pyKT 序列字段出现非整数/越界 token 时的处理策略；默认替换为 -1",
    )
    args = parser.parse_args()

    id_maps_path = args.pykt_dir / "id_maps.json"
    if not id_maps_path.exists():
        print(f"[ERROR] 缺少 {id_maps_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("pyKT 训练 fold → 10% 学生子集 过滤")
    print("=" * 60)

    subset_uids = load_subset_uids(args.ids, id_maps_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict] = {}

    # 过滤 train_valid(只动 fold 0),并清洗全部 pyKT 序列字段
    stats = clean_pykt_sequence_file(
        args.pykt_dir / args.train_valid_file,
        args.out_dir / args.train_valid_file,
        args.invalid_token_policy,
        subset_uids,
    )
    reports[args.train_valid_file] = stats
    print(
        f"[filter] train_valid: 训练序列保留 {stats['kept_train_rows']:,} / "
        f"丢弃 {stats['dropped_train_rows']:,}，验证/非训练序列保留 "
        f"{stats['kept_non_train_rows']:,}"
    )
    print_stats("[clean] train_valid", stats)

    # test 序列全量保留,但仍做严格整数清洗
    print("[filter] test 序列全量保留并清洗:")
    for fname in (args.test_file, args.test_window_file):
        src = args.pykt_dir / fname
        dst = args.out_dir / fname
        if not src.exists():
            print(f"[filter][WARN] 缺少 {src}，跳过")
            continue
        stats = clean_pykt_sequence_file(src, dst, args.invalid_token_policy)
        reports[fname] = stats
        print_stats(f"[clean] {fname}", stats)

    print("[filter] 辅助文件全量复用:")
    for fname in ("id_maps.json", "meta_sequences.json"):
        link_or_copy(args.pykt_dir / fname, args.out_dir / fname)

    # data_config 重写 dpath
    rewrite_data_config(args.pykt_dir, args.out_dir)

    report_path = args.out_dir / "filter_clean_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[filter] 清洗报告: {report_path}")

    print()
    print("=" * 60)
    print(f"完成。pyKT dataset 指向: {args.out_dir}")
    print("注意:验证(fold 1)与测试保持全量,与 SR-DKT 评估集一致。")
    print("=" * 60)


if __name__ == "__main__":
    main()
