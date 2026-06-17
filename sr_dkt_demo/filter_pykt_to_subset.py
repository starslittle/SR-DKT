#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将官方 pyKT 的训练序列过滤到与 SR-DKT 相同的 10% 学生子集。

目的(方案②的对齐核心):
  方案②让标准 KT 基线在官方 pyKT 框架内运行、SR-DKT 在自研框架内运行。为
  保证两边可比,二者必须在【同一批学生】上训练。本脚本读取 subsample_mooc.py
  产出的 10% 学生 ID 列表,把 pyKT 的 train_valid_sequences.csv 中【fold==0
  (训练)】的行过滤到这 10% 学生;【fold==1(验证)】与 test 序列保持全量不动
  —— 与 SR-DKT "只采样 train、val/test 全量" 的策略完全一致。

输入:
  - subset_train_user_ids.json   (subsample_mooc.py 产出的 10% 学生 ID)
  - pyKT pykt_ready 目录          (train_valid_sequences.csv / test_*.csv / id_maps.json / data_config / meta)

输出:
  - 新的 pykt_ready 目录:train fold 已过滤到 10%,其余文件复用/重写 dpath。

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
import shutil
import sys
from pathlib import Path


# pyKT 序列 CSV 的字段可能非常长(maxlen=200 的逗号串),放开 csv 字段上限
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


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


def filter_train_valid(src: Path, dst: Path, subset_uids: set[str]) -> None:
    """fold==0(train)只保留子集学生;fold==1(val)全部保留。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept_train = kept_val = dropped_train = bad_rows = 0
    # errors="replace": 容忍源文件中极少数非 UTF-8 字节,不让整个读取崩溃;
    # 含替换符(�)的行视为坏行直接跳过,避免把垃圾写进 pyKT 输入。
    with src.open("r", encoding="utf-8", errors="replace", newline="") as fin, \
         dst.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        try:
            fold_i = header.index("fold")
            uid_i = header.index("uid")
        except ValueError:
            fold_i, uid_i = 0, 1  # 兜底:前两列即 fold,uid
        for row in reader:
            if any("�" in cell for cell in row):
                bad_rows += 1
                continue
            fold = row[fold_i]
            uid = row[uid_i]
            if fold == "0":
                if uid in subset_uids:
                    writer.writerow(row)
                    kept_train += 1
                else:
                    dropped_train += 1
            else:
                # fold==1 (验证) 全量保留
                writer.writerow(row)
                kept_val += 1
    print(
        f"[filter] train_valid: 训练序列保留 {kept_train:,} / 丢弃 {dropped_train:,}，"
        f"验证序列全量保留 {kept_val:,}，跳过坏行 {bad_rows:,}"
    )


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

    # 过滤 train_valid(只动 fold 0)
    filter_train_valid(
        args.pykt_dir / args.train_valid_file,
        args.out_dir / args.train_valid_file,
        subset_uids,
    )

    # test 序列全量复用
    print("[filter] test 与辅助文件全量复用:")
    for fname in (args.test_file, args.test_window_file,
                  "id_maps.json", "meta_sequences.json"):
        link_or_copy(args.pykt_dir / fname, args.out_dir / fname)

    # data_config 重写 dpath
    rewrite_data_config(args.pykt_dir, args.out_dir)

    print()
    print("=" * 60)
    print(f"完成。pyKT dataset 指向: {args.out_dir}")
    print("注意:验证(fold 1)与测试保持全量,与 SR-DKT 评估集一致。")
    print("=" * 60)


if __name__ == "__main__":
    main()
