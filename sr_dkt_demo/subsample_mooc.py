#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOOCCubeX train 子集分层采样脚本。

用途:
  从【全量】预处理好的 train.pkl 中,按序列长度分层、固定种子,采样出一个
  指定比例(默认 10%)的子集。val/test 保持全量不动,保证所有实验在同一个
  评估集上可比。

设计原则(与论文实验设置一致):
  1. 只采样 train,不动 val/test —— 所有模型在同一 val/test 上评估。
  2. 按序列长度分位数分层 —— 子集长度分布 ≈ 全量,避免采样偏差。
  3. 固定随机种子 —— 可复现。
  4. 不覆盖全量 train.pkl —— 输出到独立目录。

为什么从全量(100%)直接采,而不是嵌套在 20% 里:
  随机样本的随机子样本仍是无偏样本,代表性相同;直接从全量采更简单,
  且不依赖此前手动切分的 20% 学生名单。仅当需要做"10% vs 20% 数据量
  消融"的配对对比时,嵌套才有降噪意义 —— 本实验不做该对比。

用法(在云端全量数据所在机器上运行):
  python subsample_mooc.py \
      --src-dir data/mooc \
      --out-dir data/mooc_10pct \
      --ratio 0.1 --seed 42

  随后训练时把数据目录指向 data/mooc_10pct 即可(train 为 10% 子集,
  val/test/meta/encoder 通过硬链接复用全量文件,不额外占盘)。

作者: SR-DKT 项目组
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def stratified_sample_students(
    sequences: dict,
    ratio: float,
    seed: int,
    n_bins: int = 10,
) -> list:
    """按序列长度分位数分层,从每层采样 ratio 比例的学生。

    Args:
        sequences: dict[student_id, seq]
        ratio: 采样比例 (0, 1)
        seed: 随机种子
        n_bins: 分层数(按序列长度分位数)

    Returns:
        选中的 student_id 列表
    """
    rng = np.random.default_rng(seed)

    student_ids = np.array(list(sequences.keys()), dtype=object)
    lengths = np.array([len(sequences[sid]) for sid in student_ids])

    # 按序列长度的分位数划分层边界(去重,避免重复分位点导致空桶)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(lengths, quantiles))
    # digitize 到 [1, len(edges)-1],再压回 0-based 桶号
    bin_idx = np.clip(np.digitize(lengths, edges[1:-1], right=False), 0, len(edges) - 2)

    selected: list = []
    print(f"[subsample] 分层采样: ratio={ratio}, seed={seed}, 实际层数={len(edges) - 1}")
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        bin_students = student_ids[mask]
        n_bin = len(bin_students)
        if n_bin == 0:
            continue
        n_take = max(1, int(round(n_bin * ratio)))
        chosen = rng.choice(bin_students, size=n_take, replace=False)
        selected.extend(chosen.tolist())
        lo = edges[b]
        hi = edges[b + 1]
        print(
            f"  层{b:>2d} [len {lo:>6.0f}~{hi:>6.0f}]: "
            f"{n_bin:>8,} 人 → 采 {n_take:>7,} 人"
        )

    return selected


def link_or_copy(src: Path, dst: Path) -> None:
    """优先硬链接(零拷贝),失败则复制。val/test 全量复用,避免额外占盘。"""
    if dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        print(f"  硬链接 {src.name} → {dst}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"  复制   {src.name} → {dst}")


def describe(name: str, sequences: dict) -> None:
    n = len(sequences)
    if n == 0:
        print(f"  {name}: 0 学生")
        return
    lengths = np.array([len(s) for s in sequences.values()])
    print(
        f"  {name}: {n:,} 学生 | 交互 {int(lengths.sum()):,} | "
        f"均长 {lengths.mean():.1f} | 中位 {np.median(lengths):.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MOOCCubeX train 子集分层采样")
    parser.add_argument("--src-dir", type=Path, default=Path("data/mooc"),
                        help="全量数据目录(含 train/val/test.pkl + meta + encoder)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/mooc_10pct"),
                        help="输出目录")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="train 采样比例(默认 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子(默认 42)")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="分层数,按序列长度分位数(默认 10)")
    parser.add_argument("--train-name", default="train.pkl")
    parser.add_argument("--val-name", default="val.pkl")
    parser.add_argument("--test-name", default="test.pkl")
    args = parser.parse_args()

    if not (0.0 < args.ratio < 1.0):
        print(f"[ERROR] ratio 必须在 (0,1) 之间,当前 {args.ratio}", file=sys.stderr)
        sys.exit(1)

    src_train = args.src_dir / args.train_name
    if not src_train.exists():
        print(f"[ERROR] 找不到全量 train: {src_train}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("MOOCCubeX train 分层子集采样")
    print("=" * 60)
    print(f"源目录:   {args.src_dir}")
    print(f"输出目录: {args.out_dir}")
    print()

    print(f"[subsample] 加载全量 train: {src_train}")
    train_full = load_pickle(src_train)
    describe("全量 train", train_full)

    selected = stratified_sample_students(
        train_full, ratio=args.ratio, seed=args.seed, n_bins=args.n_bins
    )
    train_subset = {sid: train_full[sid] for sid in selected}

    print()
    describe("子集 train", train_subset)
    actual_ratio = len(train_subset) / max(len(train_full), 1)
    print(f"  实际采样比例: {actual_ratio:.4f}")

    # 写出子集 train
    out_train = args.out_dir / args.train_name
    save_pickle(train_subset, out_train)
    print(f"[subsample] 已写出子集 train: {out_train}")

    # 保存选中的学生 ID 列表(JSON),供 pyKT 训练 fold 过滤对齐使用,并保证可复现
    import json
    ids_path = args.out_dir / "subset_train_user_ids.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(
        json.dumps(
            {"ratio": args.ratio, "seed": args.seed, "n_bins": args.n_bins,
             "n_students": len(selected), "user_ids": [str(s) for s in selected]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[subsample] 已保存 10% 学生 ID 列表: {ids_path}")

    # val/test 全量复用(硬链接优先)
    print("[subsample] 复用全量 val/test 及 meta/encoder:")
    for fname in (args.val_name, args.test_name):
        src = args.src_dir / fname
        if src.exists():
            link_or_copy(src, args.out_dir / fname)
        else:
            print(f"  [WARN] 缺少 {src},跳过")

    # meta 与 encoder 一并复用,保证 KC 编码与全量一致
    for fname in ("meta_mooc.json", "skill_encoder_mooc.pkl"):
        src = args.src_dir / fname
        if src.exists():
            link_or_copy(src, args.out_dir / fname)

    print()
    print("=" * 60)
    print(f"完成。训练时把数据目录指向: {args.out_dir}")
    print("注意:所有 baseline 也必须在此 10% 子集上重跑,才能与 SR-DKT 公平对比。")
    print("=" * 60)


if __name__ == "__main__":
    main()
