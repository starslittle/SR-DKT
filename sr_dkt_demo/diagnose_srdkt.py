#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SR-DKT 训练后诊断：检查架构修订是否真正生效。

修订后的关键预期（与旧架构对照）：
  1. λ_s、λ_r 收敛到稳定正值（旧架构里二者持续发散不收敛）。
  2. λ_r > λ_s 成立，且是【学出来的】（软约束鼓励，非写死恒等式）。
  3. constraint_loss 不再恒为 0（λ 独立后该惩罚项才有意义）。
  4. Val AUC 轨迹合理、best epoch 不在最后一轮（否则可能欠拟合/早停过早）。

用法：
  # 读取训练好的 checkpoint（train.py 保存的 best_model_mooc.pt）
  python diagnose_srdkt.py --ckpt data/mooc_10pct/best_model_mooc.pt

  # 同时分析训练曲线日志（含 Val AUC 轨迹）
  python diagnose_srdkt.py --ckpt data/mooc_10pct/best_model_mooc.pt \
      --logs data/training_logs.json

作者: SR-DKT 项目组
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import SRDKT


def load_model_from_ckpt(ckpt_path: Path) -> SRDKT:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        num_kc = int(ckpt.get("num_kc", 0)) or _infer_num_kc(ckpt["model_state_dict"])
        cfg = ckpt.get("config", {}) or {}
        hidden = int(cfg.get("hidden_size", 128))
        layers = int(cfg.get("num_layers", 1))
        model = SRDKT(num_kc=num_kc, hidden_size=hidden, num_layers=layers)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError("checkpoint 格式不识别：缺少 model_state_dict")
    model.eval()
    return model


def _infer_num_kc(state: dict) -> int:
    """从 kc_embed_s 权重形状推断 num_kc（Embedding(num_kc+1, H)）。"""
    for k, v in state.items():
        if k.endswith("kc_embed_s.weight"):
            return int(v.shape[0]) - 1
    raise ValueError("无法从 state_dict 推断 num_kc")


def diagnose_lambda(model: SRDKT) -> None:
    lam_s = float(model.lambda_s.item())
    lam_r = float(model.lambda_r.item())
    gap = lam_r - lam_s
    constraint = float(model.get_lambda_constraint_loss().item())

    print("=" * 56)
    print("差异化遗忘参数诊断")
    print("=" * 56)
    print(f"  λ_s (Storage, 慢) = {lam_s:.4f}")
    print(f"  λ_r (Retrieval, 快) = {lam_r:.4f}")
    print(f"  lambda_r - lambda_s = {gap:+.4f}")
    print(f"  constraint_loss     = {constraint:.6f}")
    print("-" * 56)

    ok_order = lam_r > lam_s
    print(f"  [{'✓' if ok_order else '✗'}] λ_r > λ_s {'成立' if ok_order else '不成立（差异化遗忘方向错误！）'}")
    # 软约束下 constraint_loss 应趋近 0（被满足）但参数是学出来的；
    # 若 λ_r 远大于 λ_s，constraint 自然为 0，这是【期望】结果。
    if ok_order and constraint == 0.0:
        print("  [✓] 软约束已被满足（constraint=0 且 λ_r>λ_s 由学习达成，符合预期）")
    elif constraint > 0.0:
        print(f"  [!] constraint>0：当前 λ_r 仅略大于/未超过 λ_s+margin，约束仍在施压")
    # 合理量级检查（避免发散到极端值）
    if lam_s > 50 or lam_r > 50:
        print(f"  [!] λ 量级偏大（>50），可能未收敛/发散，建议检查训练曲线与学习率")
    else:
        print(f"  [✓] λ 量级正常（未发散到极端值）")
    print()


def diagnose_logs(logs_path: Path, model_name: str = "SR-DKT") -> None:
    if not logs_path.exists():
        print(f"[diag] 未找到训练日志: {logs_path}，跳过 Val AUC 轨迹分析")
        return
    logs = json.loads(logs_path.read_text(encoding="utf-8"))
    hist = logs.get(model_name)
    if not hist:
        # 兼容独立日志目录格式
        for k, v in logs.items():
            if model_name.lower() in k.lower():
                hist = v
                break
    if not hist:
        print(f"[diag] 日志中未找到 {model_name} 的 Val AUC 轨迹")
        return

    best = max(hist)
    best_ep = hist.index(best) + 1
    n = len(hist)
    print("=" * 56)
    print("Val AUC 轨迹诊断")
    print("=" * 56)
    print(f"  轮数: {n} | 最佳 Val AUC = {best:.4f} (epoch {best_ep})")
    print(f"  首轮 {hist[0]:.4f} → 末轮 {hist[-1]:.4f}")
    if best_ep >= n:
        print("  [!] 最佳出现在最后一轮：可能欠拟合，建议增大 epochs/patience")
    elif best_ep <= 2:
        print("  [!] 最佳出现在极早期：可能很快过拟合或学习率偏大")
    else:
        print("  [✓] 最佳出现在中段，收敛形态正常")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SR-DKT 训练后诊断")
    parser.add_argument("--ckpt", type=Path, required=True, help="best_model_*.pt 路径")
    parser.add_argument("--logs", type=Path, default=None, help="training_logs.json 路径（可选）")
    parser.add_argument("--model-name", default="SR-DKT", help="日志中的模型名")
    args = parser.parse_args()

    if not args.ckpt.exists():
        print(f"[ERROR] 找不到 checkpoint: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    print(f"[diag] 加载 checkpoint: {args.ckpt}")
    model = load_model_from_ckpt(args.ckpt)
    print(f"[diag] num_kc={model.num_kc}, hidden_size={model.hidden_size}\n")

    diagnose_lambda(model)
    if args.logs:
        diagnose_logs(args.logs, args.model_name)

    print("诊断完成。理想结果：λ_r>λ_s 成立、λ 量级正常、constraint 已满足、")
    print("Val AUC 在中段收敛。若全部 ✓，说明架构修订生效，可继续批量实验。")


if __name__ == "__main__":
    main()
