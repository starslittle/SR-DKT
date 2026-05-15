# 功能：训练 SR-DKT 模型，支持多输出、lambda约束损失、可选输入、early stopping。
# 已修复 Bug 2：evaluate_model 和 train_sr_dkt 中 mask 参数位置错误
# 作者：SR-DKT 项目组
# 日期：2026-05-11
# 训练配置：batch_size=64, max_seq_len=200, hidden_size=128, epochs=100, patience=10, seed=42
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import SRDKT


# 训练配置
CONFIG = {
    "hidden_size": 128,
    "num_layers": 1,
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 100,
    "patience": 10,  # early stopping patience
    "seed": 42,
    "max_seq_len": 200,
    "lambda_constraint_weight": 0.01,  # lambda约束损失权重
}

DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = DATA_DIR / "best_model.pt"
TRAINING_LOG_PATH = DATA_DIR / "training_logs.json"


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path: Path):
    """加载 pickle 文件"""
    with path.open("rb") as f:
        return pickle.load(f)


class SequenceDataset(Dataset):
    """
    序列数据集，用于知识追踪训练

    每条数据格式: (student_id, seq)
    seq: list of tuples [(kc_id, correct, delta_t, hint, attempt), ...]
    """

    def __init__(self, sequences: dict) -> None:
        self.items = list(sequences.items())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        student_id, seq = self.items[idx]
        return student_id, seq


def collate_batch(batch, max_seq_len: int = 200):
    """
    将批次数据打包为 tensor

    Args:
        batch: list of (student_id, seq)
        max_seq_len: 最大序列长度，超长截断

    Returns:
        dict: 包含所有特征的 batch dict
    """
    # 截断超长序列
    batch = [
        (student_id, seq[-max_seq_len:] if len(seq) > max_seq_len else seq)
        for student_id, seq in batch
    ]

    student_ids, seqs = zip(*batch)
    max_len = max(len(seq) for seq in seqs)
    batch_size = len(seqs)

    sequences = torch.zeros(batch_size, max_len, 2, dtype=torch.float32)
    corrects = torch.zeros(batch_size, max_len, dtype=torch.float32)
    delta_ts = torch.zeros(batch_size, max_len, dtype=torch.float32)
    hints = torch.zeros(batch_size, max_len, dtype=torch.float32)
    attempts = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(seqs):
        for t, (kc_id, correct, delta_t, hint, attempt) in enumerate(seq):
            sequences[i, t, 0] = float(kc_id)
            sequences[i, t, 1] = float(correct)
            corrects[i, t] = float(correct)
            delta_ts[i, t] = float(delta_t)
            hints[i, t] = float(hint)
            attempts[i, t] = float(attempt)
            mask[i, t] = True

    return {
        "student_ids": student_ids,
        "sequences": sequences,
        "corrects": corrects,
        "delta_ts": delta_ts,
        "hints": hints,
        "attempts": attempts,
        "mask": mask,
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    """将 batch 中的 tensor 移动到指定设备"""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def get_next_step_loss(
    preds: torch.Tensor,
    corrects: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module
) -> torch.Tensor:
    """
    计算下一步预测损失（知识追踪标准做法）

    预测第 t+1 步的正确率，使用第 t 步的标签
    """
    if preds.shape[1] <= 1:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    pred_next = preds[:, :-1]
    label_next = corrects[:, 1:]
    label_mask = mask[:, 1:]
    return criterion(pred_next.squeeze(-1)[label_mask], label_next[label_mask])


def infer_num_kc() -> int:
    """推断知识点数量"""
    meta_path = DATA_DIR / "meta.json"
    if meta_path.exists():
        return int(json.loads(meta_path.read_text(encoding="utf-8"))["num_kc"])
    encoder = load_pickle(DATA_DIR / "skill_encoder.pkl")
    return int(len(encoder.classes_))


@torch.no_grad()
def evaluate_model(
    model: SRDKT,
    loader: DataLoader,
    device: torch.device
) -> dict[str, float]:
    """
    评估模型性能

    Returns:
        dict: {"AUC": float, "ACC": float, "F1": float}
    """
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        preds, _, _ = model(
            batch["sequences"],
            batch["delta_ts"],
            batch["hints"],
            batch["attempts"],
            mask=batch["mask"],
        )
        if preds.shape[1] <= 1:
            continue
        label_mask = batch["mask"][:, 1:]
        y_true.extend(batch["corrects"][:, 1:][label_mask].detach().cpu().numpy().tolist())
        y_pred.extend(preds[:, :-1, 0][label_mask].detach().cpu().numpy().tolist())

    if not y_true:
        return {"AUC": 0.5, "ACC": 0.0, "F1": 0.0}

    y_hat = np.array(y_pred) >= 0.5
    return {
        "AUC": float(roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5),
        "ACC": float(accuracy_score(y_true, y_hat)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def save_training_log(model_name: str, auc_history: list[float]) -> None:
    """保存训练日志"""
    logs = {}
    if TRAINING_LOG_PATH.exists():
        logs = json.loads(TRAINING_LOG_PATH.read_text(encoding="utf-8"))
    logs[model_name] = [float(v) for v in auc_history]
    TRAINING_LOG_PATH.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")


def train_sr_dkt(
    model: SRDKT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    lambda_constraint_weight: float = 0.01,
) -> tuple[SRDKT, list[float]]:
    """
    训练 SR-DKT 模型

    Args:
        model: SR-DKT 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备
        config: 训练配置
        lambda_constraint_weight: lambda约束损失权重

    Returns:
        model: 训练后的模型（加载最佳权重）
        val_auc_history: 验证集 AUC 历史
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss()

    best_auc = -1.0
    best_state = None
    no_improve = 0
    val_auc_history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_constraint_loss = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad()

            # SR-DKT 多输出：p_seq, s_seq, r_seq
            # loss 只用 p_seq 计算 BCE
            preds, s_seq, r_seq = model(
                batch["sequences"],
                batch["delta_ts"],
                batch["hints"],
                batch["attempts"],
                batch["mask"],
            )

            # 计算预测损失
            bce_loss = get_next_step_loss(preds, batch["corrects"], batch["mask"], criterion)

            # 计算 lambda 约束损失
            constraint_loss = model.get_lambda_constraint_loss()

            # 总损失 = BCE + lambda_constraint_weight * constraint_loss
            loss = bce_loss + lambda_constraint_weight * constraint_loss

            loss.backward()
            optimizer.step()

            total_loss += float(bce_loss.item())
            total_constraint_loss += float(constraint_loss.item())
            steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", constraint=f"{constraint_loss.item():.4f}")

        avg_loss = total_loss / max(steps, 1)
        avg_constraint = total_constraint_loss / max(steps, 1)
        val_metrics = evaluate_model(model, val_loader, device)
        val_auc = val_metrics["AUC"]
        val_auc_history.append(val_auc)

        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"BCE Loss: {avg_loss:.4f} | "
            f"Constraint: {avg_constraint:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val ACC: {val_metrics['ACC']:.4f} | "
            f"λ_s: {model.lambda_s.item():.3f} | "
            f"λ_r: {model.lambda_r.item():.3f}"
        )

        # Early stopping 监控验证集 AUC
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"[train] Early stopping 触发，最佳 Val AUC: {best_auc:.3f}")
                break

    # 加载最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, val_auc_history


def main() -> None:
    """主函数：训练 SR-DKT 模型"""
    parser = argparse.ArgumentParser(description="SR-DKT Training Script")
    parser.add_argument("--dataset", default="2009", choices=["2009", "2017"],
                        help="选择数据集: 2009 或 2017")
    parser.add_argument("--lambda_weight", type=float, default=0.01,
                        help="lambda约束损失权重")
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 使用设备: {device}")
    print(f"[train] 数据集: ASSISTments {args.dataset}")

    # 加载数据
    if args.dataset == "2017":
        train_data = load_pickle(DATA_DIR / "train_2017.pkl")
        val_data = load_pickle(DATA_DIR / "val_2017.pkl")
        meta_path = DATA_DIR / "meta_2017.json"
        model_save_path = DATA_DIR / "best_model_2017.pt"
    else:
        train_data = load_pickle(DATA_DIR / "train.pkl")
        val_data = load_pickle(DATA_DIR / "val.pkl")
        meta_path = DATA_DIR / "meta.json"
        model_save_path = MODEL_PATH

    num_kc = int(json.loads(meta_path.read_text(encoding="utf-8"))["num_kc"])
    print(f"[train] 知识点数量: {num_kc}")

    # 创建 DataLoader
    train_loader = DataLoader(
        SequenceDataset(train_data),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, CONFIG["max_seq_len"]),
    )
    val_loader = DataLoader(
        SequenceDataset(val_data),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, CONFIG["max_seq_len"]),
    )

    # 创建模型
    model = SRDKT(
        num_kc=num_kc,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"]
    )

    # 训练模型
    model, val_auc_history = train_sr_dkt(
        model, train_loader, val_loader, device, CONFIG, args.lambda_weight
    )

    # 保存模型
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_kc": num_kc,
            "config": CONFIG,
        },
        model_save_path,
    )
    print(f"[train] 已保存最佳模型: {model_save_path}")

    # 保存训练日志
    save_training_log("SR-DKT", val_auc_history)
    print(f"[train] 已保存训练日志: {TRAINING_LOG_PATH}")
    print(f"[train] 训练结束，最佳 Val AUC: {max(val_auc_history):.3f}")


if __name__ == "__main__":
    main()