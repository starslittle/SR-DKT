# 功能：SR-DKT 消融实验，验证各组件贡献（5组配置）
# 通过 ablation_mode 参数控制，不创建新模型类
# 作者：SR-DKT 项目组
# 日期：2026-05-12
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
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import SRDKT
from train import SequenceDataset, collate_batch, move_batch


CONFIG = {
    "hidden_size": 128,
    "batch_size": 128,
    "lr": 0.001,
    "epochs": 50,
    "patience": 8,
    "seed": 42,
    "max_seq_len": 200,
    "lambda_constraint_weight": 0.01,
}

DATA_DIR = ROOT_DIR / "data"
RESULT_PATH = DATA_DIR / "ablation_results.json"
LOG_PATH = DATA_DIR / "training_logs.json"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def collate_recent_batch(batch):
    return collate_batch(batch, CONFIG["max_seq_len"])


def save_training_log(model_name: str, auc_history: list[float]) -> None:
    logs = {}
    if LOG_PATH.exists():
        logs = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    logs[model_name] = [float(v) for v in auc_history]
    LOG_PATH.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")


def sequence_loss(preds: torch.Tensor, corrects: torch.Tensor, mask: torch.Tensor, criterion) -> torch.Tensor:
    if preds.shape[1] <= 1:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    valid = mask[:, 1:]
    return criterion(preds[:, :-1, 0][valid], corrects[:, 1:][valid])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, ablation_mode: str = 'full') -> dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        preds, _, _ = model(
            batch["sequences"],
            batch["delta_ts"],
            batch["hints"],
            batch["attempts"],
            watch_ratio_seq=batch.get("watch_ratios"),
            is_replay_seq=batch.get("is_replays"),
            mask=batch["mask"],
            ablation_mode=ablation_mode,
        )
        if preds.shape[1] <= 1:
            continue
        valid = batch["mask"][:, 1:]
        y_true.extend(batch["corrects"][:, 1:][valid].detach().cpu().numpy().tolist())
        y_pred.extend(preds[:, :-1, 0][valid].detach().cpu().numpy().tolist())

    if not y_true:
        return {"AUC": 0.5, "ACC": 0.0, "F1": 0.0}

    y_hat = np.array(y_pred) >= 0.5
    return {
        "AUC": float(roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5),
        "ACC": float(accuracy_score(y_true, y_hat)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def train_ablation(
    model: SRDKT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    ablation_mode: str = 'full',
) -> dict[str, float]:
    """
    训练指定消融模式的 SR-DKT 模型

    Args:
        model: SR-DKT 模型实例
        train_loader: 训练数据
        val_loader: 验证数据
        test_loader: 测试数据
        device: 设备
        ablation_mode: 消融模式

    Returns:
        metrics: {AUC, ACC, F1}
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.BCELoss()

    best_auc = -1.0
    best_state = None
    no_improve = 0
    auc_history = []
    lambda_weight = CONFIG["lambda_constraint_weight"]

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in tqdm(train_loader, desc=f"[{ablation_mode}] Epoch {epoch}", leave=False):
            batch = move_batch(batch, device)
            optimizer.zero_grad()

            preds, _, _ = model(
                batch["sequences"],
                batch["delta_ts"],
                batch["hints"],
                batch["attempts"],
                watch_ratio_seq=batch.get("watch_ratios"),
                is_replay_seq=batch.get("is_replays"),
                mask=batch["mask"],
                ablation_mode=ablation_mode,
            )

            bce_loss = sequence_loss(preds, batch["corrects"], batch["mask"], criterion)

            constraint_loss = model.get_lambda_constraint_loss()
            loss = bce_loss + lambda_weight * constraint_loss

            loss.backward()
            optimizer.step()
            total_loss += float(bce_loss.item())
            steps += 1

        val_metrics = evaluate(model, val_loader, device, ablation_mode=ablation_mode)
        val_auc = float(val_metrics["AUC"])
        auc_history.append(val_auc)

        print(f"[{ablation_mode}] Epoch {epoch}/{CONFIG['epochs']} | "
              f"Loss: {total_loss / max(steps, 1):.4f} | "
              f"Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"[{ablation_mode}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_training_log(ablation_mode, auc_history)
    metrics = evaluate(model, test_loader, device, ablation_mode=ablation_mode)
    print(f"[{ablation_mode}] 训练完成 | Test AUC: {metrics['AUC']:.3f} | ACC: {metrics['ACC']:.3f} | F1: {metrics['F1']:.3f}")
    return metrics


def print_results(results: dict[str, dict[str, float]], dataset: str = "2009") -> None:
    baseline = results.get("Full SR-DKT", {})
    best_auc = baseline.get("AUC", 0.0)
    dataset_names = {"2009": "ASSISTments 2009", "2017": "ASSISTments 2017", "mooc": "MOOCCubeX"}
    ds_name = dataset_names.get(dataset, dataset)
    print("=" * 50)
    print(f"SR-DKT 消融实验结果（{ds_name}）")
    print("=" * 50)
    print(f"{'配置':<16}{'AUC':>8}{'ACC':>8}{'F1':>8}")
    print("-" * 50)
    for name, metrics in results.items():
        marker = "  ★ 基准" if name == "Full SR-DKT" else ""
        print(f"{name:<16}{metrics['AUC']:>8.3f}{metrics['ACC']:>8.3f}{metrics['F1']:>8.3f}{marker}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="SR-DKT Ablation Study")
    parser.add_argument("--dataset", default="2009", choices=["2009", "2017", "mooc"],
                        help="选择数据集: 2009, 2017 或 mooc")
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] 使用设备: {device}")
    dataset_labels = {"2009": "ASSISTments 2009", "2017": "ASSISTments 2017", "mooc": "MOOCCubeX"}
    print(f"[ablation] 数据集: {dataset_labels.get(args.dataset, args.dataset)}")

    # 加载数据
    if args.dataset == "2017":
        train_data = load_pickle(DATA_DIR / "2017" / "train_2017.pkl")
        val_data = load_pickle(DATA_DIR / "2017" / "val_2017.pkl")
        test_data = load_pickle(DATA_DIR / "2017" / "test_2017.pkl")
        meta_path = DATA_DIR / "2017" / "meta_2017.json"
    elif args.dataset == "mooc":
        train_data = load_pickle(DATA_DIR / "mooc" / "train.pkl")
        val_data = load_pickle(DATA_DIR / "mooc" / "val.pkl")
        test_data = load_pickle(DATA_DIR / "mooc" / "test.pkl")
        meta_path = DATA_DIR / "mooc" / "meta_mooc.json"
    else:
        train_data = load_pickle(DATA_DIR / "2009" / "train.pkl")
        val_data = load_pickle(DATA_DIR / "2009" / "val.pkl")
        test_data = load_pickle(DATA_DIR / "2009" / "test.pkl")
        meta_path = DATA_DIR / "2009" / "meta.json"

    num_kc = int(json.loads(meta_path.read_text(encoding="utf-8"))["num_kc"])
    print(f"[ablation] 知识点数量: {num_kc}")

    train_loader = DataLoader(
        SequenceDataset(train_data),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_recent_batch,
    )
    val_loader = DataLoader(
        SequenceDataset(val_data),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_recent_batch,
    )
    test_loader = DataLoader(
        SequenceDataset(test_data),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_recent_batch,
    )

    # 6 组消融配置，每个配置创建一个新模型实例
    # 必须为每个配置创建独立模型，防止权重继承影响
    ablation_configs = [
        ("Full SR-DKT",     'full'),
        ("w/o Storage",     'no_storage'),
        ("w/o Retrieval",   'no_retrieval'),
        ("w/o 差异化遗忘",  'shared_decay'),
        ("w/o 注意力融合",  'no_attention'),
        ("单路径退化版",    'single_path'),
    ]

    results = {}
    for display_name, mode in ablation_configs:
        print(f"\n{'='*50}")
        print(f"训练消融变体: {display_name} (mode={mode})")
        print(f"{'='*50}")
        model = SRDKT(
            num_kc=num_kc,
            hidden_size=CONFIG["hidden_size"],
            num_layers=1,
        )
        results[display_name] = train_ablation(
            model, train_loader, val_loader, test_loader, device,
            ablation_mode=mode,
        )

    # 保存结果（仅保留指标）
    save_results = {
        name: {"AUC": m["AUC"], "ACC": m["ACC"], "F1": m["F1"]}
        for name, m in results.items()
    }
    if args.dataset == "mooc":
        result_path = DATA_DIR / "mooc" / "ablation_results_mooc.json"
    elif args.dataset == "2017":
        result_path = DATA_DIR / "2017" / "ablation_results_2017.json"
    else:
        result_path = DATA_DIR / "2009" / "ablation_results.json"
    result_path.write_text(json.dumps(save_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ablation] 已保存结果: {result_path}")

    # 打印对比表格
    print_results(results, dataset=args.dataset)


if __name__ == "__main__":
    main()
