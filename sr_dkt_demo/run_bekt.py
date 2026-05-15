# 功能：单独训练 BEKT 模型（ASSISTments 2017）
from __future__ import annotations

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

from baselines.baseline_models import BEKT
from train import SequenceDataset, collate_batch, move_batch

CONFIG = {
    "hidden_size": 128,
    "batch_size": 64,
    "lr": 5e-4,
    "epochs": 100,
    "patience": 10,
    "seed": 42,
    "max_seq_len": 200,
}

DATA_DIR = ROOT_DIR / "data"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def sequence_loss(preds, corrects, mask, criterion):
    if preds.shape[1] <= 1:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    valid = mask[:, 1:]
    return criterion(preds[:, :-1, 0][valid], corrects[:, 1:][valid])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        preds, _ = model(
            batch["sequences"][..., 0].long(),
            batch["corrects"],
            batch["mask"],
            hint_count=batch["hints"],
            attempt_count=batch["attempts"],
            delta_t=batch["delta_ts"],
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


def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BEKT] 使用设备: {device}")
    print(f"[BEKT] 数据集: ASSISTments 2017")

    train_data = load_pickle(DATA_DIR / "train_2017.pkl")
    val_data = load_pickle(DATA_DIR / "val_2017.pkl")
    test_data = load_pickle(DATA_DIR / "test_2017.pkl")
    meta = json.loads((DATA_DIR / "meta_2017.json").read_text(encoding="utf-8"))
    num_kc = int(meta["num_kc"])
    print(f"[BEKT] 知识点数量: {num_kc}")

    train_loader = DataLoader(
        SequenceDataset(train_data), batch_size=CONFIG["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_batch(b, CONFIG["max_seq_len"]),
    )
    val_loader = DataLoader(
        SequenceDataset(val_data), batch_size=CONFIG["batch_size"], shuffle=False,
        collate_fn=lambda b: collate_batch(b, CONFIG["max_seq_len"]),
    )
    test_loader = DataLoader(
        SequenceDataset(test_data), batch_size=CONFIG["batch_size"], shuffle=False,
        collate_fn=lambda b: collate_batch(b, CONFIG["max_seq_len"]),
    )

    model = BEKT(num_kc, CONFIG["hidden_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.BCELoss()

    best_auc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in tqdm(train_loader, desc=f"[BEKT] Epoch {epoch}", leave=False):
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            preds, _ = model(
                batch["sequences"][..., 0].long(),
                batch["corrects"],
                batch["mask"],
                hint_count=batch["hints"],
                attempt_count=batch["attempts"],
                delta_t=batch["delta_ts"],
            )
            loss = sequence_loss(preds, batch["corrects"], batch["mask"], criterion)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        val_metrics = evaluate(model, val_loader, device)
        val_auc = val_metrics["AUC"]
        avg_loss = total_loss / max(steps, 1)
        print(
            f"[BEKT] Epoch {epoch}/{CONFIG['epochs']} | "
            f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val ACC: {val_metrics['ACC']:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"[BEKT] Early stopping 触发，最佳 Val AUC: {best_auc:.3f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    CKPT_DIR = DATA_DIR / "ckpt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": CONFIG}, CKPT_DIR / "BEKT.pt")

    test_metrics = evaluate(model, test_loader, device)
    print(f"[BEKT] 训练完成 | Test AUC: {test_metrics['AUC']:.3f} | ACC: {test_metrics['ACC']:.3f} | F1: {test_metrics['F1']:.3f}")


if __name__ == "__main__":
    main()