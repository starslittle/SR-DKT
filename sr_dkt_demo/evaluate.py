# 功能：计算 SR-DKT 模型层指标，以及 KT-Harness 推荐层评估指标。
# 作者：Cursor AI
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, ndcg_score, roc_auc_score
from torch.utils.data import DataLoader

from harness_agent import HarnessAgent
from model import SRDKT
from train import SequenceDataset, collate_batch, move_batch


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "best_model.pt"
DATA_MODEL_PATH = DATA_DIR / "best_model.pt"
EVAL_PATH = DATA_DIR / "eval_results.pkl"


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


@torch.no_grad()
def evaluate_kt_model() -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = DATA_MODEL_PATH if DATA_MODEL_PATH.exists() else MODEL_PATH
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = SRDKT(
        num_kc=int(checkpoint["num_kc"]),
        hidden_size=int(checkpoint["config"]["hidden_size"]),
        num_layers=int(checkpoint["config"].get("num_layers", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_data = load_pickle(DATA_DIR / "test.pkl")
    loader = DataLoader(SequenceDataset(test_data), batch_size=64, shuffle=False, collate_fn=collate_batch)
    y_true, y_pred = [], []

    for batch in loader:
        batch = move_batch(batch, device)
        preds, _, _ = model(
            batch["sequences"],
            batch["delta_ts"],
            batch["hints"],
            batch["attempts"],
            batch["mask"],
        )
        if preds.shape[1] <= 1:
            continue
        label_mask = batch["mask"][:, 1:]
        y_true.extend(batch["corrects"][:, 1:][label_mask].detach().cpu().numpy().tolist())
        y_pred.extend(preds[:, :-1, 0][label_mask].detach().cpu().numpy().tolist())

    if not y_true:
        return 0.5, 0.0
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
    return float(auc), float(acc)


def kc_midpoint_gain(seq: list[tuple], kc_id: int) -> float | None:
    kc_corrects = [int(correct) for item_kc, correct, *_ in seq if int(item_kc) == int(kc_id)]
    if len(kc_corrects) < 4:
        return None

    mid = len(kc_corrects) // 2
    before = kc_corrects[max(0, mid - 10) : mid]
    after = kc_corrects[mid : min(len(kc_corrects), mid + 10)]
    if not before or not after:
        return None
    return float(np.mean(after) - np.mean(before))


def evaluate_harness() -> tuple[float, float, float]:
    test_data = load_pickle(DATA_DIR / "test.pkl")
    student_states = load_pickle(DATA_DIR / "student_states.pkl")
    agent = HarnessAgent(student_states)

    gains = []
    hr_values = []
    ndcg_values = []

    for student_id, seq in test_data.items():
        try:
            recs = agent.recommend(student_id, top_k=10)
        except KeyError:
            continue
        weak_kcs = [
            int(kc_id)
            for kc_id, state in student_states.get(student_id, {}).items()
            if float(state.get("S", 1.0)) < 0.5
        ]
        for kc_id in weak_kcs:
            gain = kc_midpoint_gain(seq, kc_id)
            if gain is not None:
                gains.append(gain)
        if not recs:
            continue

        rec_gains = []
        scores = []
        hit_count = 0
        for rec in recs:
            kc_id = int(rec["kc_id"])
            score = float(rec["score"])
            gain = kc_midpoint_gain(seq, kc_id)
            if gain is None:
                gain = 0.0
            rec_gains.append(max(gain, 0.0))
            scores.append(score)
            if gain > 0.0:
                hit_count += 1

        hr_values.append(hit_count / 10.0)
        if len(rec_gains) >= 2 and np.sum(rec_gains) > 0:
            ndcg_values.append(float(ndcg_score([rec_gains], [scores], k=min(10, len(rec_gains)))))
        else:
            ndcg_values.append(0.0)

    wcg = float(np.mean(gains)) if gains else 0.0
    hr10 = float(np.mean(hr_values)) if hr_values else 0.0
    ndcg10 = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    return wcg, hr10, ndcg10


def main() -> None:
    if EVAL_PATH.exists():
        eval_results = load_pickle(EVAL_PATH)
        auc = float(eval_results.get("AUC", 0.5))
        acc = float(eval_results.get("ACC", 0.0))
    else:
        auc, acc = evaluate_kt_model()
    wcg, hr10, ndcg10 = evaluate_harness()
    with EVAL_PATH.open("wb") as f:
        pickle.dump({"AUC": auc, "ACC": acc, "WCG": wcg, "HR10": hr10, "NDCG10": ndcg10}, f)

    print("========== SR-DKT 评估报告 ==========")
    print("[模型层 - Knowledge Tracing]")
    print(f"AUC  = {auc:.3f}")
    print(f"ACC  = {acc:.3f}（阈值=0.5）")
    print()
    print("[推荐层 - KT-Harness]")
    print(f"WeakConcept-Gain (WCG) = {wcg:+.3f}")
    print(f"HR@10                  = {hr10:.3f}")
    print(f"NDCG@10                = {ndcg10:.3f}")
    print("=====================================")


if __name__ == "__main__":
    main()
