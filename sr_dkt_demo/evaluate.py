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


def get_dataset_dir(dataset: str) -> Path:
    """根据数据集名称返回对应的子目录"""
    if dataset == "mooc":
        return DATA_DIR / "mooc"
    return DATA_DIR / ("2017" if dataset == "2017" else "2009")


def get_state_name(dataset: str) -> str:
    """返回 export_states.py 产出的学生状态文件名。"""
    if dataset == "mooc":
        return "student_states_mooc.pkl"
    if dataset == "2017":
        return "student_states_2017.pkl"
    return "student_states.pkl"


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


@torch.no_grad()
def evaluate_kt_model(dataset: str = "2009") -> tuple[float, float]:
    ds_dir = get_dataset_dir(dataset)
    if dataset == "mooc":
        model_name = "best_model_mooc.pt"
    elif dataset == "2017":
        model_name = "best_model_2017.pt"
    else:
        model_name = "best_model.pt"
    model_path = ds_dir / model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not model_path.exists():
        model_path = DATA_MODEL_PATH if DATA_MODEL_PATH.exists() else MODEL_PATH
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = SRDKT(
        num_kc=int(checkpoint["num_kc"]),
        hidden_size=int(checkpoint["config"]["hidden_size"]),
        num_layers=int(checkpoint["config"].get("num_layers", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_name = "test_2017.pkl" if dataset == "2017" else "test.pkl"
    test_data = load_pickle(ds_dir / test_name)
    loader = DataLoader(SequenceDataset(test_data), batch_size=64, shuffle=False, collate_fn=collate_batch)
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


def evaluate_harness(dataset: str = "2009") -> tuple[float, float, float]:
    ds_dir = get_dataset_dir(dataset)
    test_name = "test_2017.pkl" if dataset == "2017" else "test.pkl"
    states_name = get_state_name(dataset)
    states_path = ds_dir / states_name
    if not states_path.exists():
        raise FileNotFoundError(
            f"缺少学生状态文件 {states_path}；请先运行 "
            f"`python export_states.py --dataset {dataset}`。"
        )
    test_data = load_pickle(ds_dir / test_name)
    student_states = load_pickle(states_path)
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

        hr_values.append(hit_count / max(min(10, len(recs)), 1))
        if len(rec_gains) >= 2 and np.sum(rec_gains) > 0:
            ndcg_values.append(float(ndcg_score([rec_gains], [scores], k=min(10, len(rec_gains)))))
        else:
            ndcg_values.append(0.0)

    wcg = float(np.mean(gains)) if gains else 0.0
    hr10 = float(np.mean(hr_values)) if hr_values else 0.0
    ndcg10 = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    return wcg, hr10, ndcg10


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SR-DKT Evaluation")
    parser.add_argument("--dataset", default="2009", choices=["2009", "2017", "mooc"],
                        help="选择数据集: 2009, 2017 或 mooc")
    args = parser.parse_args()

    ds_dir = get_dataset_dir(args.dataset)
    if args.dataset == "mooc":
        eval_name, model_name = "eval_results_mooc.pkl", "best_model_mooc.pt"
    elif args.dataset == "2017":
        eval_name, model_name = "eval_results_2017.pkl", "best_model_2017.pt"
    else:
        eval_name, model_name = "eval_results.pkl", "best_model.pt"
    eval_path = ds_dir / eval_name
    model_path = ds_dir / model_name

    need_eval = True
    if eval_path.exists() and model_path.exists():
        eval_mtime = eval_path.stat().st_mtime
        model_mtime = model_path.stat().st_mtime
        if eval_mtime > model_mtime:
            eval_results = load_pickle(eval_path)
            auc = float(eval_results.get("AUC", 0.5))
            acc = float(eval_results.get("ACC", 0.0))
            need_eval = False
            print("[evaluate] 使用缓存的评估结果（模型未更新）")

    if need_eval:
        auc, acc = evaluate_kt_model(args.dataset)
    wcg, hr10, ndcg10 = evaluate_harness(args.dataset)
    with eval_path.open("wb") as f:
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
