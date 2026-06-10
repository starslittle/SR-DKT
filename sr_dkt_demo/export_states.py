# 功能：加载最佳模型，在测试集上导出每个学生各知识点最新的 S/R 状态。
# 作者：Cursor AI
from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import SRDKT
from train import SequenceDataset, collate_batch, evaluate_model, move_batch


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = DATA_DIR / "best_model.pt"
STATE_PATH = DATA_DIR / "student_states.pkl"
EVAL_PATH = DATA_DIR / "eval_results.pkl"


def get_dataset_dir(dataset: str) -> Path:
    """根据数据集名称返回对应的子目录"""
    if dataset == "mooc":
        return DATA_DIR / "mooc"
    return DATA_DIR / ("2017" if dataset == "2017" else "2009")


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


@torch.no_grad()
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export Student States")
    parser.add_argument("--dataset", default="2009", choices=["2009", "2017", "mooc"],
                        help="选择数据集: 2009, 2017 或 mooc")
    args = parser.parse_args()

    ds_dir = get_dataset_dir(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mooc":
        model_name = "best_model_mooc.pt"
    elif args.dataset == "2017":
        model_name = "best_model_2017.pt"
    else:
        model_name = "best_model.pt"
    model_path = ds_dir / model_name
    if not model_path.exists():
        model_path = MODEL_PATH

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = SRDKT(
        num_kc=int(checkpoint["num_kc"]),
        hidden_size=int(checkpoint["config"]["hidden_size"]),
        num_layers=int(checkpoint["config"].get("num_layers", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.dataset == "mooc":
        test_name, enc_name = "test.pkl", "skill_encoder_mooc.pkl"
    elif args.dataset == "2017":
        test_name, enc_name = "test_2017.pkl", "skill_encoder_2017.pkl"
    else:
        test_name, enc_name = "test.pkl", "skill_encoder.pkl"
    test_data = load_pickle(ds_dir / test_name)
    encoder = load_pickle(ds_dir / enc_name)
    kc_names = {i: str(name) for i, name in enumerate(encoder.classes_)}

    loader = DataLoader(SequenceDataset(test_data), batch_size=64, shuffle=False, collate_fn=collate_batch)
    student_states = {}

    for batch in loader:
        student_ids = batch["student_ids"]
        batch = move_batch(batch, device)
        _, s_seq, r_seq = model(
            batch["sequences"],
            batch["delta_ts"],
            batch["hints"],
            batch["attempts"],
            watch_ratio_seq=batch.get("watch_ratios"),
            is_replay_seq=batch.get("is_replays"),
            mask=batch["mask"],
        )

        for i, student_id in enumerate(student_ids):
            valid_len = int(batch["mask"][i].sum().item())
            states_for_student = {}
            for t in range(valid_len):
                kc_id = int(batch["sequences"][i, t, 0].item())
                states_for_student[kc_id] = {
                    "S": float(s_seq[i, t, 0].item()),
                    "R": float(r_seq[i, t, 0].item()),
                    "delta_t": float(batch["delta_ts"][i, t].item()),
                    "kc_name": kc_names.get(kc_id, f"KC_{kc_id}"),
                }
            student_states[student_id] = states_for_student

    if args.dataset == "mooc":
        state_name = "student_states_mooc.pkl"
    elif args.dataset == "2017":
        state_name = "student_states_2017.pkl"
    else:
        state_name = "student_states.pkl"
    state_path = ds_dir / state_name
    with state_path.open("wb") as f:
        pickle.dump(student_states, f)

    if args.dataset == "mooc":
        eval_name = "eval_results_mooc.pkl"
    elif args.dataset == "2017":
        eval_name = "eval_results_2017.pkl"
    else:
        eval_name = "eval_results.pkl"
    eval_path = ds_dir / eval_name
    metrics = evaluate_model(model, loader, device)
    with eval_path.open("wb") as f:
        pickle.dump(
            {
                "AUC": float(metrics["AUC"]),
                "ACC": float(metrics["ACC"]),
                "F1": float(metrics["F1"]),
            },
            f,
        )

    print(f"[export_states] 已导出学生状态: {state_path}")
    print(f"[export_states] 已保存模型评估: {eval_path}")
    print(f"[export_states] 学生数: {len(student_states)}")


if __name__ == "__main__":
    main()
