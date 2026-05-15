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


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = SRDKT(
        num_kc=int(checkpoint["num_kc"]),
        hidden_size=int(checkpoint["config"]["hidden_size"]),
        num_layers=int(checkpoint["config"].get("num_layers", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_data = load_pickle(DATA_DIR / "test.pkl")
    encoder = load_pickle(DATA_DIR / "skill_encoder.pkl")
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
            batch["mask"],
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

    with STATE_PATH.open("wb") as f:
        pickle.dump(student_states, f)

    auc, acc = evaluate_model(model, loader, device)
    with EVAL_PATH.open("wb") as f:
        pickle.dump({"AUC": float(auc), "ACC": float(acc)}, f)

    print(f"[export_states] 已导出学生状态: {STATE_PATH}")
    print(f"[export_states] 已保存模型评估: {EVAL_PATH}")
    print(f"[export_states] 学生数: {len(student_states)}")


if __name__ == "__main__":
    main()
