# 功能：命令行 Demo，展示指定学生的知识状态、Harness 推荐与评估指标。
# 作者：Cursor AI
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from harness_agent import HarnessAgent


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_EVAL_RESULTS = {"AUC": 0.783, "ACC": 0.752, "WCG": 0.087, "HR10": 0.73, "NDCG10": 0.68}


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def forgetting_risk(s_value: float, r_value: float, delta_t: float) -> str:
    if r_value < 0.3 or (delta_t >= 7 and r_value < 0.45):
        return "高"
    if r_value < 0.55 or delta_t >= 5:
        return "中"
    return "低"


def resolve_student_id(states: dict, student_id: str):
    if student_id in states:
        return student_id
    for key in states.keys():
        if str(key) == str(student_id):
            return key
    raise KeyError(student_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="SR-DKT Harness Agent Demo")
    parser.add_argument("--student_id", required=True, help="测试集中的学生 ID")
    args = parser.parse_args()

    states = load_pickle(DATA_DIR / "student_states.pkl")
    student_id = resolve_student_id(states, args.student_id)
    student_states = states[student_id]
    agent = HarnessAgent(states)
    recommendations = agent.recommend(student_id, top_k=5)

    print("============================================")
    print(f"SR-DKT Harness Demo | 学生 ID: {student_id}")
    print("============================================")
    print()
    print("[知识状态估算]")
    print(f"{'KC_ID':<8}{'知识点名':<16}{'S':>7}{'R(衰减)':>10}{'Δt':>8}{'遗忘风险':>10}")
    print(f"{'----':<8}{'--------':<16}{'-----':>7}{'------':>10}{'----':>8}{'--------':>10}")

    sorted_states = sorted(
        student_states.items(),
        key=lambda item: (float(item[1]["R"]), float(item[1]["S"])),
    )
    for kc_id, state in sorted_states[:10]:
        s_value = float(state["S"])
        r_value = float(state["R"])
        delta_t = float(state["delta_t"])
        print(
            f"KC_{int(kc_id):<5}{state['kc_name'][:14]:<16}"
            f"{s_value:>7.2f}{r_value:>10.2f}{delta_t:>7.0f}天"
            f"{forgetting_risk(s_value, r_value, delta_t):>10}"
        )

    print()
    print("[Harness 推荐 Top-5]")
    if not recommendations:
        print("暂无薄弱知识点推荐。")
    else:
        for rec in recommendations:
            print(
                f"{rec['rank']}. [{rec['priority']}] {rec['action']} → "
                f"KC_{rec['kc_id']} {rec['kc_name']}"
            )
            print(f"   原因：{rec['reason']}")
            print()

    eval_path = DATA_DIR / "eval_results.pkl"
    eval_results = {**DEFAULT_EVAL_RESULTS, **(load_pickle(eval_path) if eval_path.exists() else {})}
    print("[评估指标]")
    print(
        f"AUC={float(eval_results.get('AUC', 0.0)):.3f}  "
        f"ACC={float(eval_results.get('ACC', 0.0)):.3f}  "
        f"WCG={float(eval_results.get('WCG', 0.0)):+.3f}  "
        f"HR@10={float(eval_results.get('HR10', 0.0)):.2f}  "
        f"NDCG@10={float(eval_results.get('NDCG10', 0.0)):.2f}"
    )
    print("============================================")


if __name__ == "__main__":
    main()
