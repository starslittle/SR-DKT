# 功能：基于学生知识点 S/R 状态生成个性化学习推荐。
# 作者：Cursor AI
from __future__ import annotations

import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


class HarnessAgent:
    def __init__(self, student_states: dict | str | Path | None = None) -> None:
        if student_states is None:
            states_path = DATA_DIR / "student_states.pkl"
            with states_path.open("rb") as f:
                self.student_states = pickle.load(f)
        elif isinstance(student_states, (str, Path)):
            with Path(student_states).open("rb") as f:
                self.student_states = pickle.load(f)
        else:
            self.student_states = student_states

    @staticmethod
    def _classify_state(s_value: float, r_value: float) -> tuple[str, str, str, str]:
        if s_value < 0.35 and r_value < 0.25:
            return "WATCH", "看视频", "还没掌握，先建立基础认知", "必要"
        elif s_value < 0.5 and r_value >= 0.4:
            return "RETRIEVAL_TEST", "主动检索", "趁R高固化S，Testing Effect", "推荐"
        elif s_value >= 0.6 and r_value < 0.4:
            return "SPACED_REVIEW", "间隔复习", "S在但R快消失，重建提取能力", "紧急"
        elif s_value >= 0.6 and r_value < 0.55:
            return "PRACTICE", "低难度练习", "维持状态，无需紧急复习", "建议"
        elif s_value >= 0.7 and r_value >= 0.7:
            return "CHALLENGE", "进阶挑战", "已掌握，推进更难知识点", "进阶"
        else:
            return "DISCUSS", "讨论复述", "强制输出，加速S建立", "推荐"

    classify_state = _classify_state

    def _resolve_student_id(self, student_id):
        if student_id in self.student_states:
            return student_id
        for key in self.student_states.keys():
            if str(key) == str(student_id):
                return key
        raise KeyError(f"未找到学生 ID: {student_id}")

    def recommend(self, student_id: int, top_k: int = 10) -> list[dict]:
        resolved_id = self._resolve_student_id(student_id)
        states = self.student_states[resolved_id]
        candidates = []

        for kc_id, state in states.items():
            s_value = float(state["S"])
            r_value = float(state["R"])
            if not (s_value < 0.5 or r_value < 0.4):
                continue
            action, label, reason, priority = self._classify_state(s_value, r_value)
            score = (1.0 - s_value) * 0.6 + (1.0 - r_value) * 0.4
            candidates.append(
                {
                    "rank": 0,
                    "kc_id": int(kc_id),
                    "kc_name": str(state.get("kc_name", f"KC_{kc_id}")),
                    "action": action,
                    "action_label": label,
                    "reason": reason,
                    "S": s_value,
                    "R": r_value,
                    "delta_t": float(state.get("delta_t", 0.0)),
                    "priority": priority,
                    "score": float(score),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        top_items = candidates[:top_k]
        for rank, item in enumerate(top_items, start=1):
            item["rank"] = rank
        return top_items
