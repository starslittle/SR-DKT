# 功能：读取 ASSISTments 数据或生成模拟数据，并转换为 SR-DKT 可训练序列。
# 作者：Cursor AI
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_FILE = DATA_DIR / "skill_builder_data_corrected.csv"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


MATH_KC_NAMES = [
    "一元一次方程",
    "二元一次方程",
    "函数基础",
    "一次函数",
    "反比例函数",
    "二次函数",
    "分数运算",
    "小数运算",
    "百分数应用",
    "比例与相似",
    "几何基础",
    "三角形性质",
    "四边形性质",
    "圆的性质",
    "勾股定理",
    "概率初步",
    "统计图表",
    "代数式化简",
    "因式分解",
    "不等式",
    "有理数运算",
    "实数与根式",
    "坐标系",
    "面积计算",
    "体积计算",
    "数列基础",
    "应用题建模",
    "逻辑推理",
    "图形变换",
    "综合复习",
]


def generate_mock_data(path: Path, num_students: int = 200, num_kc: int = 30) -> pd.DataFrame:
    """Generate a small but learnable interaction log when the real CSV is absent."""
    rng = np.random.default_rng(SEED)
    rows = []
    base_time = pd.Timestamp("2024-01-01 08:00:00")

    for student_id in range(1, num_students + 1):
        ability = rng.normal(0.0, 0.7)
        kc_mastery = rng.beta(2.0, 4.0, size=num_kc)
        current_time = base_time + pd.Timedelta(days=int(rng.integers(0, 5)))
        seq_len = 50

        for step in range(seq_len):
            kc = int(rng.integers(0, num_kc))
            current_time += pd.Timedelta(hours=float(rng.uniform(1, 72)))
            progress = min(step / max(seq_len - 1, 1), 1.0)
            p_correct = 1 / (1 + np.exp(-(ability + 2.5 * kc_mastery[kc] + 1.2 * progress - 1.4)))
            correct = int(rng.random() < p_correct)
            if correct:
                kc_mastery[kc] = min(1.0, kc_mastery[kc] + rng.uniform(0.02, 0.08))
            else:
                kc_mastery[kc] = min(1.0, kc_mastery[kc] + rng.uniform(0.00, 0.03))

            rows.append(
                {
                    "student_id": student_id,
                    "skill_id": f"KC_{kc:02d}",
                    "correct": correct,
                    "start_time": current_time.isoformat(),
                    "hint_count": int(rng.integers(0, 6)),
                    "attempt_count": int(rng.integers(1, 4)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[preprocess] 未找到真实数据，已生成模拟数据: {path}")
    return df


def generate_mock_states(n_students: int = 50, n_kc: int = 20) -> dict[int, dict[int, dict[str, float | str]]]:
    """Generate app-ready S/R states so Streamlit can launch without training artifacts."""
    rng = np.random.default_rng(SEED)
    states: dict[int, dict[int, dict[str, float | str]]] = {}
    for student_offset in range(n_students):
        student_id = 10000 + student_offset
        weak_count = int(rng.integers(3, 9))
        weak_kcs = set(rng.choice(n_kc, size=weak_count, replace=False).tolist())
        student_states = {}
        for kc_id in range(n_kc):
            if kc_id in weak_kcs:
                s_value = float(rng.uniform(0.1, 0.55))
                r_value = float(rng.uniform(0.1, 0.45))
            else:
                s_value = float(rng.uniform(0.45, 0.9))
                r_value = float(rng.uniform(0.35, 0.8))
            student_states[kc_id] = {
                "S": s_value,
                "R": r_value,
                "delta_t": float(rng.uniform(1, 20)),
                "kc_name": MATH_KC_NAMES[kc_id % len(MATH_KC_NAMES)],
            }
        states[student_id] = student_states
    return states


def load_raw_data() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        print(f"[preprocess] 读取原始数据: {RAW_FILE}")
        for encoding in ("utf-8", "utf-8-sig", "latin1", "gbk"):
            try:
                print(f"[preprocess] 尝试编码: {encoding}")
                return pd.read_csv(RAW_FILE, low_memory=False, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(RAW_FILE, low_memory=False, encoding_errors="replace")
    return generate_mock_data(RAW_FILE)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "user_id": "student_id",
        "problem_log_id": "student_id",
        "skill": "skill_id",
        "first_action": "start_time",
        "ms_first_response": "start_time",
    }
    for old, new in aliases.items():
        if new not in df.columns and old in df.columns:
            df[new] = df[old]

    required = ["student_id", "skill_id", "correct", "start_time"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    if "hint_count" not in df.columns:
        df["hint_count"] = 0
    if "attempt_count" not in df.columns:
        df["attempt_count"] = 1

    keep = ["student_id", "skill_id", "correct", "start_time", "hint_count", "attempt_count"]
    if "order_id" in df.columns:
        keep.append("order_id")
    return df[keep]


def compute_delta_t(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw = df["start_time"]
    # 优先尝试字符串解析（适配 ASSISTments 2009 的 "2009-09-17 10:03:24" 格式）
    parsed_time = pd.to_datetime(raw, errors="coerce")
    # 若字符串解析大面积失败，再尝试数值型毫秒时间戳（适配 ASSISTments 2017）
    if parsed_time.isna().mean() > 0.9:
        numeric_time = pd.to_numeric(raw, errors="coerce")
        parsed_time = pd.to_datetime(numeric_time, unit="ms", errors="coerce")
    df["_parsed_time"] = parsed_time
    df["_sort_time"] = parsed_time.fillna(pd.Timestamp("1970-01-01"))
    # 同一时间戳多条记录时需稳定次序，否则 diff 多为 0
    df["_orig_idx"] = np.arange(len(df))
    df = df.sort_values(["student_id", "_sort_time", "_orig_idx"]).reset_index(drop=True)

    delta_days = df.groupby("student_id")["_parsed_time"].diff().dt.total_seconds() / 86400.0
    first_mask = df.groupby("student_id").cumcount() == 0
    df["delta_t"] = delta_days.fillna(1.0).clip(lower=0.0)
    df.loc[first_mask, "delta_t"] = 0.0
    # 同学内相邻两行时间完全相同（或未解析 NaT）：给最小正间隔，避免 SR-DKT 衰减项恒为 1
    min_step_day = 1.0 / 86400.0  # 1 秒（天）
    dup_mask = (~first_mask) & (df["delta_t"] <= 0)
    df.loc[dup_mask, "delta_t"] = min_step_day
    nz = df.loc[~first_mask, "delta_t"]
    if len(nz):
        print(
            f"[preprocess] delta_t(天) 非首条: median={nz.median():.6f}, "
            f"p95={nz.quantile(0.95):.6f}, 仍<=0占比={(nz <= 0).mean():.4f}"
        )
    df = df.drop(columns=["_parsed_time", "_sort_time", "_orig_idx"])
    print("[preprocess] 已按 student_id + start_time 排序并计算 delta_t(天)")
    return df


def build_sequences(df: pd.DataFrame) -> dict:
    sequences = {}
    for student_id, group in df.groupby("student_id", sort=False):
        records = []
        for row in group.itertuples(index=False):
            records.append(
                (
                    int(row.kc_id),
                    int(row.correct),
                    float(row.delta_t),
                    float(row.hint_count),
                    float(row.attempt_count),
                )
            )
        sequences[student_id] = records
    return sequences


def split_by_student(sequences: dict) -> tuple[dict, dict, dict]:
    student_ids = list(sequences.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(student_ids)

    n_total = len(student_ids)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    train_ids = student_ids[:n_train]
    val_ids = student_ids[n_train : n_train + n_val]
    test_ids = student_ids[n_train + n_val :]

    return (
        {sid: sequences[sid] for sid in train_ids},
        {sid: sequences[sid] for sid in val_ids},
        {sid: sequences[sid] for sid in test_ids},
    )


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def main() -> None:
    set_seed()
    df = normalize_columns(load_raw_data())
    print(f"[preprocess] 原始记录数: {len(df)}")

    df = df.dropna(subset=["skill_id"])
    df = df[df["skill_id"].astype(str).str.strip() != ""].copy()
    print(f"[preprocess] 删除 skill_id 为空后记录数: {len(df)}")

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    # 只保留 correct 明确为 0 或 1 的记录（标准做法）
    df = df[df["correct"].isin([0, 1])].copy()
    print(f"[preprocess] 过滤 correct 非0/1 后记录数: {len(df)}")
    df["correct"] = df["correct"].astype(int)
    df["hint_count"] = pd.to_numeric(df["hint_count"], errors="coerce").fillna(0).clip(lower=0)
    df["attempt_count"] = pd.to_numeric(df["attempt_count"], errors="coerce").fillna(1).clip(lower=0)

    df = compute_delta_t(df)

    encoder = LabelEncoder()
    df["kc_id"] = encoder.fit_transform(df["skill_id"].astype(str))

    # 输出到 data/2009/ 子目录
    out_dir = DATA_DIR / "2009"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_pickle(encoder, out_dir / "skill_encoder.pkl")
    print(f"[preprocess] 已保存 skill_encoder.pkl，知识点数: {len(encoder.classes_)}")

    counts = df.groupby("student_id").size()
    valid_students = counts[counts >= 3].index
    df = df[df["student_id"].isin(valid_students)].copy()
    print(f"[preprocess] 仅保留行为数 >= 3 的学生，记录数: {len(df)}")

    sequences = build_sequences(df)
    train, val, test = split_by_student(sequences)
    save_pickle(train, out_dir / "train.pkl")
    save_pickle(val, out_dir / "val.pkl")
    save_pickle(test, out_dir / "test.pkl")

    lengths = [len(seq) for seq in sequences.values()]
    stats = {
        "num_students": len(sequences),
        "num_kc": int(len(encoder.classes_)),
        "avg_seq_len": float(np.mean(lengths)) if lengths else 0.0,
        "train_students": len(train),
        "val_students": len(val),
        "test_students": len(test),
    }
    (out_dir / "meta.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== 数据统计 ==========")
    print(f"总学生数: {stats['num_students']}")
    print(f"知识点数: {stats['num_kc']}")
    print(f"平均序列长度: {stats['avg_seq_len']:.2f}")
    print(f"Train/Val/Test 学生数: {len(train)}/{len(val)}/{len(test)}")
    print("============================")


if __name__ == "__main__":
    main()
