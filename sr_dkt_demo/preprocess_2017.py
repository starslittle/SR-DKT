# 功能：读取 ASSISTments 2017 数据集，转换为 SR-DKT 可训练序列。
# 数据文件：anonymized_full_release_competition_dataset.csv
# 作者：Cursor AI
# 日期：2026-05-11
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
RAW_FILE = DATA_DIR / "anonymized_full_release_competition_dataset.csv"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_raw_data() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"找不到数据文件: {RAW_FILE}")
    for encoding in ("utf-8", "latin1"):
        try:
            print(f"[preprocess_2017] 尝试编码: {encoding}")
            return pd.read_csv(RAW_FILE, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(RAW_FILE, low_memory=False, encoding_errors="replace")


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "studentId": "student_id",
        "skill": "skill_id",
        "startTime": "start_time",
        "hintCount": "hint_count",
        "attemptCount": "attempt_count",
    }
    df = df.rename(columns=rename_map)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["skill_id"])
    df = df[df["skill_id"].astype(str).str.strip() != ""].copy()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df[df["correct"].isin([0, 1])].copy()
    df["correct"] = df["correct"].astype(int)
    df["hint_count"] = pd.to_numeric(df["hint_count"], errors="coerce").fillna(0).clip(lower=0)
    df["attempt_count"] = pd.to_numeric(df["attempt_count"], errors="coerce").fillna(1).clip(lower=0)
    return df


def compute_delta_t(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # startTime 是 Unix 时间戳（秒），转为 datetime
    parsed_time = pd.to_datetime(df["start_time"], unit="s", errors="coerce")
    df["_parsed_time"] = parsed_time
    df["_sort_time"] = parsed_time.fillna(pd.Timestamp("1970-01-01"))
    df["_orig_idx"] = np.arange(len(df))
    df = df.sort_values(["student_id", "_sort_time", "_orig_idx"]).reset_index(drop=True)

    delta_days = df.groupby("student_id")["_parsed_time"].diff().dt.total_seconds() / 86400.0
    first_mask = df.groupby("student_id").cumcount() == 0
    df["delta_t"] = delta_days.fillna(1.0).clip(lower=0.0)
    df.loc[first_mask, "delta_t"] = 0.0
    # 同学内相邻两行时间完全相同（或未解析 NaT）：给最小正间隔
    min_step_day = 1.0 / 86400.0  # 1 秒（天）
    dup_mask = (~first_mask) & (df["delta_t"] <= 0)
    df.loc[dup_mask, "delta_t"] = min_step_day

    nz = df.loc[~first_mask, "delta_t"]
    if len(nz):
        print(
            f"[preprocess_2017] delta_t(天) 非首条: median={nz.median():.6f}, "
            f"p95={nz.quantile(0.95):.6f}, 仍<=0占比={(nz <= 0).mean():.4f}"
        )

    df = df.drop(columns=["_parsed_time", "_sort_time", "_orig_idx"])
    print("[preprocess_2017] 已按 student_id + start_time 排序并计算 delta_t(天)")
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
    df = load_raw_data()
    print(f"[preprocess_2017] 原始记录数: {len(df)}")

    df = rename_columns(df)
    df = clean(df)
    print(f"[preprocess_2017] 清洗后记录数: {len(df)}")

    df = compute_delta_t(df)

    encoder = LabelEncoder()
    df["kc_id"] = encoder.fit_transform(df["skill_id"].astype(str))

    # 输出到 data/2017/ 子目录
    out_dir = DATA_DIR / "2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_pickle(encoder, out_dir / "skill_encoder_2017.pkl")
    print(f"[preprocess_2017] 已保存 skill_encoder_2017.pkl，知识点数: {len(encoder.classes_)}")

    counts = df.groupby("student_id").size()
    valid_students = counts[counts >= 3].index
    df = df[df["student_id"].isin(valid_students)].copy()
    print(f"[preprocess_2017] 仅保留行为数 >= 3 的学生，记录数: {len(df)}")

    sequences = build_sequences(df)
    train, val, test = split_by_student(sequences)
    save_pickle(train, out_dir / "train_2017.pkl")
    save_pickle(val, out_dir / "val_2017.pkl")
    save_pickle(test, out_dir / "test_2017.pkl")

    lengths = [len(seq) for seq in sequences.values()]
    stats = {
        "num_students": len(sequences),
        "num_kc": int(len(encoder.classes_)),
        "avg_seq_len": float(np.mean(lengths)) if lengths else 0.0,
        "train_students": len(train),
        "val_students": len(val),
        "test_students": len(test),
    }
    (out_dir / "meta_2017.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("========== ASSISTments 2017 数据统计 ==========")
    print(f"总学生数: {stats['num_students']}")
    print(f"知识点数: {stats['num_kc']}")
    print(f"平均序列长度: {stats['avg_seq_len']:.2f}")
    print(f"Train/Val/Test 学生数: {stats['train_students']}/{stats['val_students']}/{stats['test_students']}")
    print("===============================================")


if __name__ == "__main__":
    main()
