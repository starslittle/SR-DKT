#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOOCCubeX 数据预处理脚本
将学堂在线 MOOC 交互数据转换为 SR-DKT 可训练序列格式。

数据文件 (位于 data/mooc/):
  - relations/user-problem.json   (22.5GB, ~1.33亿条做题记录)
  - relations/user-video.json     (3.2GB, ~31万条视频观看记录)
  - entities/problem.json         (1.3GB, 问题元数据, 含 exercise_id)
  - entities/video.json           (608MB, 视频片段信息)

输出文件 (保存至 data/mooc/):
  - train.pkl, val.pkl, test.pkl
    格式: dict[str, list[tuple]]
    tuple: (kc_id, correct, delta_t, hint, attempt, watch_ratio, is_replay)
  - skill_encoder_mooc.pkl (LabelEncoder)
  - meta_mooc.json (数据集统计信息)

核心设计:
  - 流式读取 user-problem.json, 避免一次性加载 22.5GB 文件
  - 使用 numpy 结构化数组存储每个用户的交互记录, 降低内存占用
  - 自定义时间解析器 (纯 Python, 避免 timezone 开销)

用法:
  python preprocess_mooc.py
  python preprocess_mooc.py --chunk-size 10000000 --min-interactions 5

作者: SR-DKT 项目组
日期: 2026-06-06
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 配置
# ============================================================
SEED = 42
MOOC_DIR = Path(__file__).resolve().parent / "data" / "mooc"
ENTITIES_DIR = MOOC_DIR / "entities"
RELATIONS_DIR = MOOC_DIR / "relations"

CHUNK_SIZE = 5_000_000          # 每次读取行数 (可按内存调整)
MIN_INTERACTIONS = 3            # 最低交互次数阈值


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_datetime_to_epoch(s: str) -> float:
    """
    快速解析 "YYYY-MM-DD HH:MM:SS" 格式的 datetime 字符串。

    比 datetime.strptime 快约 5x, 因为跳过了 timezone/locale 处理。
    返回 Unix epoch 秒数 (float)。解析失败返回 NaN。
    """
    try:
        year = int(s[0:4])
        month = int(s[5:7])
        day = int(s[8:10])
        hour = int(s[11:13])
        minute = int(s[14:16])
        second = int(s[17:19])

        # 累积天数 (非闰年近似, 足够用于排序和 delta_t 计算)
        days_in_month = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
        total_days = (year - 1970) * 365 + (year - 1969) // 4
        total_days += days_in_month[month - 1] + day - 1
        # 闰年修正 (简化版)
        if month > 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            total_days += 1

        return float(
            total_days * 86400 + hour * 3600 + minute * 60 + second
        )
    except (ValueError, IndexError):
        return float("nan")


# ============================================================
# Step 1: 加载 problem_id → exercise_id 映射
# ============================================================
def load_problem_to_exercise() -> dict[int, str]:
    """
    加载 problem_id → exercise_id 映射。

    优先从 relations/exercise-problem.txt 加载 (覆盖率 ~99.95%),
    再从 entities/problem.json 补充遗漏。

    exercise-problem.txt 使用二进制模式读取, 因为该文件中的 Tab 分隔符
    在某些环境下无法被 str.split('\\t') 正确解析。

    Returns:
        dict: {problem_id (int): exercise_id (str)}
    """
    print("=" * 60)
    print("[Step 1] 加载问题-练习映射...")
    p2e: dict[int, str] = {}

    # 主映射源: exercise-problem.txt (覆盖率 ~99.95%)
    txt_path = RELATIONS_DIR / "exercise-problem.txt"
    if txt_path.exists():
        count = 0
        with open(txt_path, "rb") as f:
            for raw_line in f:
                # 查找 Tab 字节 (0x09): 限定在前 20 字节内搜索
                # 因为 Ex_XXXXX/Pm_XXXXX 部分不超过 15 字符
                tab_pos = -1
                search_limit = min(len(raw_line), 20)
                for i in range(search_limit):
                    if raw_line[i] == 0x09:
                        tab_pos = i
                        break
                if tab_pos < 0:
                    continue
                ex_id = raw_line[:tab_pos].decode("utf-8", errors="replace")
                pm_part = raw_line[tab_pos + 1:].strip().decode(
                    "utf-8", errors="replace"
                )
                if pm_part.startswith("Pm_"):
                    try:
                        pm_id = int(pm_part[3:])
                        p2e[pm_id] = ex_id
                        count += 1
                    except ValueError:
                        pass
        print("  exercise-problem.txt: {:,} 条映射".format(count))
    else:
        print("  [WARN] exercise-problem.txt 不存在")

    # 补充映射源: problem.json (补充 exercise-problem.txt 中未覆盖的问题)
    json_path = ENTITIES_DIR / "problem.json"
    json_added = 0
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                pid = obj.get("problem_id")
                eid = obj.get("exercise_id")
                if pid is not None and eid and int(pid) not in p2e:
                    p2e[int(pid)] = eid
                    json_added += 1
        print("  problem.json 补充: {:,} 条映射".format(json_added))

    print("  合计: {:,} 个问题 → 练习映射".format(len(p2e)))
    return p2e


# ============================================================
# Step 1.5: 加载 video_id → ccid 映射
# ============================================================
def load_video_id_to_ccid() -> dict[str, str]:
    """
    加载 video_id (V_XXXXX) → ccid (hex) 映射。

    user-video.json 中的 video_id 使用 "V_XXXXX" 格式,
    而 video.json (entities) 中的视频 ID 使用 ccid (hex 字符串)。
    两者是完全不同的 ID 系统, 需要 video_id-ccid.txt 桥接映射,
    否则无法查询视频时长, 导致 watch_ratio 恒为 0。

    Returns:
        dict: {video_id (str): ccid (str)}
    """
    print("=" * 60)
    print("[Step 1.5] 加载视频 ID 映射 (video_id-ccid.txt)...")
    vid_to_ccid: dict[str, str] = {}
    txt_path = RELATIONS_DIR / "video_id-ccid.txt"

    if txt_path.exists():
        count = 0
        with open(txt_path, "rb") as f:
            for raw_line in f:
                # 查找 Tab 字节 (0x09): 限定在前 20 字节内搜索
                # 因为 V_XXXXX 部分不超过 15 字符, Tab 一定在前面
                tab_pos = -1
                search_limit = min(len(raw_line), 20)
                for i in range(search_limit):
                    if raw_line[i] == 0x09:
                        tab_pos = i
                        break
                if tab_pos < 0:
                    continue
                vid = raw_line[:tab_pos].decode("utf-8", errors="replace")
                ccid = raw_line[tab_pos + 1:].strip().decode(
                    "utf-8", errors="replace"
                )
                vid_to_ccid[vid] = ccid
                count += 1

        print("  已加载 {:,} 条 video_id → ccid 映射".format(count))
    else:
        print("  [WARN] video_id-ccid.txt 不存在")

    return vid_to_ccid


# ============================================================
# Step 2: 加载视频时长索引
# ============================================================
def load_video_durations() -> dict[str, float]:
    """
    从 entities/video.json 加载每个视频的总时长 (秒)。

    视频总时长 = end 数组中的最大值。

    Returns:
        dict: {video_ccid (str): total_duration_seconds (float)}
    """
    print("=" * 60)
    print("[Step 2] 加载视频时长索引 (video.json)...")
    durations: dict[str, float] = {}
    skipped = 0
    path = ENTITIES_DIR / "video.json"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ccid = obj.get("ccid")
            ends = obj.get("end")
            if ccid and ends and len(ends) > 0:
                durations[ccid] = float(max(ends))
            else:
                skipped += 1
    print("  已加载 {:,} 个视频时长, {:,} 个跳过".format(
        len(durations), skipped))
    return durations


# ============================================================
# Step 3: 处理用户视频观看数据
# ============================================================
def process_user_videos(
    video_durations: dict[str, float],
    vid_to_ccid: dict[str, str] | None = None,
) -> dict[str, list[tuple[str, float, int, float]]]:
    """
    处理 relations/user-video.json, 提取每个用户观看过的视频信息。

    对于每个用户的每个视频:
      - watch_ratio: 观看时长 / 视频总时长, 截断到 [0, 1]
      - is_replay: 是否有重叠的观看片段 (1=回看过)

    重要: user-video.json 中的 video_id 使用 "V_XXXXX" 格式,
    而 video.json (entities) 的时长索引使用 ccid (hex) 作为键。
    必须通过 vid_to_ccid 映射将 V_XXXXX 转换为 ccid,
    才能正确查询视频时长, 否则 watch_ratio 恒为 0。

    Args:
        video_durations: {ccid (hex): duration_seconds} — 来自 video.json
        vid_to_ccid: {V_XXXXX: ccid (hex)} — 来自 video_id-ccid.txt

    Returns:
        dict: {
            user_id: [
                (video_id, watch_ratio, is_replay, last_watch_epoch),
                ...
            ]
        }
    """
    print("=" * 60)
    print("[Step 3] 处理用户视频数据 (user-video.json)...")
    t0 = time.time()

    # 如果没有提供 vid_to_ccid 映射, 尝试加载
    if vid_to_ccid is None:
        vid_to_ccid = load_video_id_to_ccid()

    duration_hit = 0  # 成功通过 vid→ccid 查到时长的计数
    duration_miss = 0  # 无法查到时长的计数

    user_videos: dict[str, list[tuple[str, float, int, float]]] = {}
    record_count = 0
    video_interaction_count = 0

    path = RELATIONS_DIR / "user-video.json"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user_id = obj.get("user_id")
            if not user_id:
                continue

            video_entries: list[tuple[str, float, int, float]] = []

            for item in obj.get("seq", []):
                vid = item.get("video_id")
                if not vid:
                    continue

                segments = item.get("segment", [])
                if not segments:
                    continue

                # 计算总观看时长
                total_watched = 0.0
                max_epoch = 0.0
                for seg in segments:
                    start = float(seg.get("start_point", 0))
                    end = float(seg.get("end_point", 0))
                    total_watched += max(end - start, 0.0)
                    epoch = seg.get("local_start_time", 0)
                    if isinstance(epoch, (int, float)) and epoch > max_epoch:
                        max_epoch = float(epoch)

                # 视频总时长: V_XXXXX → ccid (hex) → duration
                ccid = vid_to_ccid.get(vid, "")
                duration = video_durations.get(ccid, 0.0) if ccid else 0.0
                if duration > 0:
                    duration_hit += 1
                else:
                    duration_miss += 1
                watch_ratio = (
                    min(total_watched / duration, 1.0)
                    if duration > 0
                    else 0.0
                )

                # 检测回放: 按起始时间排序后检查片段重叠
                is_replay = 0
                if len(segments) > 1:
                    sorted_segs = sorted(
                        segments,
                        key=lambda s: float(s.get("start_point", 0)),
                    )
                    for j in range(1, len(sorted_segs)):
                        prev_end = float(
                            sorted_segs[j - 1].get("end_point", 0)
                        )
                        curr_start = float(
                            sorted_segs[j].get("start_point", 0)
                        )
                        if curr_start < prev_end:
                            is_replay = 1
                            break

                video_entries.append((vid, watch_ratio, is_replay, max_epoch))
                video_interaction_count += 1

            if video_entries:
                user_videos[user_id] = video_entries
            record_count += 1

            if record_count % 50_000 == 0:
                print("  已处理 {:,} 条视频记录...".format(record_count))

    print("  完成: {:,} 个用户, {:,} 次视频交互, 耗时 {:.1f}s".format(
        len(user_videos), video_interaction_count, time.time() - t0))
    print("  视频时长查询: {:,} 次命中, {:,} 次未命中".format(
        duration_hit, duration_miss))
    return user_videos


# ============================================================
# Step 4: 流式加载用户做题数据
# ============================================================
def load_user_problems_streaming(
    problem_to_exercise: dict[int, str],
    chunk_size: int = CHUNK_SIZE,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """
    流式读取 user-problem.json (22.5GB), 按用户分组并转为 numpy 结构化数组。

    每个用户的交互记录存储为 numpy structured array:
      dtype: [(pid, i8), (epoch, f8), (correct, i1), (attempt, i2)]

    仅保留能映射到 exercise_id 的问题。
    同时统计每个 exercise_id 的出现次数, 用于后续 top-K 过滤。

    Args:
        problem_to_exercise: problem_id → exercise_id 映射
        chunk_size: 每次处理的行数

    Returns:
        tuple: (
            {user_id: numpy_structured_array},
            {exercise_id: interaction_count},
        )
    """
    print("=" * 60)
    print("[Step 4] 流式加载做题数据 (user-problem.json)...")
    print("  文件大小 ~22.5GB, chunk_size={:,}".format(chunk_size))
    t0 = time.time()

    INTERACTION_DTYPE = np.dtype([
        ("pid", np.int64),
        ("epoch", np.float64),
        ("correct", np.int8),
        ("attempt", np.int16),
    ])

    # 阶段 1: 用 list 累积, 之后转 numpy
    user_data: dict[str, list] = {}
    exercise_counts: dict[str, int] = {}  # exercise_id → 出现次数
    total_lines = 0
    kept_lines = 0
    skipped_no_exercise = 0
    skipped_parse_error = 0

    path = RELATIONS_DIR / "user-problem.json"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1

            if total_lines % chunk_size == 0:
                elapsed = time.time() - t0
                speed = total_lines / elapsed if elapsed > 0 else 0
                print(
                    "  {:,} 行已处理 ({:,.0f} 行/s), "
                    "已保留 {:,}, 当前用户数 {:,}".format(
                        total_lines, speed, kept_lines, len(user_data)
                    )
                )

            # 快速 JSON 解析: 直接提取需要的字段
            # 完整解析 (json.loads 对于简单对象足够快)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_parse_error += 1
                continue

            # 提取 problem_id → 转为整数
            pid_str = obj.get("problem_id")
            if not pid_str:
                skipped_parse_error += 1
                continue
            # "Pm_6906522" → 6906522
            pid = int(pid_str[3:]) if pid_str.startswith("Pm_") else 0

            # 检查是否有 exercise_id 映射
            exercise_id = problem_to_exercise.get(pid)
            if not exercise_id:
                skipped_no_exercise += 1
                continue

            # 统计 exercise 出现次数 (用于后续 top-K 过滤)
            exercise_counts[exercise_id] = (
                exercise_counts.get(exercise_id, 0) + 1
            )

            user_id = obj.get("user_id")
            if not user_id:
                skipped_parse_error += 1
                continue

            correct_raw = obj.get("is_correct", 0)
            correct = 1 if correct_raw else 0

            attempt_raw = obj.get("attempts", 1)
            attempt = max(int(attempt_raw) if attempt_raw is not None else 1, 1)

            submit_time = obj.get("submit_time", "")
            epoch = parse_datetime_to_epoch(submit_time)
            if epoch != epoch:  # NaN check
                skipped_parse_error += 1
                continue

            # 追加到用户列表
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append((pid, epoch, correct, attempt))
            kept_lines += 1

    # 阶段 2: 转换为 numpy 结构化数组
    print("  转换 {:,} 个用户的数据为 numpy 数组...".format(len(user_data)))
    user_arrays: dict[str, np.ndarray] = {}
    for uid, records in user_data.items():
        arr = np.empty(len(records), dtype=INTERACTION_DTYPE)
        for i, (p, e, c, a) in enumerate(records):
            arr[i] = (p, e, c, a)
        user_arrays[uid] = arr
    # 释放原始 list 内存
    del user_data

    elapsed = time.time() - t0
    print("  完成: {:,} 个用户, {:,} 条有效交互".format(
        len(user_arrays), kept_lines))
    print("  跳过: {:,} 无 exercise_id, {:,} 解析失败".format(
        skipped_no_exercise, skipped_parse_error))
    print("  不同 exercise_id 数: {:,}".format(len(exercise_counts)))
    print("  总耗时: {:.1f}s".format(elapsed))
    return user_arrays, exercise_counts


# ============================================================
# Step 5: 构建序列
# ============================================================
def build_sequences(
    user_data: dict[str, np.ndarray],
    problem_to_exercise: dict[int, str],
    user_videos: dict[str, list[tuple[str, float, int, float]]],
) -> dict[str, list[tuple]]:
    """
    为每个用户构建有序交互序列。

    对每个用户:
      1. 按时间排序
      2. 计算 delta_t (天)
      3. 映射 problem_id → exercise_id (作为 KC)
      4. 合并视频特征 (按最近观看时间匹配)

    输出 tuple 格式:
      (kc_id_str, correct, delta_t, hint, attempt, watch_ratio, is_replay)

    Returns:
        dict: {user_id: [(kc_str, correct, delta_t, hint, attempt,
                          watch_ratio, is_replay), ...]}
    """
    print("=" * 60)
    print("[Step 5] 构建交互序列...")
    t0 = time.time()

    sequences: dict[str, list[tuple]] = {}
    processed = 0

    for user_id, arr in user_data.items():
        # 按时间排序
        sort_idx = np.argsort(arr["epoch"], kind="mergesort")
        sorted_arr = arr[sort_idx]

        # 计算 delta_t (天)
        n = len(sorted_arr)
        delta_t_days = np.empty(n, dtype=np.float64)
        delta_t_days[0] = 0.0
        if n > 1:
            diff_seconds = np.diff(sorted_arr["epoch"])
            # 同时间戳给最小正间隔 (1 秒 = 1/86400 天)
            diff_seconds = np.where(diff_seconds <= 0, 1.0, diff_seconds)
            delta_t_days[1:] = diff_seconds / 86400.0

        # 获取用户视频数据 (按最近观看时间排序)
        u_videos = user_videos.get(user_id, [])
        # 按最后观看时间排序, 用于二分查找
        u_videos_sorted = sorted(u_videos, key=lambda x: x[3])

        records: list[tuple] = []
        for i in range(n):
            pid = int(sorted_arr["pid"][i])
            exercise_id = problem_to_exercise.get(pid)
            if not exercise_id:
                continue

            correct = int(sorted_arr["correct"][i])
            dt = float(delta_t_days[i])
            attempt = int(sorted_arr["attempt"][i])
            current_epoch = float(sorted_arr["epoch"][i])

            # hint: MOOC 数据无 hint 信息, 用 0 填充
            hint = 0.0

            # 合并视频特征: 找到该做题时间之前最近一次视频观看
            watch_ratio = 0.0
            is_replay = 0
            if u_videos_sorted:
                # 二分查找: 找最后一个 epoch <= current_epoch 的视频
                lo, hi = 0, len(u_videos_sorted) - 1
                best = -1
                while lo <= hi:
                    mid = (lo + hi) >> 1
                    if u_videos_sorted[mid][3] <= current_epoch:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1

                if best >= 0:
                    _, wr, rep, _ = u_videos_sorted[best]
                    watch_ratio = wr
                    is_replay = rep

            records.append((
                exercise_id,    # str: 作为 KC 名称, 之后用 LabelEncoder 编码
                correct,
                dt,
                hint,
                float(attempt),
                watch_ratio,
                is_replay,
            ))

        if records:
            sequences[user_id] = records

        processed += 1
        if processed % 100_000 == 0:
            print("  已处理 {:,}/{:,} 个用户...".format(
                processed, len(user_data)))

    print("  完成: {:,} 个用户有序序列, 耗时 {:.1f}s".format(
        len(sequences), time.time() - t0))
    return sequences


# ============================================================
# 数据划分
# ============================================================
def split_by_student(
    sequences: dict,
) -> tuple[dict, dict, dict]:
    """按 80/10/10 比例划分 train/val/test"""
    student_ids = list(sequences.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(student_ids)

    n_total = len(student_ids)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    train_ids = student_ids[:n_train]
    val_ids = student_ids[n_train:n_train + n_val]
    test_ids = student_ids[n_train + n_val:]

    return (
        {sid: sequences[sid] for sid in train_ids},
        {sid: sequences[sid] for sid in val_ids},
        {sid: sequences[sid] for sid in test_ids},
    )


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOOCCubeX 数据预处理 → SR-DKT 可训练序列"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help="流式读取每次处理的行数 (默认 {:,})".format(CHUNK_SIZE),
    )
    parser.add_argument(
        "--min-interactions", type=int, default=MIN_INTERACTIONS,
        help="最少交互次数阈值 (默认 {})".format(MIN_INTERACTIONS),
    )
    parser.add_argument(
        "--max-kc", type=int, default=5000,
        help="最大知识点数: 保留出现频率最高的 N 个 exercise_id, "
             "其余归入 OTHER (默认 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="随机种子 (默认 {})".format(SEED),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    total_start = time.time()

    print("=" * 60)
    print("MOOCCubeX → SR-DKT 预处理")
    print("=" * 60)
    print("数据目录: {}".format(MOOC_DIR))
    print("chunk_size: {:,}".format(args.chunk_size))
    print("min_interactions: {}".format(args.min_interactions))
    print("max_kc: {:,}".format(args.max_kc))
    print("seed: {}".format(args.seed))
    print()

    # 验证数据目录
    if not MOOC_DIR.exists():
        print("[ERROR] 数据目录不存在: {}".format(MOOC_DIR))
        sys.exit(1)
    for sub in ["entities", "relations"]:
        sub_dir = MOOC_DIR / sub
        if not sub_dir.exists():
            print("[ERROR] 子目录不存在: {}".format(sub_dir))
            sys.exit(1)

    # Step 1: 问题 → 练习映射
    problem_to_exercise = load_problem_to_exercise()

    # Step 2: 视频时长索引
    video_durations = load_video_durations()

    # Step 2.5: 加载 video_id → ccid 映射 (桥接两种 ID 格式)
    vid_to_ccid = load_video_id_to_ccid()

    # Step 3: 用户视频特征
    user_videos = process_user_videos(video_durations, vid_to_ccid)
    # 释放视频时长索引和 ID 映射 (不再需要)
    del video_durations
    del vid_to_ccid

    # Step 4: 流式加载做题数据
    user_data, exercise_counts = load_user_problems_streaming(
        problem_to_exercise,
        chunk_size=args.chunk_size,
    )

    # Step 4.5: Top-K KC 过滤
    # 保留出现频率最高的 max_kc 个 exercise_id, 其余归入 "OTHER"
    print()
    print("=" * 60)
    print("[Step 4.5] Top-{:} KC 过滤...".format(args.max_kc))
    sorted_exercises = sorted(
        exercise_counts.items(), key=lambda x: x[1], reverse=True
    )
    if len(sorted_exercises) > args.max_kc:
        top_kc_set = set(
            eid for eid, _ in sorted_exercises[:args.max_kc]
        )
        other_count = sum(
            cnt for _, cnt in sorted_exercises[args.max_kc:]
        )
        print("  原始 exercise_id 数: {:,}".format(len(sorted_exercises)))
        print("  保留 top {:,} (覆盖 {:.1f}% 交互)".format(
            args.max_kc,
            sum(cnt for _, cnt in sorted_exercises[:args.max_kc])
            / max(sum(exercise_counts.values()), 1) * 100,
        ))
        print("  OTHER 类交互数: {:,}".format(other_count))
    else:
        top_kc_set = set(eid for eid, _ in sorted_exercises)
        print("  exercise_id 总数 {:,} <= max_kc {:}, 无需过滤".format(
            len(sorted_exercises), args.max_kc))

    # 构建过滤后的 problem_to_exercise 映射
    filtered_p2e: dict[int, str] = {}
    for pid, eid in problem_to_exercise.items():
        filtered_p2e[pid] = eid if eid in top_kc_set else "OTHER"
    del top_kc_set

    # Step 5: 构建序列
    sequences = build_sequences(
        user_data, filtered_p2e, user_videos
    )
    # 释放原始数据和视频特征 (不再需要)
    del user_data
    del user_videos
    del filtered_p2e

    print()
    print("=" * 60)
    print("[Step 6] 编码 KC 并保存...")

    # 统计过滤前
    all_lengths = [len(s) for s in sequences.values()]
    print("  过滤前: {:,} 个用户, 平均序列长度 {:.1f}".format(
        len(sequences),
        float(np.mean(all_lengths)) if all_lengths else 0,
    ))

    # 过滤: 保留交互数 >= min_interactions 的学生
    sequences = {
        uid: seq
        for uid, seq in sequences.items()
        if len(seq) >= args.min_interactions
    }
    print("  过滤后 (>= {} 次交互): {:,} 个用户".format(
        args.min_interactions, len(sequences)))

    if not sequences:
        print("[ERROR] 过滤后无有效用户, 请检查数据或降低 min_interactions")
        sys.exit(1)

    # 用 LabelEncoder 编码 exercise_id → 整数 kc_id
    # 优化: 先收集 unique exercise_id 做 fit, 再建 dict 映射做 O(1) 查表
    # 避免 O(N) 的 all_exercise_ids 和 exercise_positions 中间数组
    all_exercise_ids = set()
    for seq in sequences.values():
        for record in seq:
            all_exercise_ids.add(record[0])

    encoder = LabelEncoder()
    encoder.fit(list(all_exercise_ids))
    num_kc = len(encoder.classes_)
    print("  知识点数 (exercise_id 编码): {:,}".format(num_kc))

    # 构建 dict 映射: exercise_id (str) → kc_id (int), O(1) 查表
    id_to_kc: dict[str, int] = {}
    for eid, kc_id in zip(encoder.classes_, range(num_kc)):
        id_to_kc[eid] = int(kc_id)
    del all_exercise_ids  # 释放内存

    # 就地编码: 直接修改 sequences 中的 tuple, 不建第二份 copy
    # 这是避免 Step 6 OOM 的关键优化
    for uid, seq in sequences.items():
        for i, record in enumerate(seq):
            seq[i] = (id_to_kc[record[0]],) + record[1:]
    del id_to_kc

    # 保存 encoder
    MOOC_DIR.mkdir(parents=True, exist_ok=True)
    save_pickle(encoder, MOOC_DIR / "skill_encoder_mooc.pkl")
    print("  已保存 skill_encoder_mooc.pkl")

    # 划分数据集 (sequences 已编码为 int kc_id)
    train, val, test = split_by_student(sequences)
    del sequences

    # 保存 pkl 文件
    save_pickle(train, MOOC_DIR / "train.pkl")
    save_pickle(val, MOOC_DIR / "val.pkl")
    save_pickle(test, MOOC_DIR / "test.pkl")
    print("  已保存 train.pkl, val.pkl, test.pkl")

    # 统计信息
    train_lengths = [len(s) for s in train.values()]
    val_lengths = [len(s) for s in val.values()]
    test_lengths = [len(s) for s in test.values()]
    all_lengths_final = train_lengths + val_lengths + test_lengths

    # 统计视频特征覆盖率
    total_interactions = sum(all_lengths_final)
    video_covered = 0
    for seq in train.values():
        for rec in seq:
            if rec[5] > 0 or rec[6] > 0:  # watch_ratio > 0 or is_replay > 0
                video_covered += 1
    for seq in val.values():
        for rec in seq:
            if rec[5] > 0 or rec[6] > 0:
                video_covered += 1
    for seq in test.values():
        for rec in seq:
            if rec[5] > 0 or rec[6] > 0:
                video_covered += 1

    video_coverage = (
        video_covered / total_interactions
        if total_interactions > 0
        else 0
    )

    stats = {
        "dataset": "MOOCCubeX",
        "num_students": len(train) + len(val) + len(test),
        "num_kc": num_kc,
        "max_kc_threshold": args.max_kc,
        "avg_seq_len": float(np.mean(all_lengths_final)) if all_lengths_final else 0,
        "median_seq_len": float(np.median(all_lengths_final)) if all_lengths_final else 0,
        "min_seq_len": int(min(all_lengths_final)) if all_lengths_final else 0,
        "max_seq_len": int(max(all_lengths_final)) if all_lengths_final else 0,
        "total_interactions": total_interactions,
        "video_feature_coverage": round(video_coverage, 4),
        "min_interactions_filter": args.min_interactions,
        "train_students": len(train),
        "val_students": len(val),
        "test_students": len(test),
        "sequence_format": "(kc_id, correct, delta_t, hint, attempt, watch_ratio, is_replay)",
        "kc_source": "exercise_id from exercise-problem.txt + problem.json (top-K filtered)",
    }
    meta_path = MOOC_DIR / "meta_mooc.json"
    meta_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("  已保存 meta_mooc.json")

    # 打印统计
    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print("MOOCCubeX 数据集统计")
    print("=" * 60)
    print("  总学生数:         {:,}".format(stats["num_students"]))
    print("  知识点数:         {:,}".format(num_kc))
    print("  总交互数:         {:,}".format(total_interactions))
    print("  平均序列长度:     {:.1f}".format(stats["avg_seq_len"]))
    print("  中位数序列长度:   {:.1f}".format(stats["median_seq_len"]))
    print("  最短/最长序列:    {}/{}".format(stats["min_seq_len"], stats["max_seq_len"]))
    print("  视频特征覆盖率:   {:.2%}".format(video_coverage))
    print("-" * 60)
    print("  Train 学生数:     {:,}".format(len(train)))
    print("  Val   学生数:     {:,}".format(len(val)))
    print("  Test  学生数:     {:,}".format(len(test)))
    print("-" * 60)
    print("  总耗时:           {:.1f}s ({:.1f} min)".format(
        total_elapsed, total_elapsed / 60))
    print("=" * 60)
    print()
    print("输出文件:")
    print("  {}/train.pkl".format(MOOC_DIR))
    print("  {}/val.pkl".format(MOOC_DIR))
    print("  {}/test.pkl".format(MOOC_DIR))
    print("  {}/skill_encoder_mooc.pkl".format(MOOC_DIR))
    print("  {}/meta_mooc.json".format(MOOC_DIR))


if __name__ == "__main__":
    main()
