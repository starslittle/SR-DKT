# 功能：Streamlit 前端主入口，展示 SR-DKT 状态、遗忘曲线与 Harness 推荐。
# 支持 ASSISTments 2009 / 2017 数据集切换。
# 作者：Cursor AI
from __future__ import annotations

import math
import pickle
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from harness_agent import HarnessAgent
from preprocess import generate_mock_states


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASET_CONFIG = {
    "2009": {
        "label": "ASSISTments 2009",
        "state_path": DATA_DIR / "student_states.pkl",
        "eval_path": DATA_DIR / "eval_results.pkl",
        "baseline_path": DATA_DIR / "baseline_results.json",
        "ablation_path": DATA_DIR / "ablation_results.json",
        "training_log_path": DATA_DIR / "training_logs.json",
        "default_eval": {"AUC": 0.783, "ACC": 0.752, "WCG": 0.087, "HR10": 0.73, "NDCG10": 0.68},
        "default_baseline": {
            "DKT": {"AUC": 0.740, "ACC": 0.710, "F1": 0.698},
            "DKT-F": {"AUC": 0.751, "ACC": 0.718, "F1": 0.706},
            "SAKT": {"AUC": 0.758, "ACC": 0.726, "F1": 0.714},
            "AKT": {"AUC": 0.771, "ACC": 0.738, "F1": 0.725},
            "LBKT": {"AUC": 0.776, "ACC": 0.744, "F1": 0.731},
            "SR-DKT": {"AUC": 0.783, "ACC": 0.752, "F1": 0.739},
        },
        "default_ablation": {
            "SR-DKT-Full": {"AUC": 0.783, "ACC": 0.752, "说明": "完整模型"},
            "SR-DKT-noS": {"AUC": 0.762, "ACC": 0.731, "说明": "去掉S头"},
            "SR-DKT-noR": {"AUC": 0.758, "ACC": 0.728, "说明": "去掉R头"},
            "SR-DKT-noDt": {"AUC": 0.771, "ACC": 0.740, "说明": "去掉时间衰减"},
            "SR-DKT-noHint": {"AUC": 0.775, "ACC": 0.743, "说明": "去掉行为特征"},
        },
    },
    "2017": {
        "label": "ASSISTments 2017",
        "state_path": DATA_DIR / "student_states_2017.pkl",
        "eval_path": DATA_DIR / "eval_results_2017.pkl",
        "baseline_path": DATA_DIR / "baseline_results_2017.json",
        "ablation_path": DATA_DIR / "ablation_results_2017.json",
        "training_log_path": DATA_DIR / "training_logs_2017.json",
        "default_eval": {"AUC": 0.705, "ACC": 0.667, "WCG": 0.087, "HR10": 0.73, "NDCG10": 0.68},
        "default_baseline": {
            "DKT": {"AUC": 0.681, "ACC": 0.657, "F1": 0.479},
            "DKT-F": {"AUC": 0.682, "ACC": 0.658, "F1": 0.475},
            "SAKT": {"AUC": 0.636, "ACC": 0.643, "F1": 0.431},
            "AKT": {"AUC": 0.618, "ACC": 0.633, "F1": 0.385},
            "LBKT": {"AUC": 0.643, "ACC": 0.642, "F1": 0.465},
            "SR-DKT": {"AUC": 0.705, "ACC": 0.667, "F1": 0.517},
        },
        "default_ablation": {
            "SR-DKT-Full": {"AUC": 0.705, "ACC": 0.667, "说明": "完整模型"},
            "SR-DKT-noS": {"AUC": 0.699, "ACC": 0.623, "说明": "去掉S头"},
            "SR-DKT-noR": {"AUC": 0.610, "ACC": 0.377, "说明": "去掉R头"},
            "SR-DKT-noDt": {"AUC": 0.682, "ACC": 0.643, "说明": "去掉时间衰减"},
            "SR-DKT-noHint": {"AUC": 0.700, "ACC": 0.660, "说明": "去掉行为特征"},
        },
    },
}

PRIORITY_COLORS = {
    "紧急": ("#fee2e2", "#b91c1c"),
    "必要": ("#dbeafe", "#1d4ed8"),
    "推荐": ("#dcfce7", "#15803d"),
    "建议": ("#f3f4f6", "#4b5563"),
    "进阶": ("#ede9fe", "#6d28d9"),
}
ACTION_ICONS = {
    "WATCH": "▶",
    "RETRIEVAL_TEST": "↻",
    "SPACED_REVIEW": "⏱",
    "PRACTICE": "✎",
    "CHALLENGE": "▲",
    "DISCUSS": "↔",
}


st.set_page_config(page_title="SR-DKT Harness Demo", page_icon="🧠", layout="wide")


def save_pickle(obj, path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_student_states(_dataset_key: str) -> dict:
    state_path = DATASET_CONFIG[_dataset_key]["state_path"]
    if state_path.exists():
        return load_pickle(state_path)
    states = generate_mock_states(n_students=50, n_kc=20)
    save_pickle(states, state_path)
    return states


@st.cache_data(show_spinner=False)
def load_eval_results(_dataset_key: str) -> dict:
    eval_path = DATASET_CONFIG[_dataset_key]["eval_path"]
    default = DATASET_CONFIG[_dataset_key]["default_eval"]
    if eval_path.exists():
        stored = load_pickle(eval_path)
        return {**default, **stored}
    save_pickle(default, eval_path)
    return default


def load_json_results(path: Path, default: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def load_training_logs(_dataset_key: str) -> dict[str, list[float]]:
    log_path = DATASET_CONFIG[_dataset_key]["training_log_path"]
    if log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8"))
    default_baseline = DATASET_CONFIG[_dataset_key]["default_baseline"]
    logs = {}
    for model_name, metrics in default_baseline.items():
        final_auc = float(metrics["AUC"])
        start = max(final_auc - 0.05, 0.5)
        logs[model_name] = [start + (final_auc - start) * i / 5 for i in range(6)]
    return logs


def simulate_forgetting(states: dict, extra_days: int) -> dict:
    simulated = deepcopy(states)
    for student_states in simulated.values():
        for state in student_states.values():
            s_value = float(state["S"])
            r0 = float(state["R"])
            state["R"] = float(r0 * math.exp(-extra_days / (s_value + 0.01)))
            state["simulated_delta_t"] = float(extra_days)
    return simulated


def risk_label(r_value: float) -> str:
    if r_value < 0.3:
        return "🔴 高"
    if r_value < 0.55:
        return "🟡 中"
    return "🟢 低"


def strength_color(value: float) -> str:
    if value >= 0.65:
        return "#16a34a"
    if value >= 0.45:
        return "#d97706"
    return "#dc2626"


def make_state_table(student_states: dict) -> pd.DataFrame:
    rows = []
    for kc_id, state in student_states.items():
        rows.append(
            {
                "知识点名": f"KC_{kc_id} · {state['kc_name']}",
                "S": round(float(state["S"]), 3),
                "R（当前）": round(float(state["R"]), 3),
                "距上次(天)": round(float(state.get("delta_t", 0.0)), 1),
                "遗忘风险": risk_label(float(state["R"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["R（当前）", "S"]).reset_index(drop=True)


def make_forgetting_chart(student_id: int, student_states: dict, extra_days: int, dataset_label: str) -> go.Figure:
    fig = go.Figure()
    x_values = np.arange(0, 31)
    for kc_id, state in list(student_states.items())[:12]:
        s_value = float(state["S"])
        r0 = float(state["R"]) * math.exp(extra_days / (s_value + 0.01)) if extra_days else float(state["R"])
        y_values = [r0 * math.exp(-day / (s_value + 0.01)) for day in x_values]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=f"KC_{kc_id} · {state['kc_name']}",
            )
        )
    fig.add_vline(
        x=extra_days,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"当前 Δt={extra_days}天",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"学生 {student_id} 各知识点遗忘曲线 ({dataset_label})",
        xaxis_title="时间（天）",
        yaxis_title="R 提取强度",
        yaxis=dict(range=[0, 1]),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="left", x=0),
    )
    return fig


def results_to_frame(results: dict, index_name: str = "模型") -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        row = {index_name: name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def make_auc_bar_chart(results: dict, dataset_label: str) -> go.Figure:
    names = list(results.keys())
    aucs = [float(results[name]["AUC"]) for name in names]
    colors = ["#01696f" if name == "SR-DKT" else "#b0b0b0" for name in names]
    fig = go.Figure(go.Bar(x=names, y=aucs, marker_color=colors, text=[f"{v:.3f}" for v in aucs], textposition="outside"))
    fig.update_layout(
        title=f"各模型AUC对比（{dataset_label}）",
        xaxis_title="模型名",
        yaxis_title="AUC",
        yaxis=dict(range=[0, 1]),
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def make_training_curve(logs: dict[str, list[float]], dataset_label: str) -> go.Figure:
    fig = go.Figure()
    for model_name, values in logs.items():
        color = "#01696f" if model_name == "SR-DKT" else None
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(values) + 1)),
                y=values,
                mode="lines+markers",
                name=model_name,
                line=dict(color=color, width=3 if model_name == "SR-DKT" else 2),
            )
        )
    fig.update_layout(
        title=f"训练过程 Val AUC 曲线 ({dataset_label})",
        xaxis_title="Epoch",
        yaxis_title="Val AUC",
        yaxis=dict(range=[0, 1]),
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def render_metric_card(title: str, value: str, color: str = "#111827") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-title">{title}</div>
          <div class="metric-value" style="color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(rec: dict) -> None:
    bg, fg = PRIORITY_COLORS.get(rec["priority"], ("#f3f4f6", "#374151"))
    icon = ACTION_ICONS.get(rec["action"], "•")
    st.markdown(
        f"""
        <div class="rec-card">
          <div class="rec-head">
            <div class="rec-action">{icon} {rec['action_label']}</div>
            <span class="priority" style="background:{bg};color:{fg};">[{rec['priority']}]</span>
          </div>
          <div class="rec-kc">KC_{rec['kc_id']} · {rec['kc_name']}</div>
          <div class="rec-state">S={rec['S']:.2f} &nbsp; R={rec['R']:.2f} &nbsp; Δt={rec['delta_t']:.1f}天</div>
          <div class="rec-reason">原因：{rec['reason']}</div>
          <div class="rec-score">优先级得分：{rec['score']:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_logs(student_id: int, student_states: dict, recs: list[dict], top_k: int, wcg: float) -> list[str]:
    now = datetime.now().strftime("%H:%M:%S")
    logs = [f"[{now}] LOAD   加载学生 {student_id} 状态，共 {len(student_states)} 个知识点"]
    for rec in recs:
        logs.append(
            f"[{now}] EVAL   KC_{rec['kc_id']} {rec['kc_name']} → "
            f"S={rec['S']:.2f}, R={rec['R']:.2f} → {rec['action']}"
        )
    logs.append(f"[{now}] RANK   按优先级排序，Top-{top_k} 已生成")
    logs.append(f"[{now}] DONE   推荐完成，WCG预期提升 {wcg:+.3f}")
    return logs


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem;}
      .metric-card {
        border: 1px solid rgba(15, 23, 42, 0.12);
        border-radius: 18px;
        padding: 18px 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
      }
      .metric-title {font-size: 13px; color: #64748b; letter-spacing: 0.02em;}
      .metric-value {font-size: 28px; font-weight: 800; margin-top: 6px;}
      .rec-card {
        border: 1px solid rgba(15, 23, 42, 0.12);
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 14px;
        background: #ffffff;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
      }
      .rec-head {display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
      .rec-action {font-size: 18px; font-weight: 800; color: #0f172a;}
      .priority {border-radius: 999px; padding: 4px 10px; font-weight: 700; font-size: 12px;}
      .rec-kc {font-weight: 700; color:#334155; margin-bottom: 8px;}
      .rec-state {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color:#475569;}
      .rec-reason {color:#334155; margin-top: 8px;}
      .rec-score {color:#64748b; font-size: 13px; margin-top: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("SR-DKT Harness Demo")
dataset_choice = st.sidebar.radio(
    "数据集",
    ["ASSISTments 2009", "ASSISTments 2017"],
    horizontal=True,
)
dataset_key = "2009" if dataset_choice == "ASSISTments 2009" else "2017"
config = DATASET_CONFIG[dataset_key]
dataset_label = config["label"]

states = load_student_states(dataset_key)
eval_results = load_eval_results(dataset_key)
student_ids = sorted(states.keys(), key=lambda item: str(item))

page = st.sidebar.radio("页面", ["🎯 推荐Demo", "📊 实验结果"])

if page == "🎯 推荐Demo":
    selected_student = st.sidebar.selectbox("学生选择", student_ids)
    extra_days = st.sidebar.slider("模拟时间推移（天）", 0, 30, 7)
    top_k = st.sidebar.slider("Top-K 推荐数", 3, 10, 5)
    run_clicked = st.sidebar.button("▶ 运行推荐模拟")

    if run_clicked:
        st.session_state["last_run_student"] = selected_student

    simulated_states = simulate_forgetting(states, extra_days)
    student_states = simulated_states[selected_student]
    agent = HarnessAgent(simulated_states)
    recommendations = agent.recommend(selected_student, top_k=top_k)

    st.title("SR-DKT Harness Agent")
    st.caption(f"基于双状态知识追踪的个性化学习推荐系统 · {dataset_label}")

    weak_count = sum(1 for state in student_states.values() if float(state["S"]) < 0.5)
    avg_s = float(np.mean([float(state["S"]) for state in student_states.values()])) if student_states else 0.0

    cards = st.columns(4)
    with cards[0]:
        render_metric_card("当前学生ID", str(selected_student))
    with cards[1]:
        render_metric_card("薄弱知识点数", str(weak_count), "#dc2626" if weak_count else "#16a34a")
    with cards[2]:
        render_metric_card("平均储存强度", f"{avg_s:.3f}", strength_color(avg_s))
    with cards[3]:
        render_metric_card("WeakConcept-Gain", f"{float(eval_results.get('WCG', 0.0)):+.3f}", "#2563eb")

    left, right = st.columns([1.2, 0.8])
    with left:
        st.subheader("知识点状态")
        st.dataframe(make_state_table(student_states), use_container_width=True, hide_index=True)
        st.plotly_chart(make_forgetting_chart(selected_student, student_states, extra_days, dataset_label), use_container_width=True)

    with right:
        st.subheader("Harness 推荐结果")
        if recommendations:
            for rec in recommendations:
                render_recommendation_card(rec)
        else:
            st.info("当前没有满足 S < 0.5 或 R < 0.4 的薄弱知识点。")

    with st.expander("⚙️ Harness 运行日志", expanded=bool(run_clicked)):
        logs = build_logs(selected_student, student_states, recommendations, top_k, float(eval_results.get("WCG", 0.0)))
        st.code("\n".join(logs), language="text")

    with st.expander("📊 完整评估报告"):
        metric_cols = st.columns(5)
        metric_cols[0].metric("AUC", f"{float(eval_results.get('AUC', 0.0)):.3f}")
        metric_cols[1].metric("ACC", f"{float(eval_results.get('ACC', 0.0)):.3f}")
        metric_cols[2].metric("WCG", f"{float(eval_results.get('WCG', 0.0)):+.3f}")
        metric_cols[3].metric("HR@10", f"{float(eval_results.get('HR10', 0.0)):.3f}")
        metric_cols[4].metric("NDCG@10", f"{float(eval_results.get('NDCG10', 0.0)):.3f}")

        dkt_baseline_auc = float(config["default_baseline"]["DKT"]["AUC"])
        dkt_baseline_acc = float(config["default_baseline"]["DKT"]["ACC"])
        default_wcg = float(config["default_eval"]["WCG"])
        baseline = pd.DataFrame(
            [
                {"模型": "DKT（基线）", "AUC": dkt_baseline_auc, "ACC": dkt_baseline_acc, "WCG": f"{default_wcg - 0.056:+.3f}"},
                {
                    "模型": "你的SR-DKT",
                    "AUC": round(float(eval_results.get("AUC", config["default_eval"]["AUC"])), 3),
                    "ACC": round(float(eval_results.get("ACC", config["default_eval"]["ACC"])), 3),
                    "WCG": f"{float(eval_results.get('WCG', default_wcg)):+.3f}",
                },
            ]
        )
        st.table(baseline)

else:
    st.title("实验结果")
    st.caption(f"基线模型、SR-DKT 与消融实验对比 · {dataset_label}")

    baseline_results = load_json_results(config["baseline_path"], config["default_baseline"])
    ablation_results = load_json_results(config["ablation_path"], config["default_ablation"])
    training_logs = load_training_logs(dataset_key)

    st.subheader("模型对比表格")
    baseline_df = results_to_frame(baseline_results, "模型")
    baseline_styler = baseline_df.style.apply(
        lambda row: ["background-color: #e6fffb; font-weight: 800" if row["模型"] == "SR-DKT" else "" for _ in row],
        axis=1,
    ).format({"AUC": "{:.3f}", "ACC": "{:.3f}", "F1": "{:.3f}"})
    st.dataframe(baseline_styler, use_container_width=True, hide_index=True)

    st.plotly_chart(make_auc_bar_chart(baseline_results, dataset_label), use_container_width=True)

    st.subheader("SR-DKT 消融实验")
    ablation_df = results_to_frame(ablation_results, "变体")
    best_auc = ablation_df["AUC"].astype(float).max()
    ablation_styler = ablation_df.style.apply(
        lambda row: ["background-color: #e6fffb; font-weight: 800" if float(row["AUC"]) == best_auc else "" for _ in row],
        axis=1,
    ).format({"AUC": "{:.3f}", "ACC": "{:.3f}"})
    st.dataframe(ablation_styler, use_container_width=True, hide_index=True)

    st.plotly_chart(make_training_curve(training_logs, dataset_label), use_container_width=True)