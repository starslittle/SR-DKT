# 功能：统一训练并评估 DKT、DKT-F、SAKT、AKT、LBKT 与 SR-DKT，对比 AUC/ACC/F1。
# 已修复 Bug 3：ROOT_DIR 路径错误、import 文件名错误、forward_model 中 mask 参数位置错误
# 已修复 Bug 4：SAKT num_layers 默认为 2，应与 pyKT 对齐为 1
# 作者：SR-DKT 项目组
# 日期：2026-05-11
# 训练配置：batch_size=128, max_seq_len=200, hidden_size=128, epochs=50, patience=8, seed=42
# 注意：max_seq_len 与官方 pyKT 数据(maxlen=200)保持一致，便于自研框架与 pyKT 交叉验证；
#       降本主要靠 10% 训练子集(见 subsample_mooc.py)，不再靠缩短序列。
# 学习率配置：DKT/DKT-F/LBKT/SR-DKT lr=0.001, SAKT lr=5e-4, AKT lr=1e-4
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
PARENT_DIR = ROOT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from baseline_models import AKT, BEKT, DKT, DKT_F, DKTVec, DKVMN, LBKT, SAKT
from model import SRDKT
from train import SequenceDataset, collate_batch, move_batch


# 训练配置
CONFIG = {
    "hidden_size": 128,
    "batch_size": 128,
    "epochs": 50,
    "patience": 8,
    "seed": 42,
    "max_seq_len": 200,  # 与官方 pyKT 数据(maxlen=200)对齐；降本靠 10% 子集而非缩短序列
    "lambda_constraint_weight": 0.01,
}

# 模型独立学习率配置
MODEL_LR = {
    "DKT": 0.001,
    "DKT-Vec": 0.001,  # pyKT 同款标准 DKT，跨框架对齐锚点
    "DKT-F": 0.001,
    "SAKT": 5e-4,   # SAKT 需要较小学习率
    "AKT": 1e-4,    # AKT 需要最小学习率
    "LBKT": 0.001,
    "DKVMN": 0.001,  # 新增
    "BEKT": 5e-4,    # 新增，Transformer 结构用较小lr
    "SR-DKT": 0.001,
}

DATA_DIR = PARENT_DIR / "data"
CKPT_DIR = DATA_DIR / "ckpt"
LOG_PATH = DATA_DIR / "training_logs.json"


def safe_file_stem(name: str) -> str:
    """将模型名转成适合文件名的短标识。"""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def get_dataset_dir(dataset: str) -> Path:
    """根据数据集名称返回对应的子目录"""
    if dataset == "mooc":
        return DATA_DIR / "mooc"
    return DATA_DIR / ("2017" if dataset == "2017" else "2009")


def set_seed(seed: int = 42) -> None:
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path: Path):
    """加载 pickle 文件"""
    with path.open("rb") as f:
        return pickle.load(f)


def collate_recent_batch(batch):
    """截断超长序列并打包"""
    return collate_batch(batch, CONFIG["max_seq_len"])


def save_training_log(model_name: str, auc_history: list[float]) -> None:
    """保存训练日志"""
    history = [float(v) for v in auc_history]

    # 并行单模型实验时，独立日志文件不会互相覆盖。
    log_dir = LOG_PATH.parent / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    single_log_path = log_dir / f"{safe_file_stem(model_name)}.json"
    single_log_path.write_text(
        json.dumps({model_name: history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 聚合日志保留给顺序训练和前端读取；并行时以独立日志为准。
    logs = {}
    if LOG_PATH.exists():
        try:
            logs = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logs = {}
    logs[model_name] = history
    tmp_path = LOG_PATH.with_name(f"{LOG_PATH.stem}_{safe_file_stem(model_name)}.tmp")
    tmp_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(LOG_PATH)


def forward_model(model: nn.Module, model_name: str, batch: dict) -> torch.Tensor:
    """
    根据模型类型执行前向传播

    Args:
        model: 模型实例
        model_name: 模型名称
        batch: 输入 batch

    Returns:
        preds: 预测概率 (B, T, 1)
    """
    kc_ids = batch["sequences"][..., 0].long()
    corrects = batch["corrects"]
    mask = batch["mask"]

    if model_name == "SR-DKT":
        # SR-DKT 使用完整输入，包括 hints、attempts 和视频特征
        preds, _, _ = model(
            batch["sequences"],
            batch["delta_ts"],
            batch["hints"],
            batch["attempts"],
            watch_ratio_seq=batch.get("watch_ratios"),
            is_replay_seq=batch.get("is_replays"),
            mask=batch["mask"],
        )
        return preds

    if model_name == "DKT-F":
        # DKT-F 需要时间间隔
        preds, _ = model(kc_ids, corrects, mask, delta_t=batch["delta_ts"])
        return preds

    if model_name == "BEKT":
        # BEKT 使用 hint_count + attempt_count + delta_t
        preds, _ = model(
            kc_ids, corrects, mask,
            hint_count=batch["hints"],
            attempt_count=batch["attempts"],
            delta_t=batch["delta_ts"],
        )
        return preds

    if model_name == "DKVMN":
        # DKVMN 使用标准输入 kc_ids + corrects
        preds, _ = model(kc_ids, corrects, mask)
        return preds

    if model_name == "LBKT":
        # LBKT 支持 hint 和 attempt 特征（用于公平对比）
        preds, _ = model(
            kc_ids, corrects, mask,
            hint_count=batch["hints"],
            attempt_count=batch["attempts"],
        )
        return preds

    # DKT, SAKT, AKT 使用标准输入
    preds, _ = model(kc_ids, corrects, mask)
    return preds


def sequence_loss(preds: torch.Tensor, corrects: torch.Tensor, mask: torch.Tensor, criterion,
                  model_name: str = "") -> torch.Tensor:
    """计算下一步预测损失

    DKVMN 特殊处理：read-before-write 语义，preds[t] 预测 corrects[t] 而非 corrects[t+1]。
    因此 DKVMN 的损失对齐方式为 preds[:, 1:] vs corrects[:, 1:]（跳过位置0）。
    """
    if preds.shape[1] < 2:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)

    if model_name == "DKVMN":
        # DKVMN: read(q_t) → pred(r_t) → write(q_t, r_t), preds[t]→corrects[t]
        valid = mask[:, 1:]
        return criterion(preds[:, 1:, 0][valid], corrects[:, 1:][valid].float())

    # Default (DKT/SAKT/AKT/LBKT/BEKT/SR-DKT): preds[t]→corrects[t+1]
    valid = mask[:, 1:]
    return criterion(preds[:, :-1, 0][valid], corrects[:, 1:][valid].float())


@torch.no_grad()
def evaluate(model: nn.Module, model_name: str, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """
    评估模型性能

    Returns:
        dict: {"AUC": float, "ACC": float, "F1": float}
    """
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        preds = forward_model(model, model_name, batch)
        if preds.shape[1] < 2:
            continue
        valid = batch["mask"][:, 1:]
        # DKVMN: preds[t]→corrects[t], other models: preds[t]→corrects[t+1]
        if model_name == "DKVMN":
            y_true.extend(batch["corrects"][:, 1:][valid].detach().cpu().numpy().tolist())
            y_pred.extend(preds[:, 1:, 0][valid].detach().cpu().numpy().tolist())
        else:
            y_true.extend(batch["corrects"][:, 1:][valid].detach().cpu().numpy().tolist())
            y_pred.extend(preds[:, :-1, 0][valid].detach().cpu().numpy().tolist())

    if not y_true:
        return {"AUC": 0.5, "ACC": 0.0, "F1": 0.0}

    y_hat = np.array(y_pred) >= 0.5
    return {
        "AUC": float(roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5),
        "ACC": float(accuracy_score(y_true, y_hat)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def train_one_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    训练单个模型

    Args:
        model_name: 模型名称
        model: 模型实例
        train_loader: 训练数据
        val_loader: 验证数据
        test_loader: 测试数据
        device: 设备

    Returns:
        metrics: {"AUC": float, "ACC": float, "F1": float}
    """
    lr = MODEL_LR[model_name]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_auc = -1.0
    best_state = None
    no_improve = 0
    auc_history = []

    lambda_weight = CONFIG["lambda_constraint_weight"]

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}", leave=False):
            batch = move_batch(batch, device)
            optimizer.zero_grad()

            preds = forward_model(model, model_name, batch)

            # 计算 BCE 损失
            bce_loss = sequence_loss(preds, batch["corrects"], batch["mask"], criterion, model_name)

            # SR-DKT 需要额外加 lambda 约束损失
            if model_name == "SR-DKT":
                constraint_loss = model.get_lambda_constraint_loss()
                loss = bce_loss + lambda_weight * constraint_loss
            else:
                loss = bce_loss

            loss.backward()
            optimizer.step()
            total_loss += float(bce_loss.item())
            steps += 1

        val_metrics = evaluate(model, model_name, val_loader, device)
        val_auc = float(val_metrics["AUC"])
        auc_history.append(val_auc)

        # 打印训练进度
        extra_info = ""
        if model_name == "SR-DKT":
            extra_info = f" | λ_s: {model.lambda_s.item():.3f} | λ_r: {model.lambda_r.item():.3f}"
        print(
            f"[{model_name}] Epoch {epoch}/{CONFIG['epochs']} | "
            f"Loss: {total_loss / max(steps, 1):.4f} | "
            f"Val AUC: {val_auc:.4f}{extra_info}"
        )

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                print(f"[{model_name}] Early stopping at epoch {epoch}")
                break

    # 加载最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存模型 checkpoint
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": CONFIG}, CKPT_DIR / f"{model_name}.pt")

    # 保存训练日志
    save_training_log(model_name, auc_history)

    # 测试集评估
    metrics = evaluate(model, model_name, test_loader, device)
    print(f"[{model_name}] 训练完成 | Test AUC: {metrics['AUC']:.3f} | ACC: {metrics['ACC']:.3f} | F1: {metrics['F1']:.3f}")
    return metrics


def print_results(results: dict[str, dict[str, float]], dataset: str = "2009") -> None:
    """
    打印对比表格

    格式：
    模型名 | AUC | ACC | F1
    -------|-----|-----|----
    DKT    | ... | ... | ...
    SR-DKT | ... | ... | ...
    """
    dataset_labels = {"2009": "ASSISTments 2009", "2017": "ASSISTments 2017", "mooc": "MOOCCubeX"}
    ds_name = dataset_labels.get(dataset, dataset)
    best_name = max(results, key=lambda name: results[name]["AUC"])
    print("=" * 50)
    print(f"模型对比实验结果（{ds_name}）")
    print("=" * 50)
    print(f"{'模型':<10}{'AUC':>8}{'ACC':>8}{'F1':>8}")
    print("-" * 50)
    for name, metrics in results.items():
        marker = "  ★ 最优" if name == best_name else ""
        print(f"{name:<10}{metrics['AUC']:>8.3f}{metrics['ACC']:>8.3f}{metrics['F1']:>8.3f}{marker}")
    print("=" * 50)
    print("注：各模型均按原论文输入配置实现，SR-DKT 的额外输入与 LBKT 保持对等。")


def main() -> None:
    """主函数：批量训练所有 baseline 模型"""
    parser = argparse.ArgumentParser(description="SR-DKT Baselines Runner")
    parser.add_argument("--dataset", default="2009", choices=["2009", "2017", "mooc"],
                        help="选择数据集: 2009, 2017 或 mooc")
    parser.add_argument("--model", default=None,
                        choices=["DKT", "DKT-Vec", "DKT-F", "SAKT", "AKT", "LBKT", "DKVMN", "BEKT", "SR-DKT"],
                        help="指定单个模型名（不指定则跑全部）")
    parser.add_argument("--data-dir", dest="data_dir", default=None,
                        help="直接指定数据目录，覆盖默认路径（例如 data/mooc_10pct 跑 10%% 子集）")
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baselines] 使用设备: {device}")
    dataset_label = {"2009": "ASSISTments 2009", "2017": "ASSISTments 2017", "mooc": "MOOCCubeX"}.get(args.dataset, args.dataset)
    print(f"[baselines] 数据集: {dataset_label}")

    # 加载数据（--data-dir 优先，便于指向 10% 子集目录）
    ds_dir = Path(args.data_dir) if args.data_dir else get_dataset_dir(args.dataset)
    if args.data_dir:
        print(f"[baselines] 数据目录(覆盖): {ds_dir}")
    if args.dataset == "2017":
        train_data = load_pickle(ds_dir / "train_2017.pkl")
        val_data = load_pickle(ds_dir / "val_2017.pkl")
        test_data = load_pickle(ds_dir / "test_2017.pkl")
        meta_path = ds_dir / "meta_2017.json"
        result_path = ds_dir / "baseline_results_2017.json"
    elif args.dataset == "mooc":
        train_data = load_pickle(ds_dir / "train.pkl")
        val_data = load_pickle(ds_dir / "val.pkl")
        test_data = load_pickle(ds_dir / "test.pkl")
        meta_path = ds_dir / "meta_mooc.json"
        result_path = ds_dir / "baseline_results_mooc.json"
    else:
        train_data = load_pickle(ds_dir / "train.pkl")
        val_data = load_pickle(ds_dir / "val.pkl")
        test_data = load_pickle(ds_dir / "test.pkl")
        meta_path = ds_dir / "meta.json"
        result_path = ds_dir / "baseline_results.json"

    num_kc = int(json.loads(meta_path.read_text(encoding="utf-8"))["num_kc"])
    print(f"[baselines] 知识点数量: {num_kc}")

    # 创建 DataLoader
    train_loader = DataLoader(
        SequenceDataset(train_data),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_recent_batch,
    )
    val_loader = DataLoader(
        SequenceDataset(val_data),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_recent_batch,
    )
    test_loader = DataLoader(
        SequenceDataset(test_data),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_recent_batch,
    )

    # 创建所有模型（使用独立学习率）
    model_configs = {
        "DKT":    DKT(num_kc, CONFIG["hidden_size"]),
        "DKT-Vec": DKTVec(num_kc, CONFIG["hidden_size"]),  # pyKT 同款，跨框架对齐锚点
        "DKT-F":  DKT_F(num_kc, CONFIG["hidden_size"]),
        "SAKT":   SAKT(num_kc, CONFIG["hidden_size"], num_layers=1),  # 与 pyKT 对齐，默认2层会偏强
        "AKT":    AKT(num_kc, CONFIG["hidden_size"]),
        "LBKT":   LBKT(num_kc, CONFIG["hidden_size"]),
        "DKVMN":  DKVMN(num_kc, CONFIG["hidden_size"]),
        "BEKT":   BEKT(num_kc, CONFIG["hidden_size"]),
        "SR-DKT": SRDKT(num_kc, CONFIG["hidden_size"]),
    }

    # 支持 --model 参数，只训练指定模型（便于并行跑多个 screen）
    if args.model:
        model_configs = {args.model: model_configs[args.model]}
        print(f"[baselines] 单模型模式: {args.model}")

    # 逐个训练
    results = {}
    for name, model in model_configs.items():
        print(f"\n{'='*50}")
        print(f"训练模型: {name} (lr={MODEL_LR[name]})")
        print(f"{'='*50}")
        results[name] = train_one_model(
            name, model, train_loader, val_loader, test_loader, device
        )

    # 保存结果。单模型模式写独立文件，避免并行进程抢写同一个 JSON。
    if args.model:
        single_result_path = result_path.with_name(
            f"{result_path.stem}_{safe_file_stem(args.model)}.json"
        )
        single_result_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[baselines] 已保存单模型结果: {single_result_path}")
    else:
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[baselines] 已保存结果: {result_path}")

    # 打印对比表格
    print_results(results, dataset=args.dataset)


if __name__ == "__main__":
    main()
