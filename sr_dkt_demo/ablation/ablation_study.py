# 功能：训练并评估 SR-DKT 的 7 组消融/公平性变体，验证各组件贡献。
# 作者：SR-DKT 项目组
# 日期：2026-05-11
# 消融配置：
#   1. Full SR-DKT    - 完整模型：双路径LSTM + 差异化遗忘 + 注意力融合
#   2. w/o Storage    - 去掉lstm_s，只用lstm_r，S固定为0.5
#   3. w/o Retrieval  - 去掉lstm_r，只用lstm_s，R置零
#   4. Uniform Decay  - lambda_s = lambda_r（去掉差异化）
#   5. Mean Fusion    - 注意力融合改为简单平均 alpha_s=alpha_r=0.5
#   6. Single Path    - 单路径退化，只保留Storage路径
#   7. w/o Video      - 去掉 watch_ratio / is_replay，验证视频特征贡献
#
# 注意：历史版本在多个消融类中使用 one-hot KC 输入，MOOCCubeX 的
# num_kc 可达 30001，会造成显存/内存风险。主流程现在统一复用
# model.py 中基于 Embedding 的 SRDKT(ablation_mode=...) 实现。
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import SRDKT
from train import SequenceDataset, collate_batch, move_batch


# 训练配置
CONFIG = {
    "hidden_size": 128,
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 100,
    "patience": 10,
    "seed": 42,
    "max_seq_len": 200,
    "lambda_constraint_weight": 0.01,
}

DATA_DIR = ROOT_DIR / "data"
RESULT_PATH = DATA_DIR / "ablation_results.json"
LOG_PATH = DATA_DIR / "training_logs.json"
FIG_PATH = ROOT_DIR / "results" / "ablation_bar.png"


def safe_file_stem(name: str) -> str:
    """将实验名转成适合文件名的短标识。"""
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


class SRDKTAblationWrapper(nn.Module):
    """Embedding 版 SR-DKT 消融包装器，避免大规模 KC one-hot 展开。"""

    _CONSTRAINT_MODES = {"full", "no_attention", "no_storage", "no_video"}

    def __init__(self, num_kc: int, hidden_size: int, ablation_mode: str) -> None:
        super().__init__()
        self.ablation_mode = ablation_mode
        self.model = SRDKT(num_kc, hidden_size)

    def forward(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
        watch_ratio_seq: torch.Tensor | None = None,
        is_replay_seq: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forward_mode = "full" if self.ablation_mode == "no_video" else self.ablation_mode
        if self.ablation_mode == "no_video":
            watch_ratio_seq = torch.zeros_like(delta_t_seq)
            is_replay_seq = torch.zeros_like(delta_t_seq)

        return self.model(
            sequences,
            delta_t_seq,
            hints_seq,
            attempts_seq,
            watch_ratio_seq=watch_ratio_seq,
            is_replay_seq=is_replay_seq,
            mask=mask,
            ablation_mode=forward_mode,
        )

    def get_lambda_constraint_loss(self) -> torch.Tensor:
        if self.ablation_mode not in self._CONSTRAINT_MODES:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.model.get_lambda_constraint_loss()


def save_training_log(model_name: str, auc_history: list[float]) -> None:
    """保存训练日志"""
    history = [float(v) for v in auc_history]

    # 并行单变体实验时，独立日志文件不会互相覆盖。
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


# =============================================================================
# 消融变体 1: Full SR-DKT（完整模型）
# =============================================================================

# 使用原始 SRDKT 类，无需额外定义


# =============================================================================
# 消融变体 2: w/o Storage（去掉 lstm_s）
# =============================================================================

class SRDKTWithoutStorage(nn.Module):
    """
    去掉 Storage 分支，只用 Retrieval LSTM

    - S 固定为 0.5（常数）
    - 只使用 lstm_r 进行预测
    - 注意力权重 alpha_r = 1.0（完全依赖 Retrieval）
    """

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size

        # 只保留 Retrieval LSTM
        retrieval_input_size = num_kc + 3
        self.lstm_r = nn.LSTM(
            input_size=retrieval_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Retrieval Head
        self.fc_R = nn.Linear(hidden_size, 1)

        # 单一遗忘速率
        self.log_lambda = nn.Parameter(torch.tensor(-1.0))

        # 预测头（只用 R）
        self.fc_pred = nn.Linear(hidden_size, 1)

    @property
    def lambda_val(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def encode_retrieval_inputs(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
    ) -> torch.Tensor:
        kc_ids = sequences[..., 0].long().clamp(min=0)
        kc_one_hot = F.one_hot(kc_ids, num_classes=self.num_kc).float()
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        # ASSISTments 没有 watch_ratio 和 is_replay，填零
        watch = torch.zeros_like(delta_t_seq)
        replay = torch.zeros_like(delta_t_seq)
        return torch.cat([
            kc_one_hot,
            delta_norm.unsqueeze(-1),
            watch.unsqueeze(-1),
            replay.unsqueeze(-1),
        ], dim=-1)

    def forward(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = sequences.shape[:2]
        device = sequences.device

        # S 固定为 0.5
        s_seq = torch.full((B, T, 1), 0.5, device=device)

        # Retrieval 分支
        x_r = self.encode_retrieval_inputs(sequences, delta_t_seq)
        h_r, _ = self.lstm_r(x_r)
        r_raw = torch.sigmoid(self.fc_R(h_r))

        # 时间衰减
        if delta_t_seq.dim() == 2:
            delta_t_seq = delta_t_seq.unsqueeze(-1)
        last_delta = torch.zeros(B, 1, 1, device=device, dtype=delta_t_seq.dtype)
        delta_t_next = torch.cat([delta_t_seq[:, 1:, :], last_delta], dim=1)
        delta_t_norm = torch.log1p(delta_t_next.clamp(min=0)) / torch.log1p(torch.tensor(30.0, device=device))

        decay = torch.exp(-self.lambda_val * delta_t_norm).clamp(min=0.01, max=1.0)
        r_seq = r_raw * decay

        # 预测（对隐状态施加衰减后再预测，与完整模型一致）
        h_r_decayed = h_r * decay
        p_seq = torch.sigmoid(self.fc_pred(h_r_decayed))

        return p_seq, s_seq, r_seq

    def get_lambda_constraint_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


# =============================================================================
# 消融变体 3: w/o Retrieval（去掉 lstm_r）
# =============================================================================

class SRDKTWithoutRetrieval(nn.Module):
    """
    去掉 Retrieval 分支，只用 Storage LSTM

    - R 固定为 0.5（常数）
    - 只使用 lstm_s 进行预测
    - 注意力权重 alpha_s = 1.0（完全依赖 Storage）
    """

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size

        # 只保留 Storage LSTM
        storage_input_size = num_kc + 4
        self.lstm_s = nn.LSTM(
            input_size=storage_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Storage Head
        self.fc_S = nn.Linear(hidden_size, 1)

        # 单一遗忘速率（不实际使用，但保留参数）
        self.log_lambda = nn.Parameter(torch.tensor(-1.0))

        # 预测头（只用 S）
        self.fc_pred = nn.Linear(hidden_size, 1)

    @property
    def lambda_val(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def encode_storage_inputs(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
    ) -> torch.Tensor:
        kc_ids = sequences[..., 0].long().clamp(min=0)
        corrects = sequences[..., 1].float()
        kc_one_hot = F.one_hot(kc_ids, num_classes=self.num_kc).float()
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        hint_norm = hints_seq.float().clamp(0, 5) / 5.0
        attempt_norm = attempts_seq.float().clamp(0, 3) / 3.0
        return torch.cat([
            kc_one_hot,
            corrects.unsqueeze(-1),
            delta_norm.unsqueeze(-1),
            hint_norm.unsqueeze(-1),
            attempt_norm.unsqueeze(-1),
        ], dim=-1)

    def forward(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = sequences.shape[:2]
        device = sequences.device

        # R 固定为 0.5
        r_seq = torch.full((B, T, 1), 0.5, device=device)

        # Storage 分支
        x_s = self.encode_storage_inputs(sequences, delta_t_seq, hints_seq, attempts_seq)
        h_s, _ = self.lstm_s(x_s)
        s_raw = torch.sigmoid(self.fc_S(h_s))

        # 时间衰减（Storage 使用较慢的衰减）
        if delta_t_seq.dim() == 2:
            delta_t_seq = delta_t_seq.unsqueeze(-1)
        last_delta = torch.zeros(B, 1, 1, device=device, dtype=delta_t_seq.dtype)
        delta_t_next = torch.cat([delta_t_seq[:, 1:, :], last_delta], dim=1)
        delta_t_norm = torch.log1p(delta_t_next.clamp(min=0)) / torch.log1p(torch.tensor(30.0, device=device))

        decay = torch.exp(-self.lambda_val * delta_t_norm).clamp(min=0.01, max=1.0)
        s_seq = s_raw * decay

        # 预测（对隐状态施加衰减后再预测，与完整模型一致）
        h_s_decayed = h_s * decay
        p_seq = torch.sigmoid(self.fc_pred(h_s_decayed))

        return p_seq, s_seq, r_seq

    def get_lambda_constraint_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


# =============================================================================
# 消融变体 4: Shared LSTM（合并双路径）
# =============================================================================

class SRDKTSharedLSTM(nn.Module):
    """
    使用单一共享 LSTM 替代双路径 LSTM

    - lstm_s 和 lstm_r 合并为一个共享 LSTM
    - S 和 R 从同一个 LSTM 输出中提取
    - 保留差异化遗忘和注意力融合
    """

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size

        # 共享 LSTM（合并 Storage 和 Retrieval 的输入）
        # 输入: kc_one_hot + correct + delta_t + hint + attempt
        input_size = num_kc + 4
        self.lstm_shared = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # S 和 R 头
        self.fc_S = nn.Linear(hidden_size, 1)
        self.fc_R = nn.Linear(hidden_size, 1)

        # 差异化遗忘速率
        self.log_lambda_s = nn.Parameter(torch.tensor(-2.0))
        self.log_lambda_r = nn.Parameter(torch.tensor(-1.0))

        # 注意力融合
        self.fusion_attn = nn.Linear(2, 2)

        # S 和 R 的独立投影头（使 SharedLSTM 的融合操作有意义）
        self.proj_s = nn.Linear(hidden_size, hidden_size)
        self.proj_r = nn.Linear(hidden_size, hidden_size)

        # 预测头
        self.fc_pred = nn.Linear(hidden_size, 1)

    @property
    def lambda_s(self) -> torch.Tensor:
        return F.softplus(self.log_lambda_s)

    @property
    def lambda_r(self) -> torch.Tensor:
        delta = F.softplus(self.log_lambda_r)
        return self.lambda_s + delta

    def encode_inputs(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
    ) -> torch.Tensor:
        kc_ids = sequences[..., 0].long().clamp(min=0)
        corrects = sequences[..., 1].float()
        kc_one_hot = F.one_hot(kc_ids, num_classes=self.num_kc).float()
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        hint_norm = hints_seq.float().clamp(0, 5) / 5.0
        attempt_norm = attempts_seq.float().clamp(0, 3) / 3.0
        return torch.cat([
            kc_one_hot,
            corrects.unsqueeze(-1),
            delta_norm.unsqueeze(-1),
            hint_norm.unsqueeze(-1),
            attempt_norm.unsqueeze(-1),
        ], dim=-1)

    def forward(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = sequences.shape[:2]
        device = sequences.device

        # 共享 LSTM
        x = self.encode_inputs(sequences, delta_t_seq, hints_seq, attempts_seq)
        h_shared, _ = self.lstm_shared(x)

        # S 和 R 从同一隐状态提取
        s_raw = torch.sigmoid(self.fc_S(h_shared))
        r_raw = torch.sigmoid(self.fc_R(h_shared))

        # 时间衰减
        if delta_t_seq.dim() == 2:
            delta_t_seq = delta_t_seq.unsqueeze(-1)
        last_delta = torch.zeros(B, 1, 1, device=device, dtype=delta_t_seq.dtype)
        delta_t_next = torch.cat([delta_t_seq[:, 1:, :], last_delta], dim=1)
        delta_t_norm = torch.log1p(delta_t_next.clamp(min=0)) / torch.log1p(torch.tensor(30.0, device=device))

        s_decay = torch.exp(-self.lambda_s * delta_t_norm).clamp(min=0.01, max=1.0)
        r_decay = torch.exp(-self.lambda_r * delta_t_norm).clamp(min=0.01, max=1.0)

        s_seq = s_raw * s_decay
        r_seq = r_raw * r_decay

        # 注意力融合
        sr_cat = torch.cat([s_seq, r_seq], dim=-1)
        alpha = torch.softmax(self.fusion_attn(sr_cat), dim=-1)
        alpha_s = alpha[..., 0:1]
        alpha_r = alpha[..., 1:2]

        h_fused = alpha_s * torch.tanh(self.proj_s(h_shared)) + alpha_r * torch.tanh(self.proj_r(h_shared))

        # 预测
        p_seq = torch.sigmoid(self.fc_pred(h_fused))

        return p_seq, s_seq, r_seq

    def get_lambda_constraint_loss(self) -> torch.Tensor:
        margin = 0.1
        violation = torch.relu(self.lambda_s - self.lambda_r + margin)
        return violation


# =============================================================================
# 消融变体 5: Uniform Decay（统一遗忘速率）
# =============================================================================

class SRDKTUniformDecay(nn.Module):
    """
    去掉差异化遗忘，lambda_s = lambda_r

    - 使用单一可学习遗忘速率 lambda
    - S 和 R 使用相同的衰减函数
    - 保留双路径 LSTM 和注意力融合
    """

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size

        # 双路径 LSTM（保留）
        storage_input_size = num_kc + 4
        retrieval_input_size = num_kc + 3

        self.lstm_s = nn.LSTM(
            input_size=storage_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.lstm_r = nn.LSTM(
            input_size=retrieval_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # S 和 R 头
        self.fc_S = nn.Linear(hidden_size, 1)
        self.fc_R = nn.Linear(hidden_size, 1)

        # 单一遗忘速率（lambda_s = lambda_r）
        self.log_lambda = nn.Parameter(torch.tensor(-1.5))

        # 注意力融合（保留）
        self.fusion_attn = nn.Linear(2, 2)

        # 预测头
        self.fc_pred = nn.Linear(hidden_size, 1)

    @property
    def lambda_val(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def encode_storage_inputs(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
    ) -> torch.Tensor:
        kc_ids = sequences[..., 0].long().clamp(min=0)
        corrects = sequences[..., 1].float()
        kc_one_hot = F.one_hot(kc_ids, num_classes=self.num_kc).float()
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        hint_norm = hints_seq.float().clamp(0, 5) / 5.0
        attempt_norm = attempts_seq.float().clamp(0, 3) / 3.0
        return torch.cat([
            kc_one_hot,
            corrects.unsqueeze(-1),
            delta_norm.unsqueeze(-1),
            hint_norm.unsqueeze(-1),
            attempt_norm.unsqueeze(-1),
        ], dim=-1)

    def encode_retrieval_inputs(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
    ) -> torch.Tensor:
        kc_ids = sequences[..., 0].long().clamp(min=0)
        kc_one_hot = F.one_hot(kc_ids, num_classes=self.num_kc).float()
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        watch = torch.zeros_like(delta_t_seq)
        replay = torch.zeros_like(delta_t_seq)
        return torch.cat([
            kc_one_hot,
            delta_norm.unsqueeze(-1),
            watch.unsqueeze(-1),
            replay.unsqueeze(-1),
        ], dim=-1)

    def forward(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = sequences.shape[:2]
        device = sequences.device

        # Storage 分支
        x_s = self.encode_storage_inputs(sequences, delta_t_seq, hints_seq, attempts_seq)
        h_s, _ = self.lstm_s(x_s)
        s_raw = torch.sigmoid(self.fc_S(h_s))

        # Retrieval 分支
        x_r = self.encode_retrieval_inputs(sequences, delta_t_seq)
        h_r, _ = self.lstm_r(x_r)
        r_raw = torch.sigmoid(self.fc_R(h_r))

        # 统一时间衰减（lambda_s = lambda_r = lambda）
        if delta_t_seq.dim() == 2:
            delta_t_seq = delta_t_seq.unsqueeze(-1)
        last_delta = torch.zeros(B, 1, 1, device=device, dtype=delta_t_seq.dtype)
        delta_t_next = torch.cat([delta_t_seq[:, 1:, :], last_delta], dim=1)
        delta_t_norm = torch.log1p(delta_t_next.clamp(min=0)) / torch.log1p(torch.tensor(30.0, device=device))

        decay = torch.exp(-self.lambda_val * delta_t_norm).clamp(min=0.01, max=1.0)
        s_seq = s_raw * decay
        r_seq = r_raw * decay

        # 注意力融合
        sr_cat = torch.cat([s_seq, r_seq], dim=-1)
        alpha = torch.softmax(self.fusion_attn(sr_cat), dim=-1)
        alpha_s = alpha[..., 0:1]
        alpha_r = alpha[..., 1:2]

        h_fused = alpha_s * h_s + alpha_r * h_r

        # 预测
        p_seq = torch.sigmoid(self.fc_pred(h_fused))

        return p_seq, s_seq, r_seq

    def get_lambda_constraint_loss(self) -> torch.Tensor:
        # 不需要约束，因为 lambda_s = lambda_r
        return torch.tensor(0.0, device=next(self.parameters()).device)


# =============================================================================
# 消融变体 6: Mean Fusion（简单平均融合）
# =============================================================================

class SRDKTMeanFusion(nn.Module):
    """
    注意力融合改为简单平均

    - alpha_s = alpha_r = 0.5（固定）
    - 保留双路径 LSTM 和差异化遗忘
    """

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size

        # 双路径 LSTM（保留）
        storage_input_size = num_kc + 4
        retrieval_input_size = num_kc + 3

        self.lstm_s = nn.LSTM(
            input_size=storage_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.lstm_r = nn.LSTM(
            input_size=retrieval_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # S 和 R 头
        self.fc_S = nn.Linear(hidden_size, 1)
        self.fc_R = nn.Linear(hidden_size, 1)

        # 差异化遗忘速率（保留）
        self.log_lambda_s = nn.Parameter(torch.tensor(-2.0))
        self.log_lambda_r = nn.Parameter(torch.tensor(-1.0))

        # 不需要注意力融合层（固定 alpha_s = alpha_r = 0.5）

        # 预测头
        self.fc_pred = nn.Linear(hidden_size, 1)

    @property
    def lambda_s(self) -> torch.Tensor:
        return F.softplus(self.log_lambda_s)

    @property
    def lambda_r(self) -> torch.Tensor:
        delta = F.softplus(self.log_lambda_r)
        return self.lambda_s + delta

    def encode_storage_inputs(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
    ) -> torch.Tensor:
        kc_ids = sequences[..., 0].long().clamp(min=0)
        corrects = sequences[..., 1].float()
        kc_one_hot = F.one_hot(kc_ids, num_classes=self.num_kc).float()
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        hint_norm = hints_seq.float().clamp(0, 5) / 5.0
        attempt_norm = attempts_seq.float().clamp(0, 3) / 3.0
        return torch.cat([
            kc_one_hot,
            corrects.unsqueeze(-1),
            delta_norm.unsqueeze(-1),
            hint_norm.unsqueeze(-1),
            attempt_norm.unsqueeze(-1),
        ], dim=-1)

    def encode_retrieval_inputs(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
    ) -> torch.Tensor:
        kc_ids = sequences[..., 0].long().clamp(min=0)
        kc_one_hot = F.one_hot(kc_ids, num_classes=self.num_kc).float()
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        watch = torch.zeros_like(delta_t_seq)
        replay = torch.zeros_like(delta_t_seq)
        return torch.cat([
            kc_one_hot,
            delta_norm.unsqueeze(-1),
            watch.unsqueeze(-1),
            replay.unsqueeze(-1),
        ], dim=-1)

    def forward(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = sequences.shape[:2]
        device = sequences.device

        # Storage 分支
        x_s = self.encode_storage_inputs(sequences, delta_t_seq, hints_seq, attempts_seq)
        h_s, _ = self.lstm_s(x_s)
        s_raw = torch.sigmoid(self.fc_S(h_s))

        # Retrieval 分支
        x_r = self.encode_retrieval_inputs(sequences, delta_t_seq)
        h_r, _ = self.lstm_r(x_r)
        r_raw = torch.sigmoid(self.fc_R(h_r))

        # 差异化时间衰减（保留）
        if delta_t_seq.dim() == 2:
            delta_t_seq = delta_t_seq.unsqueeze(-1)
        last_delta = torch.zeros(B, 1, 1, device=device, dtype=delta_t_seq.dtype)
        delta_t_next = torch.cat([delta_t_seq[:, 1:, :], last_delta], dim=1)
        delta_t_norm = torch.log1p(delta_t_next.clamp(min=0)) / torch.log1p(torch.tensor(30.0, device=device))

        s_decay = torch.exp(-self.lambda_s * delta_t_norm).clamp(min=0.01, max=1.0)
        r_decay = torch.exp(-self.lambda_r * delta_t_norm).clamp(min=0.01, max=1.0)

        s_seq = s_raw * s_decay
        r_seq = r_raw * r_decay

        # 固定平均融合（alpha_s = alpha_r = 0.5）
        h_fused = 0.5 * h_s + 0.5 * h_r

        # 预测
        p_seq = torch.sigmoid(self.fc_pred(h_fused))

        return p_seq, s_seq, r_seq

    def get_lambda_constraint_loss(self) -> torch.Tensor:
        margin = 0.1
        violation = torch.relu(self.lambda_s - self.lambda_r + margin)
        return violation


# =============================================================================
# 训练与评估函数
# =============================================================================

def sequence_loss(preds: torch.Tensor, corrects: torch.Tensor, mask: torch.Tensor, criterion) -> torch.Tensor:
    """计算下一步预测损失"""
    if preds.shape[1] <= 1:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    valid = mask[:, 1:]
    return criterion(preds[:, :-1, 0][valid], corrects[:, 1:][valid])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """评估模型"""
    model.eval()
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
        valid = batch["mask"][:, 1:]
        y_true.extend(batch["corrects"][:, 1:][valid].detach().cpu().numpy().tolist())
        y_pred.extend(preds[:, :-1, 0][valid].detach().cpu().numpy().tolist())

    if not y_true:
        return {"AUC": 0.5, "ACC": 0.0}

    y_hat = np.array(y_pred) >= 0.5
    return {
        "AUC": float(roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5),
        "ACC": float(accuracy_score(y_true, y_hat)),
    }


def train_variant(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    note: str,
) -> dict:
    """训练单个消融变体"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
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

        for batch in tqdm(train_loader, desc=f"[{name}] Epoch {epoch}", leave=False):
            batch = move_batch(batch, device)
            optimizer.zero_grad()

            preds, _, _ = model(
                batch["sequences"],
                batch["delta_ts"],
                batch["hints"],
                batch["attempts"],
                watch_ratio_seq=batch.get("watch_ratios"),
                is_replay_seq=batch.get("is_replays"),
                mask=batch["mask"],
            )

            bce_loss = sequence_loss(preds, batch["corrects"], batch["mask"], criterion)

            # 如果模型有 lambda 约束损失，加入总损失
            if hasattr(model, 'get_lambda_constraint_loss'):
                constraint_loss = model.get_lambda_constraint_loss()
                loss = bce_loss + lambda_weight * constraint_loss
            else:
                loss = bce_loss

            loss.backward()
            optimizer.step()
            total_loss += float(bce_loss.item())
            steps += 1

        val_metrics = evaluate(model, val_loader, device)
        val_auc = val_metrics["AUC"]
        auc_history.append(val_auc)

        print(f"[{name}] Epoch {epoch}/{CONFIG['epochs']} | "
              f"Loss: {total_loss / max(steps, 1):.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CONFIG["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_training_log(name, auc_history)
    metrics = evaluate(model, test_loader, device)
    metrics["说明"] = note
    print(f"[{name}] 训练完成 | Test AUC: {metrics['AUC']:.3f}")
    return metrics


def print_results(results: dict[str, dict]) -> None:
    """打印消融实验对比表格"""
    print("=" * 70)
    print("SR-DKT 消融实验结果")
    print("=" * 70)
    print(f"{'配置':<20}{'AUC':>10}{'ACC':>10}   说明")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<20}{metrics['AUC']:>10.3f}{metrics['ACC']:>10.3f}   {metrics['说明']}")
    print("=" * 70)


def plot_ablation_bar(results: dict[str, dict], save_path: Path) -> None:
    """
    绘制消融实验 AUC 柱状图

    Args:
        results: 实验结果字典
        save_path: 图片保存路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    names = list(results.keys())
    aucs = [results[name]["AUC"] for name in names]

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 柱状图颜色：Full 模型用深色，其他用浅色
    colors = ['#2E86AB' if 'Full' in name else '#A8DADC' for name in names]

    bars = ax.bar(names, aucs, color=colors, edgecolor='black', linewidth=1.2)

    # 添加数值标签
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.annotate(f'{auc:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    # 设置标题和标签
    ax.set_title('SR-DKT 消融实验 AUC 对比', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_xlabel('消融配置', fontsize=12)

    # 设置 Y 轴范围
    ax.set_ylim(min(aucs) - 0.05, max(aucs) + 0.05)

    # 旋转 X 轴标签
    plt.xticks(rotation=15, ha='right')

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # 添加注释说明
    ax.text(0.02, 0.98, '深色柱 = 完整模型', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 保存图片
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[ablation] 已保存柱状图: {save_path}")
    plt.close()


def main() -> None:
    """主函数：运行所有消融实验"""
    parser = argparse.ArgumentParser(description="SR-DKT Ablation Study")
    parser.add_argument("--dataset", default="2009", choices=["2009", "2017", "mooc"],
                        help="选择数据集")
    parser.add_argument(
        "--variant",
        default=None,
        choices=[
            "full",
            "no_storage",
            "no_retrieval",
            "shared_decay",
            "no_attention",
            "single_path",
            "no_video",
        ],
        help="指定单个消融变体（不指定则跑全部 7 组）",
    )
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] 使用设备: {device}")
    dataset_labels = {"2009": "ASSISTments 2009", "2017": "ASSISTments 2017", "mooc": "MOOCCubeX"}
    print(f"[ablation] 数据集: {dataset_labels.get(args.dataset, args.dataset)}")

    # 加载数据
    ds_dir = get_dataset_dir(args.dataset)
    if args.dataset == "2017":
        train_data = load_pickle(ds_dir / "train_2017.pkl")
        val_data = load_pickle(ds_dir / "val_2017.pkl")
        test_data = load_pickle(ds_dir / "test_2017.pkl")
        meta_path = ds_dir / "meta_2017.json"
    elif args.dataset == "mooc":
        train_data = load_pickle(ds_dir / "train.pkl")
        val_data = load_pickle(ds_dir / "val.pkl")
        test_data = load_pickle(ds_dir / "test.pkl")
        meta_path = ds_dir / "meta_mooc.json"
    else:
        train_data = load_pickle(ds_dir / "train.pkl")
        val_data = load_pickle(ds_dir / "val.pkl")
        test_data = load_pickle(ds_dir / "test.pkl")
        meta_path = ds_dir / "meta.json"

    num_kc = int(json.loads(meta_path.read_text(encoding="utf-8"))["num_kc"])
    print(f"[ablation] 知识点数量: {num_kc}")

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

    # 定义 7 组消融 / 公平性配置。
    # 全部使用 model.py 中的 Embedding 版 ablation_mode，避免 MOOCCubeX
    # 大知识点词表下 legacy one-hot 消融类造成 OOM。
    variants = {
        "full": (
            "Full SR-DKT",
            SRDKTAblationWrapper(num_kc, CONFIG["hidden_size"], "full"),
            "完整模型：双路径LSTM + 差异化遗忘 + 注意力融合"
        ),
        "no_storage": (
            "w/o Storage",
            SRDKTAblationWrapper(num_kc, CONFIG["hidden_size"], "no_storage"),
            "去掉lstm_s，只用lstm_r，S固定为0.5"
        ),
        "no_retrieval": (
            "w/o Retrieval",
            SRDKTAblationWrapper(num_kc, CONFIG["hidden_size"], "no_retrieval"),
            "去掉lstm_r，只用lstm_s，R置零"
        ),
        "shared_decay": (
            "Uniform Decay",
            SRDKTAblationWrapper(num_kc, CONFIG["hidden_size"], "shared_decay"),
            "lambda_s = lambda_r（去掉差异化遗忘）"
        ),
        "no_attention": (
            "Mean Fusion",
            SRDKTAblationWrapper(num_kc, CONFIG["hidden_size"], "no_attention"),
            "注意力融合改为简单平均 alpha_s=alpha_r=0.5"
        ),
        "single_path": (
            "Single Path",
            SRDKTAblationWrapper(num_kc, CONFIG["hidden_size"], "single_path"),
            "单路径退化，只保留Storage路径"
        ),
        "no_video": (
            "w/o Video Features",
            SRDKTAblationWrapper(num_kc, CONFIG["hidden_size"], "no_video"),
            "去掉 watch_ratio / is_replay，验证视频特征贡献"
        ),
    }

    if args.variant:
        variants = {args.variant: variants[args.variant]}
        print(f"[ablation] 单变体模式: {args.variant}")

    # 训练所有变体
    results = {}
    for _variant_key, (name, model, note) in variants.items():
        print(f"\n{'='*50}")
        print(f"训练消融变体: {name}")
        print(f"说明: {note}")
        print(f"{'='*50}")
        results[name] = train_variant(
            name, model, train_loader, val_loader, test_loader, device, note
        )

    # 保存结果
    if args.dataset == "mooc":
        result_path = ds_dir / "ablation_results_mooc.json"
        fig_path = ROOT_DIR / "results" / "ablation_bar_mooc.png"
    elif args.dataset == "2017":
        result_path = ds_dir / "ablation_results_2017.json"
        fig_path = ROOT_DIR / "results" / "ablation_bar_2017.png"
    else:
        result_path = ds_dir / "ablation_results.json"
        fig_path = FIG_PATH
    if args.variant:
        single_result_path = result_path.with_name(
            f"{result_path.stem}_{safe_file_stem(args.variant)}.json"
        )
        single_result_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[ablation] 已保存单变体结果: {single_result_path}")
    else:
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[ablation] 已保存结果: {result_path}")

    # 打印对比表格
    print_results(results)

    # 全量消融才生成柱状图；单变体并行跑时最终汇总后再画图。
    if not args.variant:
        plot_ablation_bar(results, fig_path)


if __name__ == "__main__":
    main()
