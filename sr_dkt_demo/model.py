# 功能：定义 SR-DKT 双路径 LSTM 知识追踪模型，显式建模 S/R 与差异化遗忘。
# 已检查 Bug 1：LSTM 初始化括号闭合正确，无需修复
# 核心特性：双路径LSTM + 差异化遗忘 + 注意力融合
# 作者：SR-DKT 项目组
# 日期：2026-05-11
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class SRDKT(nn.Module):
    """
    SR-DKT: Storage-Retrieval Deep Knowledge Tracing

    双路径LSTM：
    - lstm_s: 处理主动检索行为（做题、论坛发帖、测验）
    - lstm_r: 处理被动接收行为（看视频、浏览材料）

    差异化遗忘：lambda_r > lambda_s（可学习参数，软约束）
    双状态融合：注意力加权 alpha_s * S_t + alpha_r * R_t
    """

    def __init__(self, num_kc: int, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Storage分支输入：kc_one_hot + correct + delta_t + attempt + hint
        storage_input_size = num_kc + 4
        # Retrieval分支输入：kc_one_hot + watch_ratio + delta_t + is_replay
        retrieval_input_size = num_kc + 3

        # 双路径LSTM，参数完全独立
        self.lstm_s = nn.LSTM(
            input_size=storage_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.lstm_r = nn.LSTM(
            input_size=retrieval_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Storage Head -> S_t
        self.fc_S = nn.Linear(hidden_size, 1)
        # Retrieval Head -> R_t (before decay)
        self.fc_R = nn.Linear(hidden_size, 1)

        # 差异化遗忘速率（可学习，初始化保证 lambda_r > lambda_s）
        # 使用 softplus 保证正值
        self.log_lambda_s = nn.Parameter(torch.tensor(-2.0))  # ~0.13
        self.log_lambda_r = nn.Parameter(torch.tensor(-1.0))  # ~0.37，初始 > lambda_s

        # 注意力融合层：输入 [S_t, R_t] -> 权重 [alpha_s, alpha_r]
        self.fusion_attn = nn.Linear(2, 2)

        # 最终预测头：输入 h_final（hidden_size）-> P(correct)
        self.fc_pred = nn.Linear(hidden_size, 1)

        # 融合后的线性映射（用于消融实验中的shared LSTM）
        self.fc_fusion = nn.Linear(hidden_size * 2, hidden_size)

    @property
    def lambda_s(self) -> torch.Tensor:
        """Storage遗忘速率，保证为正值"""
        return F.softplus(self.log_lambda_s)

    @property
    def lambda_r(self) -> torch.Tensor:
        """Retrieval遗忘速率，保证 > lambda_s"""
        # lambda_r = lambda_s + softplus(delta)，确保 lambda_r > lambda_s
        delta = F.softplus(self.log_lambda_r)
        return self.lambda_s + delta

    def encode_storage_inputs(
        self,
        sequences: torch.Tensor,      # (B, T, 2) [kc_id, correct]
        delta_t_seq: torch.Tensor,     # (B, T)
        hints_seq: torch.Tensor,       # (B, T)
        attempts_seq: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:
        """编码 Storage 分支输入特征"""
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
        sequences: torch.Tensor,      # (B, T, 2) [kc_id, correct]
        delta_t_seq: torch.Tensor,     # (B, T)
        watch_ratio_seq: torch.Tensor, # (B, T) 观看比例 0-1
        is_replay_seq: torch.Tensor,   # (B, T) 是否回放 0/1
    ) -> torch.Tensor:
        """编码 Retrieval 分支输入特征"""
        kc_ids = sequences[..., 0].long().clamp(min=0)
        kc_one_hot = F.one_hot(kc_ids, num_classes=self.num_kc).float()
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        watch = watch_ratio_seq.float().clamp(0, 1)
        replay = is_replay_seq.float().clamp(0, 1)
        return torch.cat([
            kc_one_hot,
            delta_norm.unsqueeze(-1),
            watch.unsqueeze(-1),
            replay.unsqueeze(-1),
        ], dim=-1)

    def apply_decay(
        self,
        state_seq: torch.Tensor,   # (B, T, 1) 原始状态
        delta_t_seq: torch.Tensor, # (B, T, 1) 时间间隔
        lambda_val: torch.Tensor,  # scalar
    ) -> torch.Tensor:
        """对状态序列施加指数遗忘衰减"""
        decay = torch.exp(-lambda_val * delta_t_seq.clamp(min=0))
        decay = decay.clamp(min=0.01, max=1.0)
        return state_seq * decay

    def forward(
        self,
        sequences: torch.Tensor,
        delta_t_seq: torch.Tensor,
        hints_seq: torch.Tensor,
        attempts_seq: torch.Tensor,
        watch_ratio_seq: torch.Tensor | None = None,
        is_replay_seq: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        ablation_mode: str = 'full',
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            sequences: (B, T, 2) [kc_id, correct]
            delta_t_seq: (B, T) 时间间隔序列
            hints_seq: (B, T) 提示次数
            attempts_seq: (B, T) 尝试次数
            watch_ratio_seq: (B, T) 可选，观看比例（无数据时自动填零）
            is_replay_seq: (B, T) 可选，是否回放（无数据时自动填零）
            mask: (B, T) 可选，有效位置掩码
            ablation_mode: 消融模式
                'full'          - 完整模型
                'no_retrieval'  - 去掉Retrieval分支
                'shared_decay'  - 共享遗忘速率
                'no_attention'  - 简单平均融合
                'single_path'   - 单路径退化

        Returns:
            p_seq: (B, T, 1) 预测答对概率
            s_seq: (B, T, 1) Storage强度
            r_seq: (B, T, 1) Retrieval强度（衰减后）
        """
        B, T = sequences.shape[:2]
        device = sequences.device

        # 构造被动行为输入（若未提供则用零向量）
        if watch_ratio_seq is None:
            watch_ratio_seq = torch.zeros(B, T, device=device)
        if is_replay_seq is None:
            is_replay_seq = torch.zeros(B, T, device=device)

        # ---- Storage 分支 ----
        x_s = self.encode_storage_inputs(sequences, delta_t_seq, hints_seq, attempts_seq)
        h_s, _ = self.lstm_s(x_s)  # (B, T, hidden)
        s_seq = torch.sigmoid(self.fc_S(h_s))  # (B, T, 1)

        # ---- 单路径退化：等价于DKT-F，只用Storage路径 ----
        if ablation_mode == 'single_path':
            p_seq = torch.sigmoid(self.fc_pred(h_s))
            return p_seq, s_seq, torch.zeros_like(s_seq)

        # ---- Retrieval 分支 ----
        if ablation_mode == 'no_retrieval':
            r_seq = torch.zeros(B, T, 1, device=device)
            h_r = torch.zeros(B, T, self.hidden_size, device=device)
        else:
            x_r = self.encode_retrieval_inputs(sequences, delta_t_seq, watch_ratio_seq, is_replay_seq)
            h_r, _ = self.lstm_r(x_r)  # (B, T, hidden)
            r_raw = torch.sigmoid(self.fc_R(h_r))  # (B, T, 1)

        # ---- 差异化遗忘衰减 ----
        if delta_t_seq.dim() == 2:
            delta_t_seq = delta_t_seq.unsqueeze(-1)
        last_delta = torch.zeros(B, 1, 1, device=device, dtype=delta_t_seq.dtype)
        delta_t_next = torch.cat([delta_t_seq[:, 1:, :], last_delta], dim=1)
        delta_t_norm = (torch.log1p(delta_t_next.clamp(min=0)) /
                        torch.log1p(torch.tensor(30.0, device=device)))

        # Storage衰减（慢）
        s_seq = self.apply_decay(s_seq, delta_t_norm, self.lambda_s)

        # Retrieval衰减：shared_decay模式下使用lambda_s替代lambda_r
        if ablation_mode == 'no_retrieval':
            r_seq = torch.zeros(B, T, 1, device=device)
        else:
            lambda_r_used = self.lambda_s if ablation_mode == 'shared_decay' else self.lambda_r
            r_seq = self.apply_decay(r_raw, delta_t_norm, lambda_r_used)

        # ---- 融合层 ----
        if ablation_mode == 'no_attention':
            # 简单平均融合
            h_fused_raw = 0.5 * h_s + 0.5 * h_r
        elif ablation_mode == 'no_retrieval':
            # 只用Storage
            h_fused_raw = h_s
        else:
            # 注意力融合
            sr_cat = torch.cat([s_seq, r_seq], dim=-1)  # (B, T, 2)
            alpha = torch.softmax(self.fusion_attn(sr_cat), dim=-1)  # (B, T, 2)
            alpha_s = alpha[..., 0:1]
            alpha_r = alpha[..., 1:2]
            h_fused_raw = alpha_s * h_s + alpha_r * h_r

        # ---- 最终预测 ----
        p_seq = torch.sigmoid(self.fc_pred(h_fused_raw))  # (B, T, 1)

        return p_seq, s_seq, r_seq

    def get_lambda_constraint_loss(self) -> torch.Tensor:
        """
        软约束损失：鼓励 lambda_r > lambda_s

        在训练时加入总损失：loss = bce_loss + 0.01 * constraint_loss

        Returns:
            violation: 当 lambda_s > lambda_r - margin 时的惩罚值
        """
        margin = 0.1
        violation = torch.relu(self.lambda_s - self.lambda_r + margin)
        return violation