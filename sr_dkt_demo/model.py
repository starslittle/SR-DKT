# 功能：定义 SR-DKT 双路径 LSTM 知识追踪模型，显式建模 S/R 与差异化遗忘。
# 已检查 Bug 1：LSTM 初始化括号闭合正确，无需修复
# 核心特性：双路径LSTM + 差异化遗忘 + 注意力融合
# 作者：SR-DKT 项目组
# 日期：2026-05-11
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SRDKT(nn.Module):
    """
    SR-DKT: Storage-Retrieval Deep Knowledge Tracing

    双路径LSTM：
    - lstm_s: 处理主动检索行为（做题、论坛发帖、测验）
    - lstm_r: 处理被动接收行为（看视频、浏览材料）

    差异化遗忘：lambda_s、lambda_r 为两个独立可学习参数，由软约束损失
                 鼓励 lambda_r > lambda_s（短期提取衰减更快）。遗忘衰减
                 直接作用在进入预测的隐状态 h_s / h_r 上，因此 lambda
                 真正影响最终预测，而非仅调节融合权重。
    双状态融合：注意力加权 alpha_s * h_s + alpha_r * h_r（衰减后）
    """

    def __init__(self, num_kc: int, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # KC Embedding：替代 one-hot，输入维度与 KC 数量解耦
        # +1 是为了容纳 OTHER 类（kc_id = num_kc）
        self.kc_embed_s = nn.Embedding(num_kc + 1, hidden_size, padding_idx=0)
        self.kc_embed_r = nn.Embedding(num_kc + 1, hidden_size, padding_idx=0)

        # Storage分支输入：kc_embed(128) + correct + delta_t + attempt + hint
        storage_input_size = hidden_size + 4
        # Retrieval分支输入：kc_embed(128) + watch_ratio + delta_t + is_replay
        retrieval_input_size = hidden_size + 3

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

        # 差异化遗忘速率：两个【独立】可学习参数，softplus 保证正值。
        # 初始化让 lambda_r > lambda_s，但二者各自自由学习；是否保持
        # lambda_r > lambda_s 由软约束损失鼓励（get_lambda_constraint_loss），
        # 而非写死的恒等式 —— 这样该不等式是「学出来的」，可写入论文。
        self.log_lambda_s = nn.Parameter(torch.tensor(-2.0))  # softplus ~0.13
        self.log_lambda_r = nn.Parameter(torch.tensor(-1.0))  # softplus ~0.31，初始 > lambda_s

        # 残差门控「保留底」β：经 sigmoid 限制在 (0,1)，各自可学习。
        # 衰减系数 = β + (1-β)·exp(-λ·Δt) ∈ [β, 1]，即记忆保留一个持久核心 β，
        # 只有「新鲜部分」(1-β) 随时间遗忘 —— 避免把隐状态整体压成 0 而破坏表征。
        # 初始化 β_s 高(Storage 持久) > β_r 低(Retrieval 易挥发)，差异化遗忘的第二维度。
        self.logit_beta_s = nn.Parameter(torch.tensor(1.5))   # sigmoid ~0.82
        self.logit_beta_r = nn.Parameter(torch.tensor(0.0))   # sigmoid ~0.50

        # 注意力融合层：输入 [S_t, R_t] -> 权重 [alpha_s, alpha_r]
        self.fusion_attn = nn.Linear(2, 2)

        # 最终预测头：输入 h_final（hidden_size）-> P(correct)
        self.fc_pred = nn.Linear(hidden_size, 1)

    @property
    def lambda_s(self) -> torch.Tensor:
        """Storage遗忘速率，保证为正值"""
        return F.softplus(self.log_lambda_s)

    @property
    def lambda_r(self) -> torch.Tensor:
        """Retrieval遗忘速率，独立可学习（不再写死 > lambda_s）"""
        return F.softplus(self.log_lambda_r)

    @property
    def beta_s(self) -> torch.Tensor:
        """Storage 保留底（持久核心比例），sigmoid 限制在 (0,1)"""
        return torch.sigmoid(self.logit_beta_s)

    @property
    def beta_r(self) -> torch.Tensor:
        """Retrieval 保留底，独立可学习，初始低于 beta_s"""
        return torch.sigmoid(self.logit_beta_r)

    def encode_storage_inputs(
        self,
        sequences: torch.Tensor,      # (B, T, 2) [kc_id, correct]
        delta_t_seq: torch.Tensor,     # (B, T)
        hints_seq: torch.Tensor,       # (B, T)
        attempts_seq: torch.Tensor,    # (B, T)
    ) -> torch.Tensor:
        """编码 Storage 分支输入特征"""
        kc_ids = sequences[..., 0].long().clamp(min=0, max=self.num_kc)
        corrects = sequences[..., 1].float()
        kc_emb = self.kc_embed_s(kc_ids)  # (B, T, hidden_size)
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        hint_norm = hints_seq.float().clamp(0, 5) / 5.0
        attempt_norm = attempts_seq.float().clamp(0, 3) / 3.0
        return torch.cat([
            kc_emb,
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
        kc_ids = sequences[..., 0].long().clamp(min=0, max=self.num_kc)
        kc_emb = self.kc_embed_r(kc_ids)  # (B, T, hidden_size)
        delta_norm = (torch.log1p(delta_t_seq.float().clamp(min=0)) /
                      torch.log1p(torch.tensor(30.0, device=delta_t_seq.device)))
        watch = watch_ratio_seq.float().clamp(0, 1)
        replay = is_replay_seq.float().clamp(0, 1)
        return torch.cat([
            kc_emb,
            delta_norm.unsqueeze(-1),
            watch.unsqueeze(-1),
            replay.unsqueeze(-1),
        ], dim=-1)

    def apply_decay(
        self,
        state_seq: torch.Tensor,   # (B, T, H) 隐状态
        delta_t_seq: torch.Tensor, # (B, T, 1) 归一化时间间隔
        lambda_val: torch.Tensor,  # scalar，遗忘速率
        floor: torch.Tensor,       # scalar，保留底 β ∈ (0,1)
    ) -> torch.Tensor:
        """残差门控式遗忘衰减。

        decay 系数 = β + (1-β)·exp(-λ·Δt)，取值 [β, 1]：
        - 保留底 β 表示记忆的持久核心，永不被遗忘清零；
        - 只有「新鲜部分」(1-β) 随时间按 exp(-λ·Δt) 衰减；
        - λ 仍真实调节遗忘快慢（λ_r>λ_s → Retrieval 忘得快），但不再像旧版
          纯乘法那样把整个隐状态压到 ~0、破坏预测表征。
        """
        raw = torch.exp(-lambda_val * delta_t_seq.clamp(min=0))  # (B,T,1) ∈ (0,1]
        factor = floor + (1.0 - floor) * raw                     # ∈ [β, 1]
        return state_seq * factor

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
                'no_storage'    - 去掉Storage分支（S固定0.5）
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

        # ---- Storage 分支（主动行为：做题/测验）----
        if ablation_mode == 'no_storage':
            h_s = torch.zeros(B, T, self.hidden_size, device=device)
        else:
            x_s = self.encode_storage_inputs(sequences, delta_t_seq, hints_seq, attempts_seq)
            # 用 pack_padded_sequence 避免 padding 污染 LSTM 隐状态
            if mask is not None:
                lengths = mask.sum(dim=1).clamp(min=1).long().cpu()
                x_s_packed = pack_padded_sequence(x_s, lengths, batch_first=True, enforce_sorted=False)
                h_s_packed, _ = self.lstm_s(x_s_packed)
                h_s, _ = pad_packed_sequence(h_s_packed, batch_first=True, total_length=T)
            else:
                h_s, _ = self.lstm_s(x_s)

        # ---- Retrieval 分支（被动行为：看视频/浏览）----
        if ablation_mode in ('no_retrieval', 'single_path'):
            h_r = torch.zeros(B, T, self.hidden_size, device=device)
        else:
            x_r = self.encode_retrieval_inputs(sequences, delta_t_seq, watch_ratio_seq, is_replay_seq)
            if mask is not None:
                lengths_r = mask.sum(dim=1).clamp(min=1).long().cpu()
                x_r_packed = pack_padded_sequence(x_r, lengths_r, batch_first=True, enforce_sorted=False)
                h_r_packed, _ = self.lstm_r(x_r_packed)
                h_r, _ = pad_packed_sequence(h_r_packed, batch_first=True, total_length=T)
            else:
                h_r, _ = self.lstm_r(x_r)

        # ---- 差异化遗忘：直接衰减【进入预测的隐状态】 ----
        # delta_t_norm[t] = 归一化的「到下一次交互的时间间隔」（预测时已知，非泄漏）
        if delta_t_seq.dim() == 2:
            delta_t_seq = delta_t_seq.unsqueeze(-1)
        last_delta = torch.zeros(B, 1, 1, device=device, dtype=delta_t_seq.dtype)
        delta_t_next = torch.cat([delta_t_seq[:, 1:, :], last_delta], dim=1)
        delta_t_norm = (torch.log1p(delta_t_next.clamp(min=0)) /
                        torch.log1p(torch.tensor(30.0, device=device)))

        # 残差门控衰减：系数 β+(1-β)exp(-λΔt) (B,T,1) 广播到隐状态 (B,T,H)。
        # Storage 慢忘+高保留(lambda_s, beta_s)，Retrieval 快忘+低保留(lambda_r, beta_r)。
        if ablation_mode != 'no_storage':
            h_s = self.apply_decay(h_s, delta_t_norm, self.lambda_s, self.beta_s)
        if ablation_mode not in ('no_retrieval', 'single_path'):
            # shared_decay 消融：Retrieval 也用 lambda_s/beta_s，去掉差异化遗忘
            if ablation_mode == 'shared_decay':
                lambda_r_used, beta_r_used = self.lambda_s, self.beta_s
            else:
                lambda_r_used, beta_r_used = self.lambda_r, self.beta_r
            h_r = self.apply_decay(h_r, delta_t_norm, lambda_r_used, beta_r_used)

        # ---- 可解释的双状态强度（从衰减后的隐状态导出，用于可视化/解释）----
        if ablation_mode == 'no_storage':
            s_seq = torch.full((B, T, 1), 0.5, device=device)
        else:
            s_seq = torch.sigmoid(self.fc_S(h_s))  # (B, T, 1)
        if ablation_mode in ('no_retrieval', 'single_path'):
            r_seq = torch.zeros(B, T, 1, device=device)
        else:
            r_seq = torch.sigmoid(self.fc_R(h_r))  # (B, T, 1)

        # ---- 单路径退化：等价 DKT-Forget，只用衰减后的 Storage ----
        if ablation_mode == 'single_path':
            p_seq = torch.sigmoid(self.fc_pred(h_s))
            return p_seq, s_seq, r_seq

        # ---- 融合层（对【衰减后】的隐状态融合）----
        if ablation_mode == 'no_attention':
            # 简单平均融合
            h_fused = 0.5 * h_s + 0.5 * h_r
        elif ablation_mode == 'no_storage':
            h_fused = h_r
        elif ablation_mode == 'no_retrieval':
            h_fused = h_s
        else:
            # 注意力融合：门控由衰减后的双状态强度生成
            sr_cat = torch.cat([s_seq, r_seq], dim=-1)  # (B, T, 2)
            alpha = torch.softmax(self.fusion_attn(sr_cat), dim=-1)  # (B, T, 2)
            alpha_s = alpha[..., 0:1]
            alpha_r = alpha[..., 1:2]
            h_fused = alpha_s * h_s + alpha_r * h_r

        # ---- 最终预测 ----
        p_seq = torch.sigmoid(self.fc_pred(h_fused))  # (B, T, 1)

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