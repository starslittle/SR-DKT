# 功能：实现 DKT(2015)、DKT-F(遗忘增强)、SAKT(2019)、AKT(2020) 与 LBKT(2021) 基线模型。
# 作者：Cursor AI
# 日期：2026-05-10
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_position_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(seq_len, device=device).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class DKT(nn.Module):
    """Deep Knowledge Tracing, Piech et al., 2015."""

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.lstm = nn.LSTM(num_kc + 1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, kc_ids: torch.Tensor, corrects: torch.Tensor, mask: torch.Tensor | None = None):
        kc_one_hot = F.one_hot(kc_ids.long().clamp(min=0), num_classes=self.num_kc).float()
        x = torch.cat([kc_one_hot, corrects.float().unsqueeze(-1)], dim=-1)
        hidden, _ = self.lstm(x)
        logits = torch.sigmoid(self.fc(hidden))
        return logits, hidden


class DKT_F(nn.Module):
    """DKT with Forgetting, extending DKT with time-decayed hidden states."""

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(num_kc + 1, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.lambda_param = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        kc_ids: torch.Tensor,
        corrects: torch.Tensor,
        mask: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ):
        batch_size, seq_len = kc_ids.shape
        device = kc_ids.device
        kc_one_hot = F.one_hot(kc_ids.long().clamp(min=0), num_classes=self.num_kc).float()
        x = torch.cat([kc_one_hot, corrects.float().unsqueeze(-1)], dim=-1)
        if delta_t is None:
            delta_t = torch.zeros(batch_size, seq_len, device=device)

        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs = []
        lambda_value = F.softplus(self.lambda_param)
        for t in range(seq_len):
            h = h * torch.exp(-lambda_value * delta_t[:, t].float().unsqueeze(-1).clamp(min=0.0))
            h_new, c_new = self.cell(x[:, t, :], (h, c))
            if mask is not None:
                valid = mask[:, t].float().unsqueeze(-1)
                h = h_new * valid + h * (1.0 - valid)
                c = c_new * valid + c * (1.0 - valid)
            else:
                h, c = h_new, c_new
            outputs.append(h)
        hidden = torch.stack(outputs, dim=1)
        logits = torch.sigmoid(self.fc(hidden))
        return logits, hidden


class SAKT(nn.Module):
    """Self-Attentive Knowledge Tracing, Pandey and Karypis, 2019."""

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.kc_embed = nn.Embedding(num_kc, hidden_size)
        self.correct_embed = nn.Embedding(2, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def encode(self, kc_ids: torch.Tensor, corrects: torch.Tensor) -> torch.Tensor:
        x = self.kc_embed(kc_ids.long().clamp(min=0)) + self.correct_embed(corrects.long().clamp(min=0, max=1))
        return x + sinusoidal_position_encoding(kc_ids.shape[1], self.hidden_size, kc_ids.device)

    def forward(self, kc_ids: torch.Tensor, corrects: torch.Tensor, mask: torch.Tensor | None = None):
        x = self.encode(kc_ids, corrects)
        seq_len = kc_ids.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=kc_ids.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = ~mask.bool() if mask is not None else None
        attn_output, _ = self.attn(x, x, x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        hidden = self.norm(x + attn_output)
        logits = torch.sigmoid(self.fc(hidden))
        return logits, hidden


class AKT(nn.Module):
    """Attentive Knowledge Tracing with monotonic distance penalty, Ghosh et al., 2020."""

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.kc_embed = nn.Embedding(num_kc, hidden_size)
        self.correct_embed = nn.Embedding(2, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.penalty = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, kc_ids: torch.Tensor, corrects: torch.Tensor, mask: torch.Tensor | None = None):
        x = self.kc_embed(kc_ids.long().clamp(min=0)) + self.correct_embed(corrects.long().clamp(min=0, max=1))
        x = x + sinusoidal_position_encoding(kc_ids.shape[1], self.hidden_size, kc_ids.device)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)

        seq_len = kc_ids.shape[1]
        idx = torch.arange(seq_len, device=kc_ids.device)
        distance = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)).float()
        scores = scores - F.softplus(self.penalty) * distance.unsqueeze(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=kc_ids.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        if mask is not None:
            scores = scores.masked_fill((~mask.bool()).unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, v)
        hidden = self.norm(x + attn_output)
        logits = torch.sigmoid(self.fc(hidden))
        return logits, hidden


class LBKT(nn.Module):
    """Learning Behavior-aware Knowledge Tracing, Yang et al., 2021."""

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.kc_embed = nn.Embedding(num_kc, hidden_size)
        self.correct_embed = nn.Embedding(2, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.behavior_gate = nn.Linear(2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        kc_ids: torch.Tensor,
        corrects: torch.Tensor,
        mask: torch.Tensor | None = None,
        hint_count: torch.Tensor | None = None,
        attempt_count: torch.Tensor | None = None,
    ):
        x = self.kc_embed(kc_ids.long().clamp(min=0)) + self.correct_embed(corrects.long().clamp(min=0, max=1))
        x = x + sinusoidal_position_encoding(kc_ids.shape[1], self.hidden_size, kc_ids.device)
        seq_len = kc_ids.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=kc_ids.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = ~mask.bool() if mask is not None else None
        attn_output, _ = self.attn(x, x, x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        if hint_count is None:
            hint_count = torch.zeros_like(corrects)
        if attempt_count is None:
            attempt_count = torch.ones_like(corrects)
        behavior = torch.stack([hint_count.float().clamp(0, 5) / 5.0, attempt_count.float().clamp(0, 3) / 3.0], dim=-1)
        gate = torch.sigmoid(self.behavior_gate(behavior))
        hidden = self.norm(x + gate * attn_output)
        logits = torch.sigmoid(self.fc(hidden))
        return logits, hidden


class DKVMN(nn.Module):
    """
    DKVMN: Dynamic Key-Value Memory Networks for Knowledge Tracing (2017)

    核心机制：
    - Key Memory: 存储知识概念的静态表示（固定）
    - Value Memory: 动态更新学生对每个概念的掌握状态
    - Read: 软注意力检索 → 当前知识状态
    - Write: 根据答题结果更新 Value Memory

    输入：kc_id + correct（标准输入，不需要额外特征）
    """

    def __init__(self, num_kc: int, hidden_size: int = 128,
                 memory_size: int = 20, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.memory_size = memory_size  # N：记忆槽数量

        # Key Memory（静态，可学习）
        self.key_memory = nn.Parameter(
            torch.randn(memory_size, hidden_size // 2)
        )

        # 题目 embedding → 查询 Key Memory
        self.kc_embed = nn.Embedding(num_kc + 1, hidden_size // 2, padding_idx=0)

        # 交互 embedding（kc + correct）→ 写入 Value Memory
        self.interaction_embed = nn.Embedding(
            num_kc * 2 + 2, hidden_size, padding_idx=0
        )

        # Read 网络：[value_read, kc_emb] → 知识状态
        self.read_fc = nn.Linear(hidden_size + hidden_size // 2, hidden_size)

        # Write 网络：erase + add
        self.erase_fc = nn.Linear(hidden_size, hidden_size)
        self.add_fc = nn.Linear(hidden_size, hidden_size)

        # 预测层
        self.fc_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def attention(self, kc_emb: torch.Tensor) -> torch.Tensor:
        """
        计算 kc_emb 对 Key Memory 的软注意力权重
        """
        scores = torch.matmul(kc_emb, self.key_memory.T)
        return torch.softmax(scores, dim=-1)

    def read(self, value_memory: torch.Tensor,
             weights: torch.Tensor) -> torch.Tensor:
        return torch.bmm(weights.unsqueeze(1), value_memory).squeeze(1)

    def write(self, value_memory: torch.Tensor, weights: torch.Tensor,
              interaction_emb: torch.Tensor) -> torch.Tensor:
        erase = torch.sigmoid(self.erase_fc(interaction_emb))
        add = torch.tanh(self.add_fc(interaction_emb))
        w = weights.unsqueeze(-1)
        erase_term = torch.bmm(w, erase.unsqueeze(1))
        value_memory = value_memory * (1 - erase_term)
        add_term = torch.bmm(w, add.unsqueeze(1))
        value_memory = value_memory + add_term
        return value_memory

    def forward(self, kc_ids: torch.Tensor, corrects: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, None]:
        B, T = kc_ids.shape
        device = kc_ids.device

        value_memory = torch.zeros(B, self.memory_size, self.hidden_size,
                                   device=device)
        interaction_ids = (kc_ids * 2 + corrects.long()).clamp(
            0, self.num_kc * 2 + 1
        )
        preds_list = []

        for t in range(T):
            kc_t = kc_ids[:, t].clamp(0, self.num_kc)
            inter_t = interaction_ids[:, t]
            kc_emb_t = self.kc_embed(kc_t)
            inter_emb_t = self.interaction_embed(inter_t)
            weights_t = self.attention(kc_emb_t)
            read_t = self.read(value_memory, weights_t)
            ks_t = torch.tanh(
                self.read_fc(torch.cat([read_t, kc_emb_t], dim=-1))
            )
            pred_t = torch.sigmoid(self.fc_pred(ks_t))
            preds_list.append(pred_t)
            value_memory = self.write(value_memory, weights_t, inter_emb_t)

        preds = torch.stack(preds_list, dim=1)
        return preds, None


class BEKT(nn.Module):
    """
    BEKT: Behavior-Enhanced Knowledge Tracing (2026)

    核心机制：
    1. 题目/知识双编码器（自注意力）
    2. 时间衰减注意力检索知识状态
    3. 行为波动层（hint/attempt/time → 状态修正）
    4. MLP 融合增强预测

    输入：kc_id + correct + hint_count + attempt_count + delta_t
    """

    def __init__(self, num_kc: int, hidden_size: int = 128, num_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size

        self.kc_embed = nn.Embedding(num_kc + 1, hidden_size, padding_idx=0)
        self.correct_embed = nn.Embedding(2, hidden_size)
        self.interaction_embed = nn.Embedding(num_kc * 2 + 1, hidden_size, padding_idx=0)

        self.question_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 2, dropout=dropout,
            batch_first=True
        )

        self.knowledge_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 2, dropout=dropout,
            batch_first=True
        )

        self.theta = nn.Parameter(torch.tensor(0.1))

        self.retrieval_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.behavior_proj = nn.Linear(3, hidden_size)
        self.behavior_gate = nn.Linear(3, 1)
        self.behavior_threshold = 0.5

        self.cross_proj = nn.Linear(hidden_size * 2, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

        self.fc_pred = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, kc_ids: torch.Tensor, corrects: torch.Tensor,
                mask: torch.Tensor, hint_count: torch.Tensor = None,
                attempt_count: torch.Tensor = None,
                delta_t: torch.Tensor = None) -> tuple[torch.Tensor, None]:
        B, T = kc_ids.shape
        device = kc_ids.device

        if hint_count is None:
            hint_count = torch.zeros(B, T, device=device)
        if attempt_count is None:
            attempt_count = torch.zeros(B, T, device=device)
        if delta_t is None:
            delta_t = torch.zeros(B, T, device=device)

        interaction_ids = (kc_ids * 2 + corrects.long()).clamp(0, self.num_kc * 2)

        kc_emb = self.kc_embed(kc_ids.clamp(0, self.num_kc))
        inter_emb = self.interaction_embed(interaction_ids)

        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        q_enc = self.question_encoder(
            kc_emb,
            src_mask=causal_mask,
            src_key_padding_mask=~mask
        )

        k_enc = self.knowledge_encoder(
            inter_emb,
            src_mask=causal_mask,
            src_key_padding_mask=~mask
        )

        delta_norm = torch.log1p(delta_t.float().clamp(min=0)) / \
                     torch.log1p(torch.tensor(30.0, device=device))
        time_decay = torch.exp(-F.softplus(self.theta) * delta_norm)

        k_enc_decayed = k_enc * time_decay.unsqueeze(-1)
        ks_retrieved, _ = self.retrieval_attn(
            q_enc, k_enc_decayed, k_enc_decayed,
            attn_mask=causal_mask,
            key_padding_mask=~mask
        )

        hint_norm = hint_count.float().clamp(0, 5) / 5.0
        attempt_norm = attempt_count.float().clamp(0, 3) / 3.0
        behavior_feat = torch.stack([hint_norm, attempt_norm, delta_norm], dim=-1)
        behavior_vec = torch.tanh(self.behavior_proj(behavior_feat))

        gate = torch.sigmoid(self.behavior_gate(behavior_feat))
        sign = torch.where(gate > self.behavior_threshold,
                           torch.ones_like(gate) * -1,
                           torch.ones_like(gate))
        behavior_delta = sign * behavior_vec

        ks_updated = ks_retrieved + behavior_delta

        cross_feat = torch.cat([ks_retrieved, behavior_delta], dim=-1)
        cross_out = torch.relu(self.cross_proj(cross_feat))

        mlp_input = torch.cat([ks_updated, cross_out], dim=-1)
        mlp_out = self.mlp(mlp_input)

        preds = torch.sigmoid(self.fc_pred(mlp_out))

        return preds, None
