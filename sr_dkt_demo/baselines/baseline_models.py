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
    """Deep Knowledge Tracing, Piech et al., 2015.
    
    NOTE: Use Embedding instead of one-hot encoding to support 
    large KC vocabulary (e.g. MOOCCubeX with 30,000+ KCs).
    This reduces input dimension from O(num_kc) to O(hidden_size).
    """

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.kc_embed = nn.Embedding(num_kc + 1, hidden_size)  # +1 for OOD/OTHER
        self.lstm = nn.LSTM(hidden_size + 1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, kc_ids: torch.Tensor, corrects: torch.Tensor, mask: torch.Tensor | None = None):
        kc_emb = self.kc_embed(kc_ids.long().clamp(min=0, max=self.num_kc))
        x = torch.cat([kc_emb, corrects.float().unsqueeze(-1)], dim=-1)
        hidden, _ = self.lstm(x)
        logits = torch.sigmoid(self.fc(hidden))
        return logits, hidden


class DKT_F(nn.Module):
    """DKT with Forgetting, extending DKT with time-decayed hidden states.
    
    NOTE: Use Embedding instead of one-hot encoding to support 
    large KC vocabulary (e.g. MOOCCubeX with 30,000+ KCs).
    """

    def __init__(self, num_kc: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.kc_embed = nn.Embedding(num_kc + 1, hidden_size)  # +1 for OOD/OTHER
        self.cell = nn.LSTMCell(hidden_size + 1, hidden_size)
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
        kc_emb = self.kc_embed(kc_ids.long().clamp(min=0, max=self.num_kc))
        x = torch.cat([kc_emb, corrects.float().unsqueeze(-1)], dim=-1)
        if delta_t is None:
            delta_t = torch.zeros(batch_size, seq_len, device=device)

        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        outputs = []
        lambda_value = F.softplus(self.lambda_param)
        for t in range(seq_len):
            decay_t = torch.exp(-lambda_value * delta_t[:, t].float().unsqueeze(-1).clamp(min=0.0))
            h = h * decay_t
            c = c * decay_t
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


# ──────────────────── Transformer Block (shared) ────────────────────

class TransformerBlock(nn.Module):
    """Single Transformer encoder block: MultiheadAttention + FFN + residual + LN."""

    def __init__(self, emb_size: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(emb_size)

        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Cross-attention: Q attends to K=V. Returns transformed K/V.

        Uses generate_square_subsequent_mask for correct causal masking
        and converts key_padding_mask to float to avoid type conflicts.
        """
        seq_len = k.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=k.device
        )  # float32, upper-tri = -inf

        # key_padding_mask must match dtype (float, not bool)
        kp_float: torch.Tensor | None = None
        if key_padding_mask is not None:
            kp_float = torch.zeros_like(key_padding_mask, dtype=torch.float)
            kp_float = kp_float.masked_fill(key_padding_mask, float("-inf"))

        attn_out, _ = self.attn(
            q, k, v, attn_mask=causal_mask, key_padding_mask=kp_float
        )
        attn_out = self.attn_norm(q + self.attn_dropout(attn_out))

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(attn_out + self.ffn_dropout(ffn_out))

        return ffn_out  # serves as k/v for next block


# ──────────────────── SAKT (pyKT 兼容 cross-attention) ────────────────────

class SAKT(nn.Module):
    """SAKT with cross-attention, Pandey & Karypis, 2019 (pyKT-compatible).

    Core mechanism:
    - interaction_emb(q + num_kc * r)  →  encodes history as key/value
    - exercise_emb(next_exercise)      →  next exercise as query
    - Cross-attention layers           →  Q attends to K=V with causal mask
    - Output: next-step prediction probability

    Differs from naive self-attention SAKT: uses the next exercise embedding
    as an explicit query token, which is the original paper's design.
    """

    def __init__(self, num_kc: int, hidden_size: int = 128, seq_len: int = 200,
                 num_heads: int = 4, dropout: float = 0.1, num_layers: int = 2) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size

        self.interaction_emb = nn.Embedding(num_kc * 2, hidden_size)
        self.exercise_emb = nn.Embedding(num_kc, hidden_size)

        self.pos_emb = nn.Embedding(seq_len, hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(hidden_size, 1)

    def forward(self, kc_ids: torch.Tensor, corrects: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward with cross-attention.

        Args:
            kc_ids:  (B, T)  concept/exercise IDs
            corrects: (B, T)  0/1 correctness
            mask:     (B, T)  bool padding mask

        Returns:
            preds:  (B, T, 1)  predictions, pred[t] answers corrects[t+1]
            hidden: None       (kept for API compatibility)
        """
        B, T = kc_ids.shape
        device = kc_ids.device
        T1 = T - 1  # number of prediction steps

        if T1 <= 0:
            preds = torch.zeros(B, T, 1, device=device)
            return preds, None

        # ── History (key/value) and Query ──
        q_hist = kc_ids[:, :T1].long().clamp(min=0, max=self.num_kc - 1)
        r_hist = corrects[:, :T1].long().clamp(min=0, max=1)
        q_query = kc_ids[:, 1:].long().clamp(min=0, max=self.num_kc - 1)

        # Interaction embedding for past history  →  K, V
        x_int = q_hist + self.num_kc * r_hist                     # (B, T-1)
        kv_emb = self.interaction_emb(x_int)                     # (B, T-1, E)

        # Exercise embedding for next-step query  →  Q
        q_emb = self.exercise_emb(q_query)                       # (B, T-1, E)

        # Position encoding (learnable)
        positions = torch.arange(T1, device=device).unsqueeze(0).expand(B, -1)
        kv_emb = kv_emb + self.pos_emb(positions)

        # Padding mask for history
        kp_mask = ~mask[:, :T1] if mask is not None else None

        # ── Cross-attention blocks ──
        kv_out = kv_emb
        for block in self.blocks:
            kv_out = block(q_emb, kv_out, kv_out, key_padding_mask=kp_mask)

        kv_out = self.dropout(kv_out)
        preds_step = torch.sigmoid(self.pred(kv_out)).squeeze(-1)   # (B, T-1)

        # ── Align with next-step prediction format ──
        # preds[t] should predict corrects[t+1] for t=0..T-2
        preds = torch.zeros(B, T, 1, device=device)
        preds[:, :T1, 0] = preds_step

        return preds, None


# ──────────────────── AKT (论文对齐版：双 Transformer + 距离感知注意力) ────────────────────

class AKTBlock(nn.Module):
    """Single Transformer block for AKT: distance-aware attention + FFN + residual + LN."""

    def __init__(self, emb_size: int, num_heads: int = 4, dropout: float = 0.1,
                 kq_same: bool = True) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.d_k = emb_size // num_heads
        self.num_heads = num_heads
        self.kq_same = kq_same

        # QKV projections
        self.q_linear = nn.Linear(emb_size, emb_size)
        if kq_same:
            self.k_linear = self.q_linear  # share K/Q weights (pyKT default)
        else:
            self.k_linear = nn.Linear(emb_size, emb_size)
        self.v_linear = nn.Linear(emb_size, emb_size)

        # Per-head gamma for distance-aware monotonic attention
        self.gammas = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)

        self.out_proj = nn.Linear(emb_size, emb_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(emb_size)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(emb_size)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                apply_pos: bool = True, causal_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Distance-aware multi-head attention with residual + FFN.

        Args:
            q: Query tensor (B, T_q, E)
            k: Key tensor (B, T_k, E)
            v: Value tensor (B, T_k, E)
            apply_pos: Whether to apply residual + norm (False for first sub-layer in blocks_2)
            causal_mask: (T_q, T_k) bool mask, True = do NOT attend
            key_padding_mask: (B, T_k) bool mask, True = padded (ignore)
        """
        B, T_q, _ = q.shape
        T_k = k.size(1)

        # QKV projections
        q_proj = self.q_linear(q).view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        k_proj = self.k_linear(k).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        v_proj = self.v_linear(v).view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Distance-aware exponential decay (per-head, monotonic)
        idx_q = torch.arange(T_q, device=q.device)
        idx_k = torch.arange(T_k, device=k.device)
        distance = (idx_q.unsqueeze(1) - idx_k.unsqueeze(0)).abs().float()  # (T_q, T_k)
        gamma = -F.softplus(self.gammas)  # negative → decay
        decay = torch.exp(gamma * distance.unsqueeze(0))  # (num_heads, T_q, T_k)
        scores = scores * decay  # multiplicative decay (论文原始方式)

        # Causal mask
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        attn_out = torch.matmul(weights, v_proj)  # (B, H, T_q, d_k)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, self.emb_size)
        attn_out = self.out_proj(attn_out)

        # Residual + LayerNorm
        if apply_pos:
            attn_out = self.attn_norm(q + attn_out)
        else:
            attn_out = q + attn_out

        # FFN + residual + LayerNorm
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(attn_out + self.ffn_dropout(ffn_out))

        return ffn_out


class AKT(nn.Module):
    """AKT: Context-Aware Attentive Knowledge Tracing, Ghosh et al., 2020.

    Aligned with original paper architecture:
    - blocks_1: QA interaction encoder (self-attention with distance-aware decay)
    - blocks_2: Question decoder (alternating self-attn + cross-attn with blocks_1 output)
    - Output: concat(context_output, q_embed) → 3-layer MLP (skip connection)
    - Rasch difficulty: omitted (akt_cid variant for concept-level, MOOCCubeX has no problem-id)

    This follows pyKT's AKT implementation (akt_cid mode) with:
    - n_blocks=1 default → blocks_1 has 1 layer, blocks_2 has 2 layers
    - Distance-aware attention with per-head gamma (multiplicative exponential decay)
    - Separate question embedding + QA interaction embedding
    """

    def __init__(self, num_kc: int, hidden_size: int = 128, num_heads: int = 4,
                 dropout: float = 0.1, n_blocks: int = 1,
                 final_fc_dim: int = 512, kq_same: bool = True) -> None:
        super().__init__()
        self.num_kc = num_kc
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks

        # Separate embeddings (matching original paper)
        # Question/concept embedding
        self.q_embed = nn.Embedding(num_kc, hidden_size)
        # QA interaction embedding: combines question + correctness
        self.qa_embed = nn.Embedding(num_kc * 2, hidden_size)

        # blocks_1: QA interaction encoder (self-attention)
        self.blocks_1 = nn.ModuleList([
            AKTBlock(hidden_size, num_heads, dropout, kq_same=kq_same)
            for _ in range(n_blocks)
        ])

        # blocks_2: Question decoder (2*n_blocks layers, alternating self + cross)
        self.blocks_2 = nn.ModuleList([
            AKTBlock(hidden_size, num_heads, dropout, kq_same=kq_same)
            for _ in range(n_blocks * 2)
        ])

        self.dropout_layer = nn.Dropout(dropout)

        # Output MLP with skip connection:
        # Input: concat(context_output, q_embed) = hidden_size * 2
        # Structure: d_model*2 → 512 → 256 → 1 (matching original)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, final_fc_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim // 2, 1),
        )

    def forward(self, kc_ids: torch.Tensor, corrects: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward with dual-Transformer + distance-aware attention.

        Args:
            kc_ids:   (B, T)  concept/exercise IDs
            corrects: (B, T)  0/1 correctness
            mask:     (B, T)  bool padding mask (True = valid)

        Returns:
            logits: (B, T, 1)  predictions
            hidden: (B, T, E)  contextualized representations
        """
        B, T = kc_ids.shape
        device = kc_ids.device

        # ── Embeddings ──
        q_clamped = kc_ids.long().clamp(min=0, max=self.num_kc - 1)
        r_clamped = corrects.long().clamp(min=0, max=1)

        # Question embedding
        q_emb = self.q_embed(q_clamped)  # (B, T, E)

        # QA interaction embedding: qa_id = q + num_kc * r
        qa_ids = q_clamped + self.num_kc * r_clamped  # range [0, 2*num_kc-1]
        qa_emb = self.qa_embed(qa_ids)  # (B, T, E)

        # Position encoding (sinusoidal, added to both)
        pe = sinusoidal_position_encoding(T, self.hidden_size, device)
        qa_emb = qa_emb + pe
        q_emb = q_emb + pe

        # ── Causal mask ──
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # Key padding mask: True = padded/ignore
        kp_mask = ~mask if mask is not None else None

        # ── blocks_1: QA interaction encoder ──
        # Self-attention on QA interactions
        y = qa_emb
        for block in self.blocks_1:
            y = block(y, y, y, apply_pos=True, causal_mask=causal_mask,
                      key_padding_mask=kp_mask)

        # ── blocks_2: Question decoder ──
        # Alternating: odd layers = self-attention on questions
        #              even layers = cross-attention (questions attend to encoded QA)
        x = q_emb
        for i, block in enumerate(self.blocks_2):
            if i % 2 == 0:
                # Odd layer (0-indexed even): self-attention on questions, no residual norm
                x = block(x, x, x, apply_pos=False, causal_mask=causal_mask,
                          key_padding_mask=kp_mask)
            else:
                # Even layer: cross-attention (Q=questions, K/V=encoded QA from blocks_1)
                x = block(x, y, y, apply_pos=True, causal_mask=causal_mask,
                          key_padding_mask=kp_mask)

        # ── Output with skip connection ──
        # Concatenate context output with original question embedding
        # (Skip connection matching original paper)
        q_emb_raw = self.q_embed(q_clamped)  # re-lookup without position encoding
        out_input = torch.cat([x, q_emb_raw], dim=-1)  # (B, T, 2*E)
        out_input = self.dropout_layer(out_input)
        logits = torch.sigmoid(self.out_mlp(out_input))  # (B, T, 1)

        return logits, x


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
        # Use float masks to avoid bool dtype conflicts in PyTorch 2.0+
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=kc_ids.device)
        key_padding_mask: torch.Tensor | None = None
        if mask is not None:
            key_padding_mask = torch.zeros_like(mask, dtype=torch.float)
            key_padding_mask = key_padding_mask.masked_fill(~mask, float("-inf"))
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
        self.kc_embed = nn.Embedding(num_kc + 1, hidden_size // 2)

        # 交互 embedding（kc + correct）→ 写入 Value Memory
        self.interaction_embed = nn.Embedding(
            num_kc * 2 + 2, hidden_size
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

        self.kc_embed = nn.Embedding(num_kc, hidden_size)
        self.correct_embed = nn.Embedding(2, hidden_size)
        self.interaction_embed = nn.Embedding(num_kc * 2, hidden_size)

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

        interaction_ids = (kc_ids * 2 + corrects.long()).clamp(0, self.num_kc * 2 - 1)

        kc_emb = self.kc_embed(kc_ids.clamp(0, self.num_kc - 1))
        inter_emb = self.interaction_embed(interaction_ids)

        # Use float masks to avoid bool dtype conflicts in PyTorch 2.0+
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        kp_float: torch.Tensor | None = None
        if mask is not None:
            kp_float = torch.zeros_like(mask, dtype=torch.float)
            kp_float = kp_float.masked_fill(~mask, float("-inf"))

        q_enc = self.question_encoder(
            kc_emb,
            src_mask=causal_mask,
            src_key_padding_mask=kp_float
        )

        k_enc = self.knowledge_encoder(
            inter_emb,
            src_mask=causal_mask,
            src_key_padding_mask=kp_float
        )

        delta_norm = torch.log1p(delta_t.float().clamp(min=0)) / \
                     torch.log1p(torch.tensor(30.0, device=device))
        time_decay = torch.exp(-F.softplus(self.theta) * delta_norm)

        k_enc_decayed = k_enc * time_decay.unsqueeze(-1)
        ks_retrieved, _ = self.retrieval_attn(
            q_enc, k_enc_decayed, k_enc_decayed,
            attn_mask=causal_mask,
            key_padding_mask=kp_float
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
