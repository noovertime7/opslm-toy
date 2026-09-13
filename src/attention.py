import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Single Self-Attention Head
# ============================================================

class SelfAttentionHead(nn.Module):

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.head_dim = head_dim

        # ----------------------------------------------------
        # Q / K / V Projection
        # ----------------------------------------------------

        self.query = nn.Linear(
            d_model,
            head_dim,
            bias=False,
        )

        self.key = nn.Linear(
            d_model,
            head_dim,
            bias=False,
        )

        self.value = nn.Linear(
            d_model,
            head_dim,
            bias=False,
        )

        # ----------------------------------------------------
        # Attention Dropout
        # ----------------------------------------------------

        self.attn_dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Causal Mask
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        # ----------------------------------------------------

        causal_mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        # x:
        #
        # [B, T, C]
        #
        # B = batch size
        # T = sequence length
        # C = d_model

        _, seq_len, _ = x.shape

        # ----------------------------------------------------
        # 1. Q / K / V
        # ----------------------------------------------------

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # q/k/v:
        #
        # [B, T, head_dim]

        # ----------------------------------------------------
        # 2. Attention Score
        #
        # Q @ K^T
        # ----------------------------------------------------

        scores = (
            q
            @
            k.transpose(-2, -1)
        )

        # scores:
        #
        # [B, T, T]

        # ----------------------------------------------------
        # 3. Scale
        # ----------------------------------------------------

        scores = (
            scores
            /
            math.sqrt(
                self.head_dim
            )
        )

        # ----------------------------------------------------
        # 4. Causal Mask
        # ----------------------------------------------------

        mask = self.causal_mask[
            :seq_len,
            :seq_len
        ]

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        # ----------------------------------------------------
        # 5. Softmax
        # ----------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        attention_weights = (
            self.attn_dropout(
                attention_weights
            )
        )

        # ----------------------------------------------------
        # 6. Attention Weight @ V
        # ----------------------------------------------------

        output = (
            attention_weights
            @
            v
        )

        # output:
        #
        # [B, T, head_dim]

        if return_attention:
            return (
                output,
                attention_weights,
            )

        return output


# ============================================================
# Multi-Head Self-Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 基本检查
        # ----------------------------------------------------

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.d_model = d_model
        self.n_heads = n_heads

        self.head_dim = (
            d_model
            //
            n_heads
        )

        # ----------------------------------------------------
        # 创建多个 Attention Head
        # ----------------------------------------------------

        self.heads = nn.ModuleList([
            SelfAttentionHead(
                d_model=d_model,
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            for _ in range(n_heads)
        ])

        # ----------------------------------------------------
        # Output Projection
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # ----------------------------------------------------
        # Residual Dropout
        # ----------------------------------------------------

        self.resid_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x:
        #
        # [B, T, d_model]

        # ----------------------------------------------------
        # 1. 每个 Head 独立计算
        # ----------------------------------------------------

        head_outputs = [
            head(x)
            for head in self.heads
        ]

        # 每个 head:
        #
        # [B, T, head_dim]

        # ----------------------------------------------------
        # 2. Concat
        # ----------------------------------------------------

        x = torch.cat(
            head_outputs,
            dim=-1,
        )

        # [B, T, head_dim * n_heads]
        #
        # =
        #
        # [B, T, d_model]

        # ----------------------------------------------------
        # 3. Output Projection
        # ----------------------------------------------------

        x = self.out_proj(x)

        # ----------------------------------------------------
        # 4. Dropout
        # ----------------------------------------------------

        x = self.resid_dropout(x)

        return x
