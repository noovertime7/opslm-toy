import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.mlp import MLP


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ====================================================
        # LayerNorm 1
        # ====================================================

        self.ln1 = nn.LayerNorm(
            d_model
        )

        # ====================================================
        # Multi-Head Attention
        # ====================================================

        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # ====================================================
        # LayerNorm 2
        # ====================================================

        self.ln2 = nn.LayerNorm(
            d_model
        )

        # ====================================================
        # MLP
        # ====================================================

        self.mlp = MLP(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ====================================================
        # 第一部分
        #
        # LayerNorm
        # ->
        # Attention
        # ->
        # Residual
        # ====================================================

        x = (
            x
            +
            self.attention(
                self.ln1(x)
            )
        )

        # ====================================================
        # 第二部分
        #
        # LayerNorm
        # ->
        # MLP
        # ->
        # Residual
        # ====================================================

        x = (
            x
            +
            self.mlp(
                self.ln2(x)
            )
        )

        return x
