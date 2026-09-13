import torch
import torch.nn as nn

from src.config import ModelConfig
from src.embedding import TokenAndPositionEmbedding
from src.block import TransformerBlock


class OpsLMToy(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        # ====================================================
        # 1. Embedding
        # ====================================================

        self.embedding = TokenAndPositionEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # ====================================================
        # 2. Transformer Blocks
        # ====================================================

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # ====================================================
        # 3. Final LayerNorm
        # ====================================================

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        # ====================================================
        # 4. LM Head
        # ====================================================

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # ====================================================
        # 5. 初始化全部参数
        #
        # 注意：
        # Weight Tying 要放到初始化之后
        # ====================================================

        self.apply(
            self._init_weights
        )

        # ====================================================
        # 6. Weight Tying
        #
        # 输入 Embedding
        # 和输出 LM Head
        # 共用一份权重
        # ====================================================

        self.lm_head.weight = (
            self.embedding
            .token_embedding
            .weight
        )

    # ========================================================
    # 参数初始化
    # ========================================================

    @staticmethod
    def _init_weights(
        module: nn.Module,
    ) -> None:

        # ----------------------------------------------------
        # Linear
        # ----------------------------------------------------

        if isinstance(
            module,
            nn.Linear,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        elif isinstance(
            module,
            nn.Embedding,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        # ----------------------------------------------------
        # LayerNorm
        # ----------------------------------------------------

        elif isinstance(
            module,
            nn.LayerNorm,
        ):

            nn.init.ones_(
                module.weight
            )

            nn.init.zeros_(
                module.bias
            )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        # [B,T]
        #
        # ->
        #
        # [B,T,d_model]

        x = self.embedding(
            input_ids
        )

        # Transformer Blocks

        for block in self.blocks:

            x = block(x)

        # Final Norm

        x = self.final_norm(x)

        # LM Head

        logits = self.lm_head(x)

        # [B,T,vocab_size]

        return logits


# ============================================================
# 参数统计
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


def count_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )
