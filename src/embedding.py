import torch
import torch.nn as nn


class TokenAndPositionEmbedding(
    nn.Module
):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):

        super().__init__()


        # ====================================================
        # Token Embedding
        # ====================================================

        self.token_embedding = nn.Embedding(

            num_embeddings=vocab_size,

            embedding_dim=d_model,

        )


        # ====================================================
        # Position Embedding
        # ====================================================

        self.position_embedding = nn.Embedding(

            num_embeddings=max_seq_len,

            embedding_dim=d_model,

        )


        # ====================================================
        # Dropout
        # ====================================================

        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:


        # input_ids:
        #
        # [B,T]

        batch_size, seq_len = (
            input_ids.shape
        )


        # ====================================================
        # 防止超过最大上下文
        # ====================================================

        if (
            seq_len
            >
            self.position_embedding
            .num_embeddings
        ):

            raise ValueError(

                f"Sequence length "
                f"{seq_len} exceeds "
                f"max_seq_len "
                f"{self.position_embedding.num_embeddings}"

            )


        # ====================================================
        # 生成位置
        #
        # 0 1 2 3 4 ...
        # ====================================================

        positions = torch.arange(

            seq_len,

            device=input_ids.device,

        )


        # 当前：
        #
        # [T]
        #
        # 变为：
        #
        # [1,T]

        positions = (
            positions
            .unsqueeze(0)
        )


        # 扩展到：
        #
        # [B,T]

        positions = positions.expand(

            batch_size,

            seq_len,

        )


        # ====================================================
        # Token Embedding
        # ====================================================

        token_embeddings = (

            self.token_embedding(
                input_ids
            )

        )


        # ====================================================
        # Position Embedding
        # ====================================================

        position_embeddings = (

            self.position_embedding(
                positions
            )

        )


        # ====================================================
        # 两者相加
        # ====================================================

        x = (

            token_embeddings
            +
            position_embeddings

        )


        # ====================================================
        # Dropout
        # ====================================================

        x = self.dropout(x)


        return x
