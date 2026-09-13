from dataclasses import dataclass


@dataclass
class ModelConfig:

    # Tokenizer 词表大小
    vocab_size: int

    # 最大上下文长度
    max_seq_len: int = 1024

    # 每个 Token 的向量维度
    d_model: int = 512

    # Attention Head 数量
    n_heads: int = 8

    # Transformer Block 数量
    n_layers: int = 8

    # MLP 中间层大小
    d_ff: int = 2048

    # Dropout
    dropout: float = 0.1


    def __post_init__(self):

        if self.d_model % self.n_heads != 0:

            raise ValueError(
                "d_model must be divisible by n_heads"
            )


    @property
    def head_dim(self):

        return (
            self.d_model
            //
            self.n_heads
        )
