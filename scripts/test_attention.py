import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT),
)


from src.config import ModelConfig

from src.embedding import (
    TokenAndPositionEmbedding
)

from src.attention import (
    SelfAttentionHead
)


def main():

    torch.manual_seed(42)

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer_path = (
        ROOT
        / "tokenizer"
        / "artifacts"
        / "tokenizer.json"
    )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    # ========================================================
    # Config
    # ========================================================

    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=1024,
        d_model=512,
        n_heads=8,
    )

    # ========================================================
    # Embedding
    # ========================================================

    embedding = (
        TokenAndPositionEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=0.0,
        )
    )

    # ========================================================
    # 一个 Attention Head
    # ========================================================

    attention = SelfAttentionHead(
        d_model=config.d_model,
        head_dim=config.head_dim,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    )

    # ========================================================
    # 输入
    # ========================================================

    text = (
        "kubectl get pods -n production"
    )

    encoded = tokenizer.encode(
        text
    )

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
    )

    print(
        "Tokens:",
        encoded.tokens
    )

    print(
        "Input IDs shape:",
        input_ids.shape
    )

    # ========================================================
    # Embedding
    # ========================================================

    x = embedding(
        input_ids
    )

    print(
        "Embedding shape:",
        x.shape
    )

    # ========================================================
    # Attention
    # ========================================================

    output = attention(
        x
    )

    print(
        "Attention output shape:",
        output.shape
    )


if __name__ == "__main__":
    main()
