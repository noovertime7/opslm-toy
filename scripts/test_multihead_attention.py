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
    MultiHeadAttention
)


def main():

    torch.manual_seed(42)

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer = Tokenizer.from_file(
        str(
            ROOT
            / "tokenizer"
            / "artifacts"
            / "tokenizer.json"
        )
    )

    # ========================================================
    # Config
    # ========================================================

    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=512,
        n_heads=8,
        max_seq_len=1024,
        dropout=0.0,
    )

    # ========================================================
    # Embedding
    # ========================================================

    embedding = TokenAndPositionEmbedding(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    )

    # ========================================================
    # Multi-Head Attention
    # ========================================================

    attention = MultiHeadAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    )

    # ========================================================
    # Test Input
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

    # ========================================================
    # Forward
    # ========================================================

    x = embedding(
        input_ids
    )

    output = attention(
        x
    )

    # ========================================================
    # Output
    # ========================================================

    print(
        "Tokens:",
        encoded.tokens
    )

    print()

    print(
        "Input IDs:",
        input_ids.shape
    )

    print(
        "Embedding:",
        x.shape
    )

    print(
        "MultiHead Attention:",
        output.shape
    )

    print()

    print(
        "n_heads:",
        config.n_heads
    )

    print(
        "head_dim:",
        config.head_dim
    )


if __name__ == "__main__":
    main()
