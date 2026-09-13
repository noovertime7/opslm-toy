import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


# ============================================================
# Root
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT),
)


# ============================================================
# Import
# ============================================================

from src.config import ModelConfig

from src.embedding import (
    TokenAndPositionEmbedding
)

from src.attention import (
    MultiHeadAttention
)

from src.mlp import (
    MLP
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
        d_ff=2048,
        max_seq_len=1024,
        dropout=0.0,
    )

    # ========================================================
    # Modules
    # ========================================================

    embedding = TokenAndPositionEmbedding(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    )

    attention = MultiHeadAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    )

    mlp = MLP(
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout=0.0,
    )

    # ========================================================
    # Input
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

    attention_output = attention(
        x
    )

    mlp_output = mlp(
        attention_output
    )

    # ========================================================
    # Print
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
        "Attention:",
        attention_output.shape
    )

    print(
        "MLP:",
        mlp_output.shape
    )


if __name__ == "__main__":
    main()
