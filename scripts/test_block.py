import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


# ============================================================
# Project Root
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

from src.block import (
    TransformerBlock
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
        n_layers=8,
        d_ff=2048,
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
    # Transformer Block
    # ========================================================

    block = TransformerBlock(
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
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
    # Embedding
    # ========================================================

    x = embedding(
        input_ids
    )

    # ========================================================
    # Transformer Block
    # ========================================================

    output = block(
        x
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
        "Input IDs shape:",
        input_ids.shape
    )

    print(
        "Embedding shape:",
        x.shape
    )

    print(
        "Block output shape:",
        output.shape
    )

    # ========================================================
    # 参数量
    # ========================================================

    total_params = sum(
        p.numel()
        for p in block.parameters()
    )

    print()

    print(
        "Block Parameters:",
        f"{total_params:,}"
    )

    print(
        "Block Parameters(M):",
        f"{total_params / 1_000_000:.3f}M"
    )

    # ========================================================
    # 参数名称
    # ========================================================

    print()

    print("Parameter Shapes:")

    for name, parameter in block.named_parameters():

        print(
            name,
            parameter.shape
        )


if __name__ == "__main__":
    main()
