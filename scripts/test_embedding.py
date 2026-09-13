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


# ============================================================
# Tokenizer
# ============================================================

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


# ============================================================
# Config
# ============================================================

config = ModelConfig(

    vocab_size=vocab_size,

    max_seq_len=1024,

    d_model=512,

)


# ============================================================
# Embedding
# ============================================================

embedding = (

    TokenAndPositionEmbedding(

        vocab_size=config.vocab_size,

        d_model=config.d_model,

        max_seq_len=config.max_seq_len,

        dropout=config.dropout,

    )

)


# ============================================================
# Test Text
# ============================================================

text = (
    "kubectl get pods -n production"
)


encoded = tokenizer.encode(
    text
)


print(
    "Text:",
    text
)


print(
    "Tokens:",
    encoded.tokens
)


print(
    "IDs:",
    encoded.ids
)


# ============================================================
# Tensor
# ============================================================

input_ids = torch.tensor(

    [encoded.ids],

    dtype=torch.long,

)


print()

print(
    "Input shape:",
    input_ids.shape
)


# ============================================================
# Forward
# ============================================================

embedding.eval()


with torch.no_grad():

    output = embedding(
        input_ids
    )


print(

    "Output shape:",

    output.shape,

)


# ============================================================
# 查看前几个位置
# ============================================================

print()

for i in range(
    min(
        len(encoded.tokens),
        3,
    )
):

    print(
        f"Token {i}:",
        encoded.tokens[i]
    )

    print(
        "First 10 values:"
    )

    print(
        output[0, i, :10]
    )

    print()
