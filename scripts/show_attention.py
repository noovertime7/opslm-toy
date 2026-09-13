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


torch.manual_seed(42)


tokenizer = Tokenizer.from_file(
    str(
        ROOT
        / "tokenizer"
        / "artifacts"
        / "tokenizer.json"
    )
)


config = ModelConfig(
    vocab_size=tokenizer.get_vocab_size(),
    d_model=512,
    n_heads=8,
    max_seq_len=1024,
)


embedding = (
    TokenAndPositionEmbedding(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len,
        dropout=0.0,
    )
)


attention = SelfAttentionHead(
    d_model=config.d_model,
    head_dim=config.head_dim,
    max_seq_len=config.max_seq_len,
    dropout=0.0,
)


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


x = embedding(
    input_ids
)


output, weights = attention(
    x,
    return_attention=True,
)


tokens = encoded.tokens


print(
    "Tokens:"
)

for i, token in enumerate(tokens):

    print(
        i,
        repr(token)
    )


print()

print(
    "Attention Matrix:"
)


matrix = weights[0]


for i in range(
    len(tokens)
):

    row = []

    for j in range(
        len(tokens)
    ):

        row.append(
            f"{matrix[i, j].item():.3f}"
        )

    print(
        f"{i:2d}:",
        " ".join(row)
    )
