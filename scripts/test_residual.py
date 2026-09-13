import torch

from src.block import TransformerBlock


torch.manual_seed(42)


block = TransformerBlock(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    max_seq_len=1024,
    dropout=0.0,
)


x = torch.randn(
    1,
    6,
    512,
)


output = block(
    x
)


print(
    "Input mean:",
    x.mean().item()
)

print(
    "Output mean:",
    output.mean().item()
)

print(
    "Same tensor:",
    torch.equal(
        x,
        output
    )
)

print(
    "Average absolute change:",
    (
        output - x
    )
    .abs()
    .mean()
    .item()
)
