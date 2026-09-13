import sys
from pathlib import Path

import torch
import torch.nn as nn
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


from src.config import ModelConfig
from src.model import OpsLMToy


def print_stats(
    name: str,
    tensor: torch.Tensor,
):

    data = (
        tensor
        .detach()
        .float()
    )

    print(
        f"{name}"
    )

    print(
        f"  shape = {tuple(data.shape)}"
    )

    print(
        f"  mean  = {data.mean().item():.6f}"
    )

    print(
        f"  std   = {data.std().item():.6f}"
    )

    print(
        f"  min   = {data.min().item():.6f}"
    )

    print(
        f"  max   = {data.max().item():.6f}"
    )

    print()


def main():

    # ========================================================
    # 固定随机种子
    # ========================================================

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
        max_seq_len=1024,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        dropout=0.1,
    )

    # ========================================================
    # Model
    # ========================================================

    model = OpsLMToy(
        config
    )

    print(
        "=" * 70
    )

    print(
        "OpsLM-Toy Parameter Initialization"
    )

    print(
        "=" * 70
    )

    print()

    # ========================================================
    # Token Embedding
    # ========================================================

    print_stats(
        "Token Embedding",
        model.embedding.token_embedding.weight,
    )

    # ========================================================
    # Position Embedding
    # ========================================================

    print_stats(
        "Position Embedding",
        model.embedding.position_embedding.weight,
    )

    # ========================================================
    # 第一个 Attention Query
    # ========================================================

    print_stats(
        "Block0 Head0 Query Weight",
        model
        .blocks[0]
        .attention
        .heads[0]
        .query
        .weight,
    )

    # ========================================================
    # 第一个 Attention Key
    # ========================================================

    print_stats(
        "Block0 Head0 Key Weight",
        model
        .blocks[0]
        .attention
        .heads[0]
        .key
        .weight,
    )

    # ========================================================
    # 第一个 Attention Value
    # ========================================================

    print_stats(
        "Block0 Head0 Value Weight",
        model
        .blocks[0]
        .attention
        .heads[0]
        .value
        .weight,
    )

    # ========================================================
    # 第一个 Block 的 MLP
    # ========================================================

    print_stats(
        "Block0 MLP FC1 Weight",
        model
        .blocks[0]
        .mlp
        .fc1
        .weight,
    )

    print_stats(
        "Block0 MLP FC2 Weight",
        model
        .blocks[0]
        .mlp
        .fc2
        .weight,
    )

    # ========================================================
    # LayerNorm
    # ========================================================

    print_stats(
        "Block0 LayerNorm1 Weight",
        model
        .blocks[0]
        .ln1
        .weight,
    )

    print_stats(
        "Block0 LayerNorm1 Bias",
        model
        .blocks[0]
        .ln1
        .bias,
    )

    # ========================================================
    # Weight Tying 验证
    # ========================================================

    same_storage = (
        model.lm_head.weight.data_ptr()
        ==
        model.embedding
        .token_embedding
        .weight
        .data_ptr()
    )

    print(
        "LM Head and Token Embedding share weights:",
        same_storage
    )

    # ========================================================
    # 检查参数是否可训练
    # ========================================================

    print()

    print(
        "All trainable:"
    )

    all_trainable = all(
        parameter.requires_grad
        for parameter
        in model.parameters()
    )

    print(
        all_trainable
    )

    # ========================================================
    # 查看参数总量
    # ========================================================

    total = sum(
        p.numel()
        for p
        in model.parameters()
    )

    print()

    print(
        "Total parameters:",
        f"{total:,}"
    )

    print(
        "Total parameters(M):",
        f"{total / 1_000_000:.3f}M"
    )


if __name__ == "__main__":
    main()
