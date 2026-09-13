import sys
from pathlib import Path

import torch


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
from src.mlp import MLP


def main():

    torch.manual_seed(42)

    # ========================================================
    # Config
    # ========================================================

    config = ModelConfig(
        vocab_size=16000,
        d_model=512,
        d_ff=2048,
        dropout=0.0,
    )

    # ========================================================
    # Create MLP
    # ========================================================

    mlp = MLP(
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout=0.0,
    )

    # ========================================================
    # 模拟 Attention 的输出
    #
    # batch = 2
    # sequence = 6
    # hidden = 512
    # ========================================================

    x = torch.randn(
        2,
        6,
        config.d_model,
    )

    # ========================================================
    # Forward
    # ========================================================

    output = mlp(
        x
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print()

    print(
        "d_model:",
        config.d_model
    )

    print(
        "d_ff:",
        config.d_ff
    )

    # ========================================================
    # 参数量
    # ========================================================

    total_params = sum(
        p.numel()
        for p in mlp.parameters()
    )

    print()

    print(
        "MLP Parameters:",
        f"{total_params:,}"
    )

    print(
        "MLP Parameters(M):",
        f"{total_params / 1_000_000:.3f}M"
    )

    # ========================================================
    # 显示参数名称
    # ========================================================

    print()

    print(
        "Parameter Shapes:"
    )

    for name, parameter in mlp.named_parameters():

        print(
            name,
            parameter.shape
        )


if __name__ == "__main__":
    main()
