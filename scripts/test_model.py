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

from src.model import (
    OpsLMToy,
    count_parameters,
    count_trainable_parameters,
)


def main():

    # ========================================================
    # 随机种子
    # ========================================================

    torch.manual_seed(42)

    # ========================================================
    # Device
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

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

    print(
        "Vocab size:",
        vocab_size
    )

    # ========================================================
    # Config
    # ========================================================

    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=1024,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        dropout=0.0,
    )

    print()
    print("Model Config:")
    print(
        "d_model:",
        config.d_model
    )

    print(
        "n_heads:",
        config.n_heads
    )

    print(
        "head_dim:",
        config.head_dim
    )

    print(
        "n_layers:",
        config.n_layers
    )

    print(
        "d_ff:",
        config.d_ff
    )

    print(
        "max_seq_len:",
        config.max_seq_len
    )

    # ========================================================
    # Create Model
    # ========================================================

    model = OpsLMToy(
        config
    )

    model = model.to(
        device
    )

    # ========================================================
    # 参数数量
    # ========================================================

    total_params = (
        count_parameters(
            model
        )
    )

    trainable_params = (
        count_trainable_parameters(
            model
        )
    )

    print()

    print(
        "Total Parameters:",
        f"{total_params:,}"
    )

    print(
        "Total Parameters(M):",
        f"{total_params / 1_000_000:.3f}M"
    )

    print(
        "Trainable Parameters:",
        f"{trainable_params:,}"
    )

    # ========================================================
    # 输入文本
    # ========================================================

    text = (
        "kubectl get pods -n production"
    )

    encoded = tokenizer.encode(
        text
    )

    print()

    print(
        "Text:",
        text
    )

    print(
        "Tokens:",
        encoded.tokens
    )

    print(
        "Token IDs:",
        encoded.ids
    )

    # ========================================================
    # Tensor
    # ========================================================

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    print()

    print(
        "Input IDs shape:",
        input_ids.shape
    )

    # ========================================================
    # Forward
    # ========================================================

    model.eval()

    with torch.no_grad():

        logits = model(
            input_ids
        )

    # ========================================================
    # Output
    # ========================================================

    print(
        "Logits shape:",
        logits.shape
    )

    print()

    print(
        "Expected:"
    )

    print(
        "[batch_size, sequence_length, vocab_size]"
    )

    # ========================================================
    # 最后一个位置
    # ========================================================

    last_logits = logits[
        0,
        -1
    ]

    print()

    print(
        "Last token logits shape:",
        last_logits.shape
    )

    # ========================================================
    # Top 5
    #
    # 注意：
    # 模型还没训练
    # 所以预测没有意义
    # ========================================================

    top_values, top_indices = (
        torch.topk(
            last_logits,
            k=5,
        )
    )

    print()

    print(
        "Random Top-5 predictions:"
    )

    for score, token_id in zip(
        top_values.tolist(),
        top_indices.tolist(),
    ):

        token = tokenizer.id_to_token(
            token_id
        )

        print(
            f"ID={token_id:<6} "
            f"Token={repr(token):<20} "
            f"Logit={score:.4f}"
        )


if __name__ == "__main__":
    main()
