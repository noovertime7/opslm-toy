import sys
from pathlib import Path

import torch

from torch.utils.data import DataLoader
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

from src.dataset import (
    LanguageModelDataset
)

from src.config import (
    ModelConfig
)

from src.model import (
    OpsLMToy
)

from src.prediction import (
    get_next_token_predictions
)


def main():

    # ========================================================
    # Seed
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
    # Dataset
    # ========================================================

    dataset = LanguageModelDataset(
        text_path=(
            ROOT
            / "data"
            / "cleaned.txt"
        ),
        tokenizer_path=tokenizer_path,
        seq_len=128,
    )

    # ========================================================
    # DataLoader
    #
    # 为了方便观察，这里 batch_size=1
    # ========================================================

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # ========================================================
    # Config
    # ========================================================

    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        dropout=0.0,
    )

    # ========================================================
    # Model
    # ========================================================

    model = OpsLMToy(
        config
    ).to(
        device
    )

    model.eval()

    # ========================================================
    # 获取第一个 Batch
    # ========================================================

    batch = next(
        iter(loader)
    )

    input_ids = (
        batch["input_ids"]
        .to(device)
    )

    target_ids = (
        batch["target_ids"]
        .to(device)
    )

    # ========================================================
    # Forward
    # ========================================================

    with torch.no_grad():

        logits = model(
            input_ids
        )

    # ========================================================
    # Next Token Prediction
    # ========================================================

    predicted_ids = (
        get_next_token_predictions(
            logits
        )
    )

    # ========================================================
    # Shape
    # ========================================================

    print()

    print(
        "Input shape:",
        input_ids.shape
    )

    print(
        "Target shape:",
        target_ids.shape
    )

    print(
        "Logits shape:",
        logits.shape
    )

    print(
        "Predicted IDs shape:",
        predicted_ids.shape
    )

    # ========================================================
    # 显示前 30 个位置
    # ========================================================

    print()

    print(
        "=" * 110
    )

    print(
        f"{'POS':<6}"
        f"{'INPUT':<30}"
        f"{'TARGET/NEXT TOKEN':<30}"
        f"{'MODEL PREDICTION':<30}"
    )

    print(
        "=" * 110
    )

    max_positions = min(
        30,
        input_ids.size(1),
    )

    for position in range(
        max_positions
    ):

        input_id = (
            input_ids[
                0,
                position
            ]
            .item()
        )

        target_id = (
            target_ids[
                0,
                position
            ]
            .item()
        )

        predicted_id = (
            predicted_ids[
                0,
                position
            ]
            .item()
        )

        input_token = (
            tokenizer.id_to_token(
                input_id
            )
        )

        target_token = (
            tokenizer.id_to_token(
                target_id
            )
        )

        predicted_token = (
            tokenizer.id_to_token(
                predicted_id
            )
        )

        print(
            f"{position:<6}"
            f"{repr(input_token):<30}"
            f"{repr(target_token):<30}"
            f"{repr(predicted_token):<30}"
        )

    # ========================================================
    # 当前随机模型预测正确率
    #
    # 只是做观察
    # 现在模型没训练，所以通常接近随机
    # ========================================================

    correct = (
        predicted_ids
        ==
        target_ids
    )

    accuracy = (
        correct
        .float()
        .mean()
        .item()
    )

    print()

    print(
        "Random model token accuracy:",
        f"{accuracy * 100:.4f}%"
    )


if __name__ == "__main__":

    main()
