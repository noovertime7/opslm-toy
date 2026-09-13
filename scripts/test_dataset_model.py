import sys
from pathlib import Path

import torch

from torch.utils.data import DataLoader

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


from src.dataset import (
    LanguageModelDataset
)

from src.config import (
    ModelConfig
)

from src.model import (
    OpsLMToy
)


def main():

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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
    # Dataset
    # ========================================================

    dataset = LanguageModelDataset(
        text_path=(
            ROOT
            / "data"
            / "cleaned.txt"
        ),
        tokenizer_path=(
            ROOT
            / "tokenizer"
            / "artifacts"
            / "tokenizer.json"
        ),
        seq_len=128,
    )

    # ========================================================
    # DataLoader
    # ========================================================

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
    )

    # ========================================================
    # Model Config
    #
    # 测试时 max_seq_len 只需要 >=128
    # ========================================================

    config = ModelConfig(
        vocab_size=(
            tokenizer.get_vocab_size()
        ),
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

    # ========================================================
    # Batch
    # ========================================================

    batch = next(
        iter(
            loader
        )
    )

    input_ids = (
        batch[
            "input_ids"
        ]
        .to(device)
    )

    target_ids = (
        batch[
            "target_ids"
        ]
        .to(device)
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
    # Print
    # ========================================================

    print(
        "Device:",
        device
    )

    print()

    print(
        "Input IDs:",
        input_ids.shape
    )

    print(
        "Target IDs:",
        target_ids.shape
    )

    print(
        "Logits:",
        logits.shape
    )

    print()

    print(
        "Expected:"
    )

    print(
        "[B,T,Vocab]"
    )


if __name__ == "__main__":

    main()
