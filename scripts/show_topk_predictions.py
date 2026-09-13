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

from src.prediction import (
    get_topk_predictions
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

    tokenizer_path = (
        ROOT
        / "tokenizer"
        / "artifacts"
        / "tokenizer.json"
    )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
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

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # ========================================================
    # Model
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

    model = OpsLMToy(
        config
    ).to(
        device
    )

    model.eval()

    # ========================================================
    # Batch
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
    # Top 5
    # ========================================================

    values, indices = (
        get_topk_predictions(
            logits,
            k=5,
        )
    )

    # ========================================================
    # 前10个位置
    # ========================================================

    for position in range(
        min(
            10,
            input_ids.size(1),
        )
    ):

        input_id = (
            input_ids[
                0,
                position
            ].item()
        )

        target_id = (
            target_ids[
                0,
                position
            ].item()
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

        print()

        print(
            "=" * 70
        )

        print(
            "Position:",
            position
        )

        print(
            "Current Token:",
            repr(input_token)
        )

        print(
            "Correct Next Token:",
            repr(target_token)
        )

        print()

        print(
            "Model Top-5:"
        )

        for rank in range(5):

            token_id = (
                indices[
                    0,
                    position,
                    rank
                ].item()
            )

            score = (
                values[
                    0,
                    position,
                    rank
                ].item()
            )

            token = (
                tokenizer.id_to_token(
                    token_id
                )
            )

            print(
                f"  {rank + 1}. "
                f"{repr(token):<25} "
                f"logit={score:.4f}"
            )


if __name__ == "__main__":

    main()
