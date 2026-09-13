import sys
from pathlib import Path

from torch.utils.data import DataLoader


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


def main():

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

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    print(
        "Dataset samples:",
        len(dataset)
    )

    print(
        "DataLoader batches:",
        len(dataloader)
    )

    print()

    # ========================================================
    # 取第一个 Batch
    # ========================================================

    batch = next(
        iter(
            dataloader
        )
    )

    input_ids = (
        batch[
            "input_ids"
        ]
    )

    target_ids = (
        batch[
            "target_ids"
        ]
    )

    print(
        "Input batch shape:",
        input_ids.shape
    )

    print(
        "Target batch shape:",
        target_ids.shape
    )

    print()

    print(
        "Input dtype:",
        input_ids.dtype
    )

    print(
        "Target dtype:",
        target_ids.dtype
    )


if __name__ == "__main__":

    main()
