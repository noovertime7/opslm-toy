import sys
from pathlib import Path


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


from src.dataset import (
    PackedLanguageModelDataset
)


def main():


    dataset = PackedLanguageModelDataset(

        jsonl_path=(
            ROOT
            /
            "data"
            /
            "phase2"
            /
            "processed"
            /
            "train.jsonl"
        ),

        tokenizer_path=(
            ROOT
            /
            "tokenizer"
            /
            "artifacts_v2"
            /
            "tokenizer.json"
        ),

        seq_len=512,

    )


    print()

    print(
        "Dataset length:",
        len(dataset)
    )


    sample = dataset[0]


    print()

    print(
        "Input shape:",
        sample["input_ids"].shape
    )


    print(
        "Target shape:",
        sample["target_ids"].shape
    )


    print()

    print(
        "First 50 input ids:"
    )

    print(
        sample["input_ids"][:50]
    )


    print()

    print(
        "First 50 target ids:"
    )

    print(
        sample["target_ids"][:50]
    )


    print()

    print(
        "EOS token id:",
        dataset.eos_id
    )


    # ========================================================
    # 检查 input / target 是否错位
    # ========================================================

    correct = (
        sample["input_ids"][1:]
        ==
        sample["target_ids"][:-1]
    ).all().item()


    print()

    print(
        "Shift relationship correct:",
        correct
    )


if __name__ == "__main__":

    main()
