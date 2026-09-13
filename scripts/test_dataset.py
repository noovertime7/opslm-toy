import sys
from pathlib import Path

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


def main():

    text_path = (
        ROOT
        / "data"
        / "cleaned.txt"
    )

    tokenizer_path = (
        ROOT
        / "tokenizer"
        / "artifacts"
        / "tokenizer.json"
    )

    # ========================================================
    # 测试阶段先使用128
    # ========================================================

    seq_len = 128

    dataset = LanguageModelDataset(
        text_path=text_path,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        add_bos=True,
        add_eos=True,
    )

    tokenizer = (
        Tokenizer.from_file(
            str(
                tokenizer_path
            )
        )
    )

    # ========================================================
    # 基本信息
    # ========================================================

    print(
        "Total tokens:",
        len(dataset.tokens)
    )

    print(
        "Sequence length:",
        seq_len
    )

    print(
        "Dataset samples:",
        len(dataset)
    )

    # ========================================================
    # 第一条数据
    # ========================================================

    sample = dataset[0]

    input_ids = (
        sample[
            "input_ids"
        ]
    )

    target_ids = (
        sample[
            "target_ids"
        ]
    )

    print()

    print(
        "Input shape:",
        input_ids.shape
    )

    print(
        "Target shape:",
        target_ids.shape
    )

    # ========================================================
    # 前20个 Token
    # ========================================================

    print()

    print(
        "First 20 Input IDs:"
    )

    print(
        input_ids[:20]
    )

    print()

    print(
        "First 20 Target IDs:"
    )

    print(
        target_ids[:20]
    )

    # ========================================================
    # 验证错位关系
    # ========================================================

    same = (
        input_ids[1:]
        ==
        target_ids[:-1]
    ).all().item()

    print()

    print(
        "Shift relationship correct:",
        same
    )

    # ========================================================
    # Decode
    # ========================================================

    input_text = (
        tokenizer.decode(
            input_ids.tolist()
        )
    )

    target_text = (
        tokenizer.decode(
            target_ids.tolist()
        )
    )

    print()

    print(
        "=" * 70
    )

    print(
        "INPUT TEXT"
    )

    print(
        "=" * 70
    )

    print(
        input_text[:500]
    )

    print()

    print(
        "=" * 70
    )

    print(
        "TARGET TEXT"
    )

    print(
        "=" * 70
    )

    print(
        target_text[:500]
    )


if __name__ == "__main__":

    main()
