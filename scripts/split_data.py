import argparse
import random
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="data/cleaned.txt",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def resolve_path(path: str) -> Path:

    path = Path(path)

    if path.is_absolute():
        return path

    return ROOT / path


def main():

    args = parse_args()

    input_path = resolve_path(
        args.input
    )

    output_dir = resolve_path(
        args.output_dir
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if not (
        0.0
        <
        args.val_ratio
        <
        1.0
    ):

        raise ValueError(
            "val-ratio must be between 0 and 1"
        )

    # ========================================================
    # 读取文本
    # ========================================================

    lines = (
        input_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    # 去空行

    lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    if len(lines) < 10:

        raise ValueError(
            "Dataset is too small to split reliably"
        )

    # ========================================================
    # 固定随机种子
    # ========================================================

    rng = random.Random(
        args.seed
    )

    rng.shuffle(
        lines
    )

    # ========================================================
    # 划分
    # ========================================================

    val_size = max(
        1,
        int(
            len(lines)
            *
            args.val_ratio
        )
    )

    val_lines = lines[
        :val_size
    ]

    train_lines = lines[
        val_size:
    ]

    # ========================================================
    # 输出目录
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        output_dir
        /
        "train.txt"
    )

    val_path = (
        output_dir
        /
        "val.txt"
    )

    train_path.write_text(
        "\n".join(train_lines)
        +
        "\n",
        encoding="utf-8",
    )

    val_path.write_text(
        "\n".join(val_lines)
        +
        "\n",
        encoding="utf-8",
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "Dataset Split Finished"
    )

    print(
        "=" * 70
    )

    print(
        "Total lines:",
        len(lines)
    )

    print(
        "Train lines:",
        len(train_lines)
    )

    print(
        "Validation lines:",
        len(val_lines)
    )

    print()

    print(
        "Train:",
        train_path
    )

    print(
        "Validation:",
        val_path
    )


if __name__ == "__main__":
    main()
