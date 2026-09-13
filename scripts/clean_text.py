from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "cleaned.txt"


def clean_line(line: str) -> str:

    # 去除首尾空白
    line = line.strip()

    # 多个空格压缩成一个
    line = re.sub(
        r"[ \t]+",
        " ",
        line,
    )

    # 多个空行后续统一处理

    return line


def main():

    files = sorted(
        INPUT_DIR.glob("*.txt")
    )

    if not files:
        raise RuntimeError(
            f"No txt files in {INPUT_DIR}"
        )

    all_lines = []

    for file in files:

        print(
            f"Reading: {file.name}"
        )

        text = file.read_text(
            encoding="utf-8"
        )

        for line in text.splitlines():

            line = clean_line(line)

            if not line:
                continue

            all_lines.append(line)


    # 简单去重
    seen = set()

    cleaned = []

    for line in all_lines:

        if line in seen:
            continue

        seen.add(line)

        cleaned.append(line)


    # 每句话之间空一行
    output_text = (
        "\n\n".join(cleaned)
        + "\n"
    )


    OUTPUT_PATH.write_text(
        output_text,
        encoding="utf-8",
    )


    print()
    print(
        "Original lines:",
        len(all_lines)
    )

    print(
        "Unique lines:",
        len(cleaned)
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
