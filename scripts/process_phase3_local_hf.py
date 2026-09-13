import json
import re
import hashlib
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data/phase3/hf_raw"
OUTPUT_DIR = ROOT / "data/phase3/hf"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def normalize_text(value):

    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\x00",
        "",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_for_dedup(text):

    text = normalize_text(
        text
    ).lower()

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    text = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        "",
        text,
    )

    return text


def make_id(
    source,
    instruction,
):

    raw = (
        source
        +
        "\n"
        +
        instruction
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def load_parquet(path):

    table = pq.read_table(
        path
    )

    print()
    print("=" * 80)
    print("Reading:", path)
    print("Rows:", table.num_rows)
    print("Columns:", table.column_names)
    print("=" * 80)

    return table.to_pylist()


def convert_skilln():

    path = (
        RAW_DIR
        /
        "skilln_devops.parquet"
    )

    rows = load_parquet(
        path
    )

    result = []

    rejected_empty = 0
    rejected_quality = 0

    for item in rows:

        question = normalize_text(
            item.get(
                "question",
                "",
            )
        )

        answer = normalize_text(
            item.get(
                "answer",
                "",
            )
        )

        if not question or not answer:

            rejected_empty += 1
            continue

        quality = item.get(
            "quality_score"
        )

        try:

            quality = (
                float(quality)
                if quality is not None
                else None
            )

        except Exception:

            quality = None

        # -----------------------------------------------
        # 先保留质量 >= 0.75
        # -----------------------------------------------

        if (
            quality is not None
            and
            quality < 0.75
        ):

            rejected_quality += 1
            continue

        tags = item.get(
            "tags"
        )

        if tags is None:

            tags = []

        elif isinstance(
            tags,
            str,
        ):

            tags = [
                x.strip()
                for x in tags.split(",")
                if x.strip()
            ]

        elif not isinstance(
            tags,
            list,
        ):

            tags = [
                str(tags)
            ]

        result.append(
            {
                "id":
                    make_id(
                        "Skilln/devops-qa-dataset",
                        question,
                    ),

                "instruction":
                    question,

                "output":
                    answer,

                "source":
                    "Skilln/devops-qa-dataset",

                "category":
                    item.get(
                        "category"
                    ),

                "difficulty":
                    item.get(
                        "difficulty"
                    ),

                "quality_score":
                    quality,

                "tags":
                    tags,

                "task_type":
                    "qa",

                "language":
                    "en",

                "license":
                    "CC-BY-SA-4.0",
            }
        )

    print()
    print("Skilln accepted:", len(result))
    print("Skilln empty rejected:", rejected_empty)
    print("Skilln quality rejected:", rejected_quality)

    return result


def convert_devops_v1():

    path = (
        RAW_DIR
        /
        "devops_v1.parquet"
    )

    rows = load_parquet(
        path
    )

    result = []

    for item in rows:

        question = normalize_text(
            item.get(
                "question",
                "",
            )
        )

        solution = normalize_text(
            item.get(
                "solution",
                "",
            )
        )

        if not question:
            continue

        if not solution:
            continue

        result.append(
            {
                "id":
                    make_id(
                        "pavanmantha/devops-v1",
                        question,
                    ),

                "instruction":
                    question,

                "output":
                    solution,

                "source":
                    "pavanmantha/devops-v1",

                "category":
                    "docker-kubernetes",

                "difficulty":
                    None,

                "quality_score":
                    None,

                "tags":
                    [
                        "docker",
                        "kubernetes",
                        "troubleshooting",
                    ],

                "task_type":
                    "troubleshooting",

                "language":
                    "en",

                "license":
                    "Apache-2.0",
            }
        )

    print()
    print(
        "devops-v1 accepted:",
        len(result),
    )

    return result


def deduplicate(rows):

    seen = set()

    result = []

    duplicates = 0

    for row in rows:

        key = normalize_for_dedup(
            row[
                "instruction"
            ]
        )

        if not key:
            continue

        if key in seen:

            duplicates += 1
            continue

        seen.add(
            key
        )

        result.append(
            row
        )

    print()
    print(
        "Exact duplicate questions:",
        duplicates,
    )

    return result


def write_jsonl(
    path,
    rows,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                +
                "\n"
            )


def main():

    skilln = convert_skilln()

    devops_v1 = convert_devops_v1()

    write_jsonl(
        OUTPUT_DIR
        /
        "skilln_devops_qa.jsonl",

        skilln,
    )

    write_jsonl(
        OUTPUT_DIR
        /
        "devops_v1.jsonl",

        devops_v1,
    )

    all_rows = (
        skilln
        +
        devops_v1
    )

    print()
    print("=" * 80)
    print("Deduplicating")
    print("=" * 80)

    all_rows = deduplicate(
        all_rows
    )

    write_jsonl(
        OUTPUT_DIR
        /
        "all_sft.jsonl",

        all_rows,
    )

    sources = Counter(
        row["source"]
        for row in all_rows
    )

    categories = Counter(
        str(
            row.get(
                "category"
            )
            or
            "unknown"
        )
        for row in all_rows
    )

    manifest = {

        "examples":
            len(all_rows),

        "sources":
            dict(
                sources
            ),

        "categories":
            dict(
                categories
            ),

    }

    with (
        OUTPUT_DIR
        /
        "manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("Phase3 HF Dataset Ready")
    print("=" * 80)

    print(
        "Total:",
        len(all_rows),
    )

    print()

    for source, count in (
        sources
        .most_common()
    ):

        print(
            source,
            count,
        )

    print()
    print(
        "Output:",
        OUTPUT_DIR
        /
        "all_sft.jsonl"
    )


if __name__ == "__main__":
    main()
