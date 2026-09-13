import json
import math
import random
import re
import hashlib
import argparse

from pathlib import Path
from collections import Counter, defaultdict

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="data/phase3/hf/all_sft.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/phase3/processed",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/artifacts_v2/tokenizer.json",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--near-dup-distance",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--include-seed-data",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Text helpers
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

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
        r"\n{4,}",
        "\n\n\n",
        text,
    )

    return text.strip()


def normalize_for_dedup(text):

    text = clean_text(
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


# ============================================================
# Simple quality rules
# ============================================================

def quality_check(
    instruction,
    output,
):

    if len(instruction) < 8:
        return False, "question_too_short"

    if len(instruction) > 1500:
        return False, "question_too_long"

    if len(output) < 30:
        return False, "answer_too_short"

    if len(output) > 8000:
        return False, "answer_too_long"

    instruction_lower = (
        instruction.lower()
    )

    output_lower = (
        output.lower()
    )

    bad_patterns = [
        "as an ai language model",
        "as a language model",
        "i cannot provide",
        "i'm unable to",
        "根据上述文档",
        "根据给定文档",
        "根据知识片段",
        "根据提供的文档",
        "作为一个ai",
        "作为人工智能",
    ]

    combined = (
        instruction_lower
        +
        "\n"
        +
        output_lower
    )

    for pattern in bad_patterns:

        if pattern in combined:

            return (
                False,
                "bad_pattern",
            )

    # --------------------------------------------------------
    # 检查答案是否出现非常明显的连续重复
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]

    if len(lines) >= 8:

        unique_ratio = (
            len(set(lines))
            /
            len(lines)
        )

        if unique_ratio < 0.45:

            return (
                False,
                "repetitive_answer",
            )

    return (
        True,
        "ok",
    )


# ============================================================
# Token length
# ============================================================

def calculate_sequence_tokens(
    tokenizer,
    instruction,
    output,
):

    prompt = (
        "用户："
        +
        instruction
        +
        "\n\n助手："
    )

    prompt_ids = (
        tokenizer
        .encode(prompt)
        .ids
    )

    answer_ids = (
        tokenizer
        .encode(output)
        .ids
    )

    # BOS + Prompt + Answer + EOS
    return (
        1
        +
        len(prompt_ids)
        +
        len(answer_ids)
        +
        1
    )


# ============================================================
# SimHash Near Dedup
# ============================================================

def build_shingles(text):

    normalized = (
        normalize_for_dedup(
            text
        )
    )

    if not normalized:
        return set()

    # --------------------------------------------------------
    # 同时兼容中英文：
    # 字符级 3-gram
    # --------------------------------------------------------

    if len(normalized) <= 3:
        return {
            normalized
        }

    return {
        normalized[i:i + 3]
        for i in range(
            len(normalized) - 2
        )
    }


def simhash64(text):

    shingles = build_shingles(
        text
    )

    if not shingles:
        return 0

    weights = [
        0
        for _ in range(64)
    ]

    for shingle in shingles:

        digest = hashlib.blake2b(
            shingle.encode(
                "utf-8"
            ),
            digest_size=8,
        ).digest()

        value = int.from_bytes(
            digest,
            byteorder="big",
        )

        for bit in range(64):

            if (
                value
                &
                (1 << bit)
            ):

                weights[
                    bit
                ] += 1

            else:

                weights[
                    bit
                ] -= 1

    result = 0

    for bit in range(64):

        if weights[bit] >= 0:

            result |= (
                1 << bit
            )

    return result


def hamming_distance(
    left,
    right,
):

    return (
        left
        ^
        right
    ).bit_count()


class NearDeduper:

    def __init__(
        self,
        threshold=3,
    ):

        self.threshold = (
            threshold
        )

        self.buckets = defaultdict(
            list
        )

        self.values = []


    def is_duplicate(
        self,
        text,
    ):

        value = simhash64(
            text
        )

        candidates = set()

        # ----------------------------------------------------
        # 64 bit 分成 4 个 16bit band
        #
        # Hamming <= 3 时至少会有一个 band 完全一样。
        # ----------------------------------------------------

        for band in range(4):

            band_value = (
                value
                >>
                (band * 16)
            ) & 0xFFFF

            key = (
                band,
                band_value,
            )

            for index in (
                self.buckets[
                    key
                ]
            ):

                candidates.add(
                    index
                )

        for index in candidates:

            existing = (
                self.values[
                    index
                ]
            )

            if (
                hamming_distance(
                    value,
                    existing,
                )
                <=
                self.threshold
            ):

                return True

        index = len(
            self.values
        )

        self.values.append(
            value
        )

        for band in range(4):

            band_value = (
                value
                >>
                (band * 16)
            ) & 0xFFFF

            key = (
                band,
                band_value,
            )

            self.buckets[
                key
            ].append(
                index
            )

        return False


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(path):

    rows = []

    if not path.exists():
        return rows

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                item = json.loads(
                    line
                )

            except Exception:

                continue

            rows.append(
                item
            )

    return rows


def load_seed_data():

    candidates = [
        ROOT
        /
        "data/phase3/sft_seed/train.jsonl",

        ROOT
        /
        "data/phase3/sft_seed/val.jsonl",

        ROOT
        /
        "data/phase3/sft/train.jsonl",

        ROOT
        /
        "data/phase3/sft/val.jsonl",
    ]

    rows = []

    used = set()

    for path in candidates:

        if not path.exists():
            continue

        resolved = str(
            path.resolve()
        )

        if resolved in used:
            continue

        used.add(
            resolved
        )

        for item in load_jsonl(
            path
        ):

            instruction = clean_text(
                item.get(
                    "instruction",
                    "",
                )
            )

            output = clean_text(
                item.get(
                    "output",
                    "",
                )
            )

            if not instruction:
                continue

            if not output:
                continue

            rows.append(
                {
                    "instruction":
                        instruction,

                    "output":
                        output,

                    "source":
                        "opslm-phase3-seed",

                    "category":
                        "sre-cn",

                    "difficulty":
                        None,

                    "quality_score":
                        1.0,

                    "tags":
                        [
                            "sre",
                            "chinese",
                        ],

                    "task_type":
                        "troubleshooting",

                    "language":
                        "zh",

                    "license":
                        "project-owned",
                }
            )

    return rows


# ============================================================
# Split
# ============================================================

def split_train_val(
    rows,
    val_ratio,
    seed,
):

    random.seed(
        seed
    )

    # --------------------------------------------------------
    # 按 source + category 分层
    # --------------------------------------------------------

    groups = defaultdict(
        list
    )

    for row in rows:

        source = str(
            row.get(
                "source",
                "unknown",
            )
        )

        category = str(
            row.get(
                "category",
                "unknown",
            )
            or
            "unknown"
        )

        key = (
            source,
            category,
        )

        groups[
            key
        ].append(
            row
        )

    train_rows = []
    val_rows = []

    for key, group_rows in (
        groups.items()
    ):

        random.shuffle(
            group_rows
        )

        if len(group_rows) < 10:

            train_rows.extend(
                group_rows
            )

            continue

        val_count = max(
            1,
            int(
                round(
                    len(group_rows)
                    *
                    val_ratio
                )
            ),
        )

        val_count = min(
            val_count,
            len(group_rows) - 1,
        )

        val_rows.extend(
            group_rows[
                :val_count
            ]
        )

        train_rows.extend(
            group_rows[
                val_count:
            ]
        )

    random.shuffle(
        train_rows
    )

    random.shuffle(
        val_rows
    )

    return (
        train_rows,
        val_rows,
    )


# ============================================================
# Write
# ============================================================

def write_jsonl(
    path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    input_path = (
        ROOT
        /
        args.input
    )

    output_dir = (
        ROOT
        /
        args.output_dir
    )

    tokenizer_path = (
        ROOT
        /
        args.tokenizer
    )

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    tokenizer = (
        Tokenizer.from_file(
            str(
                tokenizer_path
            )
        )
    )

    rows = load_jsonl(
        input_path
    )

    print(
        "=" * 80
    )

    print(
        "OpsLM Phase3 Dataset Builder"
    )

    print(
        "=" * 80
    )

    print(
        "Input:",
        input_path,
    )

    print(
        "Raw samples:",
        len(rows),
    )

    if args.include_seed_data:

        seed_rows = (
            load_seed_data()
        )

        print(
            "Seed samples:",
            len(seed_rows),
        )

        rows.extend(
            seed_rows
        )

    # ========================================================
    # Filter
    # ========================================================

    accepted = []

    reject_counter = Counter()

    exact_seen = set()

    near_deduper = NearDeduper(
        threshold=(
            args.near_dup_distance
        )
    )

    token_lengths = []

    for index, item in enumerate(
        rows,
        start=1,
    ):

        instruction = clean_text(
            item.get(
                "instruction",
                "",
            )
        )

        output = clean_text(
            item.get(
                "output",
                "",
            )
        )

        ok, reason = quality_check(
            instruction,
            output,
        )

        if not ok:

            reject_counter[
                reason
            ] += 1

            continue

        total_tokens = (
            calculate_sequence_tokens(
                tokenizer,
                instruction,
                output,
            )
        )

        # ----------------------------------------------------
        # 正式 SFT 不希望答案被 Dataset 静默截断。
        # ----------------------------------------------------

        if (
            total_tokens
            >
            args.seq_len
        ):

            reject_counter[
                "sequence_too_long"
            ] += 1

            continue

        exact_key = (
            normalize_for_dedup(
                instruction
            )
        )

        if not exact_key:

            reject_counter[
                "empty_after_normalize"
            ] += 1

            continue

        if (
            exact_key
            in
            exact_seen
        ):

            reject_counter[
                "exact_duplicate"
            ] += 1

            continue

        if (
            near_deduper
            .is_duplicate(
                instruction
            )
        ):

            reject_counter[
                "near_duplicate"
            ] += 1

            continue

        exact_seen.add(
            exact_key
        )

        row = dict(
            item
        )

        row[
            "instruction"
        ] = instruction

        row[
            "output"
        ] = output

        row[
            "sequence_tokens"
        ] = total_tokens

        accepted.append(
            row
        )

        token_lengths.append(
            total_tokens
        )

        if (
            index
            %
            5000
            ==
            0
        ):

            print(
                f"processed={index} "
                f"accepted={len(accepted)}",
                flush=True,
            )

    # ========================================================
    # Split
    # ========================================================

    train_rows, val_rows = (
        split_train_val(
            accepted,
            args.val_ratio,
            args.seed,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        output_dir
        /
        "train.jsonl"
    )

    val_path = (
        output_dir
        /
        "val.jsonl"
    )

    manifest_path = (
        output_dir
        /
        "manifest.json"
    )

    write_jsonl(
        train_path,
        train_rows,
    )

    write_jsonl(
        val_path,
        val_rows,
    )

    # ========================================================
    # Stats
    # ========================================================

    source_counter = Counter(
        str(
            row.get(
                "source",
                "unknown",
            )
        )
        for row in accepted
    )

    category_counter = Counter(
        str(
            row.get(
                "category",
                "unknown",
            )
            or
            "unknown"
        )
        for row in accepted
    )

    language_counter = Counter(
        str(
            row.get(
                "language",
                "unknown",
            )
            or
            "unknown"
        )
        for row in accepted
    )

    task_counter = Counter(
        str(
            row.get(
                "task_type",
                "unknown",
            )
            or
            "unknown"
        )
        for row in accepted
    )

    if token_lengths:

        sorted_lengths = sorted(
            token_lengths
        )

        avg_tokens = (
            sum(token_lengths)
            /
            len(token_lengths)
        )

        p50 = sorted_lengths[
            int(
                len(sorted_lengths)
                *
                0.50
            )
        ]

        p90 = sorted_lengths[
            min(
                len(sorted_lengths) - 1,
                int(
                    len(sorted_lengths)
                    *
                    0.90
                ),
            )
        ]

        p95 = sorted_lengths[
            min(
                len(sorted_lengths) - 1,
                int(
                    len(sorted_lengths)
                    *
                    0.95
                ),
            )
        ]

        max_tokens = max(
            token_lengths
        )

    else:

        avg_tokens = 0
        p50 = 0
        p90 = 0
        p95 = 0
        max_tokens = 0

    manifest = {

        "raw_samples":
            len(rows),

        "accepted_samples":
            len(accepted),

        "train_samples":
            len(train_rows),

        "val_samples":
            len(val_rows),

        "val_ratio":
            args.val_ratio,

        "seq_len":
            args.seq_len,

        "near_dup_distance":
            args.near_dup_distance,

        "rejections":
            dict(
                reject_counter
            ),

        "sources":
            dict(
                source_counter
            ),

        "categories":
            dict(
                category_counter
            ),

        "languages":
            dict(
                language_counter
            ),

        "task_types":
            dict(
                task_counter
            ),

        "sequence_tokens":
            {
                "average":
                    avg_tokens,

                "p50":
                    p50,

                "p90":
                    p90,

                "p95":
                    p95,

                "max":
                    max_tokens,
            },

    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Output
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "Phase3 Dataset Finished"
    )

    print(
        "=" * 80
    )

    print(
        "Raw:",
        len(rows),
    )

    print(
        "Accepted:",
        len(accepted),
    )

    print(
        "Train:",
        len(train_rows),
    )

    print(
        "Val:",
        len(val_rows),
    )

    print()

    print(
        "Rejections:"
    )

    for key, value in (
        reject_counter
        .most_common()
    ):

        print(
            f"  {key}: {value}"
        )

    print()

    print(
        "Languages:"
    )

    for key, value in (
        language_counter
        .most_common()
    ):

        print(
            f"  {key}: {value}"
        )

    print()

    print(
        "Sources:"
    )

    for key, value in (
        source_counter
        .most_common()
    ):

        print(
            f"  {key}: {value}"
        )

    print()

    print(
        "Sequence tokens:"
    )

    print(
        f"  avg: {avg_tokens:.1f}"
    )

    print(
        f"  p50: {p50}"
    )

    print(
        f"  p90: {p90}"
    )

    print(
        f"  p95: {p95}"
    )

    print(
        f"  max: {max_tokens}"
    )

    print()

    print(
        "Train:"
    )

    print(
        train_path
    )

    print()

    print(
        "Val:"
    )

    print(
        val_path
    )

    print()

    print(
        "Manifest:"
    )

    print(
        manifest_path
    )


if __name__ == "__main__":
    main()
