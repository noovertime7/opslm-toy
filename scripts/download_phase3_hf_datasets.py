import os

# ============================================================
# 国内 Hugging Face 镜像
#
# 必须放在 huggingface_hub / datasets import 之前
# ============================================================

os.environ.setdefault(
    "HF_ENDPOINT",
    "https://hf-mirror.net",
)

os.environ.setdefault(
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "300",
)

os.environ.setdefault(
    "HF_HUB_ETAG_TIMEOUT",
    "60",
)

os.environ.setdefault(
    "HF_HUB_DISABLE_TELEMETRY",
    "1",
)


import re
import json
import hashlib
import argparse

from pathlib import Path
from collections import Counter

from datasets import load_dataset

from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/phase3/hf",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/phase3/hf_cache",
    )

    parser.add_argument(
        "--skilln-min-quality",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--skip-linux-tools",
        action="store_true",
    )

    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Utils
# ============================================================

def normalize_text(text):

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
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_for_dedup(text):

    text = (
        normalize_text(text)
        .lower()
    )

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
# Download exact Parquet
#
# 核心：
# 不使用
#
# load_dataset("Skilln/devops-qa-dataset")
#
# 因为这个仓库现在自动 schema 推断有问题。
# ============================================================

def download_exact_parquet(
    repo_id,
    filename,
    cache_dir,
):

    print()

    print(
        "Downloading exact file:"
    )

    print(
        repo_id
    )

    print(
        filename
    )

    local_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(
            cache_dir
        ),
    )

    print(
        "Local file:"
    )

    print(
        local_file
    )

    dataset = load_dataset(
        "parquet",
        data_files={
            "train":
                local_file,
        },
        split="train",
        cache_dir=str(
            cache_dir
        ),
    )

    print(
        "Rows:",
        len(dataset),
    )

    print(
        "Columns:",
        dataset.column_names,
    )

    return dataset


# ============================================================
# Skilln/devops-qa-dataset
# ============================================================

def load_skilln(
    cache_dir,
    min_quality,
):

    print()

    print(
        "=" * 80
    )

    print(
        "Skilln/devops-qa-dataset"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 重要：
    #
    # 直接下载明确的 parquet。
    #
    # 不让 datasets 扫描 dataset_metadata.json。
    # --------------------------------------------------------

    dataset = download_exact_parquet(

        repo_id=(
            "Skilln/"
            "devops-qa-dataset"
        ),

        filename=(
            "devops_dataset.parquet"
        ),

        cache_dir=cache_dir,

    )

    rows = []

    rejected_quality = 0
    rejected_empty = 0

    for item in dataset:

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

        if not question:
            rejected_empty += 1
            continue

        if not answer:
            rejected_empty += 1
            continue

        quality = item.get(
            "quality_score"
        )

        if quality is not None:

            try:

                quality = float(
                    quality
                )

            except Exception:

                quality = None

        if (
            quality is not None
            and
            quality
            <
            min_quality
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
                value.strip()
                for value
                in tags.split(",")
                if value.strip()
            ]

        elif not isinstance(
            tags,
            list,
        ):

            tags = [
                str(tags)
            ]

        row = {

            "id":
                make_id(
                    "skilln-devops-qa",
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

            "url":
                item.get(
                    "url"
                ),

            "has_code":
                item.get(
                    "has_code"
                ),

            "original_source":
                item.get(
                    "source"
                ),

            "task_type":
                "qa",

            "language":
                "en",

            "license":
                "CC-BY-SA-4.0",

        }

        rows.append(
            row
        )

    print()

    print(
        "Skilln accepted:",
        len(rows),
    )

    print(
        "Rejected empty:",
        rejected_empty,
    )

    print(
        "Rejected quality:",
        rejected_quality,
    )

    return rows


# ============================================================
# pavanmantha/devops-v1
# ============================================================

def load_devops_v1(
    cache_dir,
):

    print()

    print(
        "=" * 80
    )

    print(
        "pavanmantha/devops-v1"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 同样直接读取明确的 parquet。
    #
    # 当前仓库：
    #
    # data/train-00000-of-00001.parquet
    # --------------------------------------------------------

    dataset = download_exact_parquet(

        repo_id=(
            "pavanmantha/"
            "devops-v1"
        ),

        filename=(
            "data/"
            "train-00000-of-00001.parquet"
        ),

        cache_dir=cache_dir,

    )

    rows = []

    for item in dataset:

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

        rows.append(
            {

                "id":
                    make_id(
                        "pavanmantha-devops-v1",
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

                "url":
                    None,

                "has_code":
                    None,

                "original_source":
                    None,

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
        len(rows),
    )

    return rows


# ============================================================
# Linux terminal tool calling
# ============================================================

def parse_tool_arguments(
    arguments,
):

    if arguments is None:
        return ""

    if isinstance(
        arguments,
        dict,
    ):

        for key in (
            "command",
            "cmd",
            "shell",
            "script",
            "input",
        ):

            value = arguments.get(
                key
            )

            if isinstance(
                value,
                str,
            ):

                value = (
                    value.strip()
                )

                if value:
                    return value

        return json.dumps(
            arguments,
            ensure_ascii=False,
        )

    if isinstance(
        arguments,
        str,
    ):

        raw = (
            arguments.strip()
        )

        if not raw:
            return ""

        try:

            parsed = json.loads(
                raw
            )

            return parse_tool_arguments(
                parsed
            )

        except Exception:

            return raw

    return str(
        arguments
    ).strip()


def extract_tool_calls(
    message,
):

    tool_calls = message.get(
        "tool_calls"
    )

    if not tool_calls:
        return ""

    if not isinstance(
        tool_calls,
        list,
    ):

        tool_calls = [
            tool_calls
        ]

    outputs = []

    for call in tool_calls:

        if not isinstance(
            call,
            dict,
        ):
            continue

        function = call.get(
            "function"
        )

        if isinstance(
            function,
            dict,
        ):

            arguments = (
                function.get(
                    "arguments"
                )
            )

        else:

            arguments = call.get(
                "arguments"
            )

        command = parse_tool_arguments(
            arguments
        )

        if command:

            outputs.append(
                command
            )

    return "\n".join(
        outputs
    ).strip()


def load_linux_tools(
    cache_dir,
):

    print()

    print(
        "=" * 80
    )

    print(
        "iselabvn/Linux-terminal-tool-calling"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 这个仓库当前 Viewer 工作正常，
    # 可以正常使用 load_dataset。
    # --------------------------------------------------------

    dataset_dict = load_dataset(

        "iselabvn/"
        "Linux-terminal-tool-calling",

        cache_dir=str(
            cache_dir
        ),

    )

    dataset = dataset_dict[
        "train"
    ]

    print(
        "Rows:",
        len(dataset),
    )

    print(
        "Columns:",
        dataset.column_names,
    )

    rows = []

    skipped = 0

    for item in dataset:

        messages = item.get(
            "messages",
            []
        )

        metadata = item.get(
            "metadata",
            {}
        )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}

        if not isinstance(
            messages,
            list,
        ):

            skipped += 1
            continue

        user_text = ""
        assistant_text = ""

        for message in messages:

            if not isinstance(
                message,
                dict,
            ):
                continue

            role = message.get(
                "role"
            )

            if (
                role == "user"
                and
                not user_text
            ):

                user_text = normalize_text(
                    message.get(
                        "content",
                        "",
                    )
                )

            elif (
                role == "assistant"
                and
                not assistant_text
            ):

                # --------------------------------------------
                # 只拿最终答案。
                #
                # 不读取 reasoning_content，
                # 不把隐藏推理过程训练进去。
                # --------------------------------------------

                assistant_text = normalize_text(
                    message.get(
                        "content",
                        "",
                    )
                )

                if not assistant_text:

                    assistant_text = (
                        normalize_text(
                            extract_tool_calls(
                                message
                            )
                        )
                    )

        if not user_text:
            skipped += 1
            continue

        if not assistant_text:
            skipped += 1
            continue

        rows.append(
            {

                "id":
                    make_id(
                        "linux-terminal-tool-calling",
                        user_text,
                    ),

                "instruction":
                    user_text,

                "output":
                    assistant_text,

                "source":
                    "iselabvn/Linux-terminal-tool-calling",

                "category":
                    metadata.get(
                        "category",
                        "linux-command",
                    ),

                "difficulty":
                    None,

                "quality_score":
                    None,

                "tags":
                    [
                        "linux",
                        "shell",
                    ],

                "url":
                    metadata.get(
                        "man_reference"
                    ),

                "has_code":
                    True,

                "original_source":
                    metadata.get(
                        "original_description"
                    ),

                "task_type":
                    "linux_command",

                "language":
                    "en",

                "license":
                    None,

            }
        )

    print()

    print(
        "Linux accepted:",
        len(rows),
    )

    print(
        "Linux skipped:",
        skipped,
    )

    return rows


# ============================================================
# Dedup
# ============================================================

def deduplicate(
    rows,
):

    seen_questions = set()
    seen_pairs = set()

    result = []

    duplicate_question = 0
    duplicate_pair = 0

    for row in rows:

        question_key = (
            normalize_for_dedup(
                row[
                    "instruction"
                ]
            )
        )

        answer_key = (
            normalize_for_dedup(
                row[
                    "output"
                ]
            )
        )

        if not question_key:
            continue

        if not answer_key:
            continue

        pair_key = (
            question_key
            +
            "\n"
            +
            answer_key
        )

        if (
            pair_key
            in
            seen_pairs
        ):

            duplicate_pair += 1
            continue

        if (
            question_key
            in
            seen_questions
        ):

            duplicate_question += 1
            continue

        seen_pairs.add(
            pair_key
        )

        seen_questions.add(
            question_key
        )

        result.append(
            row
        )

    print()

    print(
        "Duplicate pair:",
        duplicate_pair,
    )

    print(
        "Duplicate question:",
        duplicate_question,
    )

    return result


# ============================================================
# Benchmark
# ============================================================

def download_itbench(
    output_dir,
):

    print()

    print(
        "=" * 80
    )

    print(
        "Downloading Benchmark"
    )

    print(
        "ibm-research/ITBench-Lite"
    )

    print(
        "=" * 80
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = snapshot_download(

        repo_id=(
            "ibm-research/"
            "ITBench-Lite"
        ),

        repo_type="dataset",

        local_dir=str(
            output_dir
        ),

    )

    print(
        "Benchmark saved:"
    )

    print(
        result
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    output_dir = (
        ROOT
        /
        args.output_dir
    )

    cache_dir = (
        ROOT
        /
        args.cache_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "OpsLM Phase3"
    )

    print(
        "Hugging Face Dataset Downloader"
    )

    print(
        "=" * 80
    )

    print(
        "HF_ENDPOINT:",
        os.environ.get(
            "HF_ENDPOINT"
        ),
    )

    print(
        "Cache:",
        cache_dir,
    )

    all_rows = []


    # ========================================================
    # Dataset 1
    # ========================================================

    skilln_rows = load_skilln(
        cache_dir,
        args.skilln_min_quality,
    )

    write_jsonl(

        output_dir
        /
        "skilln_devops_qa.jsonl",

        skilln_rows,

    )

    all_rows.extend(
        skilln_rows
    )


    # ========================================================
    # Dataset 2
    # ========================================================

    devops_rows = load_devops_v1(
        cache_dir
    )

    write_jsonl(

        output_dir
        /
        "devops_v1.jsonl",

        devops_rows,

    )

    all_rows.extend(
        devops_rows
    )


    # ========================================================
    # Dataset 3
    # ========================================================

    if not args.skip_linux_tools:

        try:

            linux_rows = (
                load_linux_tools(
                    cache_dir
                )
            )

            write_jsonl(

                output_dir
                /
                "linux_terminal_tools.jsonl",

                linux_rows,

            )

            all_rows.extend(
                linux_rows
            )

        except Exception as exc:

            print()

            print(
                "[WARNING]"
            )

            print(
                "Linux tool dataset failed:"
            )

            print(
                repr(exc)
            )

            print()

            print(
                "Continue without Linux dataset."
            )


    # ========================================================
    # Dedup
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "Exact Dedup"
    )

    print(
        "=" * 80
    )

    deduped = deduplicate(
        all_rows
    )

    final_path = (
        output_dir
        /
        "all_sft.jsonl"
    )

    write_jsonl(
        final_path,
        deduped,
    )


    # ========================================================
    # Stats
    # ========================================================

    source_counter = Counter(

        row[
            "source"
        ]

        for row
        in deduped

    )

    category_counter = Counter(

        str(
            row.get(
                "category"
            )
            or
            "unknown"
        )

        for row
        in deduped

    )

    language_counter = Counter(

        str(
            row.get(
                "language"
            )
            or
            "unknown"
        )

        for row
        in deduped

    )

    manifest = {

        "hf_endpoint":
            os.environ.get(
                "HF_ENDPOINT"
            ),

        "examples_before_dedup":
            len(all_rows),

        "examples_after_dedup":
            len(deduped),

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

    }

    manifest_path = (
        output_dir
        /
        "manifest.json"
    )

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
    # Benchmark
    # ========================================================

    if not args.skip_benchmark:

        benchmark_dir = (

            ROOT
            /
            "data"
            /
            "phase3"
            /
            "eval"
            /
            "ITBench-Lite"

        )

        try:

            download_itbench(
                benchmark_dir
            )

        except Exception as exc:

            print()

            print(
                "[WARNING]"
            )

            print(
                "ITBench download failed:"
            )

            print(
                repr(exc)
            )

            print()

            print(
                "SFT dataset is still usable."
            )


    # ========================================================
    # Summary
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "Phase3 HF Download Finished"
    )

    print(
        "=" * 80
    )

    print(
        "Before dedup:",
        len(all_rows),
    )

    print(
        "After dedup:",
        len(deduped),
    )

    print()

    print(
        "Sources:"
    )

    for source, count in (
        source_counter
        .most_common()
    ):

        print(
            f"  {source}: {count}"
        )

    print()

    print(
        "Final:"
    )

    print(
        final_path
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
