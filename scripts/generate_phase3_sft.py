import os
import re
import json
import time
import random
import argparse
import urllib.request
import urllib.error

from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


SYSTEM_PROMPT = """
你是一名资深 SRE、Linux、Kubernetes、云原生和 AI Infra 工程师。

你的任务不是总结文档，而是根据给定的技术知识片段，
构造适合训练 SRE 指令模型的高质量 instruction / answer 数据。

必须遵守以下要求：

1. 所有答案必须能够从给定知识片段或其直接技术含义推出。
2. 不允许凭空补充片段完全没有依据的产品版本、参数或事实。
3. 不要说“根据给定文档”“根据上述内容”“原文提到”等。
4. 问题必须像真实工程师会提出的问题。
5. 答案必须直接回答问题，不要复述问题。
6. 尽可能包含排查思路、判断逻辑和为什么。
7. 不要所有问题都写成“怎么排查”。
8. 数据类型要多样化。
9. 不要生成简单复制原文即可回答的问题。
10. 不要生成答案依赖片段之外私有上下文的问题。

任务类型可以包括：

- troubleshooting：故障排查
- concept：原理解释
- command_analysis：命令输出分析
- log_analysis：日志分析
- scenario：真实运维场景
- correction：判断错误观点
- comparison：方案对比
- architecture：架构与设计
- best_practice：最佳实践
- risk_analysis：风险分析

只返回 JSON 数组。

禁止 Markdown 代码围栏。
禁止额外解释。
"""


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=(
            "data/phase3/chunks/"
            "knowledge_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/phase3/generated/"
            "teacher_output.jsonl"
        ),
    )

    parser.add_argument(
        "--errors",
        type=str,
        default=(
            "data/phase3/generated/"
            "teacher_errors.jsonl"
        ),
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--examples-per-chunk",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2200,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


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
                rows.append(
                    json.loads(line)
                )
            except Exception:
                continue

    return rows


def select_chunks(
    chunks,
    limit,
    seed,
):

    random.seed(
        seed
    )

    groups = defaultdict(
        list
    )

    for chunk in chunks:

        source = str(
            chunk.get(
                "source",
                "unknown",
            )
        )

        groups[
            source
        ].append(
            chunk
        )

    queues = {}

    for source, rows in groups.items():

        random.shuffle(
            rows
        )

        queues[
            source
        ] = deque(
            rows
        )

    sources = sorted(
        queues.keys()
    )

    random.shuffle(
        sources
    )

    selected = []

    while (
        len(selected)
        <
        limit
    ):

        added = False

        for source in sources:

            queue = queues[
                source
            ]

            if not queue:
                continue

            selected.append(
                queue.popleft()
            )

            added = True

            if (
                len(selected)
                >=
                limit
            ):
                break

        if not added:
            break

    return selected


def strip_reasoning(text):

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.S,
    )

    return text.strip()


def extract_json_array(text):

    text = strip_reasoning(
        text
    )

    text = text.strip()

    if text.startswith(
        "```"
    ):

        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"```$",
            "",
            text,
        )

        text = text.strip()

    try:

        result = json.loads(
            text
        )

        if isinstance(
            result,
            list,
        ):
            return result

    except Exception:
        pass

    start = text.find(
        "["
    )

    end = text.rfind(
        "]"
    )

    if (
        start
        ==
        -1
        or
        end
        ==
        -1
        or
        end
        <=
        start
    ):

        raise ValueError(
            "No JSON array found"
        )

    result = json.loads(
        text[
            start:end + 1
        ]
    )

    if not isinstance(
        result,
        list,
    ):

        raise ValueError(
            "Teacher result is not a JSON array"
        )

    return result


def validate_examples(
    examples,
    max_examples,
):

    valid = []

    allowed_types = {
        "troubleshooting",
        "concept",
        "command_analysis",
        "log_analysis",
        "scenario",
        "correction",
        "comparison",
        "architecture",
        "best_practice",
        "risk_analysis",
    }

    for item in examples:

        if not isinstance(
            item,
            dict,
        ):
            continue

        instruction = str(
            item.get(
                "instruction",
                "",
            )
        ).strip()

        output = str(
            item.get(
                "output",
                "",
            )
        ).strip()

        task_type = str(
            item.get(
                "task_type",
                "scenario",
            )
        ).strip()

        tags = item.get(
            "tags",
            [],
        )

        if not instruction:
            continue

        if not output:
            continue

        if (
            task_type
            not in
            allowed_types
        ):

            task_type = (
                "scenario"
            )

        if not isinstance(
            tags,
            list,
        ):

            tags = []

        valid.append(
            {
                "task_type":
                    task_type,

                "instruction":
                    instruction,

                "output":
                    output,

                "tags":
                    [
                        str(tag)
                        for tag
                        in tags[:8]
                    ],
            }
        )

        if (
            len(valid)
            >=
            max_examples
        ):
            break

    return valid


def call_teacher(
    chunk,
    args,
    api_base,
    api_key,
    model_name,
):

    user_prompt = f"""
请根据下面的技术知识片段生成最多 {args.examples_per_chunk} 条高质量 SRE 指令训练数据。

要求尽量覆盖不同任务类型。

返回格式必须严格为：

[
  {{
    "task_type": "troubleshooting",
    "instruction": "真实工程师问题",
    "output": "完整且技术正确的回答",
    "tags": ["linux", "performance"]
  }}
]

如果当前知识片段只能支持 1~2 个高质量问题，
宁可少生成，也不要凑数量。

来源：
source={chunk.get("source")}
repo={chunk.get("repo")}
path={chunk.get("path")}

知识片段：

{chunk.get("text")}
""".strip()

    endpoint = (
        api_base.rstrip("/")
        +
        "/chat/completions"
    )

    payload = {

        "model":
            model_name,

        "messages":
            [
                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT,
                },
                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ],

        "temperature":
            args.temperature,

        "max_tokens":
            args.max_output_tokens,

    }

    headers = {
        "Content-Type":
            "application/json",
    }

    if (
        api_key
        and
        api_key
        not in {
            "EMPTY",
            "NONE",
            "-",
        }
    ):

        headers[
            "Authorization"
        ] = (
            "Bearer "
            +
            api_key
        )

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    last_error = None

    for attempt in range(
        1,
        args.retries + 1,
    ):

        try:

            with urllib.request.urlopen(
                request,
                timeout=args.timeout,
            ) as response:

                body = (
                    response
                    .read()
                    .decode(
                        "utf-8"
                    )
                )

            data = json.loads(
                body
            )

            content = (
                data[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]
            )

            examples = (
                extract_json_array(
                    content
                )
            )

            examples = (
                validate_examples(
                    examples,
                    args.examples_per_chunk,
                )
            )

            if not examples:

                raise ValueError(
                    "No valid examples"
                )

            return {

                "chunk_id":
                    chunk[
                        "chunk_id"
                    ],

                "doc_id":
                    chunk[
                        "doc_id"
                    ],

                "source":
                    chunk.get(
                        "source"
                    ),

                "repo":
                    chunk.get(
                        "repo"
                    ),

                "path":
                    chunk.get(
                        "path"
                    ),

                "examples":
                    examples,

            }

        except Exception as exc:

            last_error = exc

            if (
                attempt
                <
                args.retries
            ):

                time.sleep(
                    min(
                        2 ** attempt,
                        10,
                    )
                )

    raise RuntimeError(
        str(
            last_error
        )
    )


def main():

    args = parse_args()

    api_base = os.environ.get(
        "TEACHER_API_BASE",
        "",
    ).strip()

    api_key = os.environ.get(
        "TEACHER_API_KEY",
        "",
    ).strip()

    model_name = os.environ.get(
        "TEACHER_MODEL",
        "",
    ).strip()

    if not api_base:

        raise RuntimeError(
            "Missing TEACHER_API_BASE"
        )

    if not model_name:

        raise RuntimeError(
            "Missing TEACHER_MODEL"
        )

    input_path = (
        ROOT
        /
        args.input
    )

    output_path = (
        ROOT
        /
        args.output
    )

    error_path = (
        ROOT
        /
        args.errors
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_jsonl(
        input_path
    )

    if not chunks:

        raise RuntimeError(
            "No knowledge chunks found"
        )

    completed = set()

    if output_path.exists():

        for row in load_jsonl(
            output_path
        ):

            chunk_id = row.get(
                "chunk_id"
            )

            if chunk_id:

                completed.add(
                    chunk_id
                )

    selected = select_chunks(
        chunks,
        min(
            args.max_chunks,
            len(chunks),
        ),
        args.seed,
    )

    selected = [
        chunk
        for chunk
        in selected
        if chunk[
            "chunk_id"
        ]
        not in completed
    ]

    print(
        "=" * 70
    )

    print(
        "Phase3 Teacher SFT Generation"
    )

    print(
        "=" * 70
    )

    print(
        "Teacher API:",
        api_base,
    )

    print(
        "Teacher model:",
        model_name,
    )

    print(
        "Knowledge chunks:",
        len(chunks),
    )

    print(
        "Already completed:",
        len(completed),
    )

    print(
        "Chunks this run:",
        len(selected),
    )

    print(
        "Examples/chunk:",
        args.examples_per_chunk,
    )

    print(
        "Workers:",
        args.workers,
    )

    if not selected:

        print(
            "Nothing to generate."
        )

        return

    success = 0
    failed = 0
    examples_total = 0

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as output_file, error_path.open(
        "a",
        encoding="utf-8",
    ) as error_file:

        with ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:

            future_map = {

                executor.submit(
                    call_teacher,
                    chunk,
                    args,
                    api_base,
                    api_key,
                    model_name,
                ):
                chunk

                for chunk
                in selected
            }

            for future in as_completed(
                future_map
            ):

                chunk = future_map[
                    future
                ]

                try:

                    record = (
                        future.result()
                    )

                    output_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        +
                        "\n"
                    )

                    output_file.flush()

                    success += 1

                    examples_total += len(
                        record[
                            "examples"
                        ]
                    )

                except Exception as exc:

                    failed += 1

                    error_record = {

                        "chunk_id":
                            chunk.get(
                                "chunk_id"
                            ),

                        "doc_id":
                            chunk.get(
                                "doc_id"
                            ),

                        "source":
                            chunk.get(
                                "source"
                            ),

                        "path":
                            chunk.get(
                                "path"
                            ),

                        "error":
                            str(exc),

                    }

                    error_file.write(
                        json.dumps(
                            error_record,
                            ensure_ascii=False,
                        )
                        +
                        "\n"
                    )

                    error_file.flush()

                processed = (
                    success
                    +
                    failed
                )

                if (
                    processed
                    %
                    20
                    ==
                    0
                ):

                    print(
                        f"processed={processed}/"
                        f"{len(selected)} "
                        f"success={success} "
                        f"failed={failed} "
                        f"examples={examples_total}",
                        flush=True,
                    )

    print()

    print(
        "=" * 70
    )

    print(
        "Teacher generation finished"
    )

    print(
        "=" * 70
    )

    print(
        "Success chunks:",
        success,
    )

    print(
        "Failed chunks:",
        failed,
    )

    print(
        "Generated examples:",
        examples_total,
    )

    print(
        "Output:",
        output_path,
    )


if __name__ == "__main__":
    main()
