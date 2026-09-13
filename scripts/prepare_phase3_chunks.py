import json
import hashlib
import argparse
import re
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


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
        default=(
            "data/phase2/processed/"
            "all_documents.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/phase3/chunks/"
            "knowledge_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=(
            "tokenizer/artifacts_v2/"
            "tokenizer.json"
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=120,
    )

    return parser.parse_args()


def normalize_text(text):

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
        r"\n{4,}",
        "\n\n\n",
        text,
    )

    return text.strip()


def make_doc_id(item, text):

    source = str(
        item.get(
            "source",
            "",
        )
    )

    repo = str(
        item.get(
            "repo",
            "",
        )
    )

    path = str(
        item.get(
            "path",
            "",
        )
    )

    original_sha = str(
        item.get(
            "sha256",
            "",
        )
    )

    if not original_sha:

        original_sha = hashlib.sha256(
            text.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()

    raw = (
        source
        + "\n"
        + repo
        + "\n"
        + path
        + "\n"
        + original_sha
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def main():

    args = parse_args()

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

    tokenizer_path = (
        ROOT
        /
        args.tokenizer
    )

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    if args.overlap >= args.max_tokens:

        raise ValueError(
            "--overlap must be smaller "
            "than --max-tokens"
        )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    step_size = (
        args.max_tokens
        -
        args.overlap
    )

    documents = 0
    written_chunks = 0
    skipped_documents = 0

    source_counter = Counter()

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as src, output_path.open(
        "w",
        encoding="utf-8",
    ) as dst:

        for line_number, line in enumerate(
            src,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            item = json.loads(
                line
            )

            text = normalize_text(
                str(
                    item.get(
                        "text",
                        "",
                    )
                )
            )

            if not text:

                skipped_documents += 1
                continue

            token_ids = (
                tokenizer
                .encode(text)
                .ids
            )

            if (
                len(token_ids)
                <
                args.min_tokens
            ):

                skipped_documents += 1
                continue

            documents += 1

            doc_id = make_doc_id(
                item,
                text,
            )

            source = str(
                item.get(
                    "source",
                    "unknown",
                )
            )

            source_counter[
                source
            ] += 1

            chunk_index = 0

            for start in range(
                0,
                len(token_ids),
                step_size,
            ):

                end = min(
                    start
                    +
                    args.max_tokens,

                    len(token_ids),
                )

                chunk_ids = (
                    token_ids[
                        start:end
                    ]
                )

                if (
                    len(chunk_ids)
                    <
                    args.min_tokens
                ):
                    continue

                chunk_text = (
                    tokenizer.decode(
                        chunk_ids,
                        skip_special_tokens=True,
                    )
                    .strip()
                )

                if not chunk_text:
                    continue

                chunk_id = (
                    doc_id
                    +
                    ":"
                    +
                    str(chunk_index)
                )

                record = {

                    "chunk_id":
                        chunk_id,

                    "doc_id":
                        doc_id,

                    "chunk_index":
                        chunk_index,

                    "token_start":
                        start,

                    "token_end":
                        end,

                    "token_count":
                        len(chunk_ids),

                    "source":
                        source,

                    "repo":
                        item.get(
                            "repo"
                        ),

                    "path":
                        item.get(
                            "path"
                        ),

                    "license":
                        item.get(
                            "license"
                        ),

                    "text":
                        chunk_text,

                }

                dst.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    +
                    "\n"
                )

                written_chunks += 1
                chunk_index += 1

            if (
                documents
                %
                500
                ==
                0
            ):

                print(
                    f"documents={documents} "
                    f"chunks={written_chunks}",
                    flush=True,
                )

    print()

    print(
        "=" * 70
    )

    print(
        "Phase3 chunk preparation finished"
    )

    print(
        "=" * 70
    )

    print(
        "Documents:",
        documents,
    )

    print(
        "Skipped documents:",
        skipped_documents,
    )

    print(
        "Chunks:",
        written_chunks,
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
        "Saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()
