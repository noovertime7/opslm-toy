import json
import re
import random
import hashlib
import argparse

from pathlib import Path
from collections import Counter


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        default=
        "data/phase2/processed/all_documents.jsonl",
    )


    parser.add_argument(
        "--output-dir",
        default=
        "data/phase2/processed",
    )


    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
    )


    return parser.parse_args()



def resolve_path(p):

    path = Path(p)

    if path.is_absolute():

        return path

    return ROOT / path



def clean_markdown(text):


    # 删除 yaml front matter

    text = re.sub(
        r"^---.*?---",
        "",
        text,
        flags=re.S,
    )


    # 删除 Hugo shortcode

    text = re.sub(
        r"\{\{<.*?>\}\}",
        "",
        text,
        flags=re.S,
    )


    text = re.sub(
        r"\{\{%.*?%\}\}",
        "",
        text,
        flags=re.S,
    )


    # 删除 HTML 注释

    text = re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.S,
    )


    # 图片保留 alt

    text = re.sub(
        r"!\[(.*?)\]\(.*?\)",
        r"\1",
        text,
    )


    # markdown link

    text = re.sub(
        r"\[(.*?)\]\(.*?\)",
        r"\1",
        text,
    )


    # 多余空行

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )


    return text.strip()



def quality_check(text):


    if len(text) < 500:

        return False


    if "\ufffd" in text:

        if (
            text.count("\ufffd")
            /
            len(text)
            >
            0.001
        ):

            return False


    return True



def digest(text):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()



def main():

    args = parse_args()


    input_path = resolve_path(
        args.input
    )


    output_dir = resolve_path(
        args.output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    documents = []

    seen = set()


    source_counter = Counter()


    with input_path.open(
        encoding="utf-8"
    ) as f:


        for line in f:


            item = json.loads(
                line
            )


            text = clean_markdown(
                item["text"]
            )


            if not quality_check(
                text
            ):

                continue


            h = digest(
                text
            )


            if h in seen:

                continue


            seen.add(
                h
            )


            documents.append({

                "source":
                    item["source"],

                "license":
                    item["license"],

                "text":
                    text,

                "sha256":
                    h,

            })


            source_counter[
                item["source"]
            ] += 1



    random.seed(
        42
    )


    random.shuffle(
        documents
    )


    val_size = int(
        len(documents)
        *
        args.val_ratio
    )


    val_docs = documents[
        :val_size
    ]


    train_docs = documents[
        val_size:
    ]



    # 写 jsonl

    for name, data in [

        (
            "train.jsonl",
            train_docs
        ),

        (
            "val.jsonl",
            val_docs
        )

    ]:


        with (
            output_dir
            /
            name
        ).open(
            "w",
            encoding="utf-8",
        ) as f:


            for item in data:

                f.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    +
                    "\n"
                )


    # tokenizer corpus

    with (
        output_dir
        /
        "tokenizer_corpus.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:


        for item in documents:

            f.write(
                item["text"]
            )

            f.write(
                "\n\n"
            )


    manifest = {

        "documents":
            len(documents),

        "train":
            len(train_docs),

        "val":
            len(val_docs),

        "sources":
            dict(source_counter),

    }


    (
        output_dir
        /
        "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )



if __name__ == "__main__":

    main()
