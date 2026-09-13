import json
from pathlib import Path
import argparse
import hashlib


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/phase2/raw/documents.jsonl",
            "data/phase2/github/repositories.jsonl",
        ],
    )

    parser.add_argument(
        "--output",
        default="data/phase2/processed/all_documents.jsonl",
    )

    return parser.parse_args()



def resolve_path(p):

    path = Path(p)

    if path.is_absolute():
        return path

    return ROOT / path



def normalize_doc(item):

    text = item.get(
        "text",
        ""
    )

    if not text:
        return None


    return {

        "source":
            item.get(
                "source",
                "unknown"
            ),

        "repo":
            item.get(
                "repo",
                ""
            ),

        "path":
            item.get(
                "path",
                ""
            ),

        "url":
            item.get(
                "url",
                ""
            ),

        "license":
            item.get(
                "license",
                "unknown"
            ),

        "text":
            text,

    }



def sha256(text):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()



def main():

    args = parse_args()


    output = resolve_path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    seen = set()

    total = 0

    duplicate = 0


    with output.open(
        "w",
        encoding="utf-8",
    ) as writer:


        for input_file in args.inputs:

            path = resolve_path(
                input_file
            )

            if not path.exists():

                print(
                    "[skip]",
                    path
                )

                continue


            print(
                "[merge]",
                path
            )


            with path.open(
                "r",
                encoding="utf-8",
            ) as reader:


                for line in reader:


                    try:

                        item = json.loads(
                            line
                        )

                    except:

                        continue


                    doc = normalize_doc(
                        item
                    )


                    if doc is None:

                        continue


                    digest = sha256(
                        doc["text"]
                    )


                    if digest in seen:

                        duplicate += 1

                        continue


                    seen.add(
                        digest
                    )


                    doc[
                        "sha256"
                    ] = digest


                    writer.write(
                        json.dumps(
                            doc,
                            ensure_ascii=False,
                        )
                        +
                        "\n"
                    )


                    total += 1


    print()

    print(
        "="*70
    )

    print(
        "Merge Finished"
    )

    print(
        "Documents:",
        total
    )

    print(
        "Duplicate:",
        duplicate
    )

    print(
        "Output:",
        output
    )


if __name__ == "__main__":

    main()
