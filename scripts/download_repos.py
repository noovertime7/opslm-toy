import argparse
import hashlib
import json
import re
import subprocess

from pathlib import Path


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
        "--config",
        default="data/phase2/repos.json",
    )

    parser.add_argument(
        "--repo-dir",
        default="data/phase2/repos",
    )

    parser.add_argument(
        "--output",
        default=(
            "data/phase2/github/"
            "repositories.jsonl"
        ),
    )

    return parser.parse_args()


# ============================================================
# Path
# ============================================================

def resolve_path(
    value
):

    path = Path(
        value
    )

    if path.is_absolute():

        return path

    return ROOT / path


# ============================================================
# Git Clone
# ============================================================

def clone_repo(
    repo,
    repo_dir,
):

    name = repo["name"]

    target = (
        repo_dir
        /
        name
    )


    if target.exists():

        print(
            f"[exists] {name}"
        )

        return target


    print(
        f"[clone] {name}"
    )


    subprocess.run(

        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            repo["branch"],
            repo["url"],
            str(target),
        ],

        check=True,

    )


    return target


# ============================================================
# Text Clean
# ============================================================

def clean_text(
    text
):

    # 删除 yaml front matter

    text = re.sub(
        r"^---.*?---",
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


    # 删除连续空白

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )


    return text.strip()


# ============================================================
# Hash
# ============================================================

def sha256(
    text
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Scan Files
# ============================================================

def scan_repo(
    repo_config,
    repo_path,
):

    extensions = set(
        repo_config[
            "extensions"
        ]
    )


    include_dirs = [
        repo_path
        /
        d
        for d
        in repo_config[
            "include_dirs"
        ]
    ]


    for include_dir in include_dirs:


        if not include_dir.exists():

            continue


        for file in include_dir.rglob(
            "*"
        ):


            if not file.is_file():

                continue


            if file.suffix not in extensions:

                continue


            # 太大的文件跳过

            if (
                file.stat()
                .st_size
                >
                2 * 1024 * 1024
            ):

                continue


            try:

                text = file.read_text(
                    encoding="utf-8"
                )

            except UnicodeDecodeError:

                continue


            text = clean_text(
                text
            )


            # 太短跳过

            if len(text) < 500:

                continue


            yield {

                "source":
                    repo_config[
                        "name"
                    ],

                "repo":
                    repo_config[
                        "url"
                    ],

                "path":
                    str(
                        file.relative_to(
                            repo_path
                        )
                    ),

                "license":
                    repo_config[
                        "license"
                    ],

                "sha256":
                    sha256(
                        text
                    ),

                "text":
                    text,

            }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()


    config_path = resolve_path(
        args.config
    )


    repo_dir = resolve_path(
        args.repo_dir
    )


    output_path = resolve_path(
        args.output
    )


    repo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )


    seen = set()

    total = 0


    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output:


        for repo in config["repos"]:


            repo_path = clone_repo(
                repo,
                repo_dir,
            )


            print()

            print(
                "=" * 80
            )

            print(
                "Processing:",
                repo["name"]
            )

            print(
                "=" * 80
            )


            count = 0


            for document in scan_repo(
                repo,
                repo_path,
            ):


                digest = document[
                    "sha256"
                ]


                if digest in seen:

                    continue


                seen.add(
                    digest
                )


                output.write(
                    json.dumps(
                        document,
                        ensure_ascii=False,
                    )
                    +
                    "\n"
                )


                count += 1

                total += 1


                if count % 100 == 0:

                    print(
                        f"{repo['name']} "
                        f"{count} docs"
                    )


            print(
                f"{repo['name']} finished "
                f"{count} docs"
            )


    print()

    print(
        "=" * 80
    )

    print(
        "Finished"
    )

    print(
        "Documents:",
        total
    )

    print(
        "Output:",
        output_path
    )


if __name__ == "__main__":

    main()
