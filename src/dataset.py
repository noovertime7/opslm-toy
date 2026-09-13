import json
from pathlib import Path

import torch

from torch.utils.data import Dataset
from tokenizers import Tokenizer


class PackedLanguageModelDataset(Dataset):

    def __init__(
        self,
        jsonl_path,
        tokenizer_path,
        seq_len=512,
        cache_path=None,
    ):

        super().__init__()

        self.seq_len = seq_len

        self.jsonl_path = Path(
            jsonl_path
        )

        # ====================================================
        # 自动 cache 路径
        # ====================================================

        if cache_path:

            self.cache_path = Path(
                cache_path
            )

        else:

            self.cache_path = (
                self.jsonl_path
                .parent
                /
                "cache"
                /
                (
                    self.jsonl_path.stem
                    +
                    "_tokens.pt"
                )
            )


        # ====================================================
        # Tokenizer
        # ====================================================

        self.tokenizer = Tokenizer.from_file(
            str(
                tokenizer_path
            )
        )


        self.eos_id = (
            self.tokenizer
            .token_to_id(
                "<eos>"
            )
        )


        self.bos_id = (
            self.tokenizer
            .token_to_id(
                "<bos>"
            )
        )


        if self.eos_id is None:

            raise ValueError(
                "Tokenizer missing <eos>"
            )


        # ====================================================
        # Token Cache
        # ====================================================

        if self.cache_path.exists():

            print(
                "Loading token cache:"
            )

            print(
                self.cache_path
            )

            self.tokens = torch.load(
                self.cache_path
            )

        else:

            print(
                "Building token cache..."
            )

            self.tokens = (
                self.build_token_stream()
            )


            self.cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            torch.save(
                self.tokens,
                self.cache_path,
            )


            print(
                "Saved token cache:"
            )

            print(
                self.cache_path
            )


        # ====================================================
        # Samples
        # ====================================================

        self.num_samples = (

            (
                len(
                    self.tokens
                )
                -
                1
            )
            //
            self.seq_len

        )


        print(
            "Total tokens:",
            len(
                self.tokens
            )
        )


        print(
            "Samples:",
            self.num_samples
        )


    # ========================================================
    # Build Token Stream
    # ========================================================

    def build_token_stream(
        self
    ):

        tokens = []


        with self.jsonl_path.open(
            encoding="utf-8"
        ) as file:


            for index, line in enumerate(
                file
            ):


                if index % 500 == 0:

                    print(
                        f"Tokenizing {index}"
                    )


                item = json.loads(
                    line
                )


                text = item.get(
                    "text",
                    "",
                )


                if not text:

                    continue


                ids = (
                    self.tokenizer
                    .encode(
                        text
                    )
                    .ids
                )


                # 文档 token

                if (
                    index == 0
                    and
                    self.bos_id is not None
                ):

                    tokens.append(
                        self.bos_id
                    )


                tokens.extend(
                    ids
                )


                # 文档结束

                tokens.append(
                    self.eos_id
                )


        return torch.tensor(
            tokens,
            dtype=torch.long,
        )


    # ========================================================
    # Length
    # ========================================================

    def __len__(
        self
    ):

        return self.num_samples


    # ========================================================
    # Get Item
    # ========================================================

    def __getitem__(
        self,
        index,
    ):


        start = (
            index
            *
            self.seq_len
        )


        chunk = (
            self.tokens[
                start:
                start
                +
                self.seq_len
                +
                1
            ]
        )


        return {

            "input_ids":
                chunk[:-1],

            "target_ids":
                chunk[1:],

        }
