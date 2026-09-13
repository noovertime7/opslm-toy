from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer


class LanguageModelDataset(Dataset):

    def __init__(
        self,
        text_path: str | Path,
        tokenizer_path: str | Path,
        seq_len: int = 1024,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        super().__init__()

        self.text_path = Path(
            text_path
        )

        self.tokenizer_path = Path(
            tokenizer_path
        )

        self.seq_len = seq_len

        # ====================================================
        # 文件检查
        # ====================================================

        if not self.text_path.exists():

            raise FileNotFoundError(
                f"Text file not found: "
                f"{self.text_path}"
            )

        if not self.tokenizer_path.exists():

            raise FileNotFoundError(
                f"Tokenizer not found: "
                f"{self.tokenizer_path}"
            )

        # ====================================================
        # 加载 Tokenizer
        # ====================================================

        self.tokenizer = (
            Tokenizer.from_file(
                str(
                    self.tokenizer_path
                )
            )
        )

        # ====================================================
        # Special Token ID
        # ====================================================

        self.bos_token_id = (
            self.tokenizer
            .token_to_id(
                "<bos>"
            )
        )

        self.eos_token_id = (
            self.tokenizer
            .token_to_id(
                "<eos>"
            )
        )

        # ====================================================
        # 读取文本
        # ====================================================

        text = (
            self.text_path
            .read_text(
                encoding="utf-8"
            )
        )

        if not text.strip():

            raise ValueError(
                "Training text is empty"
            )

        # ====================================================
        # Tokenize
        # ====================================================

        encoding = (
            self.tokenizer
            .encode(
                text
            )
        )

        token_ids = list(
            encoding.ids
        )

        # ====================================================
        # BOS / EOS
        # ====================================================

        if add_bos:

            if (
                self.bos_token_id
                is None
            ):

                raise ValueError(
                    "<bos> token not found"
                )

            token_ids.insert(
                0,
                self.bos_token_id,
            )

        if add_eos:

            if (
                self.eos_token_id
                is None
            ):

                raise ValueError(
                    "<eos> token not found"
                )

            token_ids.append(
                self.eos_token_id
            )

        # ====================================================
        # 保存所有 Token
        # ====================================================

        self.tokens = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        # ====================================================
        # 一个训练样本需要：
        #
        # seq_len + 1 个 Token
        #
        # 例如：
        #
        # 1025 Token
        #
        # Input:
        # 前1024
        #
        # Target:
        # 后1024
        # ====================================================

        self.chunk_size = (
            self.seq_len + 1
        )

        # ====================================================
        # 当前先采用“不重叠切块”
        #
        # 每次前进 seq_len
        # ====================================================

        usable_tokens = (
            len(self.tokens)
            - 1
        )

        self.num_samples = (
            usable_tokens
            //
            self.seq_len
        )

        if self.num_samples <= 0:

            raise ValueError(
                "Corpus is too small. "
                f"Need at least "
                f"{self.seq_len + 1} tokens, "
                f"but got {len(self.tokens)}."
            )

    def __len__(
        self,
    ) -> int:

        return self.num_samples

    def __getitem__(
        self,
        index: int,
    ):

        # ====================================================
        # 每个样本从：
        #
        # index * seq_len
        #
        # 开始
        # ====================================================

        start = (
            index
            *
            self.seq_len
        )

        end = (
            start
            +
            self.chunk_size
        )

        chunk = (
            self.tokens[
                start:end
            ]
        )

        # ====================================================
        # 如果最后不足 seq_len + 1
        # 不使用
        # ====================================================

        if (
            len(chunk)
            !=
            self.chunk_size
        ):

            raise IndexError(
                "Incomplete chunk"
            )

        # ====================================================
        # Input
        # ====================================================

        input_ids = (
            chunk[:-1]
            .clone()
        )

        # ====================================================
        # Target
        # ====================================================

        target_ids = (
            chunk[1:]
            .clone()
        )

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }
