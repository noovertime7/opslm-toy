import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class SFTDataset(Dataset):

    def __init__(
        self,
        jsonl_path,
        tokenizer_path,
        seq_len=512,
    ):
        super().__init__()

        self.jsonl_path = Path(jsonl_path)
        self.seq_len = seq_len

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

        if self.pad_id is None:
            raise ValueError("Tokenizer missing <pad>")

        if self.eos_id is None:
            raise ValueError("Tokenizer missing <eos>")

        self.samples = []

        with self.jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

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

                if not instruction:
                    continue

                if not output:
                    continue

                self.samples.append(
                    {
                        "instruction": instruction,
                        "output": output,
                    }
                )

        if not self.samples:
            raise ValueError(
                f"No valid SFT samples found in {self.jsonl_path}"
            )

        print(
            f"SFT dataset: {self.jsonl_path}"
        )

        print(
            f"SFT samples: {len(self.samples)}"
        )


    def __len__(self):
        return len(self.samples)


    def __getitem__(
        self,
        index,
    ):

        item = self.samples[index]

        instruction = item["instruction"]
        output = item["output"]

        prompt = (
            "用户："
            + instruction
            + "\n\n"
            + "助手："
        )

        prompt_ids = (
            self.tokenizer
            .encode(prompt)
            .ids
        )

        answer_ids = (
            self.tokenizer
            .encode(output)
            .ids
        )

        # ====================================================
        # Sequence
        #
        # <bos>
        # 用户：xxxx
        #
        # 助手：
        # answer...
        # <eos>
        # ====================================================

        input_ids = []
        labels = []

        if self.bos_id is not None:

            input_ids.append(
                self.bos_id
            )

            labels.append(
                -100
            )

        # ====================================================
        # Prompt
        #
        # 不计算用户问题部分的 loss
        # ====================================================

        input_ids.extend(
            prompt_ids
        )

        labels.extend(
            [-100]
            * len(prompt_ids)
        )

        # ====================================================
        # Assistant Answer
        #
        # 只学习回答部分
        # ====================================================

        input_ids.extend(
            answer_ids
        )

        labels.extend(
            answer_ids
        )

        # ====================================================
        # EOS
        # ====================================================

        input_ids.append(
            self.eos_id
        )

        labels.append(
            self.eos_id
        )

        # ====================================================
        # Truncate
        #
        # 保留固定最大长度
        # ====================================================

        if len(input_ids) > self.seq_len:

            input_ids = input_ids[
                :self.seq_len
            ]

            labels = labels[
                :self.seq_len
            ]

            # 如果被截断，最后一个有效 token 尽量设为 EOS
            if labels[-1] != -100:

                input_ids[-1] = self.eos_id
                labels[-1] = self.eos_id

        # ====================================================
        # Pad
        # ====================================================

        pad_length = (
            self.seq_len
            -
            len(input_ids)
        )

        if pad_length > 0:

            input_ids.extend(
                [self.pad_id]
                * pad_length
            )

            labels.extend(
                [-100]
                * pad_length
            )

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
