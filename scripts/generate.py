import sys
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT),
)


from src.config import ModelConfig
from src.model import OpsLMToy


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final.pt",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Linux",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    return parser.parse_args()


def sample_next_token(
    logits,
    temperature,
    top_k,
):

    # ========================================================
    # Temperature
    # ========================================================

    logits = (
        logits
        /
        max(
            temperature,
            1e-5,
        )
    )

    # ========================================================
    # Top-K
    # ========================================================

    if (
        top_k is not None
        and
        top_k > 0
    ):

        values, _ = torch.topk(
            logits,
            min(
                top_k,
                logits.size(-1),
            )
        )

        threshold = values[
            -1
        ]

        logits = torch.where(
            logits < threshold,
            torch.tensor(
                float("-inf"),
                device=logits.device,
            ),
            logits,
        )

    # ========================================================
    # Softmax
    # ========================================================

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    # ========================================================
    # Sampling
    # ========================================================

    next_token = torch.multinomial(
        probabilities,
        num_samples=1,
    )

    return next_token


def main():

    args = parse_args()

    torch.manual_seed(42)

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer_path = (
        ROOT
        /
        "tokenizer"
        /
        "artifacts"
        /
        "tokenizer.json"
    )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint_path = (
        ROOT
        /
        args.checkpoint
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    print(
        "Training step:",
        checkpoint.get(
            "step",
            "unknown",
        )
    )

    print(
        "Training loss:",
        checkpoint.get(
            "loss",
            "unknown",
        )
    )

    # ========================================================
    # Model Config
    # ========================================================

    config_dict = checkpoint[
        "model_config"
    ]

    config = ModelConfig(
        **config_dict
    )

    # ========================================================
    # Model
    # ========================================================

    model = OpsLMToy(
        config
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    # ========================================================
    # Prompt
    # ========================================================

    encoding = tokenizer.encode(
        args.prompt
    )

    input_ids = torch.tensor(
        [
            encoding.ids
        ],
        dtype=torch.long,
        device=device,
    )

    print()

    print(
        "Prompt:",
        args.prompt
    )

    print(
        "Prompt tokens:",
        encoding.tokens
    )

    # ========================================================
    # Generate
    # ========================================================

    eos_token_id = (
        tokenizer.token_to_id(
            "<eos>"
        )
    )

    with torch.no_grad():

        for _ in range(
            args.max_new_tokens
        ):

            # ------------------------------------------------
            # 不超过最大上下文
            # ------------------------------------------------

            context = input_ids[
                :,
                -config.max_seq_len:
            ]

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                context
            )

            # ------------------------------------------------
            # 只取最后一个位置
            # ------------------------------------------------

            next_token_logits = (
                logits[
                    0,
                    -1,
                    :
                ]
            )

            # ------------------------------------------------
            # Sample
            # ------------------------------------------------

            next_token = (
                sample_next_token(
                    next_token_logits,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
            )

            # ------------------------------------------------
            # 拼回输入
            # ------------------------------------------------

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token.view(
                        1,
                        1,
                    ),
                ],
                dim=1,
            )

            # ------------------------------------------------
            # EOS
            # ------------------------------------------------

            if (
                eos_token_id
                is not None
                and
                next_token.item()
                ==
                eos_token_id
            ):

                break

    # ========================================================
    # Decode
    # ========================================================

    generated_ids = (
        input_ids[0]
        .tolist()
    )

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "GENERATED"
    )

    print(
        "=" * 80
    )

    print(
        text
    )


if __name__ == "__main__":

    main()
