import sys
import argparse
from pathlib import Path
from contextlib import nullcontext

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
        default="checkpoints_v3/best.pt",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
    )

    return parser.parse_args()


def sample_token(
    logits,
    previous_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
):

    logits = (
        logits
        .float()
        .clone()
    )

    # ========================================================
    # Repetition penalty
    # ========================================================

    if (
        repetition_penalty
        !=
        1.0
    ):

        for token_id in set(
            previous_tokens
        ):

            if (
                token_id
                >=
                logits.shape[-1]
            ):
                continue

            if (
                logits[token_id]
                >
                0
            ):

                logits[token_id] /= (
                    repetition_penalty
                )

            else:

                logits[token_id] *= (
                    repetition_penalty
                )

    # ========================================================
    # Temperature
    # ========================================================

    temperature = max(
        temperature,
        1e-5,
    )

    logits = (
        logits
        /
        temperature
    )

    # ========================================================
    # Top-K
    # ========================================================

    if (
        top_k
        >
        0
    ):

        top_k = min(
            top_k,
            logits.size(-1),
        )

        values, _ = torch.topk(
            logits,
            top_k,
        )

        threshold = (
            values[-1]
        )

        logits[
            logits
            <
            threshold
        ] = float("-inf")

    # ========================================================
    # Top-P
    # ========================================================

    if (
        top_p
        <
        1.0
    ):

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        sorted_probs = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        sorted_remove = (
            cumulative_probs
            >
            top_p
        )

        sorted_remove[1:] = (
            sorted_remove[:-1]
            .clone()
        )

        sorted_remove[0] = False

        remove_indices = (
            sorted_indices[
                sorted_remove
            ]
        )

        logits[
            remove_indices
        ] = float("-inf")

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    next_token = torch.multinomial(
        probs,
        num_samples=1,
    )

    return next_token


def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    tokenizer_path = (
        ROOT
        /
        "tokenizer"
        /
        "artifacts_v2"
        /
        "tokenizer.json"
    )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    checkpoint_path = (
        ROOT
        /
        args.checkpoint
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Stage:",
        checkpoint.get(
            "stage",
            "unknown",
        ),
    )

    print(
        "Step:",
        checkpoint.get(
            "step",
            "unknown",
        ),
    )

    print(
        "Train loss:",
        checkpoint.get(
            "train_loss",
            "unknown",
        ),
    )

    print(
        "Val loss:",
        checkpoint.get(
            "val_loss",
            "unknown",
        ),
    )

    config = ModelConfig(
        **checkpoint[
            "model_config"
        ]
    )

    model = OpsLMToy(
        config
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    model.eval()

    bos_id = tokenizer.token_to_id(
        "<bos>"
    )

    eos_id = tokenizer.token_to_id(
        "<eos>"
    )

    prompt_text = (
        "用户："
        + args.prompt
        + "\n\n"
        + "助手："
    )

    prompt_ids = (
        tokenizer
        .encode(
            prompt_text
        )
        .ids
    )

    if (
        bos_id
        is not None
    ):

        prompt_ids = (
            [bos_id]
            +
            prompt_ids
        )

    input_ids = torch.tensor(
        [
            prompt_ids
        ],
        dtype=torch.long,
        device=device,
    )

    generated_ids = []

    print()

    print(
        "=" * 80
    )

    print(
        "USER"
    )

    print(
        "=" * 80
    )

    print(
        args.prompt
    )

    print()

    print(
        "=" * 80
    )

    print(
        "ASSISTANT"
    )

    print(
        "=" * 80
    )

    autocast_context = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():

        for _ in range(
            args.max_new_tokens
        ):

            context = (
                input_ids[
                    :,
                    -config.max_seq_len:
                ]
            )

            with autocast_context:

                logits = model(
                    context
                )

            next_logits = (
                logits[
                    0,
                    -1,
                    :
                ]
            )

            repetition_context = (
                prompt_ids
                +
                generated_ids
            )

            next_token = sample_token(
                next_logits,
                repetition_context,
                args.temperature,
                args.top_k,
                args.top_p,
                args.repetition_penalty,
            )

            token_id = int(
                next_token.item()
            )

            if (
                eos_id
                is not None
                and
                token_id
                ==
                eos_id
            ):

                break

            generated_ids.append(
                token_id
            )

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

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(
        answer
    )


if __name__ == "__main__":
    main()
