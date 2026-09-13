import sys
import argparse
from pathlib import Path

import torch


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT),
)


from tokenizers import Tokenizer

from src.config import ModelConfig
from src.model import OpsLMToy



# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_v2_best.pt",
    )


    parser.add_argument(
        "--prompt",
        type=str,
        default="Kubernetes",
    )


    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
    )


    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
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



# ============================================================
# Sampling
# ============================================================

def sample_token(
    logits,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    previous_tokens,
):


    logits = logits.clone()


    # ========================================================
    # repetition penalty
    # ========================================================

    if repetition_penalty != 1.0:

        for token_id in set(
            previous_tokens
        ):

            if logits[token_id] > 0:

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

    logits /= max(
        temperature,
        1e-5,
    )


    # ========================================================
    # Top-K
    # ========================================================

    if top_k > 0:


        values, _ = torch.topk(

            logits,

            min(
                top_k,
                logits.size(-1),
            ),

        )


        min_value = values[-1]


        logits = torch.where(

            logits < min_value,

            torch.tensor(
                float("-inf"),
                device=logits.device,
            ),

            logits,

        )


    # ========================================================
    # Top-P
    # ========================================================

    if top_p < 1.0:


        sorted_logits, sorted_indices = torch.sort(

            logits,

            descending=True,

        )


        probabilities = torch.softmax(

            sorted_logits,

            dim=-1,

        )


        cumulative = torch.cumsum(

            probabilities,

            dim=-1,

        )


        remove = cumulative > top_p


        remove[1:] = remove[:-1].clone()

        remove[0] = False


        indices_to_remove = (
            sorted_indices[remove]
        )


        logits[
            indices_to_remove
        ] = float("-inf")


    probabilities = torch.softmax(

        logits,

        dim=-1,

    )


    token = torch.multinomial(

        probabilities,

        num_samples=1,

    )


    return token



# ============================================================
# Main
# ============================================================

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


    # ========================================================
    # Tokenizer V2
    # ========================================================

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

        str(
            tokenizer_path
        )

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


    print()

    print(
        "Checkpoint:",
        checkpoint_path,
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


    # ========================================================
    # Model
    # ========================================================

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



    # ========================================================
    # Encode prompt
    # ========================================================

    encoded = tokenizer.encode(

        args.prompt

    )


    input_ids = torch.tensor(

        [
            encoded.ids
        ],

        dtype=torch.long,

        device=device,

    )


    print()

    print(
        "Prompt:",
        args.prompt,
    )


    print(
        "Tokens:",
        encoded.tokens,
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



    eos_id = tokenizer.token_to_id(
        "<eos>"
    )


    generated_tokens = (
        encoded.ids.copy()
    )


    with torch.no_grad():


        for _ in range(
            args.max_new_tokens
        ):


            # 限制上下文

            context = (

                input_ids[
                    :,
                    -config.max_seq_len:
                ]

            )


            with torch.autocast(

                "cuda",

                dtype=torch.bfloat16,

            ):


                logits = model(
                    context
                )


            next_logits = logits[

                0,

                -1,

                :

            ]


            next_token = sample_token(

                next_logits,

                args.temperature,

                args.top-k if False else args.top_k,

                args.top_p,

                args.repetition_penalty,

                generated_tokens,

            )


            token_id = (
                next_token.item()
            )


            generated_tokens.append(
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


            if (
                eos_id is not None
                and
                token_id == eos_id
            ):

                break



    text = tokenizer.decode(

        generated_tokens,

        skip_special_tokens=True,

    )


    print(
        text
    )



if __name__ == "__main__":

    main()
