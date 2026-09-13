import sys
import math
import time
import argparse
from pathlib import Path
from dataclasses import asdict


import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR


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
from src.dataset import PackedLanguageModelDataset
from src.model import OpsLMToy



# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )


    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=2,
    )


    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )


    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
    )


    parser.add_argument(
        "--eval-every",
        type=int,
        default=200,
    )


    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
    )


    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )


    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )


    return parser.parse_args()



# ============================================================
# Scheduler
# ============================================================

def build_scheduler(
    optimizer,
    warmup_steps,
    total_steps,
):


    def lr_lambda(step):

        if step < warmup_steps:

            return (
                float(step + 1)
                /
                float(warmup_steps)
            )


        progress = (

            step - warmup_steps

        ) / max(

            total_steps - warmup_steps,
            1,

        )


        cosine = (
            0.5
            *
            (
                1
                +
                math.cos(
                    math.pi
                    *
                    progress
                )
            )
        )


        return max(
            cosine,
            0.1,
        )


    return LambdaLR(
        optimizer,
        lr_lambda,
    )



# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    config,
    args,
    train_loss,
    val_loss,
):


    checkpoint = {

        "step":
            step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "model_config":
            asdict(config),

        "args":
            vars(args),

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

    }


    torch.save(
        checkpoint,
        path,
    )


    print(
        "[checkpoint saved]",
        path,
        flush=True,
    )



# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    vocab_size,
):


    model.eval()


    total_loss = 0.0

    total_tokens = 0


    for batch in loader:


        input_ids = (
            batch["input_ids"]
            .to(
                device,
                non_blocking=True,
            )
        )


        target_ids = (
            batch["target_ids"]
            .to(
                device,
                non_blocking=True,
            )
        )


        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):


            logits = model(
                input_ids
            )


            loss = F.cross_entropy(

                logits.reshape(
                    -1,
                    vocab_size,
                ),

                target_ids.reshape(
                    -1
                ),

                reduction="sum",

            )


        total_loss += (
            loss.item()
        )


        total_tokens += (
            target_ids.numel()
        )


    model.train()


    return (
        total_loss
        /
        total_tokens
    )



# ============================================================
# Main
# ============================================================

def main():


    args = parse_args()


    torch.manual_seed(
        42
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable"
        )


    device = torch.device(
        "cuda"
    )


    print("=" * 80)

    print(
        "OpsLM-V2 Training"
    )

    print("=" * 80)


    print(
        "GPU:",
        torch.cuda.get_device_name(0)
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


    # ========================================================
    # Dataset
    # ========================================================

    train_dataset = PackedLanguageModelDataset(

        jsonl_path=(

            ROOT
            /
            "data"
            /
            "phase2"
            /
            "processed"
            /
            "train.jsonl"

        ),

        tokenizer_path=tokenizer_path,

        seq_len=args.seq_len,

    )



    val_dataset = PackedLanguageModelDataset(

        jsonl_path=(

            ROOT
            /
            "data"
            /
            "phase2"
            /
            "processed"
            /
            "val.jsonl"

        ),

        tokenizer_path=tokenizer_path,

        seq_len=args.seq_len,

    )



    loader_kwargs = {

        "num_workers":
            args.num_workers,

        "pin_memory":
            True,

        "persistent_workers":
            args.num_workers > 0,

    }


    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        drop_last=True,

        **loader_kwargs,

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        drop_last=False,

        **loader_kwargs,

    )


    vocab_size = (
        train_dataset
        .tokenizer
        .get_vocab_size()
    )


    # ========================================================
    # Model
    # ========================================================

    config = ModelConfig(

        vocab_size=vocab_size,

        max_seq_len=args.seq_len,

        d_model=512,

        n_heads=8,

        n_layers=8,

        d_ff=2048,

        dropout=0.1,

    )


    model = OpsLMToy(
        config
    ).to(
        device
    )


    params = sum(
        p.numel()
        for p in model.parameters()
    )


    print(
        f"Parameters: {params/1e6:.2f}M"
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = AdamW(

        model.parameters(),

        lr=args.lr,

        betas=(
            0.9,
            0.95,
        ),

        weight_decay=0.1,

    )


    scheduler = build_scheduler(

        optimizer,

        args.warmup_steps,

        args.max_steps,

    )



    # ========================================================
    # Resume
    # ========================================================

    optimizer_step = 0


    if args.resume:


        checkpoint = torch.load(

            args.resume,

            map_location=device,

        )


        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]

        )


        optimizer.load_state_dict(

            checkpoint[
                "optimizer_state_dict"
            ]

        )


        scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]

        )


        optimizer_step = (
            checkpoint["step"]
        )


        print(
            "Resume step:",
            optimizer_step
        )



    # ========================================================
    # Train
    # ========================================================

    model.train()


    best_val_loss = float(
        "inf"
    )


    micro_step = 0


    optimizer.zero_grad(
        set_to_none=True
    )


    start_time = time.time()


    while optimizer_step < args.max_steps:


        for batch in train_loader:


            input_ids = (
                batch["input_ids"]
                .to(
                    device,
                    non_blocking=True,
                )
            )


            target_ids = (
                batch["target_ids"]
                .to(
                    device,
                    non_blocking=True,
                )
            )


            with torch.autocast(

                "cuda",

                dtype=torch.bfloat16,

            ):


                logits = model(
                    input_ids
                )


                loss = F.cross_entropy(

                    logits.reshape(
                        -1,
                        vocab_size,
                    ),

                    target_ids.reshape(
                        -1
                    ),

                )


                loss = (
                    loss
                    /
                    args.gradient_accumulation
                )


            loss.backward()


            micro_step += 1



            if (
                micro_step
                %
                args.gradient_accumulation
                ==
                0
            ):


                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )
                )


                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )


                optimizer_step += 1



                if (
                    optimizer_step
                    %
                    20
                    ==
                    0
                ):


                    elapsed = (
                        time.time()
                        -
                        start_time
                    )


                    tokens = (

                        optimizer_step
                        *
                        args.batch_size
                        *
                        args.seq_len
                        *
                        args.gradient_accumulation

                    )


                    lr = (
                        optimizer
                        .param_groups[0]
                        ["lr"]
                    )


                    print(

                        f"step={optimizer_step} "
                        f"loss={loss.item():.4f} "
                        f"lr={lr:.8f} "
                        f"tokens/s={tokens/elapsed:.1f} "
                        f"grad={float(grad_norm):.4f}",

                        flush=True,

                    )



                # =============================================
                # Eval
                # =============================================

                if (
                    optimizer_step
                    %
                    args.eval_every
                    ==
                    0
                ):


                    val_loss = evaluate(

                        model,

                        val_loader,

                        device,

                        vocab_size,

                    )


                    ppl = math.exp(
                        val_loss
                    )


                    print(

                        f"[eval] "
                        f"step={optimizer_step} "
                        f"val_loss={val_loss:.4f} "
                        f"ppl={ppl:.2f}",

                        flush=True,

                    )


                    if val_loss < best_val_loss:


                        best_val_loss = (
                            val_loss
                        )


                        save_checkpoint(

                            ROOT
                            /
                            "checkpoints_v2_best.pt",

                            model,

                            optimizer,

                            scheduler,

                            optimizer_step,

                            config,

                            args,

                            loss.item(),

                            val_loss,

                        )



                if (
                    optimizer_step
                    %
                    args.save_every
                    ==
                    0
                ):


                    save_checkpoint(

                        ROOT
                        /
                        "checkpoints_v2_latest.pt",

                        model,

                        optimizer,

                        scheduler,

                        optimizer_step,

                        config,

                        args,

                        loss.item(),

                        0,

                    )



                if (
                    optimizer_step
                    >=
                    args.max_steps
                ):

                    break



    save_checkpoint(

        ROOT
        /
        "checkpoints_v2_final.pt",

        model,

        optimizer,

        scheduler,

        optimizer_step,

        config,

        args,

        loss.item(),

        0,

    )


    print(
        "Training Finished"
    )



if __name__ == "__main__":

    main()
