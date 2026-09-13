import sys
import math
import argparse
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from tokenizers import Tokenizer


# ============================================================
# Project Root
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT),
)


# ============================================================
# Import
# ============================================================

from src.config import ModelConfig
from src.dataset import LanguageModelDataset
from src.model import OpsLMToy


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    return parser.parse_args()


# ============================================================
# Learning Rate Scheduler
# ============================================================

def build_lr_scheduler(
    optimizer,
    warmup_steps,
    max_steps,
):

    def lr_lambda(step):

        # Warmup
        if step < warmup_steps:

            return max(
                step,
                1,
            ) / max(
                warmup_steps,
                1,
            )

        # Cosine Decay
        progress = (
            step - warmup_steps
        ) / max(
            max_steps - warmup_steps,
            1,
        )

        progress = min(
            max(
                progress,
                0.0,
            ),
            1.0,
        )

        return (
            0.1
            +
            0.9
            *
            0.5
            *
            (
                1.0
                +
                math.cos(
                    math.pi * progress
                )
            )
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
    loss,
):

    checkpoint = {

        "step": step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "model_config":
            asdict(config),

        "train_args":
            vars(args),

        "loss":
            loss,

    }

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"[checkpoint] saved: {path}",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    torch.manual_seed(42)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(42)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available"
        )

    device = torch.device(
        "cuda"
    )

    print(
        "=" * 80
    )

    print(
        "OpsLM-Toy Training"
    )

    print(
        "=" * 80
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA Runtime:",
        torch.version.cuda
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer_path = (
        ROOT
        / "tokenizer"
        / "artifacts"
        / "tokenizer.json"
    )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    print(
        "Vocab size:",
        vocab_size
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = LanguageModelDataset(

        text_path=(
            ROOT
            / "data"
            / "cleaned.txt"
        ),

        tokenizer_path=tokenizer_path,

        seq_len=args.seq_len,

        add_bos=True,

        add_eos=True,

    )

    print(
        "Training tokens:",
        len(dataset.tokens)
    )

    print(
        "Dataset samples:",
        len(dataset)
    )

    print(
        "Sequence length:",
        args.seq_len
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=0,

        pin_memory=True,

        drop_last=False,

    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Batches / epoch:",
        len(dataloader)
    )

    # --------------------------------------------------------
    # Model Config
    # --------------------------------------------------------

    config = ModelConfig(

        vocab_size=vocab_size,

        max_seq_len=args.seq_len,

        d_model=512,

        n_heads=8,

        n_layers=8,

        d_ff=2048,

        dropout=0.1,

    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = OpsLMToy(
        config
    ).to(
        device
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Parameters(M):",
        f"{total_parameters / 1_000_000:.3f}M"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(

        model.parameters(),

        lr=args.learning_rate,

        betas=(
            0.9,
            0.95,
        ),

        eps=1e-8,

        weight_decay=0.1,

    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = build_lr_scheduler(

        optimizer=optimizer,

        warmup_steps=args.warmup_steps,

        max_steps=args.max_steps,

    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    global_step = 0

    if args.resume is not None:

        checkpoint_path = Path(
            args.resume
        )

        print(
            f"Loading checkpoint: "
            f"{checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
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

        global_step = checkpoint[
            "step"
        ]

        print(
            "Resume step:",
            global_step
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    epoch = 0

    print()

    print(
        "=" * 80
    )

    print(
        "Training Started"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # 训练直到 max_steps
    # --------------------------------------------------------

    while global_step < args.max_steps:

        epoch += 1

        for batch in dataloader:

            if global_step >= args.max_steps:

                break

            # ------------------------------------------------
            # Data -> GPU
            # ------------------------------------------------

            input_ids = (
                batch[
                    "input_ids"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target_ids = (
                batch[
                    "target_ids"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            # ------------------------------------------------
            # 清空上一轮梯度
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward
            #
            # RTX 5090 使用 BF16
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                logits = model(
                    input_ids
                )

                # --------------------------------------------
                # logits:
                #
                # [B,T,V]
                #
                # CrossEntropy 要：
                #
                # [N,V]
                #
                # Target:
                #
                # [N]
                # --------------------------------------------

                loss = F.cross_entropy(

                    logits.reshape(
                        -1,
                        vocab_size,
                    ),

                    target_ids.reshape(
                        -1
                    ),

                )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient Clipping
            # ------------------------------------------------

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
            )

            # ------------------------------------------------
            # 更新参数
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # LR Scheduler
            # ------------------------------------------------

            scheduler.step()

            global_step += 1

            current_loss = (
                loss.item()
            )

            running_loss += (
                current_loss
            )

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                global_step
                %
                args.log_every
                ==
                0
            ):

                average_loss = (
                    running_loss
                    /
                    args.log_every
                )

                current_lr = (
                    optimizer
                    .param_groups[0]
                    ["lr"]
                )

                print(

                    f"step={global_step:6d} "
                    f"epoch={epoch:4d} "
                    f"loss={average_loss:.4f} "
                    f"lr={current_lr:.8f} "
                    f"grad_norm={float(grad_norm):.4f}",

                    flush=True,

                )

                running_loss = 0.0

            # ------------------------------------------------
            # Checkpoint
            # ------------------------------------------------

            if (
                global_step
                %
                args.save_every
                ==
                0
            ):

                checkpoint_path = (

                    ROOT
                    /
                    "checkpoints"
                    /
                    f"step_{global_step:06d}.pt"

                )

                save_checkpoint(

                    path=checkpoint_path,

                    model=model,

                    optimizer=optimizer,

                    scheduler=scheduler,

                    step=global_step,

                    config=config,

                    args=args,

                    loss=current_loss,

                )

    # --------------------------------------------------------
    # 最终模型
    # --------------------------------------------------------

    final_path = (

        ROOT
        /
        "checkpoints"
        /
        "final.pt"

    )

    save_checkpoint(

        path=final_path,

        model=model,

        optimizer=optimizer,

        scheduler=scheduler,

        step=global_step,

        config=config,

        args=args,

        loss=current_loss,

    )

    print()

    print(
        "=" * 80
    )

    print(
        "Training Finished"
    )

    print(
        "=" * 80
    )

    print(
        "Final checkpoint:",
        final_path
    )


if __name__ == "__main__":

    main()
