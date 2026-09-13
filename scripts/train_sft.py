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
from src.model import OpsLMToy
from src.sft_dataset import SFTDataset


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=(
            "checkpoints_v2/archive/"
            "opslm_v2_base_33m_step10000.pt"
        ),
    )

    parser.add_argument(
        "--train-data",
        type=str,
        default=(
            "data/phase3/sft/train.jsonl"
        ),
    )

    parser.add_argument(
        "--val-data",
        type=str,
        default=(
            "data/phase3/sft/val.jsonl"
        ),
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
                float(
                    max(
                        warmup_steps,
                        1,
                    )
                )
            )

        progress = (
            step - warmup_steps
        ) / max(
            total_steps - warmup_steps,
            1,
        )

        progress = min(
            max(
                progress,
                0.0,
            ),
            1.0,
        )

        cosine = (
            0.5
            *
            (
                1.0
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
# Loss
# ============================================================

def calculate_sft_loss(
    logits,
    labels,
):

    # ========================================================
    # Causal language modeling shift
    #
    # logits[t] 预测 token[t+1]
    # ========================================================

    shift_logits = (
        logits[
            :,
            :-1,
            :
        ]
        .contiguous()
    )

    shift_labels = (
        labels[
            :,
            1:
        ]
        .contiguous()
    )

    vocab_size = (
        shift_logits
        .size(-1)
    )

    loss = F.cross_entropy(
        shift_logits.view(
            -1,
            vocab_size,
        ),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    return loss


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0
    batches = 0

    for batch in loader:

        input_ids = (
            batch["input_ids"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            logits = model(
                input_ids
            )

            loss = calculate_sft_loss(
                logits,
                labels,
            )

        total_loss += (
            loss.item()
        )

        batches += 1

    model.train()

    if batches == 0:
        return float("inf")

    return (
        total_loss
        /
        batches
    )


# ============================================================
# Save checkpoint
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
    best_val_loss,
):

    checkpoint = {

        "stage":
            "OpsLM-V3-Instruct",

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

        "best_val_loss":
            best_val_loss,

        "base_checkpoint":
            args.base_checkpoint,

    }

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"[checkpoint saved] {path}",
        flush=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        "cuda"
    )

    torch.set_float32_matmul_precision(
        "high"
    )

    print(
        "=" * 80
    )

    print(
        "OpsLM-V3-Instruct SFT"
    )

    print(
        "=" * 80
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
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

    train_dataset = SFTDataset(
        jsonl_path=(
            ROOT
            /
            args.train_data
        ),
        tokenizer_path=tokenizer_path,
        seq_len=args.seq_len,
    )

    val_dataset = SFTDataset(
        jsonl_path=(
            ROOT
            /
            args.val_data
        ),
        tokenizer_path=tokenizer_path,
        seq_len=args.seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    # ========================================================
    # Load V2 Base Model
    # ========================================================

    base_checkpoint_path = (
        ROOT
        /
        args.base_checkpoint
    )

    print()

    print(
        "Loading base checkpoint:"
    )

    print(
        base_checkpoint_path
    )

    checkpoint = torch.load(
        base_checkpoint_path,
        map_location="cpu",
    )

    print(
        "Base step:",
        checkpoint.get(
            "step",
            "unknown",
        ),
    )

    print(
        "Base train loss:",
        checkpoint.get(
            "train_loss",
            "unknown",
        ),
    )

    print(
        "Base val loss:",
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

    # 强制和训练数据保持相同上下文长度
    config.max_seq_len = (
        args.seq_len
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

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()

    print(
        f"Parameters: {params / 1e6:.2f}M"
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    # ========================================================
    # Fresh optimizer
    #
    # 注意：
    # SFT 不继承 Phase2 optimizer/scheduler
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            0.95,
        ),
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        args.warmup_steps,
        args.max_steps,
    )

    output_dir = (
        ROOT
        /
        "checkpoints_v3"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir
        /
        "best.pt"
    )

    latest_path = (
        output_dir
        /
        "latest.pt"
    )

    final_path = (
        output_dir
        /
        "final.pt"
    )

    # ========================================================
    # Initial validation
    # ========================================================

    initial_val_loss = evaluate(
        model,
        val_loader,
        device,
    )

    print()

    print(
        f"[initial eval] "
        f"val_loss={initial_val_loss:.4f}",
        flush=True,
    )

    # ========================================================
    # Training
    # ========================================================

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    optimizer_step = 0
    micro_step = 0

    best_val_loss = float(
        "inf"
    )

    last_val_loss = (
        initial_val_loss
    )

    window_loss_sum = 0.0
    window_loss_count = 0

    last_log_time = time.time()
    last_log_step = 0

    data_iterator = iter(
        train_loader
    )

    while (
        optimizer_step
        <
        args.max_steps
    ):

        try:

            batch = next(
                data_iterator
            )

        except StopIteration:

            data_iterator = iter(
                train_loader
            )

            batch = next(
                data_iterator
            )

        input_ids = (
            batch["input_ids"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):

            logits = model(
                input_ids
            )

            raw_loss = calculate_sft_loss(
                logits,
                labels,
            )

            scaled_loss = (
                raw_loss
                /
                args.gradient_accumulation
            )

        scaled_loss.backward()

        window_loss_sum += (
            raw_loss.item()
        )

        window_loss_count += 1

        micro_step += 1

        if (
            micro_step
            %
            args.gradient_accumulation
            !=
            0
        ):
            continue

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

        # ====================================================
        # Log
        # ====================================================

        if (
            optimizer_step
            %
            args.log_every
            ==
            0
        ):

            now = time.time()

            elapsed = (
                now
                -
                last_log_time
            )

            step_delta = (
                optimizer_step
                -
                last_log_step
            )

            avg_loss = (
                window_loss_sum
                /
                max(
                    window_loss_count,
                    1,
                )
            )

            samples_seen = (
                step_delta
                *
                args.batch_size
                *
                args.gradient_accumulation
            )

            samples_per_second = (
                samples_seen
                /
                max(
                    elapsed,
                    1e-6,
                )
            )

            lr = (
                optimizer
                .param_groups[0]
                ["lr"]
            )

            print(
                f"step={optimizer_step} "
                f"loss={avg_loss:.4f} "
                f"lr={lr:.8f} "
                f"samples/s={samples_per_second:.2f} "
                f"grad={float(grad_norm):.4f}",
                flush=True,
            )

            window_loss_sum = 0.0
            window_loss_count = 0

            last_log_time = now
            last_log_step = (
                optimizer_step
            )

        # ====================================================
        # Eval
        # ====================================================

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
            )

            last_val_loss = (
                val_loss
            )

            print(
                f"[eval] "
                f"step={optimizer_step} "
                f"val_loss={val_loss:.4f}",
                flush=True,
            )

            if (
                val_loss
                <
                best_val_loss
            ):

                best_val_loss = (
                    val_loss
                )

                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    optimizer_step,
                    config,
                    args,
                    raw_loss.item(),
                    val_loss,
                    best_val_loss,
                )

        # ====================================================
        # Latest
        # ====================================================

        if (
            optimizer_step
            %
            args.save_every
            ==
            0
        ):

            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                optimizer_step,
                config,
                args,
                raw_loss.item(),
                last_val_loss,
                best_val_loss,
            )

    # ========================================================
    # Final validation
    # ========================================================

    final_val_loss = evaluate(
        model,
        val_loader,
        device,
    )

    if (
        final_val_loss
        <
        best_val_loss
    ):

        best_val_loss = (
            final_val_loss
        )

        save_checkpoint(
            best_path,
            model,
            optimizer,
            scheduler,
            optimizer_step,
            config,
            args,
            raw_loss.item(),
            final_val_loss,
            best_val_loss,
        )

    save_checkpoint(
        final_path,
        model,
        optimizer,
        scheduler,
        optimizer_step,
        config,
        args,
        raw_loss.item(),
        final_val_loss,
        best_val_loss,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "SFT Finished"
    )

    print(
        "Final step:",
        optimizer_step,
    )

    print(
        "Final val loss:",
        final_val_loss,
    )

    print(
        "Best val loss:",
        best_val_loss,
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
