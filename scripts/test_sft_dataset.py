import sys

from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(ROOT),
)


from src.sft_dataset import SFTDataset


dataset = SFTDataset(

    jsonl_path=(
        ROOT
        /
        "data/phase3/sft/train.jsonl"
    ),

    tokenizer_path=(
        ROOT
        /
        "tokenizer/artifacts_v2/tokenizer.json"
    ),

    seq_len=512,

)


print(
    "Dataset length:",
    len(dataset),
)


sample = dataset[0]


print(
    "input_ids shape:",
    sample["input_ids"].shape,
)


print(
    "labels shape:",
    sample["labels"].shape,
)


train_tokens = (
    sample["labels"]
    != -100
).sum().item()


ignored_tokens = (
    sample["labels"]
    == -100
).sum().item()


print(
    "Assistant train tokens:",
    train_tokens,
)


print(
    "Ignored tokens:",
    ignored_tokens,
)


print(
    "Input text:"
)


valid_ids = [

    token_id.item()

    for token_id in sample[
        "input_ids"
    ]

    if token_id.item()
    != dataset.pad_id

]


print(
    dataset.tokenizer.decode(
        valid_ids,
        skip_special_tokens=True,
    )
)
