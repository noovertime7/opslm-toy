from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = (
    ROOT
    / "tokenizer"
    / "artifacts"
    / "tokenizer.json"
)


tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)


print(
    "Vocab size:",
    tokenizer.get_vocab_size()
)


while True:

    print()

    text = input(
        "请输入文本（exit退出）: "
    )


    if text.lower() == "exit":
        break


    encoding = tokenizer.encode(
        text
    )


    print()

    print("Tokens:")

    for index, (
        token,
        token_id,
    ) in enumerate(
        zip(
            encoding.tokens,
            encoding.ids,
        )
    ):

        print(
            f"{index:3d} "
            f"{token_id:5d} "
            f"{repr(token)}"
        )


    print()

    print(
        "Token count:",
        len(encoding.ids)
    )


    decoded = tokenizer.decode(
        encoding.ids
    )


    print(
        "Decoded:",
        decoded
    )
