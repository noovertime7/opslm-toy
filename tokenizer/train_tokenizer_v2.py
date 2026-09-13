from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteDecoder
from tokenizers.trainers import BpeTrainer


ROOT = Path(__file__).resolve().parents[1]


input_file = (
    ROOT
    /
    "data/phase2/processed/tokenizer_corpus.txt"
)


output = (
    ROOT
    /
    "tokenizer/artifacts_v2/tokenizer.json"
)


tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>"
    )
)


tokenizer.normalizer = NFKC()

tokenizer.pre_tokenizer = ByteLevel()

tokenizer.decoder = ByteDecoder()


trainer = BpeTrainer(

    vocab_size=16000,

    min_frequency=2,

    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ],

)


tokenizer.train(
    [
        str(input_file)
    ],
    trainer,
)


output.parent.mkdir(
    parents=True,
    exist_ok=True,
)


tokenizer.save(
    str(output)
)


print(
    "saved:",
    output
)

print(
    "vocab:",
    tokenizer.get_vocab_size()
)
