from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ============================================================
# 1. 项目路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

INPUT_FILE = DATA_DIR / "cleaned.txt"

OUTPUT_DIR = ROOT / "tokenizer" / "artifacts"

OUTPUT_FILE = OUTPUT_DIR / "tokenizer.json"


# ============================================================
# 2. 基本配置
# ============================================================

VOCAB_SIZE = 16000

MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


# ============================================================
# 3. 检查训练数据
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Training corpus not found: {INPUT_FILE}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 60)
print("OpsLM Tokenizer Training")
print("=" * 60)

print(f"Input:       {INPUT_FILE}")
print(f"Output:      {OUTPUT_FILE}")
print(f"Target vocab:{VOCAB_SIZE}")
print(f"Min freq:    {MIN_FREQUENCY}")

print()


# ============================================================
# 4. 创建 BPE 模型
# ============================================================

tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>"
    )
)


# ============================================================
# 5. Unicode 规范化
# ============================================================

tokenizer.normalizer = NFKC()


# ============================================================
# 6. Byte-level PreTokenizer
# ============================================================

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)


# ============================================================
# 7. Byte-level Decoder
# ============================================================

tokenizer.decoder = ByteLevelDecoder()


# ============================================================
# 8. 创建 BPE Trainer
# ============================================================

trainer = BpeTrainer(

    vocab_size=VOCAB_SIZE,

    min_frequency=MIN_FREQUENCY,

    special_tokens=SPECIAL_TOKENS,

    initial_alphabet=ByteLevel.alphabet(),

    show_progress=True,

)


# ============================================================
# 9. 开始训练
# ============================================================

print("Training tokenizer...")
print()

tokenizer.train(
    files=[str(INPUT_FILE)],
    trainer=trainer,
)


# ============================================================
# 10. 保存 Tokenizer
# ============================================================

tokenizer.save(
    str(OUTPUT_FILE)
)


# ============================================================
# 11. 输出基本信息
# ============================================================

actual_vocab_size = (
    tokenizer.get_vocab_size()
)


print()
print("=" * 60)
print("Training finished")
print("=" * 60)

print(
    f"Actual vocab size: {actual_vocab_size}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)


# ============================================================
# 12. 查看特殊 Token ID
# ============================================================

print()
print("Special tokens:")

for token in SPECIAL_TOKENS:

    token_id = tokenizer.token_to_id(
        token
    )

    print(
        f"{token:8s} -> {token_id}"
    )


# ============================================================
# 13. 简单测试
# ============================================================

samples = [

    "Linux 内存使用率很高。",

    "Kubernetes Pod 出现 CrashLoopBackOff。",

    "kubectl get pods -n production",

    "CUDA_VISIBLE_DEVICES=0",

    "systemctl status nginx",

]


print()
print("=" * 60)
print("Tokenizer test")
print("=" * 60)


for text in samples:

    encoding = tokenizer.encode(
        text
    )

    print()
    print("Text:")
    print(text)

    print("Tokens:")
    print(encoding.tokens)

    print("IDs:")
    print(encoding.ids)

    print("Decoded:")
    print(
        tokenizer.decode(
            encoding.ids
        )
    )
