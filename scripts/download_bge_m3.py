"""下载 bge-m3 模型到本地（本地部署 / CI 环境共用）

用法：
    python scripts/download_bge_m3.py

只拉取 sentence-transformers 加载所需的文件，跳过仓库中的图片/onnx 等冗余内容。
可通过 HF_ENDPOINT 指定镜像（如 https://hf-mirror.com）。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "bge-m3"
REPO_ID = "BAAI/bge-m3"

ALLOW_PATTERNS = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "sentence_bert_config.json",
    "modules.json",
    "colbert_linear.pt",
    "sparse_linear.pt",
    "1_Pooling/*",
]


def main() -> int:
    from huggingface_hub import snapshot_download

    print(f"下载 {REPO_ID} -> {TARGET_DIR}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co（默认）')}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        local_dir=str(TARGET_DIR),
        allow_patterns=ALLOW_PATTERNS,
    )
    print("✅ bge-m3 下载完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
