import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")).strip()
ZHIPU_BASE_URL = os.getenv(
    "ZHIPU_BASE_URL",
    os.getenv("DEEPSEEK_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
).strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "glm-4-flash").strip()
DATA_DIR = os.getenv("DATA_DIR", "data").strip()
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "chroma_db").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipu").strip().lower()

# 如果配置的是相对路径，则默认相对于项目根目录 ROOT_DIR
if DATA_DIR and not os.path.isabs(DATA_DIR):
    DATA_DIR = str((ROOT_DIR / DATA_DIR).resolve())
if PERSIST_DIRECTORY and not os.path.isabs(PERSIST_DIRECTORY):
    PERSIST_DIRECTORY = str((ROOT_DIR / PERSIST_DIRECTORY).resolve())

LOCAL_MODEL_DIR = os.path.join(str(ROOT_DIR), "MML12-v2")
if os.path.exists(LOCAL_MODEL_DIR):
    EMBEDDING_MODEL = LOCAL_MODEL_DIR
    print(f"使用本地嵌入模型: {EMBEDDING_MODEL}")
else:
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

USE_MINERU = os.getenv("USE_MINERU", "False").strip().lower() == "true"

DOC_PROCESSOR = "MinerU" if USE_MINERU else "EasyOCR"
LLM_MODEL = OPENAI_MODEL

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

USE_RERANKER = os.getenv("USE_RERANKER", "False").strip().lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
RERANKER_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", "0.0"))

BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.5"))

CHUNK_BY_SECTION = os.getenv("CHUNK_BY_SECTION", "False").strip().lower() == "true"


def validate_config() -> None:
    """检查运行配置中的关键环境变量是否已正确设置。"""
    if not ZHIPU_API_KEY:
        raise ValueError("缺少 ZHIPU_API_KEY，请在 .env 中补充")
    if not ZHIPU_BASE_URL:
        raise ValueError("缺少 ZHIPU_BASE_URL，请在 .env 中补充")
    if not DATA_DIR:
        raise ValueError("DATA_DIR 未配置，默认应为 data")
    if not PERSIST_DIRECTORY:
        raise ValueError("PERSIST_DIRECTORY 未配置，默认应为 chroma_db")
    
    # 验证权重配置
    if BM25_WEIGHT + VECTOR_WEIGHT != 1.0:
        print(f"⚠️ 警告: BM25_WEIGHT({BM25_WEIGHT}) + VECTOR_WEIGHT({VECTOR_WEIGHT}) != 1.0")
    
    # 验证重排序配置
    if USE_RERANKER:
        print(f"✅ 重排序已启用: {RERANKER_MODEL} (Top {RERANKER_TOP_K}, 阈值: {RERANKER_THRESHOLD})")
    
    # 验证分块策略
    if CHUNK_BY_SECTION:
        print("✅ 分块策略: 按章节/条款分块")
    else:
        print(f"✅ 分块策略: 固定长度分块 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")