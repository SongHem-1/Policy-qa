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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "glm-4-flash").strip()
DATA_DIR = os.getenv("DATA_DIR", "data").strip()
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "chroma_db").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipu").strip().lower()
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
SESSION_TTL = int(os.getenv("SESSION_TTL", "3600"))

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

USE_PARENT_CHILD = os.getenv("USE_PARENT_CHILD", "True").strip().lower() == "true"
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "500"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
USE_METADATA_AUGMENT = os.getenv("USE_METADATA_AUGMENT", "True").strip().lower() == "true"
USE_KEYWORD_EXTRACT = os.getenv("USE_KEYWORD_EXTRACT", "True").strip().lower() == "true"

USE_ADAPTIVE_RETRIEVAL = os.getenv("USE_ADAPTIVE_RETRIEVAL", "True").strip().lower() == "true"
USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "True").strip().lower() == "true"
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))


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
    if USE_PARENT_CHILD:
        print(f"✅ 分块策略: 父子块检索 (父块={PARENT_CHUNK_SIZE}, 子块={CHILD_CHUNK_SIZE}, overlap={CHILD_CHUNK_OVERLAP})")
        if USE_METADATA_AUGMENT:
            print(f"   ✅ 元数据增强已启用")
            if USE_KEYWORD_EXTRACT:
                print(f"   ✅ 关键词提取已启用")
        else:
            print(f"   ⚠️ 元数据增强已禁用")
    elif CHUNK_BY_SECTION:
        print("✅ 分块策略: 按章节/条款分块")
    else:
        print(f"✅ 分块策略: 固定长度分块 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    
    # 验证自适应检索
    if USE_ADAPTIVE_RETRIEVAL:
        print(f"✅ 自适应检索已启用（查询分类 + 策略路由）")
        if USE_QUERY_EXPANSION:
            print("   ✅ 查询扩展已启用")
        else:
            print("   ⚠️ 查询扩展已禁用")
    else:
        print("⚠️ 自适应检索已禁用，使用固定权重混合检索")

import hashlib
import json

# 参与构建指纹的配置键：任一变化都会使指纹变化，从而让旧缓存自动失效
BUILD_FINGERPRINT_KEYS = [
    "EMBEDDING_MODEL",
    "USE_MINERU",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_BY_SECTION",
    "USE_PARENT_CHILD",
    "PARENT_CHUNK_SIZE",
    "CHILD_CHUNK_SIZE",
    "CHILD_CHUNK_OVERLAP",
    "USE_METADATA_AUGMENT",
    "USE_KEYWORD_EXTRACT",
]


def compute_build_fingerprint() -> str:
    """计算构建指纹（嵌入模型 + 分块参数）。

    用于校验向量库 manifest、_documents_cache.pkl、父块缓存是否来自同一次构建。
    """
    payload = {key: globals().get(key) for key in BUILD_FINGERPRINT_KEYS}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
