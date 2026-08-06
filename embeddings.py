"""嵌入模型工厂：bge-m3 本地部署

bge-m3 仓库自带自定义 XLMRoberta 实现，必须 trust_remote_code=True；
检索采用余弦相似度，统一 normalize_embeddings=True（1024 维）。
"""
from config import EMBEDDING_MODEL


def create_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
