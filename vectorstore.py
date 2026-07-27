from pathlib import Path
from typing import List, Optional, Any
import sys

# 设置默认编码为UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import EMBEDDING_MODEL, PERSIST_DIRECTORY

import os
import ssl


class ManualRerankerRetriever(BaseRetriever):
    """手动实现的重排序检索器（LangChain兼容）"""
    
    base_retriever: Any = Field(default=None, description="基础检索器")
    reranker_model: Any = Field(default=None, description="重排序模型")
    top_k: int = Field(default=5, description="返回的文档数量")
    threshold: float = Field(default=0.0, description="相关性阈值")
    
    def __init__(self, base_retriever, reranker_model, top_k=5, threshold=0.0, **kwargs):
        super().__init__(
            base_retriever=base_retriever,
            reranker_model=reranker_model,
            top_k=top_k,
            threshold=threshold,
            **kwargs
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """检索并重排序文档（LangChain接口）"""
        # 1. 从基础检索器获取候选文档
        docs = self.base_retriever.invoke(query)
        
        if not docs:
            return []
        
        # 2. 使用重排序模型评分
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker_model.predict(pairs)
        
        # 3. 根据分数排序
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 过滤并返回Top K
        result = []
        for doc, score in scored_docs[:self.top_k]:
            if score >= self.threshold:
                # 添加相关性分数到metadata
                doc.metadata["relevance_score"] = float(score)
                result.append(doc)
        
        return result


class CombinedRetriever(BaseRetriever):
    """联合检索器 - 同时检索公用数据库和用户个人数据库"""
    
    public_retriever: BaseRetriever = Field(..., description="公用数据库检索器")
    user_retriever: Optional[BaseRetriever] = Field(None, description="用户个人数据库检索器")
    public_weight: float = Field(default=0.7, description="公用数据库权重")
    user_weight: float = Field(default=0.3, description="用户数据库权重")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """检索相关文档"""
        all_docs = []
        
        print(f"\n🔍 CombinedRetriever 开始检索...")
        print(f"   查询: {query[:50]}...")
        
        # 检索公用数据库
        try:
            print(f"   正在检索公用数据库...")
            public_docs = self.public_retriever.invoke(query)
            print(f"   公用数据库返回 {len(public_docs)} 个文档")
            for doc in public_docs:
                doc.metadata["source_type"] = "public"
            all_docs.extend(public_docs)
        except Exception as e:
            print(f"   ❌ 公用数据库检索失败: {e}")
        
        # 检索用户个人数据库
        if self.user_retriever:
            try:
                print(f"   正在检索用户个人数据库...")
                user_docs = self.user_retriever.invoke(query)
                print(f"   用户数据库返回 {len(user_docs)} 个文档")
                for doc in user_docs:
                    doc.metadata["source_type"] = "user"
                all_docs.extend(user_docs)
            except Exception as e:
                print(f"   ❌ 用户数据库检索失败: {e}")
        
        # 按来源类型排序：用户文档优先
        seen = set()
        unique_docs = []
        user_docs = []
        public_docs = []
        
        for doc in all_docs:
            doc_id = f"{doc.metadata.get('source', '')}_{doc.page_content[:50]}"
            if doc_id not in seen:
                seen.add(doc_id)
                if doc.metadata.get("source_type") == "user":
                    user_docs.append(doc)
                else:
                    public_docs.append(doc)
        
        # 用户文档排在前面
        unique_docs = user_docs + public_docs
        
        print(f"   ✅ 合并后共 {len(unique_docs)} 个唯一文档（用户: {len(user_docs)}, 公用: {len(public_docs)}）")
        print(f"🔍 CombinedRetriever 检索完成\n")
        
        return unique_docs
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索"""
        return self._get_relevant_documents(query)

def _create_ssl_context():
    """创建SSL上下文以解决证书验证问题"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def build_or_load_vectorstore(documents: List[Document], persist_directory: str = PERSIST_DIRECTORY, force_rebuild: bool = False) -> Chroma:
    """构建或加载持久化的 Chroma 向量数据库。
    
    Args:
        documents: 文档列表（为空时尝试加载已有向量库）
        persist_directory: 持久化目录路径
        force_rebuild: 是否强制重建（忽略已有数据）
    """
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    print("初始化嵌入模型...")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("使用 langchain_huggingface.HuggingFaceEmbeddings")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        print("回退到 langchain_community.HuggingFaceEmbeddings")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"嵌入模型初始化失败: {e}")
        raise

    print(f"嵌入模型: {EMBEDDING_MODEL}")

    has_existing_db = any(persist_path.iterdir())
    
    if has_existing_db and not force_rebuild and not documents:
        try:
            print("加载已存在的向量数据库...")
            db = Chroma(persist_directory=str(persist_path), embedding_function=embeddings)
            collection_count = db._collection.count()
            print(f"已加载 {collection_count} 个向量")
            return db
        except Exception as e:
            print(f"加载现有向量数据库失败，将重新构建: {e}")

    if not documents:
        raise ValueError("向量库不存在且未提供文档，无法构建向量数据库")

    print("构建新的向量数据库...")
    
    if has_existing_db:
        try:
            print(f"清空旧向量数据库...")
            # 先尝试关闭现有连接
            try:
                temp_db = Chroma(persist_directory=str(persist_path), embedding_function=embeddings)
                temp_db._client.close()
                del temp_db
            except:
                pass
            
            # 等待文件释放
            import time
            time.sleep(1)
            
            # 直接删除整个collection
            import shutil
            if persist_path.exists():
                shutil.rmtree(str(persist_path))
                persist_path.mkdir(parents=True, exist_ok=True)
                print("✅ 旧向量数据库已清空")
        except Exception as e:
            print(f"清空旧数据失败（将覆盖）: {e}")
            print("⚠️ 建议：请关闭所有API服务进程后重新启动")

    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=str(persist_path),
    )
    
    try:
        db.persist()
    except:
        pass
    
    collection_count = db._collection.count()
    print(f"向量数据库构建完成，共 {collection_count} 个向量")
    
    return db


def create_hybrid_retriever(
    vectorstore: Chroma, 
    documents: List[Document] = None, 
    k: int = 5,
    use_reranker: bool = False,
    reranker_top_k: int = 5,
    reranker_threshold: float = 0.0,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5
):
    """创建混合检索器（BM25 + 向量检索），可选重排序
    
    Args:
        vectorstore: 向量数据库
        documents: 文档列表（用于BM25）
        k: 每个检索器返回的文档数量
        use_reranker: 是否使用重排序
        reranker_top_k: 重排序后返回的文档数量
        reranker_threshold: 重排序阈值（低于此值的文档将被过滤）
        bm25_weight: BM25检索器权重
        vector_weight: 向量检索器权重
    
    Returns:
        检索器（EnsembleRetriever 或 ContextualCompressionRetriever）
    """
    print(f"创建混合检索器（BM25权重: {bm25_weight}, 向量权重: {vector_weight}）...")
    
    # 向量检索器 - 如果使用重排序，需要检索更多候选文档
    retrieval_k = k * 4 if use_reranker else k
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retrieval_k}
    )
    
    # BM25检索器（需要原始文档）
    if documents:
        try:
            # 尝试使用支持中文的BM25
            from bm25_chinese import ChineseBM25Retriever
            bm25_retriever = ChineseBM25Retriever(documents, k=retrieval_k)
            print("✅ 使用中文分词BM25检索器")
        except ImportError:
            # 回退到默认BM25
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = retrieval_k
            print("⚠️ 使用默认BM25检索器（中文效果可能不佳）")
        
        # 混合检索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_weight, vector_weight]
        )
        
        base_retriever = ensemble_retriever
        print(f"✅ 混合检索器创建成功（BM25权重: {bm25_weight}, 向量权重: {vector_weight}）")
    else:
        print("⚠️ 没有原始文档，仅使用向量检索")
        base_retriever = vector_retriever
    
    # 添加重排序层
    if use_reranker:
        try:
            from reranker import create_reranker
            reranker_model, compressor = create_reranker(
                top_k=reranker_top_k,
                threshold=reranker_threshold
            )
            
            if compressor:
                # 使用ContextualCompressionRetriever包装
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                
                print(f"✅ 重排序层已启用（Top {reranker_top_k}, 阈值: {reranker_threshold}）")
                
                return compression_retriever
            else:
                # 如果compressor创建失败，使用手动重排序
                print("⚠️ 使用手动重排序逻辑")
                manual_reranker = ManualRerankerRetriever(
                    base_retriever=base_retriever,
                    reranker_model=reranker_model,
                    top_k=reranker_top_k,
                    threshold=reranker_threshold
                )
                print(f"✅ 手动重排序层已启用（Top {reranker_top_k}, 阈值: {reranker_threshold}）")
                
                return manual_reranker
            
        except Exception as e:
            print(f"⚠️ 重排序器初始化失败: {e}")
            print("将使用基础检索器")
            import traceback
            traceback.print_exc()
    
    return base_retriever