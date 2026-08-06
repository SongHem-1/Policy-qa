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

from config import EMBEDDING_MODEL, PERSIST_DIRECTORY, compute_build_fingerprint

import os
import ssl
import json
import time

import chromadb

# 缓存重排序模型，避免重复加载
_reranker_cache: dict = {}


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
    user_weight: float = Field(default=0.15, description="用户数据库权重")
    
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
        
        # 内容去重：用户文档如果与公用文档内容高度相似，保留公用文档（文件名更规范）
        content_seen = set()
        public_content_set = set()
        
        # 先收集公用文档内容指纹
        for doc in all_docs:
            if doc.metadata.get("source_type") == "public":
                fingerprint = doc.page_content[:100].strip()
                public_content_set.add(fingerprint)
        
        user_docs = []
        public_docs = []
        for doc in all_docs:
            fingerprint = doc.page_content[:100].strip()
            if fingerprint in content_seen:
                continue
            content_seen.add(fingerprint)
            
            if doc.metadata.get("source_type") == "user":
                # 用户文档内容与公用文档重复 → 跳过（保留公用版，文件名更规范）
                if fingerprint in public_content_set:
                    continue
                user_docs.append(doc)
            else:
                public_docs.append(doc)
        
        # 公用文档优先，用户独有文档在后
        unique_docs = public_docs + user_docs
        
        print(f"   ✅ 合并后共 {len(unique_docs)} 个唯一文档（用户: {len(user_docs)}, 公用: {len(public_docs)}）")
        print(f"🔍 CombinedRetriever 检索完成\n")
        
        return unique_docs
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索"""
        return self._get_relevant_documents(query)


class ParentChildRetriever(BaseRetriever):
    """父子块检索器：用子块检索，返回父块
    
    核心逻辑：
    1. 子块已向量化入库，用 base_retriever 检索最相关的子块
    2. 通过子块的 metadata["parent_id"] 找到对应的父块
    3. 返回父块（更大上下文），去重合并
    """
    
    base_retriever: Any = Field(..., description="基础检索器（检索子块）")
    parent_documents: List[Document] = Field(default_factory=list, description="父块列表")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """检索：子块检索 → 父块返回"""
        child_docs = self.base_retriever.invoke(query)
        
        if not child_docs:
            return []
        
        if not self.parent_documents:
            print("⚠️ ParentChildRetriever: 无父块缓存，直接返回子块")
            return child_docs
        
        print(f"\n🧩 父子块检索: 检索到 {len(child_docs)} 个子块")
        
        parent_map = {}
        for pdoc in self.parent_documents:
            pid = pdoc.metadata.get("parent_id", "")
            if pid:
                parent_map[pid] = pdoc
        
        seen_parent_ids = set()
        result_docs = []
        
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get("parent_id", "")
            if parent_id and parent_id in parent_map:
                if parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    parent_doc = parent_map[parent_id]
                    new_doc = Document(
                        page_content=parent_doc.page_content,
                        metadata={
                            **parent_doc.metadata,
                            "child_source": child_doc.metadata.get("source", ""),
                            "child_page": child_doc.metadata.get("page", ""),
                            "child_score": child_doc.metadata.get("relevance_score", 0),
                            "source_type": child_doc.metadata.get("source_type", "public"),
                        }
                    )
                    result_docs.append(new_doc)
            else:
                result_docs.append(child_doc)
        
        print(f"   ✅ 映射到 {len(result_docs)} 个父块（去重后）")
        return result_docs
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

def _create_ssl_context():
    """创建SSL上下文以解决证书验证问题"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------- 集合双缓冲区与构建清单 ----------------
# 固定两个命名集合交替写入，manifest 记录谁是 active；切换前旧集合始终可用。
ACTIVE_COLLECTION_LEGACY = "langchain"  # 兼容旧版本默认集合名
COLLECTION_A = "langchain_a"
COLLECTION_B = "langchain_b"
MANIFEST_FILE = "manifest.json"


def _manifest_path(persist_directory: str) -> Path:
    return Path(persist_directory) / MANIFEST_FILE


def read_manifest(persist_directory: str) -> Optional[dict]:
    """读取构建清单；不存在或损坏时返回 None"""
    path = _manifest_path(persist_directory)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 读取清单失败: {e}")
        return None


def write_manifest(persist_directory: str, manifest: dict) -> None:
    """原子写入清单：先写临时文件，再 os.replace 原子替换"""
    path = _manifest_path(persist_directory)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def update_manifest_count(db: Chroma, persist_directory: str) -> None:
    """向量增量变更后刷新清单中的 document_count"""
    manifest = read_manifest(persist_directory) or {}
    try:
        manifest["document_count"] = db._collection.count()
        write_manifest(persist_directory, manifest)
    except Exception as e:
        print(f"⚠️ 更新清单计数失败: {e}")


def delete_documents_by_source(db: Chroma, sources: List[str]) -> None:
    """按 source 元数据物理删除向量（增量重建/级联删除共用）"""
    for source in sources:
        try:
            db._collection.delete(where={"source": source})
            print(f"  已删除 source={source} 的向量")
        except Exception as e:
            print(f"  删除 source={source} 向量失败: {e}")


def _collection_exists(client, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def _safe_delete_collection(client, name: str) -> None:
    try:
        client.delete_collection(name)
        print(f"  已删除集合: {name}")
    except Exception:
        pass


def _load_embeddings():
    """加载嵌入模型"""
    from embeddings import create_embeddings
    print("使用本地嵌入模型（bge-m3）")
    embeddings = create_embeddings()
    print(f"嵌入模型: {EMBEDDING_MODEL}")
    return embeddings


def _rebuild_double_buffer(client, embeddings, documents: List[Document], persist_directory: str) -> Chroma:
    """双缓冲区重建：写入非 active 集合 -> 验证 -> 原子切换 -> 删除旧集合

    任一步失败都不会破坏当前 active 集合，替代原来 shutil.rmtree 删库的重建方式。
    """
    manifest = read_manifest(persist_directory) or {}
    active = manifest.get("active_collection")
    target = COLLECTION_A if active == COLLECTION_B else COLLECTION_B

    print(f"双缓冲区重建：active={active or '（旧库/无清单）'}，写入目标={target}")

    # 1. 清空目标集合，保证从零写入
    _safe_delete_collection(client, target)

    # 2. 写入新集合
    db = Chroma.from_documents(
        documents,
        embeddings,
        client=client,
        collection_name=target,
    )

    # 3. 验证：数量校验 + 冒烟查询
    count = db._collection.count()
    if count != len(documents):
        raise RuntimeError(f"向量入库数量不一致: 期望 {len(documents)}，实际 {count}")
    if count > 0:
        try:
            # 用 langchain 封装执行冒烟查询，确保走我们配置的嵌入模型，
            # 而不是 Chroma collection 自带的默认 ONNX 嵌入函数
            db.similarity_search("验证", k=1)
        except Exception as e:
            raise RuntimeError(f"冒烟查询失败: {e}")
    print(f"✅ 新集合 {target} 验证通过：{count} 个向量")

    # 4. 原子切换：先写清单（指向新集合），再删除旧集合
    source_files = sorted({d.metadata.get("source", "") for d in documents})
    write_manifest(persist_directory, {
        "active_collection": target,
        "fingerprint": compute_build_fingerprint(),
        "document_count": count,
        "source_files": source_files,
        "created_at": time.time(),
    })

    if active and active != target:
        _safe_delete_collection(client, active)
    elif active is None and _collection_exists(client, ACTIVE_COLLECTION_LEGACY) and ACTIVE_COLLECTION_LEGACY != target:
        # 旧库默认集合在切换后删除，释放空间
        _safe_delete_collection(client, ACTIVE_COLLECTION_LEGACY)

    try:
        client.persist()  # 兼容旧版 chromadb；0.4+ 自动持久化
    except Exception:
        pass

    return db


def build_or_load_vectorstore(documents: List[Document], persist_directory: str = PERSIST_DIRECTORY, force_rebuild: bool = False) -> Chroma:
    """构建或加载持久化 Chroma 向量库（双缓冲区原子切换）。

    - documents 为空：加载 active 集合（按 manifest），并校验指纹一致性
    - documents 非空：双缓冲区重建（写入非 active 集合 -> 验证 -> 原子切换）
    - force_rebuild 与普通构建语义一致，均走双缓冲区
    """
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    print("初始化嵌入模型...")
    try:
        embeddings = _load_embeddings()
    except Exception as e:
        print(f"嵌入模型初始化失败: {e}")
        raise

    client = chromadb.PersistentClient(path=str(persist_path))

    # ---- 加载模式 ----
    if not documents:
        manifest = read_manifest(persist_directory)
        active = manifest.get("active_collection") if manifest else None

        # 指纹校验：向量库指纹与当前配置不符时告警（不阻塞，提示重建）
        if manifest and manifest.get("fingerprint") and manifest["fingerprint"] != compute_build_fingerprint():
            print("⚠️ 向量库构建指纹与当前配置不一致（嵌入模型或分块参数已变化），建议重建知识库")

        loaded = None
        candidates = ([active] if active else []) + [ACTIVE_COLLECTION_LEGACY]
        for name in candidates:
            if _collection_exists(client, name):
                try:
                    loaded = Chroma(client=client, collection_name=name, embedding_function=embeddings)
                    break
                except Exception as e:
                    print(f"加载集合 {name} 失败: {e}")

        if loaded is None:
            raise ValueError("向量库不存在且未提供文档，无法构建向量数据库")

        count = loaded._collection.count()
        print(f"已加载 {count} 个向量（集合: {loaded._collection.name}）")

        # 旧库迁移：补齐清单，便于后续双缓冲区重建与缓存指纹校验
        if not manifest:
            try:
                metas = loaded._collection.get()["metadatas"]
                sources = sorted({m.get("source", "") for m in metas if m})
            except Exception:
                sources = []
            write_manifest(persist_directory, {
                "active_collection": loaded._collection.name,
                "fingerprint": compute_build_fingerprint(),
                "document_count": count,
                "source_files": sources,
                "created_at": time.time(),
            })
        return loaded

    # ---- 构建/重建模式 ----
    print("强制重建（双缓冲区）..." if force_rebuild else "构建新的向量数据库（双缓冲区）...")
    return _rebuild_double_buffer(client, embeddings, documents, str(persist_path))


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
    retrieval_k = k * 2 if use_reranker else k
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
            # 使用缓存的模型，避免重复加载
            cache_key = f"{reranker_top_k}_{reranker_threshold}"
            if cache_key not in _reranker_cache:
                reranker_model, compressor = create_reranker(
                    top_k=reranker_top_k,
                    threshold=reranker_threshold
                )
                _reranker_cache[cache_key] = (reranker_model, compressor)
            else:
                reranker_model, compressor = _reranker_cache[cache_key]
                print("✅ 重排序模型从缓存加载")
            
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
