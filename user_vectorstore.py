"""用户级别向量库管理"""
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

from config import PERSIST_DIRECTORY, EMBEDDING_MODEL, ROOT_DIR
from document_processor import load_and_split_pdfs


USER_VECTOR_DIR = ROOT_DIR / "chroma_db" / "users"
PUBLIC_VECTOR_DIR = Path(PERSIST_DIRECTORY)

MAX_USER_DOCUMENTS = int(os.getenv("MAX_USER_DOCUMENTS", "10"))
MAX_USER_FILE_SIZE = int(os.getenv("MAX_USER_FILE_SIZE", "10485760"))  # 10MB


def get_user_vector_path(user_id: int) -> Path:
    """获取用户向量库路径"""
    return USER_VECTOR_DIR / f"user_{user_id}"


def init_user_vectorstore(user_id: int, embedding_function: HuggingFaceEmbeddings) -> Chroma:
    """初始化用户向量库
    
    Args:
        user_id: 用户ID
        embedding_function: 嵌入函数
    
    Returns:
        Chroma向量库实例
    """
    vector_path = get_user_vector_path(user_id)
    vector_path.mkdir(parents=True, exist_ok=True)
    
    vectorstore = Chroma(
        persist_directory=str(vector_path),
        embedding_function=embedding_function,
        collection_name=f"user_{user_id}_documents"
    )
    
    return vectorstore


def add_documents_to_user_store(
    user_id: int, 
    documents: List[Document],
    embedding_function: HuggingFaceEmbeddings
) -> bool:
    """添加文档到用户向量库（带去重）
    
    Args:
        user_id: 用户ID
        documents: 文档列表
        embedding_function: 嵌入函数
    
    Returns:
        是否成功
    """
    try:
        import hashlib
        
        vector_path = get_user_vector_path(user_id)
        
        if vector_path.exists():
            vectorstore = Chroma(
                persist_directory=str(vector_path),
                embedding_function=embedding_function,
                collection_name=f"user_{user_id}_documents"
            )
        else:
            vectorstore = init_user_vectorstore(user_id, embedding_function)
        
        # 去重：基于内容哈希
        unique_docs = []
        seen_hashes = set()
        
        for doc in documents:
            # 计算内容哈希
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        print(f"去重前: {len(documents)} 个文档块")
        print(f"去重后: {len(unique_docs)} 个唯一文档块")
        
        if unique_docs:
            vectorstore.add_documents(unique_docs)
        
        return True
    except Exception as e:
        print(f"添加文档到用户向量库失败: {e}")
        return False


def get_user_vectorstore(user_id: int, embedding_function: HuggingFaceEmbeddings) -> Optional[Chroma]:
    """获取用户向量库
    
    Args:
        user_id: 用户ID
        embedding_function: 嵌入函数
    
    Returns:
        Chroma向量库实例，如果不存在则返回None
    """
    vector_path = get_user_vector_path(user_id)
    
    if not vector_path.exists():
        return None
    
    try:
        vectorstore = Chroma(
            persist_directory=str(vector_path),
            embedding_function=embedding_function,
            collection_name=f"user_{user_id}_documents"
        )
        
        # 检查向量库是否真的有文档（空目录不算）
        try:
            collection_data = vectorstore._collection.get()
            if len(collection_data.get("ids", [])) == 0:
                print(f"⚠️ 用户 {user_id} 向量库为空，跳过")
                return None
        except Exception:
            pass
        
        return vectorstore
    except Exception as e:
        print(f"获取用户向量库失败: {e}")
        return None


def delete_user_vectorstore(user_id: int) -> bool:
    """删除用户向量库
    
    Args:
        user_id: 用户ID
    
    Returns:
        是否成功
    """
    try:
        vector_path = get_user_vector_path(user_id)
        
        if vector_path.exists():
            shutil.rmtree(vector_path)
        
        return True
    except Exception as e:
        print(f"删除用户向量库失败: {e}")
        return False


def delete_user_document_vectors(user_id: int, source_name: str, embedding_function: HuggingFaceEmbeddings) -> bool:
    """按 source 过滤物理删除用户向量库中的文档向量（级联删除）。

    Args:
        user_id: 用户ID
        source_name: 文档原始文件名（与入库时 metadata["source"] 一致）
        embedding_function: 嵌入模型实例

    Returns:
        是否删除成功（向量库不存在视为成功；删除失败返回 False 由调用方记录）
    """
    vectorstore = get_user_vectorstore(user_id, embedding_function)
    if vectorstore is None:
        return True
    try:
        vectorstore._collection.delete(where={"source": source_name})
        print(f"已删除用户 {user_id} 的文档向量: source={source_name}")
        return True
    except Exception as e:
        print(f"删除用户文档向量失败: {e}")
        return False


def process_user_document(
    user_id: int,
    file_path: str,
    embedding_function: HuggingFaceEmbeddings,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    original_filename: str = None
) -> Dict[str, Any]:
    """处理用户上传的文档
    
    Args:
        user_id: 用户ID
        file_path: 文件路径
        embedding_function: 嵌入函数
        chunk_size: 分块大小
        chunk_overlap: 重叠大小
        original_filename: 原始文件名（用于显示）
    
    Returns:
        处理结果
    """
    try:
        documents = load_and_split_pdfs(
            data_dir=str(Path(file_path).parent),
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        
        user_documents = [
            doc for doc in documents 
            if doc.metadata.get("source") == Path(file_path).name
        ]
        
        if not user_documents:
            return {
                "success": False,
                "error": "No documents extracted from file"
            }
        
        # 更新文档的source元数据，使用原始文件名
        for doc in user_documents:
            doc.metadata["user_id"] = user_id
            doc.metadata["doc_type"] = "user_document"
            if original_filename:
                doc.metadata["source"] = original_filename
                doc.metadata["original_filename"] = original_filename
        
        success = add_documents_to_user_store(user_id, user_documents, embedding_function)
        
        if success:
            return {
                "success": True,
                "num_chunks": len(user_documents)
            }
        else:
            return {
                "success": False,
                "error": "Failed to add documents to vector store"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_user_document_count(user_id: int, embedding_function: HuggingFaceEmbeddings) -> int:
    """获取用户向量库中的文档数量
    
    Args:
        user_id: 用户ID
        embedding_function: 嵌入函数
    
    Returns:
        文档数量
    """
    try:
        vectorstore = get_user_vectorstore(user_id, embedding_function)
        
        if not vectorstore:
            return 0
        
        collection = vectorstore._collection
        count = collection.count()
        
        return count
    except Exception as e:
        print(f"获取用户文档数量失败: {e}")
        return 0


if __name__ == "__main__":
    print("用户向量库管理模块")
    print(f"用户向量库目录: {USER_VECTOR_DIR}")
    print(f"公用向量库目录: {PUBLIC_VECTOR_DIR}")
    print(f"每个用户最多文档数: {MAX_USER_DOCUMENTS}")
    print(f"最大文件大小: {MAX_USER_FILE_SIZE / 1024 / 1024}MB")
