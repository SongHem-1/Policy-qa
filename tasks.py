"""RQ 后台任务：知识库重建与用户文档处理

任务在独立 worker 进程中执行，只写磁盘/SQLite 产物，不触碰 API 进程内存状态。
API 进程通过 /api/v1/tasks/{task_id} 轮询状态，任务完成后惰性重载检索链。
"""
import logging
import shutil
import tempfile
import time
from pathlib import Path

from redis import Redis
from rq import Queue

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHUNK_BY_SECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    compute_build_fingerprint,
    DATA_DIR,
    PARENT_CHUNK_SIZE,
    PERSIST_DIRECTORY,
    REDIS_URL,
    USE_METADATA_AUGMENT,
    USE_PARENT_CHILD,
)

logger = logging.getLogger("policy_qa.tasks")

QUEUE_NAME = "policy-qa"
REBUILD_JOB_TIMEOUT = 3600
DOC_JOB_TIMEOUT = 600


def _redis_conn() -> Redis:
    return Redis.from_url(REDIS_URL)


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=_redis_conn())


def fetch_job(task_id: str):
    from rq.job import Job
    return Job.fetch(task_id, connection=_redis_conn())


def enqueue_rebuild(force: bool = False):
    """入队知识库重建任务"""
    return get_queue().enqueue(
        "tasks.run_rebuild",
        force=force,
        job_timeout=REBUILD_JOB_TIMEOUT,
        result_ttl=REBUILD_JOB_TIMEOUT,
    )


def enqueue_process_user_document(
    user_id: int,
    file_path: str,
    original_filename: str,
    document_id: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """入队用户文档处理任务"""
    return get_queue().enqueue(
        "tasks.run_process_user_document",
        user_id=user_id,
        file_path=file_path,
        original_filename=original_filename,
        document_id=document_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        job_timeout=DOC_JOB_TIMEOUT,
        result_ttl=DOC_JOB_TIMEOUT,
    )


def _dedup_documents(documents):
    """按内容 md5 去重（增量新增文件时使用）"""
    import hashlib

    unique, seen = [], set()
    for doc in documents:
        h = hashlib.md5(doc.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


def run_rebuild(
    force: bool = False,
    data_dir: str = DATA_DIR,
    persist_directory: str = PERSIST_DIRECTORY,
) -> dict:
    """重建知识库（RQ 任务）。

    - force=True 或 manifest 缺失/指纹不一致：双缓冲区全量重建
    - 否则：增量重建——按文件集差异只处理新增/删除文件，
      新增向量写入 active 集合、删除文件按 source 物理删向量，
      同步合并 BM25 语料与父块缓存（修复"删除残留向量"缺陷）
    """
    from cache_utils import load_cache_with_meta, save_cache_with_meta
    from document_processor import load_and_split_pdfs
    from vectorstore import (
        build_or_load_vectorstore,
        delete_documents_by_source,
        read_manifest,
        write_manifest,
    )

    manifest = read_manifest(persist_directory)
    data_path = Path(data_dir)
    current_pdfs = sorted([f.name for f in data_path.glob("*.pdf")]) if data_path.exists() else []

    needs_full = (
        force
        or manifest is None
        or manifest.get("fingerprint") != compute_build_fingerprint()
        or not current_pdfs
    )

    if needs_full:
        logger.info("全量重建（force=%s, 有清单=%s）...", force, manifest is not None)
        documents = load_and_split_pdfs(
            data_dir,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            chunk_by_section=CHUNK_BY_SECTION,
            parent_child=USE_PARENT_CHILD,
            parent_size=PARENT_CHUNK_SIZE,
            child_size=CHILD_CHUNK_SIZE,
            child_overlap=CHILD_CHUNK_OVERLAP,
            augment_meta=USE_METADATA_AUGMENT,
        )
        if not documents:
            raise RuntimeError("没有找到可用文档")

        vectorstore = build_or_load_vectorstore(
            documents, force_rebuild=True, persist_directory=persist_directory
        )
        save_cache_with_meta(Path(data_dir) / "_documents_cache.pkl", documents)
        manifest = read_manifest(persist_directory) or {}
        result = {
            "status": "full",
            "active_collection": manifest.get("active_collection"),
            "document_count": manifest.get("document_count", 0),
            "files": sorted({d.metadata.get("source", "未知来源") for d in documents}),
        }
        logger.info("全量重建完成: %s", result)
        return result

    # ---- 增量重建 ----
    stored_files = set(manifest.get("source_files") or [])
    current_set = set(current_pdfs)
    added = sorted(current_set - stored_files)
    removed = sorted(stored_files - current_set)
    if not added and not removed:
        return {
            "status": "no_change",
            "document_count": manifest.get("document_count"),
            "files": current_pdfs,
        }

    logger.info("增量重建：新增 %s，删除 %s", added, removed)
    vectorstore = build_or_load_vectorstore([], persist_directory=persist_directory)

    docs_cache_path = Path(data_dir) / "_documents_cache.pkl"
    parent_cache_path = Path(data_dir) / "_parent_documents_cache.pkl"
    old_docs = load_cache_with_meta(docs_cache_path, manifest) or []
    old_parents = load_cache_with_meta(parent_cache_path, manifest) or [] if USE_PARENT_CHILD else []

    new_child_docs = []
    new_parent_docs = []

    if added:
        temp_dir = tempfile.mkdtemp(prefix="policy_inc_")
        try:
            for pdf_name in added:
                shutil.copy2(data_path / pdf_name, Path(temp_dir) / pdf_name)
            # load_and_split_pdfs 会把父块缓存重写为仅新文件，随后读回合并
            new_docs = load_and_split_pdfs(
                temp_dir,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
                chunk_by_section=CHUNK_BY_SECTION,
                parent_child=USE_PARENT_CHILD,
                parent_size=PARENT_CHUNK_SIZE,
                child_size=CHILD_CHUNK_SIZE,
                child_overlap=CHILD_CHUNK_OVERLAP,
                augment_meta=USE_METADATA_AUGMENT,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        unique = _dedup_documents(new_docs)
        if unique:
            vectorstore.add_documents(unique)
            new_child_docs = unique
            logger.info("已新增 %d 个文档块", len(unique))
        if USE_PARENT_CHILD:
            new_parent_docs = load_cache_with_meta(parent_cache_path, manifest) or []

    if removed:
        delete_documents_by_source(vectorstore, removed)

    # 合并缓存：剔除已删文件、追加新增文件
    kept_docs = [d for d in old_docs if d.metadata.get("source") not in removed]
    save_cache_with_meta(docs_cache_path, kept_docs + new_child_docs)

    if USE_PARENT_CHILD:
        kept_parents = [p for p in old_parents if p.metadata.get("source") not in removed]
        save_cache_with_meta(parent_cache_path, kept_parents + new_parent_docs)

    count = vectorstore._collection.count()
    write_manifest(persist_directory, {
        **manifest,
        "source_files": current_pdfs,
        "document_count": count,
        "created_at": time.time(),
    })

    result = {
        "status": "incremental",
        "added": added,
        "removed": removed,
        "document_count": count,
        "files": current_pdfs,
    }
    logger.info("增量重建完成: %s", result)
    return result


def run_process_user_document(
    user_id: int,
    file_path: str,
    original_filename: str,
    document_id: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict:
    """处理用户上传文档：写入用户向量库 + 更新 SQLite 状态"""
    import database
    from embeddings import create_embeddings
    from user_vectorstore import process_user_document

    embeddings = create_embeddings()
    result = process_user_document(
        user_id=user_id,
        file_path=file_path,
        embedding_function=embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        original_filename=original_filename,
    )
    if result.get("success"):
        database.update_document_status(document_id, "completed")
    else:
        database.update_document_status(document_id, "failed", result.get("error"))
    return result
