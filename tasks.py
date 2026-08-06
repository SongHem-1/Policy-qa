"""RQ 后台任务：知识库重建与用户文档处理

任务在独立 worker 进程中执行，只写磁盘/SQLite 产物，不触碰 API 进程内存状态。
API 进程通过 /api/v1/tasks/{task_id} 轮询状态，任务完成后惰性重载检索链。
"""
import logging
from pathlib import Path

from redis import Redis
from rq import Queue

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHUNK_BY_SECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
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


def run_rebuild(force: bool = False) -> dict:
    """重建向量库（双缓冲区）并保存缓存产物，返回构建摘要"""
    from cache_utils import save_cache_with_meta
    from document_processor import load_and_split_pdfs
    from vectorstore import build_or_load_vectorstore, read_manifest

    logger.info("开始重建知识库（force=%s）...", force)
    documents = load_and_split_pdfs(
        DATA_DIR,
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

    vectorstore = build_or_load_vectorstore(documents, force_rebuild=force)
    save_cache_with_meta(Path(DATA_DIR) / "_documents_cache.pkl", documents)
    manifest = read_manifest(PERSIST_DIRECTORY) or {}

    result = {
        "active_collection": manifest.get("active_collection"),
        "document_count": manifest.get("document_count", 0),
        "files": sorted({d.metadata.get("source", "未知来源") for d in documents}),
    }
    logger.info("重建完成: %s", result)
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
