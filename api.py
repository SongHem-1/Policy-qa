import os
import sys
import time
import logging
import structlog
import threading
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException, Query, Request, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uuid
import shutil
import json

from config import (
    DATA_DIR, 
    PERSIST_DIRECTORY, 
    validate_config, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    CHUNK_BY_SECTION,
    USE_PARENT_CHILD,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    USE_METADATA_AUGMENT,
    USE_MINERU,
    DOC_PROCESSOR as CONFIG_DOC_PROCESSOR,
    EMBEDDING_MODEL,
)
import database
from user_vectorstore import (
    process_user_document,
    get_user_vectorstore,
    get_user_document_count,
    MAX_USER_DOCUMENTS,
    MAX_USER_FILE_SIZE
)

if USE_MINERU:
    try:
        from document_processor_mineru import load_and_split_pdfs
        DOC_PROCESSOR = "mineru"
    except ImportError:
        print("⚠️ MinerU未安装，回退到EasyOCR")
        from document_processor import load_and_split_pdfs
        DOC_PROCESSOR = "easyocr"
else:
    try:
        from document_processor import load_and_split_pdfs
        DOC_PROCESSOR = "easyocr"
    except ImportError:
        from document_processor_simple import load_and_split_pdfs
        DOC_PROCESSOR = "simple"

from vectorstore import build_or_load_vectorstore, read_manifest, update_manifest_count
from qa_chain import build_retrieval_qa_chain, extract_sources, extract_cited_sources, invoke_chain_with_memory
from logging_config import setup_logging, new_trace_id, bind_trace_id, clear_trace_id
from cache_utils import load_cache_with_meta, save_cache_with_meta
from session_store import session_store
from tasks import enqueue_rebuild, enqueue_process_user_document, fetch_job

logger = logging.getLogger("policy_qa_api")
_request_logger = structlog.get_logger("policy_qa_api.request")
setup_logging()

try:
    from memory_manager import memory_manager
    MEMORY_ENABLED = True
    logger.info("长期记忆系统已启用")
except Exception as e:
    MEMORY_ENABLED = False
    memory_manager = None
    logger.warning(f"长期记忆系统初始化失败，已禁用: {e}")

_chain = None
_file_list: List[str] = []
_chain_building = False
_chain_error: Optional[str] = None
_build_start_time = 0.0
_original_documents = None  # 保存原始文档用于BM25
_parent_documents = None  # 保存父块文档用于ParentChildRetriever

# 缓存嵌入模型和用户检索器
_embeddings_cache = None
_user_retrievers_cache: dict = {}  # {user_id: chain}

# RQ 任务跟踪：重建任务完成后 API 惰性重载检索链；用户文档任务完成后失效对应检索器缓存
_rebuild_task_id: Optional[str] = None
_pending_user_tasks: dict = {}  # {user_id: task_id}


def get_embeddings():
    """获取嵌入模型实例（缓存）"""
    global _embeddings_cache
    
    if _embeddings_cache is None:
        from embeddings import create_embeddings
        _embeddings_cache = create_embeddings()
        logger.info("✅ 嵌入模型已初始化并缓存")
    
    return _embeddings_cache


def _ensure_directories() -> None:
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)


def _build_chain():
    global _chain, _file_list, _chain_building, _chain_error, _original_documents, _parent_documents

    logger.info("开始构建检索链...")

    persist_path = Path(PERSIST_DIRECTORY)
    if persist_path.exists() and any(persist_path.iterdir()):
        logger.info("发现已存在的向量数据库，尝试直接加载...")
        try:
            vectorstore = build_or_load_vectorstore([])
            collection = vectorstore._collection
            metadatas = collection.get()["metadatas"]
            _file_list = sorted(
                {m.get("source", m.get("file_name", "未知来源")) for m in metadatas}
            )
            logger.info(f"向量数据库加载完成，已加载文件：{_file_list}")
            # 从缓存加载原始文档（用于BM25和自适应检索），指纹校验不通过则降级为纯向量检索
            manifest = read_manifest(PERSIST_DIRECTORY)
            cache_path = Path(DATA_DIR) / "_documents_cache.pkl"
            cached_docs = load_cache_with_meta(cache_path, manifest)
            if cached_docs:
                _original_documents = cached_docs
                logger.info(f"从缓存加载 {len(cached_docs)} 个文档块（指纹校验通过）")
            else:
                _original_documents = None
                logger.warning("文档块缓存不可用（缺失/指纹不匹配），BM25与自适应检索降级为纯向量检索")
            
            # 加载父块缓存（用于ParentChildRetriever）
            parent_cache_path = Path(DATA_DIR) / "_parent_documents_cache.pkl"
            if USE_PARENT_CHILD:
                _parent_documents = load_cache_with_meta(parent_cache_path, manifest)
                if _parent_documents:
                    logger.info(f"从缓存加载 {len(_parent_documents)} 个父块（指纹校验通过）")
                else:
                    _parent_documents = None
                    logger.warning("父块缓存不可用（缺失/指纹不匹配），ParentChildRetriever 将直接返回子块")
            
            qa = build_retrieval_qa_chain(
                vectorstore, 
                documents=cached_docs,
                parent_documents=_parent_documents
            )
            _chain = qa
            _chain_building = False
            return
        except Exception as e:
            logger.warning(f"加载已有向量库失败: {e}，将重新构建...")

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
        logger.warning("没有找到可用文档")
        _chain_building = False
        return

    logger.info("构建向量数据库...")
    vectorstore = build_or_load_vectorstore(documents)
    
    # 保存原始文档用于BM25混合检索，并缓存到磁盘
    _original_documents = documents
    save_cache_with_meta(Path(DATA_DIR) / "_documents_cache.pkl", documents)
    
    # 加载父块缓存（由load_and_split_pdfs自动保存，已带指纹边车）
    if USE_PARENT_CHILD:
        parent_cache_path = Path(DATA_DIR) / "_parent_documents_cache.pkl"
        _parent_documents = load_cache_with_meta(parent_cache_path, read_manifest(PERSIST_DIRECTORY))
        if _parent_documents:
            logger.info(f"父块缓存已加载: {len(_parent_documents)} 个父块")
        else:
            logger.warning("父块缓存不可用（缺失/指纹不匹配）")
    
    logger.info("构建问答链（使用混合检索：BM25 + 向量）...")
    qa = build_retrieval_qa_chain(
        vectorstore, 
        documents=documents,
        parent_documents=_parent_documents
    )
    _file_list = sorted({doc.metadata.get("source", "未知来源") for doc in documents})
    _chain = qa
    _chain_building = False
    logger.info(f"检索链构建完成，已加载文件：{_file_list}")


def _reload_chain_async():
    """重建任务完成后在后台从磁盘重载检索链（读盘 + 构建检索器）"""
    global _chain_building
    _chain_building = True
    try:
        _build_chain()
    finally:
        _chain_building = False


def _refresh_user_task_cache(user_id: int) -> None:
    """用户文档任务结束后失效该用户的检索器缓存"""
    task_id = _pending_user_tasks.get(user_id)
    if not task_id:
        return
    try:
        job = fetch_job(task_id)
        status = job.get_status() if job is not None else "finished"
        if status in ("finished", "failed"):
            _user_retrievers_cache.pop(user_id, None)
            _pending_user_tasks.pop(user_id, None)
            logger.info(f"用户 {user_id} 文档任务 {status}，检索器缓存已失效")
    except Exception as e:
        logger.warning(f"检查用户任务状态失败: {e}")


def _ensure_chain(wait: bool = False, timeout: int = 60):
    global _chain, _file_list, _chain_building, _chain_error, _build_start_time, _rebuild_task_id

    # 重建任务完成 -> 后台重载检索链（双缓冲区保证旧库服务到切换完成）
    if _rebuild_task_id and _chain is not None:
        try:
            job = fetch_job(_rebuild_task_id)
            if job is not None and job.get_status() in ("finished", "failed"):
                status = job.get_status()
                _rebuild_task_id = None
                if status == "finished":
                    logger.info("重建任务完成，后台重载检索链")
                    threading.Thread(target=_reload_chain_async, daemon=True).start()
                else:
                    _chain_error = "知识库重建失败，请查看任务状态"
                    logger.error(f"重建任务失败: {job.exc_info}")
        except Exception as e:
            logger.warning(f"检查重建任务状态失败: {e}")

    if _chain is not None:
        return _chain, _file_list

    if _chain_building:
        if wait:
            start = time.time()
            while _chain_building and (time.time() - start) < timeout:
                time.sleep(0.5)
            return _chain, _file_list
        return None, []

    _chain_building = True
    _chain_error = None
    _build_start_time = time.time()

    if wait:
        _build_chain()
    else:
        threading.Thread(target=_build_chain, daemon=True).start()

    return _chain, _file_list


# --------------- Pydantic Models ---------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索返回文档数量")
    session_id: Optional[str] = Field(default=None, description="会话ID（用于多轮对话）")
    include_history: bool = Field(default=False, description="是否包含对话历史")
    user_id: Optional[int] = Field(default=None, description="用户ID（用于长期记忆）")


class SessionInfo(BaseModel):
    session_id: str = Field(..., description="会话ID")
    message_count: int = Field(..., description="消息数量")
    created_at: float = Field(..., description="创建时间戳")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="回答内容")
    sources: List[str] = Field(default_factory=list, description="引用来源文件名")
    status: str = Field(default="success", description="状态: success / error / loading")

class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    chain_ready: bool = Field(..., description="问答链是否就绪")
    chain_building: bool = Field(..., description="问答链是否正在构建")
    loaded_files: List[str] = Field(default_factory=list, description="已加载文件列表")
    doc_processor: str = Field(..., description="当前文档处理器")
    data_dir: str = Field(..., description="数据目录路径")
    persist_dir: str = Field(..., description="向量数据库路径")
    rebuild_task_id: Optional[str] = Field(None, description="进行中的重建任务ID")
    rebuild_status: Optional[str] = Field(None, description="重建任务状态")

class FileListResponse(BaseModel):
    files: List[str] = Field(default_factory=list, description="已加载文件列表")
    total: int = Field(..., description="文件总数")

class RebuildResponse(BaseModel):
    status: str = Field(..., description="重建状态")
    message: str = Field(..., description="说明信息")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误详情")
    status: str = Field(default="error", description="status")


# --------------- User Auth Models ---------------

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, max_length=100, description="密码")

class UserLoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")

class UserResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    user_id: Optional[int] = Field(None, description="用户ID")
    username: Optional[str] = Field(None, description="用户名")
    message: str = Field(..., description="消息")


# --------------- User Document Models ---------------

class UserDocumentResponse(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    document_id: Optional[int] = Field(None, description="文档ID")
    filename: Optional[str] = Field(None, description="文件名")
    status: Optional[str] = Field(None, description="状态")
    message: str = Field(..., description="消息")

class UserDocumentListResponse(BaseModel):
    documents: List[dict] = Field(default_factory=list, description="文档列表")
    total: int = Field(..., description="文档总数")

class UserDocumentInfo(BaseModel):
    id: int = Field(..., description="文档ID")
    original_filename: str = Field(..., description="原始文件名")
    file_size: int = Field(..., description="文件大小（字节）")
    upload_time: float = Field(..., description="上传时间戳")
    status: str = Field(..., description="状态")
    num_chunks: Optional[int] = Field(None, description="文档块数")

class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(None, description="对话标题")

class ConversationResponse(BaseModel):
    id: int = Field(..., description="对话ID")
    session_id: str = Field(..., description="会话ID")
    title: str = Field(..., description="对话标题")
    created_at: float = Field(..., description="创建时间")
    updated_at: float = Field(..., description="更新时间")
    message_count: int = Field(default=0, description="消息数量")

class MessageResponse(BaseModel):
    id: int = Field(..., description="消息ID")
    role: str = Field(..., description="角色")
    content: str = Field(..., description="内容")
    sources: List[str] = Field(default_factory=list, description="来源")
    created_at: float = Field(..., description="创建时间")


# --------------- App ---------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动，初始化知识库...")
    _ensure_directories()
    database.init_database()
    logger.info("数据库初始化完成")
    _ensure_chain(wait=True, timeout=120)
    logger.info("知识库初始化完成")
    yield
    logger.info("应用关闭")


app = FastAPI(
    title="Policy QA API",
    description="国家政策知识库智能问答 RESTful API — 基于 RAG 架构",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.time()
    trace_id = new_trace_id()
    bind_trace_id(trace_id)
    try:
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        response.headers["X-Trace-Id"] = trace_id
        _request_logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(elapsed, 1),
        )
        return response
    finally:
        clear_trace_id()


# --------------- Endpoints ---------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查端点，返回服务状态与知识库信息。"""
    rebuild_status = None
    if _rebuild_task_id:
        try:
            job = fetch_job(_rebuild_task_id)
            rebuild_status = job.get_status() if job is not None else "finished"
        except Exception:
            rebuild_status = "unknown"
    return HealthResponse(
        status="ok" if _chain is not None else "loading",
        chain_ready=_chain is not None,
        chain_building=_chain_building,
        loaded_files=_file_list,
        doc_processor=DOC_PROCESSOR,
        data_dir=DATA_DIR,
        persist_dir=PERSIST_DIRECTORY,
        rebuild_task_id=_rebuild_task_id,
        rebuild_status=rebuild_status,
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["问答"])
async def query(req: QueryRequest):
    """
    核心问答接口（支持多轮对话）。

    - 接收用户问题，通过 RAG 链路检索并生成回答
    - 支持session_id实现多轮对话上下文管理
    - 返回回答内容与引用来源
    """
    current_chain, current_files = _ensure_chain(wait=True, timeout=60)

    if current_chain is None:
        if _chain_building:
            elapsed = int(time.time() - _build_start_time)
            return QueryResponse(
                answer=f"知识库正在初始化（已等待 {elapsed} 秒），请稍后重试",
                status="loading",
            )
        if _chain_error:
            raise HTTPException(status_code=503, detail=f"知识库初始化失败：{_chain_error}")
        if not current_files:
            return QueryResponse(
                answer=f"当前没有可用的政策文档，请将 PDF 文件放入 data/ 目录后重试",
                status="error",
            )
        raise HTTPException(status_code=503, detail="知识库未就绪")

    try:
        session_store.cleanup_expired()
        
        chat_history = []
        if req.session_id and req.include_history:
            chat_history = session_store.get(req.session_id)
            logger.info(f"会话 {req.session_id}: 加载 {len(chat_history)} 条历史")
        
        logger.info(f"处理查询: {req.question[:50]}...")
        
        memory_context = ""
        if req.user_id and MEMORY_ENABLED and memory_manager:
            try:
                memory_context = memory_manager.build_memory_context(
                    req.user_id, req.question, chat_history
                )
                logger.info(f"用户 {req.user_id}: 构建长期记忆上下文")
            except Exception as e:
                logger.error(f"构建记忆上下文失败: {e}")
        
        # 如果有用户ID，使用缓存的用户检索器
        if req.user_id:
            _refresh_user_task_cache(req.user_id)
            try:
                if req.user_id not in _user_retrievers_cache:
                    # 构建新的检索器（内部会自动判断用户是否有个人文档）
                    from qa_chain import build_retrieval_qa_chain
                    vectorstore = build_or_load_vectorstore([])
                    user_chain = build_retrieval_qa_chain(
                        vectorstore, 
                        documents=_original_documents,
                        user_id=req.user_id,
                        parent_documents=_parent_documents
                    )
                    _user_retrievers_cache[req.user_id] = user_chain
                current_chain = _user_retrievers_cache[req.user_id]
            except Exception as e:
                logger.error(f"创建用户检索链失败: {e}，使用默认链")
        
        invoke_input = {"input": req.question, "memory_context": memory_context}
        if chat_history:
            from qa_chain import format_chat_history
            formatted_history = format_chat_history(chat_history)
            invoke_input["chat_history"] = formatted_history
            logger.info(f"  包含 {len(formatted_history)} 条对话历史")
        else:
            invoke_input["chat_history"] = []
        
        result = current_chain.invoke(invoke_input)
        answer_text = result.get("answer") or ""
        source_documents = result.get("context", [])
        
        # 强制打印检索日志
        logger.info("=" * 80)
        logger.info("📊 检索结果详情（Top 5片段）:")
        if source_documents:
            for i, doc in enumerate(source_documents[:5], 1):
                source_name = doc.metadata.get("source") or doc.metadata.get("file_name") or "未知来源"
                page = doc.metadata.get("page", "")
                content_preview = doc.page_content.strip()[:200]
                logger.info(f"  [{i}] 文件: {source_name} (第{page}页)")
                logger.info(f"      内容: {content_preview}...")
                
                # 检查是否来自用户个人数据库
                source_type = doc.metadata.get("source_type", "public")
                logger.info(f"      来源类型: {source_type}")
        else:
            logger.info("  ⚠️ 未检索到任何文档片段")
        logger.info("=" * 80)
        
        # 直接返回检索结果中最相关的文档（前3个）
        # 不再尝试从LLM回答中提取引用，因为LLM的引用格式不可控
        sources = extract_sources(source_documents[:3]) if source_documents else []
        
        if req.session_id:
            session_store.extend(req.session_id, [
                {"role": "user", "content": req.question},
                {"role": "assistant", "content": answer_text.strip()},
            ])
        
        if req.user_id and req.session_id and MEMORY_ENABLED and memory_manager:
            try:
                conversation_id = req.session_id
                memory_manager.save_long_term_memory(
                    req.user_id, conversation_id, req.question, answer_text, sources
                )
                logger.info(f"已保存长期记忆: 用户={req.user_id}")
            except Exception as e:
                logger.error(f"保存长期记忆失败: {e}")
        
        return QueryResponse(answer=answer_text.strip(), sources=sources, status="success")
    except Exception as exc:
        logger.error(f"查询失败: {exc}")
        raise HTTPException(status_code=500, detail=f"查询失败：{exc}")


@app.get("/api/v1/files", response_model=FileListResponse, tags=["知识库"])
async def list_files():
    """获取已加载的 PDF 文件列表。"""
    return FileListResponse(files=_file_list, total=len(_file_list))


@app.post("/api/v1/rebuild", response_model=RebuildResponse, tags=["知识库"])
async def rebuild_knowledge_base(force: bool = Query(default=False, description="是否强制完全重建")):
    """
    刷新知识库（RQ 异步任务，双缓冲区原子重建）。

    - 返回 task_id，可通过 /api/v1/tasks/{task_id} 轮询状态
    - 任务完成后 API 自动重载检索链；重建期间旧知识库继续服务
    """
    global _rebuild_task_id

    if _chain_building:
        return RebuildResponse(status="loading", message="知识库正在初始化中，请稍后")

    if _rebuild_task_id:
        return RebuildResponse(status="loading", message=f"已有重建任务进行中: {_rebuild_task_id}")

    if not force:
        data_path = Path(DATA_DIR)
        current_pdfs = sorted([f.name for f in data_path.glob("*.pdf")]) if data_path.exists() else []
        if set(current_pdfs) == set(_file_list):
            return RebuildResponse(status="ready", message="文件未变化，无需更新")

    try:
        task = enqueue_rebuild(force=force)
    except Exception as e:
        logger.error(f"重建任务入队失败: {e}")
        raise HTTPException(status_code=503, detail="任务队列不可用，请确认 Redis 与 worker 已启动")

    _rebuild_task_id = task.id
    logger.info(f"重建任务已入队: {task.id}（force={force}）")
    return RebuildResponse(
        status="started",
        message=f"重建任务已启动: {task.id}，可通过 /api/v1/tasks/{task.id} 轮询状态"
    )


@app.get("/api/v1/tasks/{task_id}", tags=["任务"])
async def get_task_status(task_id: str):
    """
    RQ 任务状态轮询接口。

    - status: queued / started / finished / failed
    - finished 时返回 result；failed 时返回精简错误信息（不暴露堆栈）
    """
    try:
        job = fetch_job(task_id)
    except Exception as e:
        logger.warning(f"任务查询失败: {task_id} - {e}")
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    status = job.get_status()
    result = job.result if status == "finished" else None
    error = None
    if status == "failed" and job.exc_info:
        error = str(job.exc_info).strip().splitlines()[-1][:300]

    return {"task_id": task_id, "status": status, "result": result, "error": error}


@app.get("/api/v1/search", tags=["检索"])
async def search_documents(
    query: str = Query(..., min_length=1, max_length=2000, description="检索查询"),
    top_k: int = Query(default=5, ge=1, le=20, description="返回文档数量"),
):
    """
    纯检索接口（不经过 LLM）。

    - 仅返回向量检索命中的文档片段与来源
    - 用于调试检索效果或构建自定义问答逻辑
    """
    current_chain, _ = _ensure_chain(wait=True, timeout=60)

    if current_chain is None:
        raise HTTPException(status_code=503, detail="知识库未就绪")

    try:
        from vectorstore import build_or_load_vectorstore
        vectorstore = build_or_load_vectorstore([])
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", "未知"),
            })

        return {"query": query, "total": len(results), "documents": results}
    except Exception as exc:
        logger.error(f"检索失败: {exc}")
        raise HTTPException(status_code=500, detail=f"检索失败：{exc}")


@app.post("/api/v1/session", response_model=SessionInfo, tags=["会话"])
async def create_session():
    """
    创建新会话。

    - 返回唯一的session_id用于后续多轮对话
    - Redis 后端由 TTL 自动过期（默认1小时）
    """
    session_id = session_store.create()
    logger.info(f"创建新会话: {session_id}")
    return SessionInfo(
        session_id=session_id,
        message_count=0,
        created_at=time.time()
    )


@app.get("/api/v1/session/{session_id}", response_model=SessionInfo, tags=["会话"])
async def get_session(session_id: str):
    """
    查询会话信息。

    - 返回会话的消息数量
    """
    if not session_store.exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    return SessionInfo(
        session_id=session_id,
        message_count=len(session_store.get(session_id)),
        created_at=time.time()
    )


@app.delete("/api/v1/session/{session_id}", tags=["会话"])
async def delete_session(session_id: str):
    """
    删除会话。

    - 清除会话的所有对话历史（Redis key 或内存条目）
    """
    if not session_store.exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")

    session_store.delete(session_id)
    logger.info(f"删除会话: {session_id}")
    return {"status": "deleted", "session_id": session_id}


@app.get("/api/v1/session/{session_id}/history", tags=["会话"])
async def get_session_history(session_id: str):
    """
    获取会话的完整对话历史。
    """
    history = session_store.get(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    }


# --------------- User Auth Endpoints ---------------

@app.post("/api/v1/auth/register", response_model=UserResponse, tags=["用户认证"])
async def register_user(req: UserRegisterRequest):
    """
    用户注册。
    
    - 用户名：3-50字符
    - 密码：6-100字符
    """
    result = database.create_user(req.username, req.password)
    return UserResponse(**result)


@app.post("/api/v1/auth/login", response_model=UserResponse, tags=["用户认证"])
async def login_user(req: UserLoginRequest):
    """
    用户登录。
    
    - 返回用户ID和用户名
    """
    result = database.login_user(req.username, req.password)
    return UserResponse(**result)


@app.get("/api/v1/auth/user/{user_id}", tags=["用户认证"])
async def get_user_info(user_id: int):
    """
    获取用户信息。
    """
    user = database.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


# --------------- User Document Endpoints ---------------

@app.post("/api/v1/users/{user_id}/documents", response_model=UserDocumentResponse, tags=["用户文档"])
async def upload_user_document(
    user_id: int,
    file: UploadFile = File(..., description="PDF文件")
):
    """
    上传用户文档。
    
    - 支持PDF格式
    - 最大文件大小：10MB
    - 每个用户最多10个文档
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF格式文件")
    
    file_size = 0
    temp_file_path = None
    
    try:
        user_doc_count = database.get_user_document_count(user_id)
        if user_doc_count >= MAX_USER_DOCUMENTS:
            raise HTTPException(
                status_code=400, 
                detail=f"每个用户最多上传{MAX_USER_DOCUMENTS}个文档"
            )
        
        user_doc_dir = Path(DATA_DIR) / "user_documents" / f"user_{user_id}"
        user_doc_dir.mkdir(parents=True, exist_ok=True)
        
        import uuid as uuid_lib
        file_uuid = str(uuid_lib.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        saved_filename = f"{file_uuid}{file_ext}"
        file_path = user_doc_dir / saved_filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            file_size = len(content)
            
            if file_size > MAX_USER_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件大小超过限制（最大{MAX_USER_FILE_SIZE / 1024 / 1024}MB）"
                )
            
            buffer.write(content)
        
        document_id = database.add_user_document(
            user_id=user_id,
            filename=saved_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=file_size
        )
        
        logger.info(f"用户 {user_id} 上传文档: {file.filename} (ID: {document_id})")
        
        # 异步处理文档：RQ 任务（worker 进程执行，不占 API 线程）
        try:
            task = enqueue_process_user_document(
                user_id=user_id,
                file_path=str(file_path),
                original_filename=file.filename,
                document_id=document_id,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        except Exception as e:
            logger.error(f"文档任务入队失败: {e}")
            database.update_document_status(document_id, "failed", f"任务入队失败: {e}")
            raise HTTPException(status_code=503, detail="任务队列不可用，请确认 Redis 与 worker 已启动")
        _pending_user_tasks[user_id] = task.id
        logger.info(f"文档处理任务已入队: {task.id}（文档: {file.filename}）")
        
        return UserDocumentResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            status="processing",
            task_id=task.id,
            message=f"文档上传成功，正在处理中（task_id: {task.id}）"
        )
        
    except HTTPException:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    except Exception as e:
        logger.error(f"上传文档失败: {e}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.get("/api/v1/users/{user_id}/documents", response_model=UserDocumentListResponse, tags=["用户文档"])
async def get_user_documents(user_id: int):
    """
    获取用户的所有文档。
    
    - 返回文档列表和总数
    """
    documents = database.get_user_documents(user_id)
    
    return UserDocumentListResponse(
        documents=documents,
        total=len(documents)
    )


@app.delete("/api/v1/users/{user_id}/documents/{document_id}", tags=["用户文档"])
async def delete_user_document(user_id: int, document_id: int):
    """
    删除用户文档（级联删除）。
    
    - 删除 SQLite 记录与源文件
    - 按 source 过滤物理删除用户向量库中的向量
    - 清空用户级检索器缓存，确保下次检索不命中已删内容
    """
    document = database.get_document_by_id(document_id, user_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在或无权访问")

    result = database.delete_user_document(document_id, user_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="文档不存在或无权访问")
    
    success, file_path = result
    original_filename = document["original_filename"]

    # 1. 清空用户级检索器缓存（无论后续清理是否成功，检索器都必须重建）
    _user_retrievers_cache.pop(user_id, None)
    logger.info(f"已清空用户 {user_id} 的检索器缓存")
    
    # 2. 删除源文件（尽力而为）
    if success and file_path:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"删除文件失败: {e}")

    # 3. 级联删除向量：按 source 过滤物理删除
    try:
        from user_vectorstore import delete_user_document_vectors
        ok = delete_user_document_vectors(user_id, original_filename, get_embeddings())
        if ok:
            logger.info(f"用户 {user_id} 文档向量已删除: {original_filename}")
        else:
            logger.error(f"用户 {user_id} 文档向量删除失败（SQLite 记录已删除，需手动清理）: {original_filename}")
    except Exception as e:
        logger.error(f"级联删除向量失败: {e}")
    
    return {
        "success": True,
        "message": "文档已删除"
    }


@app.get("/api/v1/users/{user_id}/documents/{document_id}", response_model=UserDocumentInfo, tags=["用户文档"])
async def get_user_document_info(user_id: int, document_id: int):
    """
    获取单个文档信息。
    """
    document = database.get_document_by_id(document_id, user_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在或无权访问")
    
    return UserDocumentInfo(
        id=document["id"],
        original_filename=document["original_filename"],
        file_size=document["file_size"],
        upload_time=document["upload_time"],
        status=document["status"]
    )


# --------------- Conversation History Endpoints ---------------

@app.post("/api/v1/conversations", response_model=ConversationResponse, tags=["对话历史"])
async def create_conversation(
    user_id: int = Query(..., description="用户ID"),
    req: ConversationCreateRequest = None
):
    """
    创建新对话。
    
    - 返回对话ID和会话ID
    """
    session_id = str(uuid.uuid4())
    title = req.title if req else None
    conversation_id = database.create_conversation(user_id, session_id, title)
    
    return ConversationResponse(
        id=conversation_id,
        session_id=session_id,
        title=title or f"对话 {time.strftime('%Y-%m-%d %H:%M')}",
        created_at=time.time(),
        updated_at=time.time(),
        message_count=0
    )


@app.get("/api/v1/conversations", response_model=List[ConversationResponse], tags=["对话历史"])
async def get_user_conversations(
    user_id: int = Query(..., description="用户ID"),
    limit: int = Query(default=50, ge=1, le=100, description="返回数量")
):
    """
    获取用户的所有对话。
    
    - 按更新时间倒序排列
    """
    conversations = database.get_user_conversations(user_id, limit)
    
    result = []
    for conv in conversations:
        messages = database.get_conversation_messages(conv['id'])
        result.append(ConversationResponse(
            id=conv['id'],
            session_id=conv['session_id'],
            title=conv['title'],
            created_at=conv['created_at'],
            updated_at=conv['updated_at'],
            message_count=len(messages)
        ))
    
    return result


@app.get("/api/v1/conversations/{conversation_id}", response_model=List[MessageResponse], tags=["对话历史"])
async def get_conversation_messages(conversation_id: int):
    """
    获取对话的所有消息。
    """
    messages = database.get_conversation_messages(conversation_id)
    
    return [
        MessageResponse(
            id=msg['id'],
            role=msg['role'],
            content=msg['content'],
            sources=msg['sources'],
            created_at=msg['created_at']
        )
        for msg in messages
    ]


@app.delete("/api/v1/conversations/{conversation_id}", tags=["对话历史"])
async def delete_conversation(
    conversation_id: int,
    user_id: int = Query(..., description="用户ID")
):
    """
    删除对话。
    
    - 同时删除所有消息
    """
    success = database.delete_conversation(conversation_id, user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="对话不存在或无权删除")
    
    return {"status": "deleted", "conversation_id": conversation_id}


@app.put("/api/v1/conversations/{conversation_id}/title", tags=["对话历史"])
async def update_conversation_title(
    conversation_id: int,
    user_id: int = Query(..., description="用户ID"),
    title: str = Query(..., min_length=1, max_length=100, description="新标题")
):
    """
    更新对话标题。
    """
    success = database.update_conversation_title(conversation_id, user_id, title)
    
    if not success:
        raise HTTPException(status_code=404, detail="对话不存在或无权修改")
    
    return {"status": "updated", "conversation_id": conversation_id, "title": title}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # 服务端保留完整异常（含堆栈）便于排查；客户端只收到统一结构，绝不暴露内部细节
    logger.error("未处理异常", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "msg": "Internal Server Error"},
    )


if __name__ == "__main__":
    import uvicorn

    validate_config()
    _ensure_directories()
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
