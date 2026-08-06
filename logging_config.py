"""JSON 结构化日志配置（structlog + stdlib logging）

将 API 日志输出统一为 JSON 行格式，并注入请求级 trace_id（uuid），
便于日志采集（ELK/Loki）、链路追踪与问题定位。
"""
import logging
import uuid

import structlog


def new_trace_id() -> str:
    """生成请求级 trace_id（uuid4 hex）"""
    return uuid.uuid4().hex


def bind_trace_id(trace_id: str) -> None:
    """将 trace_id 绑定到当前请求上下文（协程/线程隔离）"""
    structlog.contextvars.bind_contextvars(trace_id=trace_id)


def clear_trace_id() -> None:
    """清理当前请求上下文"""
    structlog.contextvars.clear_contextvars()


def get_trace_id() -> str:
    """读取当前上下文中的 trace_id（用于日志或响应头）"""
    try:
        return structlog.contextvars.get_contextvars().get("trace_id", "")
    except Exception:
        return ""


def setup_logging(level: int = logging.INFO) -> None:
    """配置标准库 logging 与 structlog，统一输出 JSON 日志行"""
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.format_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
            processor=structlog.processors.JSONRenderer(ensure_ascii=False),
        )
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
