"""pickle 缓存读写 + 构建指纹边车（API 进程与 RQ worker 共用）"""
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Optional

from config import compute_build_fingerprint

logger = logging.getLogger("policy_qa.cache")


def cache_meta_path(cache_path: Path) -> Path:
    """缓存指纹边车路径：_documents_cache.pkl -> _documents_cache.pkl.meta.json"""
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def save_cache_with_meta(cache_path: Path, data: Any) -> None:
    """保存 pickle 缓存并写入指纹边车，供加载时与向量库 manifest 互验"""
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    meta = {"fingerprint": compute_build_fingerprint(), "created_at": time.time()}
    with open(cache_meta_path(cache_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    logger.info("缓存已保存: %s（指纹 %s）", cache_path.name, meta["fingerprint"])


def load_cache_with_meta(cache_path: Path, manifest: Optional[dict]):
    """加载缓存并校验指纹与向量库 manifest 一致；不一致返回 None（绝不静默使用旧缓存）"""
    if not cache_path.exists():
        return None
    try:
        meta_path = cache_meta_path(cache_path)
        if not meta_path.exists():
            logger.warning("缓存缺少指纹元数据，已跳过: %s", cache_path.name)
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cache_fp = meta.get("fingerprint")
        manifest_fp = (manifest or {}).get("fingerprint") if manifest else None

        if manifest_fp is None:
            if cache_fp != compute_build_fingerprint():
                logger.warning("缓存指纹与当前构建配置不一致，已跳过: %s", cache_path.name)
                return None
        elif not cache_fp or cache_fp != manifest_fp:
            logger.warning("缓存指纹与向量库 manifest 不一致，已跳过: %s", cache_path.name)
            return None

        if cache_fp != compute_build_fingerprint():
            logger.warning("⚠️ 缓存与当前构建配置不一致（配置已变更），建议重建知识库: %s", cache_path.name)

        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("缓存加载失败: %s - %s", cache_path.name, e)
        return None
