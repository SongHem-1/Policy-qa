"""会话存储抽象：Redis 后端 + 磁盘持久化兜底

- Redis 可用：JSON 序列化 + 滑动 TTL 过期（生产，多实例共享）
- Redis 不可用：降级为磁盘持久化（data/session_store_fallback.json），
  服务重启不丢失会话；每 30s 自动探测 Redis，恢复后自动切换并迁移存量会话

如需启动本地 Redis，运行 scripts/start_redis.bat（Windows 便携版）。
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from config import DATA_DIR, REDIS_URL, SESSION_TTL

logger = logging.getLogger("policy_qa.session")

SESSION_KEY_PREFIX = "session:"
FALLBACK_FILE = "session_store_fallback.json"
REDIS_RECONNECT_INTERVAL = 30  # 秒


def _key(sid: str) -> str:
    return SESSION_KEY_PREFIX + sid


class SessionStore:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        ttl: int = SESSION_TTL,
        max_sessions: int = 100,
        fallback_path: Optional[Path] = None,
    ):
        self.redis_url = redis_url
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._redis = None
        self._memory: Dict[str, List[dict]] = {}
        self._memory_ts: Dict[str, float] = {}
        self._last_redis_attempt = 0.0
        self._fallback_path = fallback_path or (Path(DATA_DIR) / FALLBACK_FILE)
        self.backend = "memory"

        # 先恢复磁盘兜底会话，再尝试连接 Redis 并迁移（避免可用时清空兜底数据）
        self._load_fallback_from_disk()
        self._try_connect_redis()

    # ---------- Redis 连接与自动重连 ----------
    def _try_connect_redis(self) -> bool:
        try:
            import redis as redis_client

            client = redis_client.Redis.from_url(self.redis_url, decode_responses=True)
            client.ping()
            self._redis = client
            self.backend = "redis"
            logger.info("会话存储: Redis 后端（TTL=%ss）", self.ttl)
            # 迁移兜底中的存量会话到 Redis
            if self._memory:
                for sid, history in list(self._memory.items()):
                    self._set(sid, history)
                self._memory.clear()
                self._memory_ts.clear()
            self._clear_fallback_file()
            return True
        except Exception as e:
            self._redis = None
            self.backend = "disk" if self._fallback_path.exists() else "memory"
            logger.warning(
                "Redis 不可用（%s），会话存储降级为磁盘持久化兜底（重启不丢失）；"
                "运行 scripts/start_redis.bat 启动 Redis 后将自动切换",
                e,
            )
            return False

    def _ensure_redis(self) -> None:
        """Redis 不可用时周期性探测（默认 30s 一次），恢复后自动切换"""
        if self._redis is not None:
            return
        now = time.time()
        if now - self._last_redis_attempt < REDIS_RECONNECT_INTERVAL:
            return
        self._last_redis_attempt = now
        self._try_connect_redis()

    # ---------- 通用接口 ----------
    def create(self) -> str:
        self._ensure_redis()
        sid = uuid.uuid4().hex
        self._set(sid, [])
        return sid

    def get(self, sid: str) -> List[dict]:
        self._ensure_redis()
        if self._redis is not None:
            raw = self._redis.get(_key(sid))
            if raw is None:
                return []
            self._redis.expire(_key(sid), self.ttl)  # 滑动过期
            try:
                return json.loads(raw)
            except Exception:
                return []
        data = self._memory.get(sid, [])
        if sid in self._memory:
            self._memory_ts[sid] = time.time()
        return list(data)

    def exists(self, sid: str) -> bool:
        self._ensure_redis()
        if self._redis is not None:
            return self._redis.exists(_key(sid)) > 0
        return sid in self._memory

    def extend(self, sid: str, messages: List[dict]) -> None:
        history = self.get(sid)
        history.extend(messages)
        self._set(sid, history)

    def append(self, sid: str, message: dict) -> None:
        self.extend(sid, [message])

    def delete(self, sid: str) -> None:
        self._ensure_redis()
        if self._redis is not None:
            self._redis.delete(_key(sid))
        else:
            self._memory.pop(sid, None)
            self._memory_ts.pop(sid, None)
            self._persist_fallback()

    def cleanup_expired(self) -> None:
        """Redis 由 TTL 自动处理；磁盘/内存兜底做惰性清理"""
        if self._redis is not None:
            return
        now = time.time()
        expired = [sid for sid, ts in self._memory_ts.items() if now - ts > self.ttl]
        for sid in expired:
            self._memory.pop(sid, None)
            self._memory_ts.pop(sid, None)
        while len(self._memory) > self.max_sessions:
            oldest = min(self._memory_ts, key=self._memory_ts.get)
            self._memory.pop(oldest, None)
            self._memory_ts.pop(oldest, None)
        if expired:
            self._persist_fallback()

    # ---------- 内部 ----------
    def _set(self, sid: str, history: List[dict]) -> None:
        if self._redis is not None:
            self._redis.set(_key(sid), json.dumps(history, ensure_ascii=False), ex=self.ttl)
        else:
            self._memory[sid] = history
            self._memory_ts[sid] = time.time()
            self._persist_fallback()

    def _persist_fallback(self) -> None:
        """将内存会话原子落盘，保证 Redis 不可用时重启不丢会话"""
        try:
            self.backend = "disk"
            self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                sid: {"history": history, "ts": self._memory_ts.get(sid, time.time())}
                for sid, history in self._memory.items()
            }
            tmp = self._fallback_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp.replace(self._fallback_path)
        except Exception as e:
            logger.warning("会话兜底落盘失败: %s", e)

    def _load_fallback_from_disk(self) -> None:
        """启动时从磁盘恢复会话（仅 Redis 不可用时）"""
        if not self._fallback_path.exists():
            self.backend = "memory"
            return
        try:
            with open(self._fallback_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            now = time.time()
            for sid, item in payload.items():
                history = item.get("history", []) if isinstance(item, dict) else []
                ts = item.get("ts", now) if isinstance(item, dict) else now
                if now - ts <= self.ttl:
                    self._memory[sid] = history
                    self._memory_ts[sid] = ts
            self.backend = "disk"
            logger.info("已从磁盘恢复 %d 个会话（Redis 不可用兜底）", len(self._memory))
        except Exception as e:
            self.backend = "memory"
            logger.warning("会话兜底文件读取失败: %s", e)

    def _clear_fallback_file(self) -> None:
        try:
            if self._fallback_path.exists():
                self._fallback_path.unlink()
        except Exception:
            pass


session_store = SessionStore()
