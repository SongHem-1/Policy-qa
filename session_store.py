"""会话存储抽象：Redis 后端 + 内存兜底

生产环境使用 Redis（TTL 自动过期，支持多实例共享）；
Redis 不可用时降级为进程内存（带容量上限与惰性清理），保证本地开发可用。
"""
import json
import logging
import time
import uuid
from typing import Dict, List

from config import REDIS_URL, SESSION_TTL

logger = logging.getLogger("policy_qa.session")

SESSION_KEY_PREFIX = "session:"


def _key(sid: str) -> str:
    return SESSION_KEY_PREFIX + sid


class SessionStore:
    def __init__(self, redis_url: str = REDIS_URL, ttl: int = SESSION_TTL, max_sessions: int = 100):
        self.ttl = ttl
        self.max_sessions = max_sessions
        self.backend = "memory"
        self._redis = None
        self._memory: Dict[str, List[dict]] = {}
        self._memory_ts: Dict[str, float] = {}

        try:
            import redis as redis_client
            client = redis_client.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self._redis = client
            self.backend = "redis"
            logger.info("会话存储: Redis 后端（TTL=%ss）", ttl)
        except Exception as e:
            logger.warning("Redis 不可用（%s），会话存储降级为进程内存（重启即失）", e)

    # ---------- 通用接口 ----------
    def create(self) -> str:
        sid = uuid.uuid4().hex
        self._set(sid, [])
        return sid

    def get(self, sid: str) -> List[dict]:
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
        if self._redis is not None:
            self._redis.delete(_key(sid))
        else:
            self._memory.pop(sid, None)
            self._memory_ts.pop(sid, None)

    def cleanup_expired(self) -> None:
        """仅内存后端需要惰性清理；Redis 由 TTL 自动处理"""
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

    # ---------- 内部 ----------
    def _set(self, sid: str, history: List[dict]) -> None:
        if self._redis is not None:
            self._redis.set(_key(sid), json.dumps(history, ensure_ascii=False), ex=self.ttl)
        else:
            self._memory[sid] = history
            self._memory_ts[sid] = time.time()


session_store = SessionStore()
