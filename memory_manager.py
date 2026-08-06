"""
记忆管理模块

实现短期记忆（当前会话）+ 长期记忆（持久化）的分层架构
"""
import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from llm_provider import get_llm_provider
import database

MEMORY_DB_PATH = Path(__file__).parent / "data" / "memory.db"


class MemoryManager:
    """记忆管理器：管理短期和长期记忆"""
    
    def __init__(self):
        self._init_memory_db()
        # 供应商抽象：主备降级 + 超时重试由 llm_provider 统一负责
        self.provider = get_llm_provider()
        self.llm = self.provider.llm
    
    def _init_memory_db(self):
        """初始化长期记忆数据库"""
        MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(MEMORY_DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                conversation_id INTEGER,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                key_entities TEXT,
                embedding BLOB,
                created_at REAL NOT NULL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                importance_score REAL DEFAULT 0.5,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_user 
            ON long_term_memory(user_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type 
            ON long_term_memory(memory_type)
        """)
        
        conn.commit()
        conn.close()
    
    def extract_memory_from_conversation(
        self, 
        user_message: str, 
        assistant_message: str,
        sources: List[str] = None
    ) -> Dict[str, Any]:
        """从对话中提取记忆信息
        
        Args:
            user_message: 用户消息
            assistant_message: 助手回复
            sources: 参考来源
        
        Returns:
            包含摘要、关键实体、重要性评分的字典
        """
        try:
            prompt = f"""请分析以下对话，提取关键信息：

用户问题：{user_message}
助手回答：{assistant_message}

请以JSON格式返回以下信息：
1. summary: 对话摘要（一句话概括）
2. key_entities: 关键实体列表（如政策名称、时间、地点等）
3. importance_score: 重要性评分（0.0-1.0，1.0表示最重要）
4. memory_type: 记忆类型（fact/preference/task/reference）

返回格式示例：
{{
    "summary": "用户询问了健身相关政策",
    "key_entities": ["全民健身计划", "2026-2030年", "体育"],
    "importance_score": 0.8,
    "memory_type": "fact"
}}

只返回JSON，不要其他内容。"""

            response = self.provider.invoke(prompt)
            result_text = response.content.strip()
            
            if result_text.startswith("```json"):
                result_text = result_text[7:-3]
            
            memory_data = json.loads(result_text)
            memory_data["sources"] = sources or []
            
            return memory_data
            
        except Exception as e:
            print(f"提取记忆失败: {e}")
            return {
                "summary": f"用户询问: {user_message[:50]}",
                "key_entities": [],
                "importance_score": 0.5,
                "memory_type": "fact",
                "sources": sources or []
            }
    
    def save_long_term_memory(
        self,
        user_id: int,
        conversation_id: int,
        user_message: str,
        assistant_message: str,
        sources: List[str] = None
    ):
        """保存长期记忆
        
        Args:
            user_id: 用户ID
            conversation_id: 对话ID
            user_message: 用户消息
            assistant_message: 助手回复
            sources: 参考来源
        """
        try:
            memory_data = self.extract_memory_from_conversation(
                user_message, assistant_message, sources
            )
            
            conn = sqlite3.connect(str(MEMORY_DB_PATH))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO long_term_memory 
                (user_id, conversation_id, memory_type, content, summary, 
                 key_entities, created_at, importance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                conversation_id,
                memory_data.get("memory_type", "fact"),
                json.dumps({
                    "user": user_message,
                    "assistant": assistant_message
                }),
                memory_data.get("summary", ""),
                json.dumps(memory_data.get("key_entities", [])),
                time.time(),
                memory_data.get("importance_score", 0.5)
            ))
            
            memory_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"已保存长期记忆 ID={memory_id}: {memory_data.get('summary')}")
            return memory_id
            
        except Exception as e:
            print(f"保存长期记忆失败: {e}")
            return None
    
    def retrieve_relevant_memories(
        self,
        user_id: int,
        current_question: str,
        top_k: int = 5,
        min_importance: float = 0.3
    ) -> List[Dict[str, Any]]:
        """检索相关的长期记忆
        
        Args:
            user_id: 用户ID
            current_question: 当前问题
            top_k: 返回数量
            min_importance: 最小重要性评分
        
        Returns:
            相关记忆列表
        """
        try:
            conn = sqlite3.connect(str(MEMORY_DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, memory_type, content, summary, key_entities, 
                       created_at, importance_score, access_count
                FROM long_term_memory
                WHERE user_id = ? AND importance_score >= ?
                ORDER BY importance_score DESC, created_at DESC
                LIMIT ?
            """, (user_id, min_importance, top_k * 2))
            
            rows = cursor.fetchall()
            
            memories = []
            for row in rows:
                memory = {
                    "id": row["id"],
                    "memory_type": row["memory_type"],
                    "content": json.loads(row["content"]),
                    "summary": row["summary"],
                    "key_entities": json.loads(row["key_entities"]),
                    "created_at": row["created_at"],
                    "importance_score": row["importance_score"],
                    "access_count": row["access_count"]
                }
                memories.append(memory)
                
                cursor.execute("""
                    UPDATE long_term_memory
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (time.time(), row["id"]))
            
            conn.commit()
            conn.close()
            
            print(f"检索到 {len(memories)} 条相关记忆")
            return memories[:top_k]
            
        except Exception as e:
            print(f"检索长期记忆失败: {e}")
            return []
    
    def build_memory_context(
        self,
        user_id: int,
        current_question: str,
        short_term_history: List[Dict] = None
    ) -> str:
        """构建包含长期记忆和短期历史的上下文
        
        Args:
            user_id: 用户ID
            current_question: 当前问题
            short_term_history: 短期对话历史
        
        Returns:
            格式化的上下文字符串
        """
        long_term_memories = self.retrieve_relevant_memories(
            user_id, current_question
        )
        
        context_parts = []
        
        if long_term_memories:
            context_parts.append("### 历史记忆（长期）")
            for i, mem in enumerate(long_term_memories, 1):
                context_parts.append(
                    f"{i}. {mem['summary']} "
                    f"(类型: {mem['memory_type']}, 重要性: {mem['importance_score']:.1f})"
                )
                if mem.get("key_entities"):
                    context_parts.append(f"   关键词: {', '.join(mem['key_entities'])}")
        
        if short_term_history:
            context_parts.append("\n### 当前会话（短期）")
            for msg in short_term_history[-4:]:
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def cleanup_old_memories(self, days: int = 30, min_importance: float = 0.3):
        """清理旧的低重要性记忆
        
        Args:
            days: 保留天数
            min_importance: 清理阈值
        """
        try:
            conn = sqlite3.connect(str(MEMORY_DB_PATH))
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (days * 24 * 3600)
            
            cursor.execute("""
                DELETE FROM long_term_memory
                WHERE created_at < ? AND importance_score < ?
            """, (cutoff_time, min_importance))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"清理了 {deleted_count} 条旧记忆")
            return deleted_count
            
        except Exception as e:
            print(f"清理记忆失败: {e}")
            return 0


memory_manager = MemoryManager()
