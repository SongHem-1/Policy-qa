"""
数据库管理模块

功能：
- 用户注册/登录
- 对话历史存储
- SQLite 轻量级数据库
"""
import sqlite3
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

DATABASE_PATH = Path(__file__).parent / "data" / "policy_qa.db"


def get_connection():
    """获取数据库连接"""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """初始化数据库表"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_login REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            title TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conversation 
        ON messages(conversation_id)
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_time REAL NOT NULL,
            status TEXT DEFAULT 'processing',
            error_message TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_documents_user 
        ON user_documents(user_id)
    """)
    
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str) -> Dict[str, Any]:
    """创建新用户"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        created_at = time.time()
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, created_at)
        )
        user_id = cursor.lastrowid
        conn.commit()
        
        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "message": "注册成功"
        }
    except sqlite3.IntegrityError:
        return {
            "success": False,
            "message": "用户名已存在"
        }
    finally:
        conn.close()


def login_user(username: str, password: str) -> Dict[str, Any]:
    """用户登录"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        
        cursor.execute(
            "SELECT id, username, created_at FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        user = cursor.fetchone()
        
        if user:
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (time.time(), user['id'])
            )
            conn.commit()
            
            return {
                "success": True,
                "user_id": user['id'],
                "username": user['username'],
                "created_at": user['created_at'],
                "message": "登录成功"
            }
        else:
            return {
                "success": False,
                "message": "用户名或密码错误"
            }
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """根据ID获取用户"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT id, username, created_at, last_login FROM users WHERE id = ?",
            (user_id,)
        )
        user = cursor.fetchone()
        
        if user:
            return {
                "id": user['id'],
                "username": user['username'],
                "created_at": user['created_at'],
                "last_login": user['last_login']
            }
        return None
    finally:
        conn.close()


def create_conversation(user_id: int, session_id: str, title: str = None) -> int:
    """创建新对话"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        created_at = time.time()
        title = title or f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        cursor.execute(
            "INSERT INTO conversations (user_id, session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, session_id, title, created_at, created_at)
        )
        conversation_id = cursor.lastrowid
        conn.commit()
        
        return conversation_id
    finally:
        conn.close()


def save_message(conversation_id: int, role: str, content: str, sources: List[str] = None) -> int:
    """保存消息"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        created_at = time.time()
        sources_json = json.dumps(sources) if sources else None
        
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, sources_json, created_at)
        )
        message_id = cursor.lastrowid
        
        cursor.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (created_at, conversation_id)
        )
        
        conn.commit()
        return message_id
    finally:
        conn.close()


def get_user_conversations(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """获取用户的所有对话"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            SELECT id, session_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit)
        )
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                "id": row['id'],
                "session_id": row['session_id'],
                "title": row['title'],
                "created_at": row['created_at'],
                "updated_at": row['updated_at']
            })
        
        return conversations
    finally:
        conn.close()


def get_conversation_messages(conversation_id: int) -> List[Dict[str, Any]]:
    """获取对话的所有消息"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            SELECT id, role, content, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,)
        )
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                "id": row['id'],
                "role": row['role'],
                "content": row['content'],
                "sources": json.loads(row['sources']) if row['sources'] else [],
                "created_at": row['created_at']
            })
        
        return messages
    finally:
        conn.close()


def delete_conversation(conversation_id: int, user_id: int) -> bool:
    """删除对话"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id)
        )
        
        if not cursor.fetchone():
            return False
        
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        
        return True
    finally:
        conn.close()


def update_conversation_title(conversation_id: int, user_id: int, title: str) -> bool:
    """更新对话标题"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title, time.time(), conversation_id, user_id)
        )
        conn.commit()
        
        return cursor.rowcount > 0
    finally:
        conn.close()


# ============ 用户文档管理 ============

def add_user_document(user_id: int, filename: str, original_filename: str, 
                      file_path: str, file_size: int) -> int:
    """添加用户文档记录"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        upload_time = time.time()
        cursor.execute(
            """INSERT INTO user_documents 
               (user_id, filename, original_filename, file_path, file_size, upload_time, status)
               VALUES (?, ?, ?, ?, ?, ?, 'processing')""",
            (user_id, filename, original_filename, file_path, file_size, upload_time)
        )
        document_id = cursor.lastrowid
        conn.commit()
        
        return document_id
    finally:
        conn.close()


def update_document_status(document_id: int, status: str, error_message: str = None) -> bool:
    """更新文档状态"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE user_documents SET status = ?, error_message = ? WHERE id = ?",
            (status, error_message, document_id)
        )
        conn.commit()
        
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_user_documents(user_id: int) -> List[Dict[str, Any]]:
    """获取用户的所有文档"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """SELECT id, filename, original_filename, file_size, upload_time, status
               FROM user_documents
               WHERE user_id = ?
               ORDER BY upload_time DESC""",
            (user_id,)
        )
        
        rows = cursor.fetchall()
        documents = []
        
        for row in rows:
            documents.append({
                "id": row["id"],
                "filename": row["filename"],
                "original_filename": row["original_filename"],
                "file_size": row["file_size"],
                "upload_time": row["upload_time"],
                "status": row["status"]
            })
        
        return documents
    finally:
        conn.close()


def get_document_by_id(document_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """获取单个文档信息"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """SELECT id, user_id, filename, original_filename, file_path, file_size, 
                      upload_time, status, error_message
               FROM user_documents
               WHERE id = ? AND user_id = ?""",
            (document_id, user_id)
        )
        
        row = cursor.fetchone()
        
        if row:
            return {
                "id": row["id"],
                "user_id": row["user_id"],
                "filename": row["filename"],
                "original_filename": row["original_filename"],
                "file_path": row["file_path"],
                "file_size": row["file_size"],
                "upload_time": row["upload_time"],
                "status": row["status"],
                "error_message": row["error_message"]
            }
        
        return None
    finally:
        conn.close()


def delete_user_document(document_id: int, user_id: int) -> bool:
    """删除用户文档"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT file_path FROM user_documents WHERE id = ? AND user_id = ?",
            (document_id, user_id)
        )
        
        row = cursor.fetchone()
        
        if not row:
            return False
        
        file_path = row["file_path"]
        
        cursor.execute("DELETE FROM user_documents WHERE id = ?", (document_id,))
        conn.commit()
        
        return True, file_path
    finally:
        conn.close()


def get_user_document_count(user_id: int) -> int:
    """获取用户文档数量"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT COUNT(*) as count FROM user_documents WHERE user_id = ?",
            (user_id,)
        )
        
        result = cursor.fetchone()
        return result["count"]
    finally:
        conn.close()


init_database()