"""
带用户认证和对话历史持久化的 Gradio 界面

功能：
- 用户注册/登录
- 对话历史保存
- 多对话管理
- 用户文档上传与管理
"""
import os
import sys
import time
import threading
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import gradio as gr

from config import LLM_MODEL, validate_config

import database

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

_current_user: Optional[Dict[str, Any]] = None
_current_conversation_id: Optional[int] = None
_current_session_id: Optional[str] = None        # 当前对话对应的会话ID（登录后独立创建）
_conversation_sessions: Dict[int, str] = {}      # conversation_id -> session_id 映射


def api_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """发送 API 请求"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    
    try:
        if method == "GET":
            req = urllib.request.Request(url, headers=headers)
        else:
            json_data = json.dumps(data).encode('utf-8') if data else b'{}'
            req = urllib.request.Request(url, data=json_data, headers=headers, method=method)
        
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        try:
            return json.loads(error_body)
        except:
            return {"success": False, "message": f"HTTP {e.code}: {error_body}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def register_user(username: str, password: str) -> Tuple[str, str]:
    """用户注册"""
    if not username or len(username) < 3:
        return "用户名至少需要3个字符", ""
    if not password or len(password) < 6:
        return "密码至少需要6个字符", ""
    
    result = api_request("/auth/register", "POST", {
        "username": username,
        "password": password
    })
    
    if result.get("success"):
        return f"✅ 注册成功！用户ID: {result.get('user_id')}", ""
    else:
        return f"❌ {result.get('message', '注册失败')}", ""


def login_user(username: str, password: str) -> Tuple[str, str, gr.update, gr.update, gr.update, str]:
    """用户登录"""
    global _current_user, _current_session_id
    
    if not username or not password:
        return "请输入用户名和密码", "", gr.update(), gr.update(), gr.update(), ""
    
    result = api_request("/auth/login", "POST", {
        "username": username,
        "password": password
    })
    
    if result.get("success"):
        _current_user = {
            "user_id": result.get("user_id"),
            "username": result.get("username")
        }
        
        # 每个登录用户创建独立会话，不再共用 web_session
        session_res = api_request("/session", "POST", {})
        _current_session_id = session_res.get("session_id")
        
        conversations = load_user_conversations()
        
        return (
            f"✅ 登录成功！欢迎 {username}",
            "",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(choices=conversations, value=None),
            ""  # 清除上传状态
        )
    else:
        return f"❌ {result.get('message', '登录失败')}", "", gr.update(), gr.update(), gr.update(), ""


def logout_user() -> Tuple[str, gr.update, gr.update, gr.update, gr.update, str]:
    """用户登出"""
    global _current_user, _current_conversation_id, _current_session_id, _conversation_sessions
    
    _current_user = None
    _current_conversation_id = None
    _current_session_id = None
    _conversation_sessions = {}
    
    return (
        "已登出",
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        gr.update(value=[]),
        ""  # 清除上传状态
    )


def load_user_conversations() -> List[str]:
    """加载用户的对话列表"""
    global _current_user, _conversation_sessions
    
    if not _current_user:
        return []
    
    result = api_request(f"/conversations?user_id={_current_user['user_id']}")
    
    if isinstance(result, list):
        conversations = []
        _conversation_sessions.clear()
        for conv in result:
            _conversation_sessions[conv["id"]] = conv.get("session_id", "")
            conversations.append(f"[{conv['id']}] {conv['title']} ({conv['message_count']}条消息)")
        return conversations
    return []


def create_new_conversation() -> Tuple[str, gr.update]:
    """创建新对话"""
    global _current_user, _current_conversation_id, _current_session_id, _conversation_sessions
    
    if not _current_user:
        return "请先登录", gr.update()
    
    result = api_request(f"/conversations?user_id={_current_user['user_id']}", "POST", {})
    
    if result.get("id"):
        _current_conversation_id = result["id"]
        # 新对话使用服务端分配的独立 session_id，多轮上下文互不串扰
        _current_session_id = result.get("session_id") or _current_session_id
        if _current_session_id:
            _conversation_sessions[_current_conversation_id] = _current_session_id
        conversations = load_user_conversations()
        return f"✅ 创建新对话 (ID: {_current_conversation_id})", gr.update(choices=conversations, value=None)
    
    return f"❌ 创建失败: {result.get('message', '未知错误')}", gr.update()


def select_conversation(choice: str) -> Tuple[List, str]:
    """选择对话"""
    global _current_conversation_id, _current_session_id
    
    if not choice:
        return [], ""
    
    try:
        conv_id = int(choice.split("[")[1].split("]")[0])
        _current_conversation_id = conv_id
        _current_session_id = _conversation_sessions.get(conv_id) or _current_session_id
        
        result = api_request(f"/conversations/{conv_id}")
        
        if isinstance(result, list):
            messages = []
            for msg in result:
                if msg['role'] == 'user':
                    messages.append({"role": "user", "content": msg['content']})
                else:
                    sources = msg.get('sources', [])
                    content = msg['content']
                    if sources:
                        content += "\n\n---\n\n📚 参考来源（原文）\n"
                        for j, src in enumerate(sources, 1):
                            content += f"\n📄 **{j}.** {src}\n"
                    messages.append({"role": "assistant", "content": content})
            
            return messages, f"已加载对话 {conv_id}（{len(result)} 条消息）"
    except Exception as e:
        return [], f"加载失败: {e}"
    
    return [], ""


def delete_current_conversation() -> Tuple[str, gr.update, gr.update]:
    """删除当前对话"""
    global _current_user, _current_conversation_id
    
    if not _current_user or not _current_conversation_id:
        return "请先选择对话", gr.update(), gr.update()
    
    result = api_request(
        f"/conversations/{_current_conversation_id}?user_id={_current_user['user_id']}",
        "DELETE"
    )
    
    if result.get("status") == "deleted":
        _current_conversation_id = None
        conversations = load_user_conversations()
        return "✅ 对话已删除", gr.update(choices=conversations, value=None), gr.update(value=[])
    
    return f"❌ 删除失败: {result.get('message', '未知错误')}", gr.update(), gr.update()


def answer_query(user_message: str) -> Tuple[str, List[str]]:
    """回答问题 - 通过API服务"""
    global _current_session_id

    if not user_message.strip():
        return "请输入要查询的问题。", []
    
    import re
    cleaned = re.sub(r'[？?！!。，,. ]', '', user_message)
    if not cleaned:
        return "请输入具体的问题内容。", []
    
    try:
        print(f"正在处理查询：{user_message[:50]}...")

        # 确保有独立的会话ID（登录后由 /session 创建），不再共享 web_session
        if not _current_session_id:
            session_res = api_request("/session", "POST", {})
            _current_session_id = session_res.get("session_id")
        
        request_data = {
            "question": user_message,
            "session_id": _current_session_id or "",
            "include_history": True
        }
        
        if _current_user:
            request_data["user_id"] = _current_user["user_id"]
        
        result = api_request("/query", "POST", request_data)
        
        print(f"API返回结果: {result}")
        
        if result.get("status") == "error":
            return f"API错误：{result.get('answer', '未知错误')}", []
        
        answer_text = result.get("answer", "")
        sources = result.get("sources", [])
        
        print(f"查询完成，答案长度: {len(answer_text)}, 来源数量: {len(sources)}")
        return answer_text.strip(), sources
    except Exception as exc:
        error_msg = f"回答时发生错误：{exc}"
        print(error_msg)
        return error_msg, []


def respond(user_message: str, chat_history: List):
    """响应用户消息"""
    global _current_user, _current_conversation_id
    
    print(f"=== respond函数被调用 ===")
    print(f"用户消息: {user_message}")
    print(f"当前历史长度: {len(chat_history) if chat_history else 0}")
    
    if not user_message.strip():
        return "", chat_history
    
    answer, sources = answer_query(user_message)
    
    print(f"回答内容: {answer[:100] if len(answer) > 100 else answer}...")
    
    if _current_user and _current_conversation_id:
        database.save_message(_current_conversation_id, "user", user_message)
        database.save_message(_current_conversation_id, "assistant", answer, sources)
    
    response_text = answer
    if sources:
        response_text += "\n\n---\n\n📚 参考来源（原文）\n"
        for i, src in enumerate(sources, 1):
            response_text += f"\n📄 **{i}.** {src}\n"
    
    new_history = chat_history.copy() if chat_history else []
    new_history.append({"role": "user", "content": user_message})
    new_history.append({"role": "assistant", "content": response_text})
    
    print(f"新历史长度: {len(new_history)}")
    print(f"=== respond函数结束 ===")
    
    return "", new_history


def refresh_files():
    """刷新知识库 - 通过API服务（增量更新）"""
    try:
        result = api_request("/rebuild", "POST", {})
        
        status = result.get("status", "")
        message = result.get("message", "")
        
        if status == "success":
            return "✅ 知识库重建已启动，请稍候...", gr.update()
        elif status == "started":
            return f"✅ {message}\n\n⏳ 正在后台处理，请等待约10秒后刷新页面查看结果...", gr.update()
        elif status == "ready":
            return f"ℹ️ {message}", gr.update()
        elif status == "loading":
            return f"⏳ {message}", gr.update()
        else:
            return f"❌ 刷新失败：{message}", gr.update()
    except Exception as e:
        return f"❌ 刷新失败：{e}", gr.update()


def force_rebuild():
    """强制完全重建知识库 - 通过API服务"""
    try:
        result = api_request("/rebuild?force=true", "POST", {})
        
        status = result.get("status", "")
        message = result.get("message", "")
        
        if status == "started":
            return f"🔨 {message}\n\n⏳ 正在完全重建知识库，请等待约30秒-2分钟后刷新页面查看结果...", gr.update()
        elif status == "loading":
            return f"⏳ {message}", gr.update()
        else:
            return f"❌ 强制重建失败：{message}", gr.update()
    except Exception as e:
        return f"❌ 强制重建失败：{e}", gr.update()


def upload_user_document(file):
    """上传用户文档"""
    global _current_user
    
    if not _current_user:
        return "❌ 请先登录", gr.update()
    
    if file is None:
        return "❌ 请选择文件", gr.update()
    
    try:
        user_id = _current_user.get("user_id")
        if not user_id:
            return "❌ 用户信息错误，请重新登录", gr.update()
        
        url = f"{API_BASE_URL}/users/{user_id}/documents"
        
        files = {"file": (os.path.basename(file), open(file, "rb"), "application/pdf")}
        
        response = requests.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return f"✅ 文档上传成功！\n文件：{result.get('filename')}\n状态：{result.get('status')}\n\n⏳ 正在后台处理，请等待约10秒后刷新列表查看结果...", gr.update()
        else:
            error_detail = response.json().get("detail", "未知错误")
            return f"❌ 上传失败：{error_detail}", gr.update()
            
    except Exception as e:
        return f"❌ 上传失败：{str(e)}", gr.update()


def get_user_documents():
    """获取用户文档列表"""
    global _current_user
    
    if not _current_user:
        return []
    
    try:
        user_id = _current_user.get("user_id")
        if not user_id:
            return []
        
        result = api_request(f"/users/{user_id}/documents", "GET")
        
        documents = result.get("documents", [])
        return [f"{doc['original_filename']} (ID: {doc['id']}, 大小: {doc['file_size']/1024:.1f}KB)" 
                for doc in documents]
    except Exception as e:
        print(f"获取用户文档失败: {e}")
        return []


def delete_user_document(doc_info):
    """删除用户文档"""
    global _current_user
    
    if not _current_user:
        return "❌ 请先登录", gr.update()
    
    if not doc_info:
        return "❌ 请选择要删除的文档", gr.update()
    
    try:
        import re
        match = re.search(r'ID: (\d+)', doc_info)
        if not match:
            return "❌ 无法解析文档ID", gr.update()
        
        doc_id = int(match.group(1))
        user_id = _current_user.get("user_id")
        
        if not user_id:
            return "❌ 用户信息错误，请重新登录", gr.update()
        
        result = api_request(f"/users/{user_id}/documents/{doc_id}", "DELETE")
        
        return f"✅ 文档已删除", gr.update(choices=get_user_documents())
        
    except Exception as e:
        return f"❌ 删除失败：{str(e)}", gr.update()


def get_public_files():
    """获取公用文件列表"""
    try:
        result = api_request("/files", "GET")
        files = result.get("files", [])
        return f"📚 公用文件库（共 {len(files)} 个文件）:\n\n" + "\n".join([f"• {f}" for f in files])
    except Exception as e:
        return f"❌ 获取失败：{str(e)}"


def get_user_files_display():
    """获取用户文件显示"""
    global _current_user
    
    if not _current_user:
        return "❌ 请先登录"
    
    try:
        user_id = _current_user.get("user_id")
        if not user_id:
            return "❌ 用户信息错误，请重新登录"
        
        result = api_request(f"/users/{user_id}/documents", "GET")
        
        documents = result.get("documents", [])
        if not documents:
            return "📁 您的个人文件库为空\n\n点击下方\"上传文档\"按钮添加文档"
        
        files_info = []
        for doc in documents:
            size_kb = doc['file_size'] / 1024
            upload_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(doc['upload_time']))
            status = "✅ 已处理" if doc['status'] == 'completed' else "⏳ 处理中"
            files_info.append(f"• {doc['original_filename']}\n  大小: {size_kb:.1f}KB | 上传时间: {upload_time} | {status}")
        
        return f"📁 您的个人文件库（共 {len(documents)} 个文件）:\n\n" + "\n\n".join(files_info)
    except Exception as e:
        return f"❌ 获取失败：{str(e)}"


def get_system_status():
    """获取系统状态 - 通过API服务"""
    try:
        result = api_request("/health", "GET")
        
        chain_ready = result.get("chain_ready", False)
        chain_building = result.get("chain_building", False)
        loaded_files = result.get("loaded_files", [])
        
        if chain_ready:
            status = "✅ 已就绪"
        elif chain_building:
            status = "⏳ 正在初始化..."
        else:
            status = "❌ 未初始化"
        
        return f"""
### 系统状态
- **状态**: {status}
- **已加载文件**: {len(loaded_files)} 个
- **文档处理器**: {result.get('doc_processor', 'unknown')}
- **语言模型**: {LLM_MODEL}
"""
    except Exception as e:
        return f"""
### 系统状态
- **状态**: ❌ 无法连接API服务
- **错误**: {e}
"""


with gr.Blocks(title="国家政策知识库智能问答") as demo:
    gr.Markdown("# 🏛️ 国家政策知识库智能问答系统")
    gr.Markdown("基于 RAG 技术的政策文档智能检索与问答系统")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 用户认证")
            
            login_status = gr.Textbox(label="状态", value="未登录", interactive=False)
            
            with gr.Group(visible=True) as login_group:
                username_input = gr.Textbox(label="用户名", placeholder="请输入用户名")
                password_input = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                
                with gr.Row():
                    login_btn = gr.Button("登录", variant="primary")
                    register_btn = gr.Button("注册")
                
                auth_message = gr.Textbox(label="消息", interactive=False)
            
            with gr.Group(visible=False) as user_group:
                user_info = gr.Textbox(label="用户信息", interactive=False)
                logout_btn = gr.Button("登出")
                
                with gr.Tabs():
                    with gr.TabItem("💬 对话管理"):
                        conversation_list = gr.Dropdown(label="历史对话", choices=[], interactive=True)
                        conversation_status = gr.Textbox(label="对话状态", interactive=False)
                        
                        with gr.Row():
                            new_conv_btn = gr.Button("新建对话", variant="primary")
                            delete_conv_btn = gr.Button("删除当前对话")
                    
                    with gr.TabItem("📁 文档管理"):
                        with gr.Tabs():
                            with gr.TabItem("📚 公用文件库"):
                                public_files_display = gr.Markdown(value=get_public_files)
                                refresh_public_btn = gr.Button("🔄 刷新")
                            
                            with gr.TabItem("👤 个人文件库"):
                                user_files_display = gr.Markdown(value=get_user_files_display)
                                
                                with gr.Row():
                                    user_doc_dropdown = gr.Dropdown(
                                        label="我的文档",
                                        choices=[],
                                        interactive=True
                                    )
                                    refresh_user_docs_btn = gr.Button("🔄 刷新列表")
                                
                                with gr.Row():
                                    upload_file = gr.File(
                                        label="上传PDF文档",
                                        file_types=[".pdf"],
                                        type="filepath"
                                    )
                                    upload_status = gr.Textbox(label="上传状态", interactive=False)
                                
                                with gr.Row():
                                    upload_btn = gr.Button("📤 上传文档", variant="primary")
                                    delete_btn = gr.Button("🗑️ 删除文档")
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 对话窗口")
            
            chatbot = gr.Chatbot(height=500)
            
            with gr.Row():
                user_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入您的问题...",
                    scale=4
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新知识库", scale=1)
                force_rebuild_btn = gr.Button("🔨 强制完全重建", scale=1, variant="stop")
                status_display = gr.Markdown(get_system_status())
    
    register_btn.click(
        register_user,
        inputs=[username_input, password_input],
        outputs=[auth_message, password_input]
    )
    
    login_btn.click(
        login_user,
        inputs=[username_input, password_input],
        outputs=[login_status, auth_message, login_group, user_group, conversation_list, upload_status]
    ).then(
        lambda: gr.update(choices=get_user_documents()),
        outputs=[user_doc_dropdown]
    ).then(
        lambda: gr.update(value=get_user_files_display()),
        outputs=[user_files_display]
    )
    
    logout_btn.click(
        logout_user,
        outputs=[login_status, login_group, user_group, conversation_list, chatbot, upload_status]
    )
    
    new_conv_btn.click(
        create_new_conversation,
        outputs=[conversation_status, conversation_list]
    )
    
    conversation_list.change(
        select_conversation,
        inputs=[conversation_list],
        outputs=[chatbot, conversation_status]
    )
    
    delete_conv_btn.click(
        delete_current_conversation,
        outputs=[conversation_status, conversation_list, chatbot]
    )
    
    submit_btn.click(
        respond,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )
    
    user_input.submit(
        respond,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )
    
    refresh_btn.click(
        refresh_files,
        outputs=[conversation_status, chatbot]
    ).then(
        lambda: None,
        inputs=None,
        outputs=None
    ).then(
        lambda: gr.update(value=get_system_status()),
        outputs=[status_display]
    )
    
    force_rebuild_btn.click(
        force_rebuild,
        outputs=[conversation_status, chatbot]
    ).then(
        lambda: None,
        inputs=None,
        outputs=None
    ).then(
        lambda: gr.update(value=get_system_status()),
        outputs=[status_display]
    )
    
    upload_btn.click(
        upload_user_document,
        inputs=[upload_file],
        outputs=[upload_status, user_doc_dropdown]
    ).then(
        lambda: gr.update(value=get_user_files_display()),
        outputs=[user_files_display]
    )
    
    delete_btn.click(
        delete_user_document,
        inputs=[user_doc_dropdown],
        outputs=[upload_status, user_doc_dropdown]
    ).then(
        lambda: gr.update(value=get_user_files_display()),
        outputs=[user_files_display]
    )
    
    refresh_user_docs_btn.click(
        lambda: gr.update(choices=get_user_documents()),
        outputs=[user_doc_dropdown]
    ).then(
        lambda: gr.update(value=get_user_files_display()),
        outputs=[user_files_display]
    )
    
    refresh_public_btn.click(
        lambda: gr.update(value=get_public_files()),
        outputs=[public_files_display]
    )
    
    demo.load(
        lambda: gr.update(value=get_system_status()),
        outputs=[status_display]
    )


if __name__ == "__main__":
    validate_config()
    
    print("=" * 50)
    print("  国家政策知识库智能问答系统（带用户认证）")
    print("=" * 50)
    print()
    print("正在启动 Gradio Web 服务...")
    print(f"启动后请在浏览器访问：http://127.0.0.1:7862")
    print()
    
    demo.launch(
        share=False,
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7862")),
        favicon_path=None,
    )
