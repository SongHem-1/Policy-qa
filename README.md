# 政策问答系统 (Policy QA System)

基于RAG（检索增强生成）的智能政策问答系统，支持用户认证、对话历史管理和记忆分层架构。

## ✨ 核心功能

### 1. 用户认证系统
- 用户注册与登录
- 会话管理
- 密码安全存储（哈希加密）

### 2. 对话历史管理
- 基于SQLite的持久化存储
- 用户级别的对话历史
- 会话恢复功能

### 3. 记忆分层架构
- **短期记忆**：当前会话的对话历史
- **长期记忆**：向量数据库存储的历史对话摘要
- **智能检索**：根据当前问题检索相关历史信息

### 4. 混合检索策略
- **BM25关键词匹配**：精准匹配关键词（支持中文分词）
- **向量检索**：语义相似度匹配
- **权重融合**：灵活配置检索权重

### 5. 评估系统
- 离线评估：检索器性能评估（Recall@K, Precision@K, MRR）
- 端到端评估：完整RAG系统评估（使用Ragas框架）
- 线上监控：实时记录用户查询和检索结果

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Windows / Linux / macOS

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/yourusername/policy-qa.git
cd policy-qa
```

2. **创建虚拟环境**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
# 复制示例配置
copy .env.example .env

# 编辑.env文件，填入您的API密钥
# ZHIPU_API_KEY=your_api_key_here
```

5. **启动服务**
```bash
# 启动API服务
run_api.bat

# 启动Web界面（新终端）
run_web.bat
```

### 访问系统
- Web界面：http://127.0.0.1:7862
- API文档：http://127.0.0.1:8000/docs

## 📁 项目结构

```
policy-qa/
├── api.py                    # FastAPI服务
├── app.py                    # Gradio Web界面
├── qa_chain.py               # QA链构建
├── vectorstore.py            # 向量数据库管理
├── database.py               # 用户认证与对话历史
├── document_processor.py     # 文档处理与分块
├── bm25_chinese.py           # 中文BM25检索器
├── config.py                 # 配置管理
│
├── evaluation/               # 评估系统
│   ├── test_set_builder.py   # 测试集构建
│   ├── offline_evaluation.py # 离线评估
│   ├── end_to_end_evaluation.py # 端到端评估
│   └── monitoring.py         # 线上监控
│
├── data/                     # 数据目录
│   └── policy_pdfs/          # 政策PDF文件
│
├── run_api.bat               # 启动API服务
├── run_web.bat               # 启动Web界面
├── run_evaluation.bat        # 运行评估
└── requirements.txt          # 依赖列表
```

## 🔧 配置说明

### 核心配置参数

```bash
# 智谱AI配置
ZHIPU_API_KEY=your_api_key

# 嵌入模型
EMBEDDING_MODEL=./MML12-v2  # 本地模型路径

# 混合检索权重
BM25_WEIGHT=0.2              # BM25权重
VECTOR_WEIGHT=0.8            # 向量检索权重

# 分块策略
CHUNK_BY_SECTION=True        # 按章节分块
CHUNK_SIZE=500               # 分块大小
CHUNK_OVERLAP=50             # 重叠大小
```

### 检索优化

1. **混合检索**：结合BM25和向量检索
2. **中文分词**：使用jieba进行专业中文分词
3. **按章节分块**：保持语义完整性
4. **权重调优**：根据评估结果调整权重

## 📊 评估系统

### 运行评估

```bash
# 运行完整评估流程
run_evaluation.bat

# 选择评估步骤：
# 1. 运行完整评估
# 2. 仅运行离线评估（检索器）
# 3. 仅运行端到端评估
# 4. 显示监控报告
# 5. 构建测试集
```

### 评估指标

- **Recall@K**：召回率（检索到的相关文档比例）
- **Precision@K**：精确率（检索结果的相关性）
- **MRR**：平均倒数排名（第一个相关文档的位置）

## 🛠️ 技术栈

- **LLM**: 智谱AI GLM-4
- **Embedding**: 本地嵌入模型（MML12-v2）
- **Vector DB**: Chroma
- **Web Framework**: Gradio + FastAPI
- **Database**: SQLite
- **Evaluation**: Ragas

## 📝 使用说明

### 用户注册与登录

1. 访问Web界面
2. 点击"注册"创建账户
3. 使用账户登录
4. 开始对话

### 对话历史管理

- 系统自动保存对话历史
- 每次登录可查看历史对话
- 支持多会话管理

### 添加新政策文档

1. 将PDF文件放入`data/policy_pdfs/`目录
2. 重启API服务（自动重建向量库）
3. 或手动运行：
```python
from vectorstore import build_vectorstore
build_vectorstore()
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma](https://github.com/chroma-core/chroma)
- [Gradio](https://github.com/gradio-app/gradio)
- [智谱AI](https://www.zhipuai.cn/)