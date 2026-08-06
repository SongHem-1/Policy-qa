FROM python:3.10-slim-bookworm

LABEL maintainer="policy-qa"
LABEL description="Policy QA API (FastAPI + RAG)"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple \
    --trusted-host mirrors.aliyun.com \
    langchain-core langchain-text-splitters langchain-community langchain \
    && pip install --no-cache-dir --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple \
    --trusted-host mirrors.aliyun.com \
    chromadb sentence-transformers "transformers>=4.30.0,<4.50" \
    && pip install --no-cache-dir --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple \
    --trusted-host mirrors.aliyun.com \
    gradio fastapi uvicorn pypdf pdfplumber PyMuPDF \
    && pip install --no-cache-dir --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple \
    --trusted-host mirrors.aliyun.com \
    jieba rank_bm25 ragas datasets python-dotenv pydantic numpy structlog redis rq tenacity \
    && pip install --no-cache-dir --retries 5 \
    -i https://mirrors.aliyun.com/pypi/simple \
    --trusted-host mirrors.aliyun.com \
    langchain-zhipu

COPY . .

RUN mkdir -p data chroma_db

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
