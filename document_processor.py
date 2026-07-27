import io
import re
from pathlib import Path
from typing import List

import easyocr
import fitz
import numpy as np
from PIL import Image

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import warnings


OCR_CACHE_DIR = Path(__file__).resolve().parent / ".easyocr_cache"


def split_by_section(documents: List[Document], max_chunk_size: int = 800) -> List[Document]:
    """按章节/条款分块，保持语义完整性
    
    Args:
        documents: 原始文档列表
        max_chunk_size: 最大块大小
    
    Returns:
        分块后的文档列表
    """
    split_docs = []
    
    # 章节标题模式（中文政策文件常见格式）
    section_patterns = [
        r'^第[一二三四五六七八九十百零]+[章节条款]',  # 第一章、第一节、第一条
        r'^[一二三四五六七八九十百零]+[、.]',  # 一、二、三、
        r'^\d+[、.]',  # 1. 2. 3.
        r'^\([一二三四五六七八九十]+\)',  # （一）（二）
        r'^\(\d+\)',  # （1）（2）
        r'^[（(][一二三四五六七八九十\d]+[）)]',  # 综合匹配
    ]
    
    for doc in documents:
        text = doc.page_content
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "")
        
        # 按行分割
        lines = text.split('\n')
        
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是章节标题
            is_section_header = any(re.match(pattern, line) for pattern in section_patterns)
            
            # 如果遇到章节标题且当前块不为空，保存当前块
            if is_section_header and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) >= 50:  # 最小块大小
                    split_docs.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "source": source,
                            "page": page,
                            "chunk_id": chunk_id,
                            "is_section": True
                        }
                    ))
                    chunk_id += 1
                current_chunk = []
                current_size = 0
            
            # 添加到当前块
            current_chunk.append(line)
            current_size += len(line)
            
            # 如果超过最大大小，强制分割
            if current_size >= max_chunk_size:
                chunk_text = '\n'.join(current_chunk)
                split_docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source,
                        "page": page,
                        "chunk_id": chunk_id,
                        "is_section": False
                    }
                ))
                chunk_id += 1
                current_chunk = []
                current_size = 0
        
        # 保存最后一个块
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= 50:
                split_docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source,
                        "page": page,
                        "chunk_id": chunk_id,
                        "is_section": True
                    }
                ))
    
    return split_docs


def _ocr_pdf(pdf_path: Path, languages: List[str] = None) -> List[Document]:
    if languages is None:
        languages = ["ch_sim", "en"]

    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(OCR_CACHE_DIR))

    import torch
    use_gpu = False
    try:
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            torch.cuda.empty_cache()
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print(f"  OCR使用GPU: True")
        else:
            print(f"  OCR使用GPU: False (CUDA不可用)")
    except Exception as e:
        print(f"  GPU检测失败，将使用CPU: {e}")
        use_gpu = False
    
    reader = easyocr.Reader(
        languages,
        gpu=use_gpu,
        model_storage_directory=str(OCR_CACHE_DIR),
        download_enabled=True,
    )
    doc = fitz.open(str(pdf_path))
    documents: List[Document] = []

    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        text = page.get_text().strip()
        if not text:
            try:
                pix = page.get_pixmap(dpi=150)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                
                max_size = 1024
                width, height = image.size
                if max(width, height) > max_size:
                    ratio = max_size / max(width, height)
                    image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
                
                try:
                    ocr_lines = reader.readtext(np.array(image), detail=0)
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"  GPU失败，切换到CPU模式: {e}")
                        reader = easyocr.Reader(
                            languages,
                            gpu=False,
                            model_storage_directory=str(OCR_CACHE_DIR),
                            download_enabled=True,
                        )
                        ocr_lines = reader.readtext(np.array(image), detail=0)
                    else:
                        raise
                
                text = "\n".join(ocr_lines).strip()
                print(f"  OCR完成，识别文本长度: {len(text)}")
            except Exception as exc:
                print(f"OCR 失败：{pdf_path.name} 第 {page_number + 1} 页，{exc}")
                continue

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_number + 1},
                )
            )

    return documents


def load_and_split_pdfs(
    data_dir: str, 
    chunk_size: int = 500, 
    overlap: int = 50,
    chunk_by_section: bool = False
) -> List[Document]:
    """从 data 文件夹加载 PDF，并按指定长度与重叠分割文本块。
    
    Args:
        data_dir: 数据目录
        chunk_size: 块大小（固定长度分块时使用）
        overlap: 重叠大小（固定长度分块时使用）
        chunk_by_section: 是否按章节/条款分块
    
    Returns:
        文档块列表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"警告：数据目录不存在，正在创建：{data_dir}")
        data_path.mkdir(parents=True, exist_ok=True)
        return []

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"提示：在目录 {data_dir} 中未找到 PDF 文件，请将 PDF 文件放入该目录")
        return []

    print(f"找到 {len(pdf_files)} 个 PDF 文件：{[f.name for f in pdf_files]}")
    
    if chunk_by_section:
        print("使用按章节/条款分块策略...")
    else:
        print(f"使用固定长度分块策略 (size={chunk_size}, overlap={overlap})...")
    
    documents: List[Document] = []

    for pdf_file in pdf_files:
        try:
            print(f"正在处理：{pdf_file.name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loader = PyMuPDFLoader(str(pdf_file))
                raw_documents = loader.load()
            total_text = sum(len(doc.page_content.strip()) for doc in raw_documents)
            print(f"  直接提取文本长度：{total_text}")
            if total_text == 0:
                print(f"  直接提取为空，尝试 OCR...")
                raw_documents = _ocr_pdf(pdf_file)
                total_text = sum(len(doc.page_content.strip()) for doc in raw_documents)
                print(f"  OCR 提取文本长度：{total_text}")
                if total_text == 0:
                    print(f"  警告：PDF 文件 {pdf_file.name} OCR 结果为空，已跳过")
                    continue
            for doc in raw_documents:
                doc.metadata["source"] = pdf_file.name
            
            # 选择分块策略
            if chunk_by_section:
                split_docs = split_by_section(raw_documents, max_chunk_size=chunk_size)
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                split_docs = splitter.split_documents(raw_documents)
            
            documents.extend(split_docs)
            print(f"  成功生成 {len(split_docs)} 个文档块")
        except Exception as exc:
            print(f"错误：加载 PDF {pdf_file.name} 失败，已跳过。错误信息：{exc}")

    if not documents:
        print("警告：未能从任何 PDF 中生成有效文档块")
        return []

    print(f"共生成 {len(documents)} 个文档块")
    return documents