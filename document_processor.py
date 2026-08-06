import io
import re
from pathlib import Path
from typing import List

import fitz
import numpy as np
from PIL import Image

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import warnings
import json
import time

from config import USE_KEYWORD_EXTRACT, compute_build_fingerprint


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


def extract_section_title(text: str, source_name: str = "") -> str:
    """从文本中提取章节标题
    
    Args:
        text: 文档块文本
        source_name: 文件名（用于fallback）
    
    Returns:
        提取到的章节标题
    """
    section_patterns = [
        (r'第[一二三四五六七八九十百零]+章\s*.+', 20),       # 第X章 XXX
        (r'第[一二三四五六七八九十百零]+节\s*.+', 20),       # 第X节 XXX
        (r'第[一二三四五六七八九十百零]+条\s*', 15),          # 第X条
        (r'[一二三四五六七八九十百零]+[、．.．]\s*.{2,30}', 30),  # 一、XXX
        (r'\d+[、．.．]\s*.{2,30}', 30),                      # 1. XXX
        (r'（[一二三四五六七八九十]+）\s*.{2,30}', 30),       # （一）XXX
        (r'\(\d+\)\s*.{2,30}', 30),                           # (1) XXX
    ]
    
    for pattern, max_len in section_patterns:
        match = re.search(pattern, text)
        if match:
            title = match.group(0).strip()
            if len(title) <= max_len:
                return title
    
    return source_name if source_name else ""


def extract_keywords(text: str, llm=None) -> str:
    """使用LLM提取文本中的关键词
    
    Args:
        text: 文本内容
        llm: LLM实例（可选，如果为None则使用规则提取）
    
    Returns:
        逗号分隔的关键词
    """
    if llm is None:
        return _extract_keywords_by_rules(text)
    
    try:
        prompt = f"""从以下政策文本中提取3-5个核心关键词，以逗号分隔。
只输出关键词，不要解释。

文本：{text[:500]}

关键词："""
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            keywords = response.content.strip()
        else:
            keywords = str(response).strip()
        keywords = keywords.replace('\n', ' ').replace('，', ',')
        if len(keywords) > 100:
            return _extract_keywords_by_rules(text)
        return keywords
    except Exception:
        return _extract_keywords_by_rules(text)


def _extract_keywords_by_rules(text: str) -> str:
    """规则提取关键词（作为LLM提取的fallback）"""
    keyword_patterns = [
        (r'《([^》]+)》', 1),           # 《XX法》
        (r'第[一二三四五六七八九十百零]+[章节条款]', 1),
        (r'(国务|交通|财政|税务|海关|工商|环保|住建|教育|卫生|民政|司法|公安|水利|农业|商务|文旅)(院|部|局|委|办)', 1),
        (r'(行政|刑事|民事|经济)(处罚|许可|强制|复议|诉讼|赔偿|责任)', 1),
        (r'(企业|公司|船舶|车辆|设备|土地|房屋|资金|税收|保险|贷款|合同|许可|登记|备案|审查|监管)', 2),
    ]
    
    keywords = []
    for pattern, count in keyword_patterns:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                for m in matches[:count]:
                    keywords.append(''.join(m))
            else:
                keywords.extend(matches[:count])
    
    return ','.join(dict.fromkeys(keywords[:5]))


def augment_metadata(docs: List[Document], llm=None) -> List[Document]:
    """为文档块添加丰富的元数据
    
    Args:
        docs: 文档块列表
        llm: LLM实例（用于关键词提取）
    
    Returns:
        增强元数据后的文档块列表
    """
    for doc in docs:
        text = doc.page_content
        source = doc.metadata.get("source", "")
        
        if "section_title" not in doc.metadata or not doc.metadata.get("section_title"):
            doc.metadata["section_title"] = extract_section_title(text, source)
        
        if USE_KEYWORD_EXTRACT and "keywords" not in doc.metadata:
            doc.metadata["keywords"] = extract_keywords(text, llm)
        
        if "source" not in doc.metadata and source:
            doc.metadata["source"] = source
    
    return docs


def split_parent_child(
    raw_documents: List[Document],
    source_name: str,
    parent_size: int = 2000,
    child_size: int = 500,
    child_overlap: int = 50,
    llm=None
) -> tuple:
    """父子块拆分：父块（完整条款）+ 子块（小句子），子块用于检索，父块用于回答
    
    策略：
    1. 先合并所有页面文本
    2. 按条款/章节边界切成父块（保证语义完整）
    3. 在父块内部再切分成子块
    
    Args:
        raw_documents: 原始文档列表（按页）
        source_name: 文件名
        parent_size: 父块最大大小（字符数），超过时在条款内进一步切分
        child_size: 子块大小（字符数）
        child_overlap: 子块重叠大小
        llm: LLM实例（用于关键词提取）
    
    Returns:
        (parent_docs: 父块列表, child_docs: 子块列表（带parent_id映射）)
    """
    full_text = ""
    page_map = {}
    char_pos = 0
    
    for doc in raw_documents:
        text = doc.page_content.strip()
        page = doc.metadata.get("page", 0)
        if text:
            page_map[char_pos] = page
            full_text += text + "\n"
            char_pos += len(text) + 1
    
    # 步骤0: 清理PDF跨页artifacts（如"第二十五"→"第二十五条"的跨页拆分）
    artifact_pattern = re.compile(
        r'^第[一二三四五六七八九十百零]+$'  # 孤立的"第二十五"（无"条"字）
    )
    full_section_pattern = re.compile(
        r'^第[一二三四五六七八九十百零]+[章节条款]'  # 完整的"第二十五条"
    )
    
    raw_lines = full_text.split('\n')
    cleaned_lines = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        
        # 情况1: 当前行是孤立条款号，下一行是完整条款 → 跳过低版本
        if (artifact_pattern.match(line) and 
            i + 1 < len(raw_lines) and 
            full_section_pattern.match(raw_lines[i + 1].strip())):
            # 跳过artifact，下一行是完整版
            i += 1
            continue
        
        # 情况2: 当前行以"第"开头且很短，下一行以"条"结尾且很短 → 合并
        if (line.startswith('第') and len(line) <= 5 and
            i + 1 < len(raw_lines) and
            raw_lines[i + 1].strip().endswith(('条', '章', '节')) and
            len(raw_lines[i + 1].strip()) <= 10):
            cleaned_lines.append(line + raw_lines[i + 1].strip())
            i += 2
            continue
        
        cleaned_lines.append(raw_lines[i])
        i += 1
    
    # 步骤1: 按条款边界切分父块
    section_patterns = [
        r'^第[一二三四五六七八九十百零]+[章节条款]',  # 第一章、第一节、第一条
        r'^[一二三四五六七八九十百零]+[、.]',          # 一、二、三、
        r'^\d+[、.]',                                   # 1. 2. 3.
        r'^\([一二三四五六七八九十]+\)',               # （一）（二）
        r'^\(\d+\)',                                    # （1）（2）
        r'^[（(][一二三四五六七八九十\d]+[）)]',        # 综合匹配
    ]
    
    lines = cleaned_lines
    
    parent_chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            current_chunk.append('')
            current_size += 1
            continue
        
        is_header = any(re.match(pat, line_stripped) for pat in section_patterns)
        
        if is_header and current_chunk and current_size >= 50:
            parent_chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(line_stripped)
        current_size += len(line_stripped)
        
        # 如果当前块超过父块大小，在内部找自然断点切分
        if current_size >= parent_size:
            chunk_text = '\n'.join(current_chunk)
            # 在最近的句号处切分
            cut_point = max(
                chunk_text.rfind('。', 0, parent_size),
                chunk_text.rfind('；', 0, parent_size),
                chunk_text.rfind('\n', 0, parent_size),
            )
            if cut_point > 0:
                parent_chunks.append(chunk_text[:cut_point + 1])
                remainder = chunk_text[cut_point + 1:].strip()
                current_chunk = [remainder] if remainder else []
                current_size = len(remainder) if remainder else 0
            else:
                parent_chunks.append(chunk_text)
                current_chunk = []
                current_size = 0
    
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        if len(chunk_text) >= 50:
            parent_chunks.append(chunk_text)
    
    # 步骤2: 构建父块文档
    parent_docs = []
    parent_page_map = {}
    
    for i, parent_text in enumerate(parent_chunks):
        parent_id = f"parent_{i}"
        parent_page_map[parent_id] = parent_text
        
        page_num = 0
        parent_start = full_text.find(parent_text[:50])  # 用前50字符定位
        if parent_start >= 0:
            page_positions = sorted(page_map.keys())
            for pos in page_positions:
                if pos <= parent_start + len(parent_text) // 2:
                    page_num = page_map[pos]
        
        section_title = extract_section_title(parent_text, source_name)
        
        parent_docs.append(Document(
            page_content=parent_text,
            metadata={
                "source": source_name,
                "page": page_num,
                "parent_id": parent_id,
                "section_title": section_title,
                "chunk_type": "parent",
            }
        ))
    
    # 步骤3: 在每个父块内切分孩子块
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    
    child_docs = []
    
    for parent_doc in parent_docs:
        parent_id = parent_doc.metadata["parent_id"]
        parent_text = parent_doc.page_content
        parent_page = parent_doc.metadata.get("page", 0)
        
        # 父块已经很小，直接作为子块
        if len(parent_text) <= child_size:
            child_docs.append(Document(
                page_content=parent_text,
                metadata={
                    "source": source_name,
                    "page": parent_page,
                    "parent_id": parent_id,
                    "section_title": parent_doc.metadata.get("section_title", ""),
                    "chunk_type": "child",
                }
            ))
            continue
        
        # 父块较大，切分成子块
        sub_chunks = child_splitter.split_text(parent_text)
        
        for sub_text in sub_chunks:
            if len(sub_text.strip()) < 20:
                continue
            
            child_docs.append(Document(
                page_content=sub_text,
                metadata={
                    "source": source_name,
                    "page": parent_page,
                    "parent_id": parent_id,
                    "section_title": parent_doc.metadata.get("section_title", ""),
                    "chunk_type": "child",
                }
            ))
    
    return parent_docs, child_docs


def _ocr_pdf(pdf_path: Path, languages: List[str] = None) -> List[Document]:
    if languages is None:
        languages = ["ch_sim", "en"]

    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(OCR_CACHE_DIR))

    import easyocr
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
    chunk_by_section: bool = False,
    parent_child: bool = False,
    parent_size: int = 2000,
    child_size: int = 500,
    child_overlap: int = 50,
    augment_meta: bool = True,
    llm=None
) -> List[Document]:
    """从 data 文件夹加载 PDF，并按指定长度与重叠分割文本块。
    
    Args:
        data_dir: 数据目录
        chunk_size: 块大小（固定长度分块时使用）
        overlap: 重叠大小（固定长度分块时使用）
        chunk_by_section: 是否按章节/条款分块
        parent_child: 是否使用父子块拆分
        parent_size: 父块大小
        child_size: 子块大小
        child_overlap: 子块重叠
        augment_meta: 是否增强元数据
        llm: LLM实例（用于关键词提取）
    
    Returns:
        文档块列表（父子块模式下返回子块用于入库，父块存入全局缓存）
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
    
    if parent_child:
        print(f"使用父子块拆分策略 (父块={parent_size}, 子块={child_size}, overlap={child_overlap})...")
    elif chunk_by_section:
        print("使用按章节/条款分块策略...")
    else:
        print(f"使用固定长度分块策略 (size={chunk_size}, overlap={overlap})...")
    
    documents: List[Document] = []
    parent_documents: List[Document] = []  # 存储父块，不入库

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
            
            if parent_child:
                parents, children = split_parent_child(
                    raw_documents,
                    pdf_file.name,
                    parent_size=parent_size,
                    child_size=child_size,
                    child_overlap=child_overlap,
                    llm=llm
                )
                
                if augment_meta:
                    children = augment_metadata(children, llm)
                    parents = augment_metadata(parents, llm)
                
                documents.extend(children)
                parent_documents.extend(parents)
                print(f"  成功生成 {len(parents)} 个父块 + {len(children)} 个子块")
            elif chunk_by_section:
                split_docs = split_by_section(raw_documents, max_chunk_size=chunk_size)
                if augment_meta:
                    split_docs = augment_metadata(split_docs, llm)
                documents.extend(split_docs)
                print(f"  成功生成 {len(split_docs)} 个文档块")
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                split_docs = splitter.split_documents(raw_documents)
                if augment_meta:
                    split_docs = augment_metadata(split_docs, llm)
                documents.extend(split_docs)
                print(f"  成功生成 {len(split_docs)} 个文档块")
        except Exception as exc:
            print(f"错误：加载 PDF {pdf_file.name} 失败，已跳过。错误信息：{exc}")

    if not documents:
        print("警告：未能从任何 PDF 中生成有效文档块")
        return []

    if parent_child and parent_documents:
        print(f"共生成 {len(parent_documents)} 个父块 + {len(documents)} 个子块（子块入库）")
        from config import DATA_DIR as _DATA_DIR
        try:
            import pickle
            parent_cache_path = Path(_DATA_DIR) / "_parent_documents_cache.pkl"
            with open(parent_cache_path, "wb") as f:
                pickle.dump(parent_documents, f)
            # 指纹边车：与向量库 manifest 互验，避免旧父块缓存被误用
            parent_meta_path = Path(_DATA_DIR) / "_parent_documents_cache.meta.json"
            with open(parent_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"fingerprint": compute_build_fingerprint(), "created_at": time.time()},
                    f,
                    ensure_ascii=False,
                )
            print(f"父块缓存已保存: {len(parent_documents)} 个父块 → {parent_cache_path}（指纹 {compute_build_fingerprint()}）")
        except Exception as e:
            print(f"⚠️ 父块缓存保存失败: {e}")
    else:
        print(f"共生成 {len(documents)} 个文档块")
    
    return documents
