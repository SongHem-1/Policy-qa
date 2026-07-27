import io
from pathlib import Path
from typing import List

import fitz
from PIL import Image

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import warnings


def _mineru_ocr_pdf(pdf_path: Path) -> List[Document]:
    """使用 MinerU 进行OCR识别"""
    try:
        print("  使用 MinerU OCR...")
        
        # MinerU的正确使用方式 - 使用do_parse函数
        from mineru.cli.common import do_parse
        
        # 创建临时输出目录
        output_dir = Path(__file__).parent / "mineru_output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"  正在解析: {pdf_path.name}")
        
        # 读取PDF文件内容
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # 调用MinerU解析PDF（正确的参数格式）
        do_parse(
            pdf_file_names=[pdf_path.name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["zh"],
            output_dir=str(output_dir)
        )
        
        # 读取生成的markdown文件
        documents = []
        
        # MinerU会在output_dir中生成markdown文件
        md_files = list(output_dir.glob("*.md"))
        
        if md_files:
            print(f"  找到 {len(md_files)} 个markdown文件")
            
            for md_file in md_files:
                content = md_file.read_text(encoding='utf-8')
                
                if content.strip():
                    # 按页面分割内容
                    lines = content.split('\n')
                    current_page = 1
                    current_content = ""
                    
                    for line in lines:
                        # 检测页面分隔符
                        if line.startswith("--- Page ") or line.startswith("## Page") or line.startswith("# Page"):
                            if current_content.strip():
                                documents.append(
                                    Document(
                                        page_content=current_content.strip(),
                                        metadata={"source": pdf_path.name, "page": current_page}
                                    )
                                )
                            current_page += 1
                            current_content = ""
                        else:
                            current_content += line + "\n"
                    
                    # 添加最后一页
                    if current_content.strip():
                        documents.append(
                            Document(
                                page_content=current_content.strip(),
                                metadata={"source": pdf_path.name, "page": current_page}
                            )
                        )
                    
                    # 如果没有找到页面分隔符，将整个内容作为一页
                    if not documents and content.strip():
                        documents.append(
                            Document(
                                page_content=content.strip(),
                                metadata={"source": pdf_path.name, "page": 1}
                            )
                        )
        else:
            # 如果没有生成markdown文件，尝试读取JSON文件
            json_files = list(output_dir.glob("*.json"))
            
            if json_files:
                print(f"  找到 {len(json_files)} 个JSON文件")
                
                import json
                for json_file in json_files:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 处理JSON数据
                    if isinstance(data, dict):
                        content = data.get("content", data.get("text", ""))
                        if content:
                            documents.append(
                                Document(
                                    page_content=content,
                                    metadata={"source": pdf_path.name, "page": 1}
                                )
                            )
                    elif isinstance(data, list):
                        for page_data in data:
                            content = page_data.get("content", page_data.get("text", ""))
                            if content:
                                documents.append(
                                    Document(
                                        page_content=content,
                                        metadata={"source": pdf_path.name, "page": page_data.get("page", 1)}
                                    )
                                )
        
        print(f"  MinerU识别完成，生成 {len(documents)} 个文档")
        
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(output_dir)
        except:
            pass
        
        # 如果MinerU失败，回退到easyocr
        if not documents:
            print("  MinerU未生成文档，回退到 easyocr")
            return _easyocr_pdf(pdf_path)
        
        return documents
        
    except ImportError as e:
        print(f"  MinerU 未安装: {e}")
        print("  回退到 easyocr")
        return _easyocr_pdf(pdf_path)
    except Exception as e:
        print(f"  MinerU 失败: {type(e).__name__}: {e}")
        print("  回退到 easyocr")
        return _easyocr_pdf(pdf_path)


def _easyocr_pdf(pdf_path: Path, languages: List[str] = None) -> List[Document]:
    """使用 easyocr 进行OCR识别"""
    if languages is None:
        languages = ["ch_sim", "en"]
    
    import easyocr
    import numpy as np
    
    OCR_CACHE_DIR = Path(__file__).resolve().parent / ".easyocr_cache"
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(OCR_CACHE_DIR))
    
    import torch
    use_gpu = False
    try:
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            torch.cuda.empty_cache()
    except:
        pass
    
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
                
                max_size = 800
                width, height = image.size
                if max(width, height) > max_size:
                    ratio = max_size / max(width, height)
                    image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
                
                try:
                    ocr_lines = reader.readtext(np.array(image), detail=0, paragraph=True)
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        reader = easyocr.Reader(languages, gpu=False, model_storage_directory=str(OCR_CACHE_DIR))
                        ocr_lines = reader.readtext(np.array(image), detail=0, paragraph=True)
                    else:
                        raise
                
                text = "\n".join(ocr_lines).strip()
                print(f"  第 {page_number+1} 页OCR完成，文本长度: {len(text)}")
            except Exception as exc:
                print(f"  OCR失败：第 {page_number+1} 页，{exc}")
                continue
        
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_number + 1},
                )
            )
    
    doc.close()
    return documents


def load_and_split_pdfs(data_dir: str, chunk_size: int = 500, overlap: int = 50) -> List[Document]:
    """从 data 文件夹加载 PDF，并按指定长度与重叠分割文本块。"""
    from config import USE_MINERU
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"警告：数据目录不存在，正在创建：{data_dir}")
        data_path.mkdir(parents=True, exist_ok=True)
        return []
    
    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"提示：在目录 {data_dir} 中未找到 PDF 文件")
        return []
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件：{[f.name for f in pdf_files]}")
    
    if USE_MINERU:
        print("使用 MinerU 文档处理器")
    else:
        print("使用 easyocr 文档处理器")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
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
                print(f"  直接提取为空，尝试OCR...")
                
                # 根据配置选择OCR方式
                if USE_MINERU:
                    raw_documents = _mineru_ocr_pdf(pdf_file)
                else:
                    raw_documents = _easyocr_pdf(pdf_file)
                
                total_text = sum(len(doc.page_content.strip()) for doc in raw_documents)
                print(f"  OCR提取文本长度：{total_text}")
                
                if total_text == 0:
                    print(f"  警告：PDF文件 {pdf_file.name} 无有效文本")
                    continue
            
            for doc in raw_documents:
                doc.metadata["source"] = pdf_file.name
            
            split_docs = splitter.split_documents(raw_documents)
            documents.extend(split_docs)
            print(f"  成功生成 {len(split_docs)} 个文档块")
            
        except Exception as exc:
            print(f"错误：加载 PDF {pdf_file.name} 失败：{exc}")
    
    print(f"共生成 {len(documents)} 个文档块")
    return documents