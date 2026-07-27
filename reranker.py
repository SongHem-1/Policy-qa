"""重排序模块 - 提升检索精度"""
from typing import List, Tuple
import sys
import os
from pathlib import Path

# 在导入任何HuggingFace相关库之前设置镜像和环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 强制使用离线模式，完全禁用网络检查
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# 设置HuggingFace Hub的镜像
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker


def get_local_model_path(model_name: str) -> str:
    """获取本地模型缓存路径
    
    Args:
        model_name: 模型名称，如 "BAAI/bge-reranker-base"
    
    Returns:
        本地模型路径
    """
    # 转换模型名称为缓存目录格式
    # BAAI/bge-reranker-base -> models--BAAI--bge-reranker-base
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = "models--" + model_name.replace("/", "--")
    model_dir = cache_dir / model_dir_name / "snapshots"
    
    if model_dir.exists():
        # 获取最新的snapshot
        snapshots = list(model_dir.iterdir())
        if snapshots:
            return str(snapshots[0])
    
    return model_name


def create_reranker(top_k: int = 5, threshold: float = 0.0):
    """创建重排序器
    
    Args:
        top_k: 返回的文档数量
        threshold: 相关性阈值（低于此值的文档将被过滤）
    
    Returns:
        (CrossEncoder模型, CrossEncoderReranker)
    """
    from sentence_transformers import CrossEncoder
    
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    
    # 确保镜像站点已设置
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 强制离线模式
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    print(f"使用HuggingFace镜像: {os.environ.get('HF_ENDPOINT')}")
    print(f"离线模式: 已启用（仅使用本地缓存）")
    
    # 检测设备
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except:
        pass
    
    print(f"加载重排序模型: {model_name} (设备: {device})")
    
    try:
        # 尝试获取本地模型路径
        local_path = get_local_model_path(model_name)
        
        if local_path != model_name and Path(local_path).exists():
            print(f"✅ 找到本地缓存: {local_path}")
            # 使用本地路径加载，完全避免网络请求
            model = CrossEncoder(
                local_path, 
                max_length=512, 
                device=device,
                trust_remote_code=False
            )
            print(f"✅ 从本地缓存加载成功")
        else:
            # 如果本地路径不存在，尝试从模型名称加载
            # 这可能会触发网络请求，但我们已经设置了离线模式
            raise FileNotFoundError(f"本地缓存不存在: {model_name}")
            
    except Exception as e:
        print(f"⚠️ 从缓存加载失败: {e}")
        print("尝试从镜像站点下载...")
        
        try:
            # 临时禁用离线模式进行下载
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            os.environ["HF_HUB_OFFLINE"] = "0"
            
            # 使用镜像站点的完整URL
            mirror_url = f"https://hf-mirror.com/{model_name}"
            model = CrossEncoder(
                mirror_url, 
                max_length=512, 
                device=device,
                trust_remote_code=False
            )
            
            # 下载完成后重新启用离线模式
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            print(f"✅ 从镜像下载成功")
        except Exception as e2:
            print(f"❌ 从镜像下载也失败: {e2}")
            print("\n请手动下载模型:")
            print(f"1. 访问: https://hf-mirror.com/{model_name}")
            print("2. 下载所有文件到本地目录")
            print("3. 在.env中设置: RERANKER_MODEL=./bge-reranker-base")
            raise
    
    # 创建LangChain的CrossEncoderReranker
    # 注意：CrossEncoderReranker需要的是CrossEncoder对象，我们已经创建了
    try:
        compressor = CrossEncoderReranker(
            model=model,
            top_n=top_k
        )
        print(f"✅ 重排序器创建成功（Top {top_k}）")
    except Exception as e:
        print(f"⚠️ 创建CrossEncoderReranker失败: {e}")
        print("尝试使用备用方法...")
        
        # 备用方法：直接返回模型，不使用LangChain的包装器
        # 我们将在vectorstore.py中手动实现重排序逻辑
        compressor = None
    
    return model, compressor


if __name__ == "__main__":
    # 测试重排序器
    print("=" * 60)
    print("  测试重排序器")
    print("=" * 60)
    
    try:
        model, compressor = create_reranker(top_k=3)
        
        # 创建测试文档
        docs = [
            Document(page_content="创业补贴政策：对首次创办小微企业或从事个体经营...", metadata={"source": "test1.pdf"}),
            Document(page_content="数字经济发展规划：加快数字产业化...", metadata={"source": "test2.pdf"}),
            Document(page_content="创业担保贷款：符合条件的创业者可申请...", metadata={"source": "test3.pdf"}),
        ]
        
        query = "创业补贴"
        
        print(f"\n测试查询: {query}")
        print(f"候选文档数: {len(docs)}")
        
        # 压缩文档
        compressed_docs = compressor.compress_documents(docs, query)
        
        print(f"\n重排序结果:")
        for i, doc in enumerate(compressed_docs, 1):
            print(f"  {i}. {doc.metadata['source']}: {doc.page_content[:50]}...")
        
        print("\n✅ 测试成功")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()