"""支持中文分词的BM25检索器"""
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
from pydantic import Field

try:
    import jieba
    JIEBA_AVAILABLE = True
    print("✅ jieba分词已加载")
except ImportError:
    JIEBA_AVAILABLE = False
    print("⚠️ jieba未安装，使用简单分词")


def chinese_tokenizer(text: str) -> List[str]:
    """中文分词器"""
    if JIEBA_AVAILABLE:
        return list(jieba.cut(text))
    else:
        # 简单分词：按字符分割
        return [char for char in text if char.strip()]


class ChineseBM25Retriever(BaseRetriever):
    """支持中文的BM25检索器（LangChain接口）"""
    
    documents: List[Document] = Field(default_factory=list)
    bm25: BM25Okapi = Field(default=None)
    k: int = Field(default=5)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, documents: List[Document], k: int = 5, **kwargs):
        """
        Args:
            documents: 文档列表
            k: 返回文档数量
        """
        super().__init__(documents=documents, k=k, **kwargs)
        
        # 对文档进行分词
        tokenized_corpus = []
        print(f"对 {len(documents)} 个文档进行中文分词...")
        for doc in documents:
            tokens = chinese_tokenizer(doc.page_content)
            tokenized_corpus.append(tokens)
        
        # 创建BM25索引
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ BM25索引创建完成")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """检索相关文档（LangChain接口）
        
        Args:
            query: 查询文本
        
        Returns:
            相关文档列表
        """
        # 对查询进行分词
        query_tokens = chinese_tokenizer(query)
        
        # 使用BM25检索
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取Top K文档的索引
        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.k]
        
        # 返回对应的文档
        return [self.documents[i] for i in top_k_indices]


# 保持向后兼容
ChineseBM25 = ChineseBM25Retriever


if __name__ == "__main__":
    # 测试中文BM25
    print("=" * 60)
    print("  测试中文BM25检索器")
    print("=" * 60)
    
    # 创建测试文档
    docs = [
        Document(page_content="创业补贴政策：对首次创办小微企业或从事个体经营，正常运营1年以上的创业者，给予一次性创业补贴。", metadata={"source": "test1.pdf"}),
        Document(page_content="数字经济发展规划：加快数字产业化，推动数字技术与实体经济深度融合。", metadata={"source": "test2.pdf"}),
        Document(page_content="创业担保贷款：符合条件的创业者可申请最高20万元的创业担保贷款。", metadata={"source": "test3.pdf"}),
    ]
    
    # 创建检索器
    bm25 = ChineseBM25Retriever(docs, k=2)
    
    # 测试查询
    query = "创业补贴"
    print(f"\n查询: {query}")
    
    results = bm25._get_relevant_documents(query)
    print(f"\n检索结果 (Top 2):")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.metadata['source']}: {doc.page_content[:50]}...")
    
    print("\n✅ 测试完成")