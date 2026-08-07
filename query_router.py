"""
查询路由器 - 自适应检索策略选择 + 查询扩展
根据问题类型自动选择最优检索策略，避免一刀切的固定权重混合检索
"""

from typing import List, Dict, Optional, Any
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


QUERY_CLASSIFY_PROMPT = """判断以下政策问题属于哪种类型，只回答一个词（精确/语义/混合/多跳）：

判断规则（按优先级从高到低）：
1. 多跳：包含"同时"、"还能"、"叠加"、"是否可"、"又"、"以及"等表示多项关联的词 → 多跳
2. 混合：同时包含【具体数字/条件/税率】+【概念性词语如"优惠"、"政策"、"认定"】→ 混合
3. 精确：仅包含具体数字、条款编号、文件名、特定名称，不涉及概念解释 → 精确
4. 语义：仅涉及概念解释、定义、适用范围、概括描述，没有具体数字或条件 → 语义

示例：
- "第三条规定的税率是多少" → 精确
- "小微企业可以享受哪些税收优惠" → 语义
- "年收入低于100万的小微企业有什么增值税优惠" → 混合
- "享受A优惠的同时还能享受B优惠吗" → 多跳
- "高新技术企业15%税率需要满足什么条件" → 混合
- "高新技术企业认定后可以同时享受哪些其他优惠" → 多跳

问题：{query}
类型："""


QUERY_EXPAND_PROMPT = """将以下政策问题扩展为2个不同角度的正式检索查询，用于在政策文档库中检索：

原始问题：{query}

要求：
1. 每个查询从不同角度切入（如：定义角度、条件角度、适用范围角度）
2. 使用正式的政策术语
3. 每个查询一行，不要编号

扩展查询："""


# 规则预过滤关键词
MULTI_HOP_KEYWORDS = ["同时", "还能", "叠加", "是否可", "又", "以及", "可否", "可否同时"]
PRECISE_KEYWORDS_PATTERNS = ["第", "条规定", "第几条", "税率", "条款"]


STRATEGY_CONFIG = {
    "精确": {
        "bm25_weight": 0.8,
        "vector_weight": 0.2,
        "retrieval_k": 10,
        "description": "以关键词匹配为主，适合精确查找",
    },
    "语义": {
        "bm25_weight": 0.1,
        "vector_weight": 0.9,
        "retrieval_k": 20,
        "description": "以语义检索为主，适合概念理解",
    },
    "混合": {
        "bm25_weight": 0.5,
        "vector_weight": 0.5,
        "retrieval_k": 15,
        "description": "平衡关键词和语义，适合综合查询",
    },
    "多跳": {
        "bm25_weight": 0.5,
        "vector_weight": 0.5,
        "retrieval_k": 20,
        "description": "扩大召回范围，适合复杂推理",
    },
}


def _classify_by_rules(query: str) -> Optional[str]:
    """基于规则的快速预分类，命中则直接返回，未命中返回 None 走 LLM 分类

    优先级：多跳 > 精确（规则命中率高，LLM 分类准确率低）
    """
    # 规则1: 多跳关键词检测
    for kw in MULTI_HOP_KEYWORDS:
        if kw in query:
            print(f"   ⚡ 规则命中(多跳): 关键词 '{kw}'")
            return "多跳"

    # 规则2: 精确查询特征（纯数字/条款查询，不包含概念词）
    has_number = any(c.isdigit() for c in query)
    has_article = any(p in query for p in ["第", "条", "款"])
    has_concept = any(kw in query for kw in ["优惠", "政策", "认定", "标准", "如何", "什么", "哪些", "怎样"])
    has_mixed = has_number and has_concept

    if has_mixed:
        return None  # 可能混合，交给 LLM
    if has_article or (has_number and not has_concept):
        return None  # 可能精确，但 LLM 更准确

    return None  # 默认走 LLM


def classify_query(query: str, llm) -> str:
    """用 LLM 判断问题类型（优先使用规则预过滤）

    Args:
        query: 用户问题（或经过历史感知重写后的独立查询）
        llm: 语言模型实例

    Returns:
        问题类型: "精确" / "语义" / "混合" / "多跳"
    """
    # 先尝试规则预过滤
    rule_result = _classify_by_rules(query)
    if rule_result:
        print(f"   📌 查询分类: {rule_result} ({STRATEGY_CONFIG[rule_result]['description']})")
        return rule_result

    # 规则未命中，走 LLM 分类
    prompt = QUERY_CLASSIFY_PROMPT.format(query=query)
    try:
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            raw = result.content.strip()
        else:
            raw = str(result).strip()

        for valid_type in ["多跳", "混合", "精确", "语义"]:
            if valid_type in raw:
                print(f"   📌 查询分类: {valid_type} ({STRATEGY_CONFIG[valid_type]['description']})")
                return valid_type

        print(f"   ⚠️ 无法识别查询类型 '{raw}'，默认使用混合策略")
        return "混合"
    except Exception as e:
        print(f"   ⚠️ 查询分类失败: {e}，默认使用混合策略")
        return "混合"


def expand_queries(query: str, llm) -> List[str]:
    """用 LLM 生成多个查询变体，提高召回率

    Args:
        query: 用户问题
        llm: 语言模型实例

    Returns:
        查询变体列表（包含原始查询）
    """
    prompt = QUERY_EXPAND_PROMPT.format(query=query)
    try:
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            raw = result.content.strip()
        else:
            raw = str(result).strip()

        expanded = [line.strip() for line in raw.split("\n") if line.strip()]
        expanded = [q for q in expanded if len(q) > 3 and "：" not in q[:5]]

        if expanded:
            print(f"   📝 查询扩展: {len(expanded)} 个变体")
            for i, q in enumerate(expanded, 1):
                print(f"      {i}. {q}")
            return [query] + expanded
        else:
            return [query]
    except Exception as e:
        print(f"   ⚠️ 查询扩展失败: {e}，使用原始查询")
        return [query]


def _deduplicate_docs(docs: List[Document]) -> List[Document]:
    """文档去重（按来源+内容前80字符去重）"""
    seen = set()
    unique = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        content_preview = doc.page_content[:80]
        key = f"{source}::{content_preview}"
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


class AdaptiveRetriever(BaseRetriever):
    """自适应检索器

    根据问题类型自动选择检索策略：
    - 精确查询 → BM25 为主 (0.8/0.2)
    - 语义查询 → 向量检索为主 (0.1/0.9)
    - 混合查询 → 平衡两者 (0.5/0.5)
    - 多跳查询 → 扩大召回范围

    同时支持查询扩展，生成多个查询变体提高召回率。
    """

    retrievers: Dict[str, BaseRetriever] = Field(
        default_factory=dict, description="各策略对应的检索器"
    )
    llm: Any = Field(default=None, description="用于查询分类和扩展的 LLM")
    use_expansion: bool = Field(default=True, description="是否启用查询扩展")
    default_strategy: str = Field(default="混合", description="默认策略")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """自适应检索主流程

        1. 分类查询类型（规则优先 + LLM 兜底）
        2. 扩展查询（精确查询跳过，节省时间）
        3. 选择对应检索器
        4. 并行检索并合并去重
        """
        print(f"\n{'='*60}")
        print(f"🧠 自适应检索开始")
        print(f"   原始查询: {query[:80]}...")

        # 步骤1: 查询分类
        query_type = classify_query(query, self.llm)
        strategy = STRATEGY_CONFIG.get(query_type, STRATEGY_CONFIG[self.default_strategy])
        print(f"   策略: {query_type} ({strategy['description']})")

        # 步骤2: 查询扩展（精确查询和语义查询跳过，语义查询本身已覆盖概念空间）
        if self.use_expansion and query_type not in ("精确", "语义"):
            queries = expand_queries(query, self.llm)
            print(f"   ⚡ 查询扩展: {len(queries)} 个查询（含原始）")
        else:
            queries = [query]
            if query_type == "精确":
                print(f"   ⚡ 精确查询跳过扩展，节省时间")
            elif query_type == "语义":
                print(f"   ⚡ 语义查询跳过扩展（向量检索已覆盖语义空间），节省时间")

        # 步骤3: 选择检索器
        retriever = self.retrievers.get(query_type, self.retrievers.get(self.default_strategy))
        if retriever is None:
            print(f"   ❌ 未找到策略 '{query_type}' 对应的检索器，使用第一个可用检索器")
            retriever = list(self.retrievers.values())[0]

        # 步骤4: 并行检索
        all_docs = []
        if len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
                futures = {executor.submit(retriever.invoke, q): (i, q) for i, q in enumerate(queries)}
                for future in as_completed(futures):
                    i, q = futures[future]
                    try:
                        docs = future.result()
                        all_docs.extend(docs)
                        label = "原始查询" if i == 0 else f"扩展查询 {i}"
                        print(f"   {label}检索到 {len(docs)} 个文档")
                    except Exception as e:
                        print(f"   ⚠️ 查询 '{q[:30]}...' 检索失败: {e}")
        else:
            try:
                docs = retriever.invoke(queries[0])
                all_docs.extend(docs)
                print(f"   检索到 {len(docs)} 个文档")
            except Exception as e:
                print(f"   ⚠️ 检索失败: {e}")

        # 步骤5: 去重
        unique_docs = _deduplicate_docs(all_docs)
        print(f"   ✅ 合并去重后共 {len(unique_docs)} 个文档")
        print(f"{'='*60}\n")

        return unique_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索"""
        return self._get_relevant_documents(query)