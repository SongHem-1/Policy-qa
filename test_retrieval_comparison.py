"""
对比测试：固定权重混合检索 vs 自适应检索（查询分类 + 查询扩展）
评估两种策略在不同类型问题上的表现差异
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain_zhipu import ChatZhipuAI

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ZHIPU_API_KEY, EMBEDDING_MODEL, DATA_DIR, PERSIST_DIRECTORY,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_BY_SECTION,
    USE_RERANKER, RERANKER_TOP_K, RERANKER_THRESHOLD,
    BM25_WEIGHT, VECTOR_WEIGHT,
    USE_ADAPTIVE_RETRIEVAL, USE_QUERY_EXPANSION,
)
from vectorstore import build_or_load_vectorstore, create_hybrid_retriever
from query_router import AdaptiveRetriever, STRATEGY_CONFIG, classify_query, expand_queries
from document_processor import load_and_split_pdfs


TEST_QUERIES = [
    # 精确查询：包含具体数字、条款、文件名
    {
        "query": "第三条规定的税率是多少",
        "expected_type": "精确",
        "description": "精确条款查询",
    },
    {
        "query": "第十六条中关于最低工资标准的规定",
        "expected_type": "精确",
        "description": "精确条款编号查询",
    },
    {
        "query": "2024年研发费用加计扣除比例",
        "expected_type": "精确",
        "description": "精确数字+政策查询",
    },
    # 语义查询：概念解释、定义、概括性描述
    {
        "query": "小微企业可以享受哪些税收优惠政策",
        "expected_type": "语义",
        "description": "概念性优惠政策查询",
    },
    {
        "query": "什么是高新技术企业认定标准",
        "expected_type": "语义",
        "description": "定义概念查询",
    },
    {
        "query": "企业如何申请出口退税",
        "expected_type": "语义",
        "description": "流程性概念查询",
    },
    # 混合查询：既有具体条件又有概念性描述
    {
        "query": "年收入低于100万的小微企业有什么增值税优惠",
        "expected_type": "混合",
        "description": "条件+优惠混合查询",
    },
    {
        "query": "高新技术企业15%税率需要满足什么条件",
        "expected_type": "混合",
        "description": "税率+条件混合查询",
    },
    # 多跳查询：需要结合多条政策
    {
        "query": "享受研发加计扣除的同时还能享受小微企业税收优惠吗",
        "expected_type": "多跳",
        "description": "多政策交叉查询",
    },
    {
        "query": "高新技术企业认定后可以同时享受哪些其他优惠",
        "expected_type": "多跳",
        "description": "多政策关联查询",
    },
]


def init_environment():
    """初始化向量库和文档（智能缓存，避免重复处理）"""
    print("=" * 70)
    print("  初始化环境...")
    print("=" * 70)

    persist_path = Path(PERSIST_DIRECTORY)
    cache_path = Path("data/_documents_cache.pkl")

    # 检查向量库是否已存在
    db_exists = persist_path.exists() and any(persist_path.iterdir())

    if db_exists and cache_path.exists():
        print("\n📂 向量库+文档缓存均存在，直接加载...")
        print("🔍 加载向量数据库...")
        vectorstore = build_or_load_vectorstore(
            [],
            persist_directory=PERSIST_DIRECTORY
        )

        import pickle
        with open(cache_path, "rb") as f:
            documents = pickle.load(f)
        print(f"   ✅ 从缓存加载了 {len(documents)} 个文档块（免重复处理）")
    else:
        if db_exists:
            print("\n📂 向量库已存在，但文档缓存缺失，重建缓存...")
        else:
            print("\n📄 首次运行，加载并处理文档...")

        documents = load_and_split_pdfs(
            DATA_DIR, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
            chunk_by_section=CHUNK_BY_SECTION
        )
        print(f"   加载了 {len(documents)} 个文档块")

        # 缓存文档（核心：下次运行秒级加载）
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(documents, f)
        print(f"   💾 文档缓存已保存 ({cache_path})")

        if not db_exists:
            print("\n🔍 构建向量数据库...")
            vectorstore = build_or_load_vectorstore(
                documents,
                persist_directory=PERSIST_DIRECTORY
            )
        else:
            print("🔍 加载已有向量数据库...")
            vectorstore = build_or_load_vectorstore(
                [],
                persist_directory=PERSIST_DIRECTORY
            )

    print(f"   向量库就绪")

    # 初始化 LLM（用于查询分类和扩展）
    print("\n🤖 初始化 LLM...")
    llm = ChatZhipuAI(api_key=ZHIPU_API_KEY, model="glm-4-flash")
    print("   LLM 就绪")

    return vectorstore, documents, llm


def build_retrievers(vectorstore, documents):
    """构建两种检索器"""
    print("\n" + "=" * 70)
    print("  构建检索器...")
    print("=" * 70)

    if documents is None:
        print("   ⚠️ 无文档缓存，仅使用纯向量检索（无 BM25）")
        # 使用纯向量检索器
        from langchain_core.retrievers import BaseRetriever
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
        fixed_retriever = vector_retriever
        adaptive_retriever = vector_retriever
        print("   ⚠️ 固定与自适应检索器均降级为纯向量检索，对比意义有限")
        return fixed_retriever, adaptive_retriever

    # 固定权重检索器（旧方案）
    print("\n📌 固定权重检索器（旧方案）...")
    fixed_retriever = create_hybrid_retriever(
        vectorstore, documents, k=3,
        use_reranker=USE_RERANKER,
        reranker_top_k=RERANKER_TOP_K,
        reranker_threshold=RERANKER_THRESHOLD,
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT
    )

    # 自适应检索器（新方案）
    print("\n📌 自适应检索器（新方案）...")
    retrievers = {}
    for strategy, cfg in STRATEGY_CONFIG.items():
        ret = create_hybrid_retriever(
            vectorstore, documents, k=3,
            use_reranker=USE_RERANKER,
            reranker_top_k=RERANKER_TOP_K,
            reranker_threshold=RERANKER_THRESHOLD,
            bm25_weight=cfg["bm25_weight"],
            vector_weight=cfg["vector_weight"]
        )
        retrievers[strategy] = ret

    llm = ChatZhipuAI(api_key=ZHIPU_API_KEY, model="glm-4-flash")
    adaptive_retriever = AdaptiveRetriever(
        retrievers=retrievers,
        llm=llm,
        use_expansion=USE_QUERY_EXPANSION,
        default_strategy="混合"
    )

    return fixed_retriever, adaptive_retriever


def run_comparison(test_queries, fixed_retriever, adaptive_retriever, llm):
    """运行对比测试"""
    results = []
    
    print("\n" + "=" * 70)
    print("  开始对比测试")
    print("=" * 70)

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_type = test_case["expected_type"]
        description = test_case["description"]

        print(f"\n{'─' * 70}")
        print(f"测试 {i}/{len(test_queries)}: {description}")
        print(f"查询: {query}")
        print(f"预期类型: {expected_type}")
        print(f"{'─' * 70}")

        result = {
            "query": query,
            "description": description,
            "expected_type": expected_type,
            "fixed": {},
            "adaptive": {},
        }

        # ========== 固定权重检索 ==========
        print("\n🔵 固定权重检索（旧方案）:")
        start_time = time.time()
        try:
            fixed_docs = fixed_retriever.invoke(query)
            fixed_time = time.time() - start_time
            result["fixed"] = {
                "doc_count": len(fixed_docs),
                "time": round(fixed_time, 3),
                "sources": list(set(d.metadata.get("source", "?") for d in fixed_docs)),
                "preview": [d.page_content[:60] + "..." for d in fixed_docs[:3]],
            }
            print(f"   文档数: {len(fixed_docs)}, 耗时: {fixed_time:.3f}s")
            print(f"   来源: {result['fixed']['sources']}")
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            result["fixed"] = {"error": str(e)}

        # ========== 自适应检索 ==========
        print("\n🟢 自适应检索（新方案）:")
        start_time = time.time()
        try:
            adaptive_docs = adaptive_retriever.invoke(query)
            adaptive_time = time.time() - start_time
            result["adaptive"] = {
                "doc_count": len(adaptive_docs),
                "time": round(adaptive_time, 3),
                "sources": list(set(d.metadata.get("source", "?") for d in adaptive_docs)),
                "preview": [d.page_content[:60] + "..." for d in adaptive_docs[:3]],
            }
            print(f"   文档数: {len(adaptive_docs)}, 耗时: {adaptive_time:.3f}s")
            print(f"   来源: {result['adaptive']['sources']}")
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            result["adaptive"] = {"error": str(e)}

        # ========== 查询分类准确性 ==========
        print("\n🟡 查询分类:")
        try:
            predicted_type = classify_query(query, llm)
            result["adaptive"]["predicted_type"] = predicted_type
            result["adaptive"]["type_correct"] = (predicted_type == expected_type)
            if predicted_type == expected_type:
                print(f"   ✅ 正确: {predicted_type}")
            else:
                print(f"   ❌ 错误: 预期 {expected_type}, 实际 {predicted_type}")
        except Exception as e:
            print(f"   ⚠️ 分类失败: {e}")
            result["adaptive"]["predicted_type"] = "error"

        # ========== 查询扩展效果 ==========
        print("\n🟡 查询扩展:")
        try:
            expanded = expand_queries(query, llm)
            result["adaptive"]["expanded_count"] = len(expanded) - 1
            result["adaptive"]["expanded_queries"] = expanded[1:]
            print(f"   扩展了 {len(expanded) - 1} 个查询变体")
        except Exception as e:
            print(f"   ⚠️ 扩展失败: {e}")
            result["adaptive"]["expanded_count"] = 0

        # ========== 对比分析 ==========
        if "doc_count" in result["fixed"] and "doc_count" in result["adaptive"]:
            f_count = result["fixed"]["doc_count"]
            a_count = result["adaptive"]["doc_count"]
            diff = a_count - f_count
            
            # 计算来源重叠
            f_sources = set(result["fixed"].get("sources", []))
            a_sources = set(result["adaptive"].get("sources", []))
            overlap = f_sources & a_sources
            new_sources = a_sources - f_sources
            
            result["comparison"] = {
                "doc_diff": diff,
                "doc_ratio": round(a_count / max(f_count, 1), 2),
                "source_overlap": len(overlap),
                "new_sources": list(new_sources),
                "new_source_count": len(new_sources),
            }
            
            print(f"\n📊 对比: 固定={f_count} docs, 自适应={a_count} docs")
            print(f"   差异: {'+' if diff >= 0 else ''}{diff} docs")
            print(f"   新增来源: {new_sources if new_sources else '无'}")
        
        results.append(result)

    return results


def print_summary(results):
    """打印汇总报告"""
    print("\n\n" + "=" * 70)
    print("  📊 对比测试汇总报告")
    print("=" * 70)

    # 分类准确率
    correct = sum(1 for r in results if r["adaptive"].get("type_correct", False))
    total = len(results)
    print(f"\n🎯 查询分类准确率: {correct}/{total} = {correct/total*100:.1f}%")

    # 按类型统计
    print(f"\n📋 按查询类型统计:")
    print(f"{'类型':<8} {'查询数':<8} {'固定检索(avg docs)':<20} {'自适应检索(avg docs)':<20} {'分类准确率':<12}")
    print(f"{'─' * 68}")
    
    type_stats = {}
    for r in results:
        etype = r["expected_type"]
        if etype not in type_stats:
            type_stats[etype] = {"count": 0, "fixed_docs": [], "adaptive_docs": [], "correct": 0}
        type_stats[etype]["count"] += 1
        if "doc_count" in r["fixed"]:
            type_stats[etype]["fixed_docs"].append(r["fixed"]["doc_count"])
        if "doc_count" in r["adaptive"]:
            type_stats[etype]["adaptive_docs"].append(r["adaptive"]["doc_count"])
        if r["adaptive"].get("type_correct", False):
            type_stats[etype]["correct"] += 1

    for etype, stats in type_stats.items():
        avg_fixed = sum(stats["fixed_docs"]) / len(stats["fixed_docs"]) if stats["fixed_docs"] else 0
        avg_adaptive = sum(stats["adaptive_docs"]) / len(stats["adaptive_docs"]) if stats["adaptive_docs"] else 0
        acc = stats["correct"] / stats["count"] * 100
        print(f"{etype:<8} {stats['count']:<8} {avg_fixed:<20.1f} {avg_adaptive:<20.1f} {acc:<11.1f}%")

    # 整体对比
    all_fixed = sum(len(r["fixed"].get("sources", [])) for r in results)
    all_adaptive = sum(len(r["adaptive"].get("sources", [])) for r in results)
    all_fixed_docs = sum(r["fixed"].get("doc_count", 0) for r in results)
    all_adaptive_docs = sum(r["adaptive"].get("doc_count", 0) for r in results)
    
    print(f"\n📈 整体对比:")
    print(f"   固定权重: 总计 {all_fixed_docs} 个文档, 来源多样性 {all_fixed}")
    print(f"   自适应:   总计 {all_adaptive_docs} 个文档, 来源多样性 {all_adaptive}")
    if all_fixed_docs > 0:
        print(f"   文档数提升: {(all_adaptive_docs / all_fixed_docs - 1) * 100:+.1f}%")
    if all_fixed > 0:
        print(f"   来源多样性提升: {(all_adaptive / all_fixed - 1) * 100:+.1f}%")

    # 新增来源统计
    new_sources_total = sum(len(r.get("comparison", {}).get("new_sources", [])) for r in results)
    print(f"   新增来源文档数: {new_sources_total}")

    # 耗时对比
    fixed_times = [r["fixed"].get("time", 0) for r in results if "time" in r["fixed"]]
    adaptive_times = [r["adaptive"].get("time", 0) for r in results if "time" in r["adaptive"]]
    if fixed_times and adaptive_times:
        print(f"\n⏱️ 平均耗时:")
        print(f"   固定权重: {sum(fixed_times)/len(fixed_times):.3f}s")
        print(f"   自适应:   {sum(adaptive_times)/len(adaptive_times):.3f}s")


def save_results(results, filepath="data/retrieval_comparison.json"):
    """保存结果到 JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果已保存到: {filepath}")


def main():
    print("\n" + "=" * 70)
    print("  检索策略对比测试")
    print("  固定权重混合检索 vs 自适应检索（查询分类 + 查询扩展）")
    print("=" * 70)

    # 初始化
    vectorstore, documents, llm = init_environment()

    # 构建检索器
    fixed_retriever, adaptive_retriever = build_retrievers(vectorstore, documents)

    # 运行对比
    results = run_comparison(TEST_QUERIES, fixed_retriever, adaptive_retriever, llm)

    # 打印汇总
    print_summary(results)

    # 保存结果
    save_results(results)

    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()