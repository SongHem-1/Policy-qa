"""
离线评估工具
评估检索器性能（不调用LLM），指标：Recall@K, Precision@K, MRR
"""

import sys
import os
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Tuple, Optional

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR, PERSIST_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP,
    CHUNK_BY_SECTION, USE_PARENT_CHILD, PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, USE_METADATA_AUGMENT,
    BM25_WEIGHT, VECTOR_WEIGHT, USE_RERANKER, RERANKER_TOP_K,
    RETRIEVAL_K, USE_ADAPTIVE_RETRIEVAL, USE_QUERY_EXPANSION,
)
from vectorstore import build_or_load_vectorstore, create_hybrid_retriever
from document_processor import load_and_split_pdfs
from qa_chain import build_retrieval_qa_chain, build_retriever


TEST_SET_PATH = Path(__file__).resolve().parent / "test_set.json"
REPORT_PATH = Path(__file__).resolve().parent / "offline_evaluation_report.json"


def load_test_set() -> List[Dict]:
    if not TEST_SET_PATH.exists():
        print(f"❌ 测试集不存在: {TEST_SET_PATH}")
        return []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_recall_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: int,
) -> float:
    """召回率：前K个检索结果中覆盖了多少个期望来源（去重）"""
    if not expected_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    found_expected = set()
    for src in top_k:
        for es in expected_sources:
            if es.lower() in src.lower() or src.lower() in es.lower():
                found_expected.add(es)
    return len(found_expected) / len(expected_sources)


def calculate_precision_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
    k: int,
) -> float:
    """精确率：前K个检索结果中，有多少个文档来源是相关的（去重）"""
    actual_k = min(k, len(retrieved_sources))
    if actual_k == 0:
        return 0.0
    top_k = retrieved_sources[:actual_k]
    seen = set()
    found = 0
    for src in top_k:
        if src in seen:
            continue
        seen.add(src)
        if any(es.lower() in src.lower() or src.lower() in es.lower()
               for es in expected_sources):
            found += 1
    return found / actual_k


def calculate_mrr(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> float:
    for rank, src in enumerate(retrieved_sources, 1):
        if any(es.lower() in src.lower() or src.lower() in es.lower()
               for es in expected_sources):
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    test_cases: List[Dict],
    ks: List[int] = None,
    retriever_only: bool = False,
) -> Dict:
    if ks is None:
        ks = [1, 3, 5, 10]

    print("\n" + "=" * 60)
    print("  离线评估 - 检索器性能")
    print("=" * 60)
    print(f"  测试用例数: {len(test_cases)}")
    print(f"  评估指标: Recall@{', '.join(map(str, ks))}, Precision@{', '.join(map(str, ks))}, MRR")

    print(f"\n正在加载向量数据库...")
    persist_path = Path(PERSIST_DIRECTORY)
    if persist_path.exists() and any(persist_path.iterdir()):
        vectorstore = build_or_load_vectorstore([])
        collection = vectorstore._collection
        metadatas = collection.get()["metadatas"]
        sources = sorted({m.get("source", "") for m in metadatas})
        print(f"  已加载向量库: {len(sources)} 个文件")

        cached_docs = None
        cache_path = Path(DATA_DIR) / "_documents_cache.pkl"
        if cache_path.exists():
            import pickle
            with open(cache_path, "rb") as f:
                cached_docs = pickle.load(f)
            print(f"  从缓存加载 {len(cached_docs)} 个文档块")
    else:
        print(f"  正在构建向量数据库...")
        documents = load_and_split_pdfs(
            str(DATA_DIR),
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            chunk_by_section=CHUNK_BY_SECTION,
            parent_child=USE_PARENT_CHILD,
            parent_size=PARENT_CHUNK_SIZE,
            child_size=CHILD_CHUNK_SIZE,
            child_overlap=CHILD_CHUNK_OVERLAP,
            augment_meta=USE_METADATA_AUGMENT,
        )
        vectorstore = build_or_load_vectorstore(documents)
        cached_docs = documents

    parent_docs = None
    if USE_PARENT_CHILD:
        parent_cache_path = Path(DATA_DIR) / "_parent_documents_cache.pkl"
        if parent_cache_path.exists():
            import pickle
            with open(parent_cache_path, "rb") as f:
                parent_docs = pickle.load(f)

    print(f"\n正在构建检索器...")
    chain = None
    retriever = None
    if retriever_only:
        # 纯检索器评测：不构建 LLM 问答链，完全离线运行
        retriever = build_retriever(
            vectorstore,
            documents=cached_docs,
            parent_documents=parent_docs,
            adaptive=False,
        )
    else:
        chain = build_retrieval_qa_chain(
            vectorstore,
            documents=cached_docs,
            parent_documents=parent_docs,
        )

    metrics = {f"recall@{k}": [] for k in ks}
    metrics.update({f"precision@{k}": [] for k in ks})
    metrics["mrr"] = []
    metrics["hit_count"] = 0
    per_case_results = []

    print(f"\n正在评估 {len(test_cases)} 个测试用例...")
    print("-" * 60)

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        expected_sources = tc.get("expected_sources", [])

        try:
            if retriever_only:
                source_docs = retriever.invoke(question)
            else:
                result = chain.invoke({
                    "input": question,
                    "chat_history": [],
                    "memory_context": "",
                })
                source_docs = result.get("context", [])
            retrieved_sources = [
                doc.metadata.get("source", "未知")
                for doc in source_docs
            ]
        except Exception as e:
            print(f"  ⚠️ 查询失败 [{tc['id']}]: {e}")
            retrieved_sources = []

        case_result = {
            "id": tc["id"],
            "question": question[:80],
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
        }

        for k in ks:
            recall = calculate_recall_at_k(retrieved_sources, expected_sources, k)
            precision = calculate_precision_at_k(retrieved_sources, expected_sources, k)
            metrics[f"recall@{k}"].append(recall)
            metrics[f"precision@{k}"].append(precision)
            case_result[f"recall@{k}"] = recall
            case_result[f"precision@{k}"] = precision

        mrr = calculate_mrr(retrieved_sources, expected_sources)
        metrics["mrr"].append(mrr)
        case_result["mrr"] = mrr

        if any(
            calculate_recall_at_k(retrieved_sources, expected_sources, k) > 0
            for k in ks
        ):
            metrics["hit_count"] += 1
            case_result["hit"] = True
        else:
            case_result["hit"] = False

        per_case_results.append(case_result)

        if (i + 1) % 10 == 0 or i == len(test_cases) - 1:
            print(f"  进度: {i + 1}/{len(test_cases)}")

    _print_report(metrics, ks, len(test_cases))
    _save_report(metrics, ks, len(test_cases), per_case_results)

    return metrics


def _print_report(metrics: Dict, ks: List[int], total: int):
    print("\n" + "=" * 80)
    print("  检索器评估报告")
    print("=" * 80)
    print(f"\n测试用例数量: {total}")
    print(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 核心指标：")
    print("-" * 80)
    print(f"{'指标':<20s} {'平均值':>10s} {'标准差':>10s} {'最小值':>10s} {'最大值':>10s}")
    print("-" * 80)

    for k in ks:
        values = metrics[f"recall@{k}"]
        if values:
            print(f"{'recall@' + str(k):<20s} {statistics.mean(values):>10.4f} "
                  f"{statistics.stdev(values) if len(values) > 1 else 0:>10.4f} "
                  f"{min(values):>10.4f} {max(values):>10.4f}")

    for k in ks:
        values = metrics[f"precision@{k}"]
        if values:
            print(f"{'precision@' + str(k):<20s} {statistics.mean(values):>10.4f} "
                  f"{statistics.stdev(values) if len(values) > 1 else 0:>10.4f} "
                  f"{min(values):>10.4f} {max(values):>10.4f}")

    values = metrics["mrr"]
    if values:
        print(f"{'mrr':<20s} {statistics.mean(values):>10.4f} "
              f"{statistics.stdev(values) if len(values) > 1 else 0:>10.4f} "
              f"{min(values):>10.4f} {max(values):>10.4f}")

    print("-" * 80)

    print(f"\n📈 性能评估：")
    recall5 = statistics.mean(metrics["recall@5"]) if metrics["recall@5"] else 0
    precision5 = statistics.mean(metrics["precision@5"]) if metrics["precision@5"] else 0
    mrr_mean = statistics.mean(metrics["mrr"]) if metrics["mrr"] else 0

    if recall5 >= 0.8:
        print(f"✅ 召回率@5: {recall5:.2%} - 优秀")
    elif recall5 >= 0.6:
        print(f"⚠️ 召回率@5: {recall5:.2%} - 一般")
    else:
        print(f"❌ 召回率@5: {recall5:.2%} - 需要改进")

    if precision5 >= 0.6:
        print(f"✅ 精确率@5: {precision5:.2%} - 优秀")
    elif precision5 >= 0.4:
        print(f"⚠️ 精确率@5: {precision5:.2%} - 一般")
    else:
        print(f"❌ 精确率@5: {precision5:.2%} - 需要改进")

    if mrr_mean >= 0.7:
        print(f"✅ MRR: {mrr_mean:.4f} - 优秀")
    elif mrr_mean >= 0.5:
        print(f"⚠️ MRR: {mrr_mean:.4f} - 一般")
    else:
        print(f"❌ MRR: {mrr_mean:.4f} - 需要改进")

    hit_rate = metrics["hit_count"] / total if total > 0 else 0
    print(f"📊 命中率 (至少1条相关): {hit_rate:.2%}")

    print(f"\n💡 改进建议:")
    if precision5 < 0.4:
        print(f"  - 精确率偏低: 考虑增加BM25权重或引入重排序")
    if recall5 < 0.6:
        print(f"  - 召回率偏低: 考虑增加检索数量K或优化查询扩展")
    if mrr_mean < 0.5:
        print(f"  - MRR偏低: 考虑调整混合检索权重或优化分块策略")


def _save_report(
    metrics: Dict,
    ks: List[int],
    total: int,
    per_case: List[Dict],
):
    report = {
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_cases": total,
        "summary": {},
        "per_case": per_case,
    }

    for k in ks:
        for metric_name in [f"recall@{k}", f"precision@{k}"]:
            values = metrics[metric_name]
            if values:
                report["summary"][metric_name] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }

    values = metrics["mrr"]
    if values:
        report["summary"]["mrr"] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }

    report["summary"]["hit_rate"] = metrics["hit_count"] / total if total > 0 else 0

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📁 报告已保存: {REPORT_PATH}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="离线检索器性能评估")
    parser.add_argument(
        "--retriever-only",
        action="store_true",
        help="仅评估检索器（不构建 LLM 问答链，完全离线运行）",
    )
    args = parser.parse_args()

    test_cases = load_test_set()
    if not test_cases:
        print("❌ 请先运行 test_set_builder.py 构建测试集")
        return 1

    print(f"加载测试集: {len(test_cases)} 条")
    evaluate_retriever(test_cases, retriever_only=args.retriever_only)
    return 0


if __name__ == "__main__":
    main()
