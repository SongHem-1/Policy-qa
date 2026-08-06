"""
端到端评估工具
评估完整RAG系统性能（检索 + 生成），支持简单评估和Ragas框架评估
"""

import sys
import os
import json
import time
import statistics
import argparse
from pathlib import Path
from typing import List, Dict, Optional

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA_DIR, PERSIST_DIRECTORY, ZHIPU_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_BY_SECTION,
    USE_PARENT_CHILD, PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP, USE_METADATA_AUGMENT,
)
from vectorstore import build_or_load_vectorstore
from document_processor import load_and_split_pdfs
from qa_chain import build_retrieval_qa_chain


TEST_SET_PATH = Path(__file__).resolve().parent / "test_set.json"
E2E_REPORT_PATH = Path(__file__).resolve().parent / "e2e_evaluation_report.json"


def load_test_set() -> List[Dict]:
    if not TEST_SET_PATH.exists():
        print(f"❌ 测试集不存在: {TEST_SET_PATH}")
        return []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_keyword_coverage(
    answer: str,
    expected_keywords: List[str],
) -> float:
    if not expected_keywords:
        return 1.0
    found = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    return found / len(expected_keywords)


def calculate_source_accuracy(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> float:
    if not expected_sources:
        return 1.0
    found = sum(1 for src in retrieved_sources if any(
        es.lower() in src.lower() or src.lower() in es.lower()
        for es in expected_sources
    ))
    return found / len(retrieved_sources) if retrieved_sources else 0.0


def simple_evaluation(test_cases: List[Dict]) -> Dict:
    print("\n" + "=" * 60)
    print("  端到端评估 - 简单模式")
    print("=" * 60)

    print(f"\n正在加载向量数据库...")
    persist_path = Path(PERSIST_DIRECTORY)
    if persist_path.exists() and any(persist_path.iterdir()):
        vectorstore = build_or_load_vectorstore([])
        cached_docs = None
        cache_path = Path(DATA_DIR) / "_documents_cache.pkl"
        if cache_path.exists():
            import pickle
            with open(cache_path, "rb") as f:
                cached_docs = pickle.load(f)
    else:
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

    chain = build_retrieval_qa_chain(
        vectorstore,
        documents=cached_docs,
        parent_documents=parent_docs,
    )

    keyword_scores = []
    source_scores = []
    answer_lengths = []
    per_case_results = []

    print(f"\n正在评估 {len(test_cases)} 个测试用例...")
    print("-" * 60)

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        expected_keywords = tc.get("expected_answer_keywords", [])
        expected_sources = tc.get("expected_sources", [])

        try:
            result = chain.invoke({
                "input": question,
                "chat_history": [],
                "memory_context": "",
            })
            answer = result.get("answer", "")
            source_docs = result.get("context", [])
            retrieved_sources = [
                doc.metadata.get("source", "未知")
                for doc in source_docs
            ]
        except Exception as e:
            print(f"  ⚠️ 查询失败 [{tc['id']}]: {e}")
            answer = ""
            retrieved_sources = []

        kw_cov = calculate_keyword_coverage(answer, expected_keywords)
        src_acc = calculate_source_accuracy(retrieved_sources, expected_sources)
        ans_len = len(answer)

        keyword_scores.append(kw_cov)
        source_scores.append(src_acc)
        answer_lengths.append(ans_len)

        per_case_results.append({
            "id": tc["id"],
            "question": question[:80],
            "answer_preview": answer[:100],
            "answer_length": ans_len,
            "keyword_coverage": kw_cov,
            "source_accuracy": src_acc,
            "retrieved_sources": retrieved_sources,
            "expected_sources": expected_sources,
        })

        if (i + 1) % 10 == 0 or i == len(test_cases) - 1:
            print(f"  进度: {i + 1}/{len(test_cases)}")

    report = _print_simple_report(
        keyword_scores, source_scores, answer_lengths, len(test_cases)
    )
    _save_e2e_report(report, per_case_results, "simple")
    return report


def _print_simple_report(
    keyword_scores: List[float],
    source_scores: List[float],
    answer_lengths: List[int],
    total: int,
) -> Dict:
    print("\n" + "=" * 80)
    print("  端到端评估报告")
    print("=" * 80)
    print(f"\n测试用例数量: {total}")
    print(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 核心指标：")
    print("-" * 80)
    print(f"{'指标':<25s} {'平均值':>10s} {'标准差':>10s}")
    print("-" * 80)

    kw_mean = statistics.mean(keyword_scores) if keyword_scores else 0
    kw_std = statistics.stdev(keyword_scores) if len(keyword_scores) > 1 else 0
    print(f"{'keyword_coverage':<25s} {kw_mean:>10.4f} {kw_std:>10.4f}")

    src_mean = statistics.mean(source_scores) if source_scores else 0
    src_std = statistics.stdev(source_scores) if len(source_scores) > 1 else 0
    print(f"{'source_accuracy':<25s} {src_mean:>10.4f} {src_std:>10.4f}")

    len_mean = statistics.mean(answer_lengths) if answer_lengths else 0
    len_std = statistics.stdev(answer_lengths) if len(answer_lengths) > 1 else 0
    print(f"{'answer_length':<25s} {len_mean:>10.1f} {len_std:>10.1f}")
    print("-" * 80)

    print(f"\n📈 性能评估：")
    if kw_mean >= 0.7:
        print(f"✅ 关键词覆盖率: {kw_mean:.2%} - 优秀")
    elif kw_mean >= 0.5:
        print(f"⚠️ 关键词覆盖率: {kw_mean:.2%} - 一般")
    else:
        print(f"❌ 关键词覆盖率: {kw_mean:.2%} - 需要改进")

    if src_mean >= 0.7:
        print(f"✅ 来源准确率: {src_mean:.2%} - 优秀")
    elif src_mean >= 0.5:
        print(f"⚠️ 来源准确率: {src_mean:.2%} - 一般")
    else:
        print(f"❌ 来源准确率: {src_mean:.2%} - 需要改进")

    return {
        "keyword_coverage": {"mean": kw_mean, "stdev": kw_std},
        "source_accuracy": {"mean": src_mean, "stdev": src_std},
        "answer_length": {"mean": len_mean, "stdev": len_std},
    }


# RAGAS 全指标 CI 门槛（默认阈值，可用 --min-<metric> 覆盖）
CI_THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.70,
    "answer_correctness": 0.60,
}


def _to_float(value) -> Optional[float]:
    try:
        f = float(value)
        if f != f:  # NaN -> None（RAGAS 单样本失败时会产生 NaN）
            return None
        return f
    except (TypeError, ValueError):
        return None


def ragas_evaluation(
    test_cases: List[Dict],
    limit: Optional[int] = None,
    ci_thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """RAGAS 全指标端到端评测。

    指标：faithfulness / answer_relevancy / context_precision /
    context_recall / answer_correctness。

    - 评测 LLM 复用主备降级供应商（Zhipu -> DeepSeek）
    - embeddings 复用本地 bge-m3（与检索链路同一模型）
    - ci_thresholds 非空时执行 CI 门槛校验，不达标抛 RuntimeError
    """
    print("\n" + "=" * 60)
    print("  端到端评估 - RAGAS 全指标模式")
    print("=" * 60)

    cases = test_cases[:limit] if limit else test_cases

    print("正在加载向量数据库...")
    persist_path = Path(PERSIST_DIRECTORY)
    if persist_path.exists() and any(persist_path.iterdir()):
        vectorstore = build_or_load_vectorstore([])
        cached_docs = None
        cache_path = Path(DATA_DIR) / "_documents_cache.pkl"
        if cache_path.exists():
            import pickle
            with open(cache_path, "rb") as f:
                cached_docs = pickle.load(f)
    else:
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

    chain = build_retrieval_qa_chain(
        vectorstore,
        documents=cached_docs,
        parent_documents=parent_docs,
    )

    ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    print(f"正在评估 {len(cases)} 个测试用例...")
    print("-" * 60)

    for i, tc in enumerate(cases):
        question = tc["question"]
        ground_truth = tc.get("ground_truth", "")

        try:
            result = chain.invoke({
                "input": question,
                "chat_history": [],
                "memory_context": "",
            })
            answer = result.get("answer", "")
            source_docs = result.get("context", [])
            contexts = [doc.page_content for doc in source_docs]
        except Exception as e:
            print(f"  查询失败 [{tc['id']}]: {e}")
            answer = ""
            contexts = []

        ragas_data["question"].append(question)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(ground_truth)

        if (i + 1) % 10 == 0 or i == len(cases) - 1:
            print(f"  进度: {i + 1}/{len(cases)}")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from embeddings import create_embeddings
    from llm_provider import get_llm_provider

    llm = LangchainLLMWrapper(get_llm_provider().judge_llm())
    embeddings = LangchainEmbeddingsWrapper(create_embeddings())
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    print("\n正在计算 RAGAS 指标（评测 LLM: " + get_llm_provider().name + "）...")
    dataset = Dataset.from_dict(ragas_data)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
    )

    df = result.to_pandas()
    # 部分指标（如 answer_correctness）在 ragas 0.2.x 中按样本返回，统一从 per-sample 表取均值
    scores = {}
    for m in metrics:
        col = m.name
        if col in df.columns:
            valid = [v for v in df[col].apply(_to_float) if v is not None]
            scores[col] = sum(valid) / len(valid) if valid else 0.0
        else:
            scores[col] = 0.0

    per_case = []
    for i, tc in enumerate(cases):
        row = df.iloc[i] if i < len(df) else {}
        per_case.append({
            "id": tc["id"],
            "question": tc["question"][:80],
            "answer_preview": (ragas_data["answer"][i] or "")[:120],
            "ground_truth": (tc.get("ground_truth") or "")[:120],
            **{name: _to_float(row.get(name)) for name in scores},
        })

    print("\n" + "=" * 80)
    print("  RAGAS 全指标评估报告")
    print("=" * 80)
    print(f"\n测试用例数量: {len(cases)}")
    print(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    for name in scores:
        print(f"  {name:<25s} {scores[name]:.4f}")
    print("-" * 80)

    _save_e2e_report({"ragas_scores": scores}, per_case, "ragas")

    if ci_thresholds:
        failures = [
            f"{name}={scores.get(name, 0.0):.4f} < {threshold}"
            for name, threshold in ci_thresholds.items()
            if scores.get(name, 0.0) < threshold
        ]
        if failures:
            raise RuntimeError("CI 门槛未通过: " + "; ".join(failures))
        print("\n✅ CI 门槛全部通过")
    return scores


def _save_e2e_report(summary: Dict, per_case: List[Dict], mode: str):
    report = {
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "summary": summary,
        "per_case": per_case,
    }
    with open(E2E_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📁 报告已保存: {E2E_REPORT_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端评估工具")
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="使用 RAGAS 全指标评估（faithfulness/answer_relevancy/context_precision/context_recall/answer_correctness）",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="使用简单评估模式（默认）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅评估前 N 条用例（冒烟测试用）",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="启用 CI 门槛校验（RAGAS 模式，任一指标低于阈值则退出码非 0）",
    )
    for name, default in CI_THRESHOLDS.items():
        parser.add_argument(
            f"--min-{name}",
            type=float,
            default=default,
            help=f"{name} 最低阈值（默认 {default}）",
        )
    args = parser.parse_args()

    test_cases = load_test_set()
    if not test_cases:
        print("❌ 请先运行 test_set_builder.py 构建测试集")
        return 1

    print(f"加载测试集: {len(test_cases)} 条")

    if args.ragas or not args.simple:
        thresholds = None
        if args.ci:
            thresholds = {name: getattr(args, f"min_{name}") for name in CI_THRESHOLDS}
        try:
            ragas_evaluation(test_cases, limit=args.limit, ci_thresholds=thresholds)
        except RuntimeError as e:
            print(f"\n❌ {e}")
            return 1
        except Exception as e:
            print(f"\n❌ RAGAS 评估失败: {e}")
            return 1
    else:
        simple_evaluation(test_cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
if __name__ == "__main__":
    main()
