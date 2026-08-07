"""
测试集构建工具
交互式构建 50-100 个典型问答对，用于评估检索器和RAG系统性能
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain_zhipu import ChatZhipuAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ZHIPU_API_KEY, DATA_DIR, PERSIST_DIRECTORY
from vectorstore import build_or_load_vectorstore
from document_processor import load_and_split_pdfs


TEST_SET_PATH = Path(__file__).resolve().parent / "test_set.json"

CATEGORIES = [
    "政策目标",
    "实施措施",
    "资格条件",
    "审批流程",
    "法律责任",
    "税收优惠",
    "行业监管",
    "其他",
]

DIFFICULTIES = ["easy", "medium", "hard"]


def load_test_set() -> List[Dict]:
    if TEST_SET_PATH.exists():
        with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_test_set(test_cases: List[Dict]):
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"测试集已保存: {TEST_SET_PATH} ({len(test_cases)} 条)")


def add_test_case():
    test_cases = load_test_set()
    new_id = max([t.get("id", 0) for t in test_cases], default=0) + 1

    print("\n" + "=" * 60)
    print("  添加新测试用例")
    print("=" * 60)

    question = input("问题: ").strip()
    if not question:
        print("❌ 问题不能为空")
        return

    print("\n分类选项:")
    for i, cat in enumerate(CATEGORIES):
        print(f"  {i + 1}. {cat}")
    cat_idx = input(f"选择分类 (1-{len(CATEGORIES)}): ").strip()
    try:
        category = CATEGORIES[int(cat_idx) - 1]
    except (ValueError, IndexError):
        category = "其他"

    print(f"\n难度选项: {', '.join(DIFFICULTIES)}")
    difficulty = input("选择难度: ").strip()
    if difficulty not in DIFFICULTIES:
        difficulty = "medium"

    keywords = input("期望关键词（逗号分隔）: ").strip()
    expected_keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    sources = input("期望来源文件名（逗号分隔）: ").strip()
    expected_sources = [s.strip() for s in sources.split(",") if s.strip()]

    ground_truth = input("标准答案（可选）: ").strip()

    test_case = {
        "id": new_id,
        "question": question,
        "expected_answer_keywords": expected_keywords,
        "expected_sources": expected_sources,
        "difficulty": difficulty,
        "category": category,
        "ground_truth": ground_truth,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    test_cases.append(test_case)
    save_test_set(test_cases)
    print(f"✅ 测试用例 #{new_id} 已添加")


def view_test_set():
    test_cases = load_test_set()
    if not test_cases:
        print("\n⚠️ 测试集为空，请先添加测试用例")
        return

    print("\n" + "=" * 60)
    print(f"  测试集概览 (共 {len(test_cases)} 条)")
    print("=" * 60)

    categories = {}
    difficulties = {}
    for tc in test_cases:
        cat = tc.get("category", "未分类")
        dif = tc.get("difficulty", "medium")
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[dif] = difficulties.get(dif, 0) + 1

    print(f"\n分类分布:")
    for cat, count in sorted(categories.items()):
        bar = "█" * count
        print(f"  {cat:12s} {count:3d} {bar}")

    print(f"\n难度分布:")
    for dif in DIFFICULTIES:
        count = difficulties.get(dif, 0)
        bar = "█" * count
        print(f"  {dif:12s} {count:3d} {bar}")

    print(f"\n详细列表:")
    for tc in test_cases:
        print(f"  [{tc['id']}] [{tc.get('difficulty', '?')}] {tc['question'][:50]}...")


def delete_test_case():
    test_cases = load_test_set()
    view_test_set()
    if not test_cases:
        return

    try:
        case_id = int(input("\n输入要删除的测试用例ID: ").strip())
        test_cases = [tc for tc in test_cases if tc.get("id") != case_id]
        save_test_set(test_cases)
        print(f"✅ 测试用例 #{case_id} 已删除")
    except ValueError:
        print("❌ 无效ID")


def generate_test_cases_from_documents(num_cases: int = 50):
    print(f"\n正在从文档自动生成 {num_cases} 个测试用例...")
    print("提示: 此功能将使用LLM从文档中提取测试用例")

    try:
        documents = load_and_split_pdfs(
            str(DATA_DIR),
            chunk_size=500,
            overlap=50,
        )
    except Exception as e:
        print(f"⚠️ 无法加载文档用于生成，使用已有向量库: {e}")
        try:
            vectorstore = build_or_load_vectorstore([])
            results = vectorstore._collection.get()
            documents = [
                Document(page_content=t, metadata={"source": m.get("source", "")})
                for t, m in zip(results["documents"], results["metadatas"])
            ]
        except Exception as e2:
            print(f"❌ 无法访问文档: {e2}")
            return

    if not documents:
        print("❌ 没有可用的文档")
        return

    llm = ChatZhipuAI(
        api_key=ZHIPU_API_KEY,
        model="glm-4-flash",
        temperature=0.3,
    )

    test_cases = load_test_set()
    start_id = max([t.get("id", 0) for t in test_cases], default=0) + 1

    for i in range(start_id, start_id + num_cases):
        doc = documents[i % len(documents)]
        source = doc.metadata.get("source", "未知")
        content = doc.page_content[:800]

        prompt = f"""根据以下政策文档内容，生成一个问答测试用例。

文档来源: {source}
文档内容（片段）:
{content}

请生成:
1. 一个具体的问题（公民/企业可能问的）
2. 问题的难度分类（easy/medium/hard）
3. 答案中应包含的关键词（逗号分隔）
4. 标准答案（简短）

请按以下JSON格式输出，不要包含其他内容:
{{"question": "...", "difficulty": "medium", "keywords": "..., ...", "answer": "..."}}"""

        try:
            response = llm.invoke(prompt)
            result = response.content.strip()

            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            result = result.strip()

            data = json.loads(result)
            test_case = {
                "id": i,
                "question": data.get("question", f"关于{source}的查询"),
                "expected_answer_keywords": [
                    k.strip() for k in data.get("keywords", "").split(",") if k.strip()
                ],
                "expected_sources": [source],
                "difficulty": data.get("difficulty", "medium"),
                "category": "自动生成",
                "ground_truth": data.get("answer", ""),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            test_cases.append(test_case)
            print(f"  ✅ 生成 #{i}: {test_case['question'][:50]}...")

        except Exception as e:
            print(f"  ⚠️ 生成 #{i} 失败: {e}")
            test_case = {
                "id": i,
                "question": f"关于{source}的政策是什么？",
                "expected_answer_keywords": [],
                "expected_sources": [source],
                "difficulty": "medium",
                "category": "自动生成",
                "ground_truth": "",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            test_cases.append(test_case)

    save_test_set(test_cases)
    print(f"\n✅ 自动生成完成，共 {len(test_cases)} 条测试用例")


def export_test_set():
    test_cases = load_test_set()
    if not test_cases:
        print("\n⚠️ 测试集为空")
        return

    output_path = input(f"导出路径 (默认: test_set_export.json): ").strip()
    if not output_path:
        output_path = "test_set_export.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"✅ 已导出到 {output_path}")


def import_test_set():
    input_path = input("导入文件路径: ").strip()
    if not input_path or not Path(input_path).exists():
        print("❌ 文件不存在")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        new_cases = json.load(f)

    test_cases = load_test_set()
    start_id = max([t.get("id", 0) for t in test_cases], default=0) + 1
    for i, tc in enumerate(new_cases):
        tc["id"] = start_id + i
        test_cases.append(tc)

    save_test_set(test_cases)
    print(f"✅ 已导入 {len(new_cases)} 条测试用例")


def main():
    while True:
        print("\n" + "=" * 60)
        print("  测试集构建工具")
        print("=" * 60)
        print(f"  当前测试集: {len(load_test_set())} 条")
        print()
        print("  1. 添加测试用例")
        print("  2. 查看测试集")
        print("  3. 删除测试用例")
        print("  4. 自动生成测试用例 (LLM)")
        print("  5. 导出测试集")
        print("  6. 导入测试集")
        print("  0. 退出")
        print()

        choice = input("请选择: ").strip()

        if choice == "1":
            add_test_case()
        elif choice == "2":
            view_test_set()
        elif choice == "3":
            delete_test_case()
        elif choice == "4":
            try:
                num = int(input("生成数量 (默认50): ").strip() or "50")
                generate_test_cases_from_documents(num)
            except ValueError:
                print("❌ 无效数量")
        elif choice == "5":
            export_test_set()
        elif choice == "6":
            import_test_set()
        elif choice == "0":
            print("退出")
            break
        else:
            print("❌ 无效选择")


if __name__ == "__main__":
    main()