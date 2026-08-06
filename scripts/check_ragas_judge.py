"""RAGAS judge 自检：完美样本下 faithfulness/context_recall 应为 ~1.0

用于验证评测 LLM 与 embeddings 集成是否正常（CI 冒烟用）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from embeddings import create_embeddings
from llm_provider import get_llm_provider

llm = LangchainLLMWrapper(get_llm_provider().judge_llm())
emb = LangchainEmbeddingsWrapper(create_embeddings())

s = "到2025年，基本形成横向打通、纵向贯通、协调有力的一体化推进格局，数字中国建设取得重要进展。"
dataset = Dataset.from_dict({
    "question": ["到2025年数字中国建设的目标是什么？"],
    "answer": [s],
    "contexts": [[s]],
    "ground_truth": [s],
})
res = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=emb,
    show_progress=False,
)
df = res.to_pandas()
for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    print(col, "=", df[col].tolist())
