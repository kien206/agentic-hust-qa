import os

import nest_asyncio
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    SemanticSimilarity,
)

nest_asyncio.apply()
load_dotenv()

if __name__ == "__main__":
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    evaluator_embeddings = HuggingfaceEmbeddings("BAAI/bge-reranker-v2-m3")

    dataset = load_dataset(
        "json",
        data_files="../data/eval/test_dataset_new.json",
        field="data",
        split="train",
    )

    eval_dataset = EvaluationDataset.from_hf_dataset(dataset)

    metrics = [
        # LLMContextRecall(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings),
    ]

    results = evaluate(dataset=eval_dataset, metrics=metrics)
    # df = results.to_pandas()
    results.upload()
