import os
from datasets import load_dataset
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper, HuggingfaceEmbeddings
from langchain_openai import ChatOpenAI


evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
evaluator_embeddings = HuggingfaceEmbeddings("BAAI/BGE-M3")

dataset = load_dataset(
    "json",
    data_files="../data/eval/test_dataset.json"
)

eval_dataset = EvaluationDataset.from_hf_dataset(dataset)

os.environ["OPENAI_API_KEY"] = "your-openai-key"

metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    FactualCorrectness(llm=evaluator_llm), 
    Faithfulness(llm=evaluator_llm),
    SemanticSimilarity(embeddings=evaluator_embeddings)
]

results = evaluate(dataset=eval_dataset, metrics=metrics)
df = results.to_pandas()
