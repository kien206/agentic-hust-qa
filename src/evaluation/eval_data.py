import json
import os

import pandas as pd
import weaviate
from tqdm import tqdm

from config.settings import Settings
from src.agents import LLM, RetrievalAgent, RouterAgent, SQLAgent, WebSearchAgent
from src.database.db_init import initialize_database
from src.graph import Graph
from src.utils.utils import get_embedding, get_llm, get_websearch
from src.utils.vectordb_utils import get_retriever, get_vectorstore

client = weaviate.connect_to_local()


def build_comp(client, settings: Settings):
    model = settings.llm.model
    embedding_model = settings.vectorstore.embedding_model
    lecturer_data_path = settings.database.lecturer_data_path
    db_path = settings.database.db_path

    # GET LLM
    llm = get_llm(model=model, format="")
    llm_json_mode = get_llm(model=model, format="json")

    # BUILD RETRIEVER
    embedding = get_embedding(model_name=embedding_model)

    vectorstore = get_vectorstore(
        client=client,
        embedding_model=embedding,
        index_name="Hust_doc_md_final",
        text_dir=settings.vectorstore.text_dir,
    )

    retriever = get_retriever(vectorstore=vectorstore, k=settings.agent.top_k)
    web_search_tool = get_websearch(k=settings.websearch.search_depth)

    # Set up or connect to existing lecturer database
    if os.path.exists(lecturer_data_path):
        reload = True
        if os.path.exists("lecturers.db"):
            reload = False
        _, db = initialize_database(lecturer_data_path, db_path, reload=reload)

    return llm, llm_json_mode, retriever, db, web_search_tool


def build_rag_eval(agents, df, output_file, **kwargs):
    pipeline = Graph(agents, **kwargs)
    file = {"question": [], "answer": [], "contexts": [], "ground_truths": []}

    for _, row in tqdm(df.iterrows()):
        question = row["User input"]
        ground_truths = row["Reference (Reference answer)"]

        response = pipeline.chat(query=question)

        answer = response["generation"].content
        try:
            ctx = [r.page_content for r in response["documents"]]
            contexts = list(set(ctx))
        except:
            contexts = []

        with open(output_file, "w", encoding="utf-8") as f:
            file["question"].append(question)
            file["ground_truths"].append(ground_truths)
            file["answer"].append(answer)
            file["contexts"].append(contexts)
            content = json.dumps(file, indent=2, ensure_ascii=False)
            f.write(content)
    client.close()


if __name__ == "__main__":
    settings = Settings()
    client = weaviate.connect_to_local()

    llm, llm_json_mode, retriever, db, web_search_tool = build_comp(client, settings)
    agents = {
        "router": RouterAgent(llm_json=llm_json_mode, verbose=True),
        "retriever": RetrievalAgent(llm, llm_json_mode, retriever, verbose=True),
        "sql": SQLAgent(llm, llm_json_mode, db, verbose=True),
        "web_search": WebSearchAgent(llm, web_search_tool, verbose=True),
        "generator": LLM(llm, verbose=True),
    }

    df = pd.read_csv("data/eval/test.csv", encoding="utf-8")
    llm, llm_json_mode, retriever, db, web_search_tool = build_comp()

    build_rag_eval(agents, df=df, output_file="data/eval/qwen2.5.json")
