import json

import pandas as pd
from sqlalchemy import MetaData
from tqdm import tqdm

import weaviate
from flow.model import Model
from utils.utils import (
    get_database,
    get_embedding,
    get_llm,
    get_retriever,
    get_sql_engine,
    get_table,
    get_vectorstore,
    get_websearch,
    split_doc,
)

client = weaviate.connect_to_local()


def build_comp():
    model = "llama3.1"

    # GET LLM
    llm = get_llm(model=model, format="", temperature=0)
    llm_json_mode = get_llm(model=model, format="json", temperature=0)

    # BUILD RETRIEVER
    # doc_list = split_doc(text_dir)
    embedding = get_embedding(model_name="BAAI/BGE-M3")

    vectorstore = get_vectorstore(
        client=client, embedding_model=embedding, index_name="Hust_doc_final"
    )
    retriever = get_retriever(vectorstore=vectorstore, k=3)

    web_search_tool = get_websearch()

    # Build SQL Database
    metadata_obj = MetaData()
    table = get_table(table_name="teacher", metadata_obj=metadata_obj)
    engine = get_sql_engine()
    db = get_database(engine=engine, metadata_obj=metadata_obj)

    return llm, llm_json_mode, retriever, db, web_search_tool


def build_rag_eval(model, df, output_file):
    file = {"question": [], "answer": [], "contexts": [], "ground_truths": []}
    for _, row in tqdm(df.iterrows()):
        question = row["User input"]
        ground_truths = row["Reference (Reference answer)"]

        response = model.chat(query=question)

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
    df = pd.read_csv("data/eval/test.csv", encoding="utf-8")
    llm, llm_json_mode, retriever, db, web_search_tool = build_comp()
    pipeline = Model(llm, llm_json_mode, retriever, db, web_search_tool, verbose=True)

    build_rag_eval(model=pipeline, df=df, output_file="remaining.json")
