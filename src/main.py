from sqlalchemy import MetaData

import weaviate
from src.flow.model import Model
from src.flow.utils import (
    get_database,
    get_embedding,
    get_llm,
    get_retriever,
    get_sql_engine,
    get_table,
    get_vectorstore,
    get_websearch,
)


def build_comp():
    model = "llama3.1"
    text_dir = "data/parse/text"

    # GET LLM
    llm = get_llm(model=model, format="")
    llm_json_mode = get_llm(model=model, format="json")

    # BUILD RETRIEVER
    # doc_list = split_doc(text_dir)
    embedding = get_embedding(model_name="BAAI/BGE-M3")

    client = weaviate.connect_to_local()
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


def main():
    llm, llm_json_mode, retriever, db, web_search_tool = build_comp()
    pipeline = Model(llm, llm_json_mode, retriever, db, web_search_tool, verbose=True)

    while True:
        query = str(input("Question: "))
        if query.lower() in ["end", "exit"]:
            break
        response = pipeline.chat(query=query)
        main_answer = response["generation"].content
        reference = ""
        if "documents" in response.keys():
            reference = response["documents"].source.split("\\")[1]

        answer = main_answer + "\n\n" + "Nguồn:" + reference
        print("Answer: ", answer)


if __name__ == "__main__":
    llm, llm_json_mode, retriever, db, web_search_tool = build_comp()
    pipeline = Model(llm, llm_json_mode, retriever, db, web_search_tool, verbose=True)

    main()
