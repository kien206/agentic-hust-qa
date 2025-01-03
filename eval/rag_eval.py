
import weaviate
from sqlalchemy import (
    MetaData,
)
from utils.utils import (
    get_llm,
    get_retriever,
    split_doc,
    get_embedding,
    get_vectorstore,
    get_retriever,
    get_websearch,
    get_table,
    get_sql_engine,
    get_database
)
from pipeline.workflow import Pipeline

def eval():
    pass

if __name__ == "__main__":
    model = "llama3.1"
    text_dir = "..data/parse/test_text"

    # GET LLM
    llm = get_llm(model=model, format="")
    llm_json_mode = get_llm(model=model, format="json")

    # BUILD RETRIEVER
    doc_list = split_doc(text_dir)
    embedding = get_embedding(model_name="BAAI/BGE-M3")

    client = weaviate.connect_to_local()
    vectorstore = get_vectorstore(client=client, doc_list=doc_list, embedding_model=embedding, index_name="test")
    retriever = get_retriever(vectorstore=vectorstore, k=3)
    
    web_search_tool = get_websearch()

    # Build SQL Database
    metadata_obj = MetaData()
    table = get_table(table_name="teacher", metadata_obj=metadata_obj)
    engine = get_sql_engine()
    db = get_database(engine=engine, metadata_obj=metadata_obj)
    
    pipeline = Pipeline(llm, llm_json_mode, retriever, db, web_search_tool)

    while True:
        query = input("Question: ")
        if query.lower() == "end":
            break

        pipeline.chat(query=query)


    
    

