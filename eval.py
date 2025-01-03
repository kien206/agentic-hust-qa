import pandas as pd
import json
import weaviate
from tqdm import tqdm
from sqlalchemy import (
    MetaData,
)
from flow.utils.utils import (
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
from flow.model import Model

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
    vectorstore = get_vectorstore(client=client, embedding_model=embedding, index_name="Hust_doc_final")
    retriever = get_retriever(vectorstore=vectorstore, k=3)
    
    web_search_tool = get_websearch()

    # Build SQL Database
    metadata_obj = MetaData()
    table = get_table(table_name="teacher", metadata_obj=metadata_obj)
    engine = get_sql_engine()
    db = get_database(engine=engine, metadata_obj=metadata_obj)

    return llm, llm_json_mode, retriever, db, web_search_tool

def rag_eval(model, df, output_file="test.json"):
   
    file = {'question': [], 'answer': [], 'contexts': [], 'ground_truths': []}
    for _, row in tqdm(df.iterrows()):
        question = row['User input']
        ground_truths = row['Reference (Reference answer)']
        
        response = model.chat(query=question)

        answer = response['generation'].content
        ctx = [r.page_content for r in response['documents']]
        contexts = list(set(ctx))

        with open(output_file, 'w', encoding="utf-8") as f:
            file['question'].append(question)
            file['ground_truths'].append(ground_truths)
            file['answer'].append(answer)
            file['contexts'].append(contexts)    
            content = json.dumps(file, indent=2, ensure_ascii=False)
            f.write(content)
    return

if __name__ == "__main__":
    df = pd.read_csv('data/eval/test.csv', encoding='utf-8')    
    llm, llm_json_mode, retriever, db, web_search_tool = build_comp()
    pipeline = Model(llm, llm_json_mode, retriever, db, web_search_tool, verbose=True)

    rag_eval(model=pipeline, df=df)