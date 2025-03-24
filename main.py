import os
from src.database.db_utils import setup_database
from src.utils.utils import get_database
from src.database.sql_dataloader import initialize_database
from src.utils.utils import (
    get_embedding,
    get_llm,
    get_retriever,
    get_vectorstore,
    get_websearch,
)
from src.flow.model import Model
from src.agents.router import RouterAgent
from src.agents.retriever import RetrievalAgent
from src.agents.generator import LLMAgent
from src.agents.sql import SQLAgent
from src.agents.web_search import WebSearchAgent
import weaviate

def update_build_comp(client, lecturer_data_path=None, db_path="sqlite:///lecturers.db"):
    """
    Updated version of the build_comp function that uses the new lecturer database
    
    Args:
        lecturer_data_path (str): Path to the JSON file containing lecturer data (optional)
        db_path (str): Path to the SQLite database to use or create
        
    Returns:
        tuple: Components needed for the QA system
    """
    model = "llama3.1"
    text_dir = "data/parse/text"

    # GET LLM
    llm = get_llm(model=model, format="")
    llm_json_mode = get_llm(model=model, format="json")

    # BUILD RETRIEVER
    embedding = get_embedding(model_name="BAAI/BGE-M3")

    
    vectorstore = get_vectorstore(
        client=client, embedding_model=embedding, index_name="Hust_doc_final"
    )
    retriever = get_retriever(vectorstore=vectorstore, k=3)
    web_search_tool = get_websearch()

    # Set up or connect to existing lecturer database
    if lecturer_data_path and os.path.exists(lecturer_data_path):
        # If data file is provided and exists, initialize database with it
        engine, db = initialize_database(lecturer_data_path, db_path)
    else:
        # Otherwise, just connect to the existing database
        engine = setup_database(db_path)
        db = get_database(engine)

    return llm, llm_json_mode, retriever, db, web_search_tool

def main():
    with weaviate.connect_to_local() as client:
        llm, llm_json_mode, retriever, db, web_search_tool = update_build_comp(client, 
                                                                               "data/lecturers/soict_lecturers.json")

        agents = {
            "router": RouterAgent(llm_json=llm_json_mode, verbose=True),
            "retriever": RetrievalAgent(llm, llm_json_mode, retriever, verbose=True),
            "sql": SQLAgent(llm, llm_json_mode, db, verbose=True),
            "web_search": WebSearchAgent(llm, web_search_tool, verbose=True),
            "generator": LLMAgent(llm, verbose=True)
        }

        pipeline = Model(agents, verbose=True)
        while True:
            query = input("Question: ").lower()
            if query in ["end", "exit"]:
                break

            response = pipeline.chat(query=query)
            main_answer = response["generation"].content
            reference = ""
            try:
                if "documents" in response.keys() and len(response["documents"]) > 0:
                    reference = response["documents"][0].metadata["source"].split("\\")[1]
                    answer = main_answer + "\n\n" + "Nguồn:" + reference
                else:
                    answer = main_answer
            except:
                answer = main_answer
            print("Answer: ", answer)


if __name__ == "__main__":
    main()