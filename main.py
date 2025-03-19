import os
from sqlalchemy import MetaData
from src.utils.db_utils import setup_database, get_database
from src.sql_dataloader import load_lecturer_data, initialize_database
from src.utils.utils import (
    get_embedding,
    get_llm,
    get_retriever,
    get_vectorstore,
    get_websearch,
)
from src.flow.model import Model
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

# This is a simpler function to just set up the lecturer database
def setup_lecturer_database(lecturer_data_path, db_path="sqlite:///lecturers.db"):
    """
    Set up just the lecturer database from a JSON file
    
    Args:
        lecturer_data_path (str): Path to the JSON file containing lecturer data
        db_path (str): SQLAlchemy compatible database URL
        
    Returns:
        tuple: (engine, db) - SQLAlchemy engine and LangChain SQLDatabase object
    """
    engine, db = initialize_database(lecturer_data_path, db_path)
    return engine, db

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Set up components for the QA system')
    parser.add_argument('--data-path', help='Path to the JSON file containing lecturer data')
    parser.add_argument('--db-path', default="sqlite:///lecturers.db", help='SQLAlchemy compatible database URL')
    
    args = parser.parse_args()
    
    # Either just set up the lecturer database
    if args.data_path:
        engine, db = setup_lecturer_database(args.data_path, args.db_path)
        print(f"Successfully set up lecturer database at {args.db_path}")
        print(f"Tables: {db.get_usable_table_names()}")
    
    # Or build all components
    with weaviate.connect_to_local() as client:
        llm, llm_json_mode, retriever, db, web_search_tool = update_build_comp(
            client,
            lecturer_data_path=args.data_path, 
            db_path=args.db_path
        )
    
        # Initialize the QA pipeline
        pipeline = Model(llm, llm_json_mode, retriever, db, web_search_tool, verbose=True)
        print("QA pipeline initialized successfully")
        while True:
            query = input("Question: ").lower()
            if query in ["end", "exit"]:
                break
            with weaviate.connect_to_local():
                response = pipeline.chat(query=query)
                main_answer = response["generation"].content
                reference = ""
                try:
                    if "documents" in response.keys():
                        reference = response["documents"][0].metadata["source"].split("\\")[1]
                except:
                    pass
                answer = main_answer + "\n\n" + "Nguồn:" + reference
                print("Answer: ", answer)
  