import json
import logging
import os

import weaviate

from config.settings import Settings
from src.agents import LLM, RetrievalAgent, RouterAgent, SQLAgent, WebSearchAgent
from src.database.db_init import initialize_database
from src.graph import Graph
from src.utils.utils import get_embedding, get_llm, get_websearch
from src.utils.vectordb_utils import get_retriever, get_vectorstore

logger = logging.getLogger(__name__)


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


def main(agents, **kwargs):
    pipeline = Graph(agents, **kwargs)
    while True:
        query = input("Question: ").lower()
        if query in ["end", "exit"]:
            break

        response = pipeline.chat(query=query)
        main_answer = response["generation"].content
        reference = ""
        try:
            if "documents" in response.keys() and len(response["documents"]) > 0:
                # reference = response["documents"][0].metadata["source"].split("\\")[1]
                reference = response["documents"][0].metatdata
                answer = main_answer + "\n\n" + "Nguồn:" + reference
            else:
                answer = main_answer
        except:
            answer = main_answer
        print("Answer: ", answer)


def main_stream(agents, **kwargs):
    pipeline = Graph(agents, verbose=True)

    while True:
        query = input("Question: ").lower()
        if query in ["end", "exit"] or not query:
            break

        for mode, payload in pipeline.graph.stream(
            {"question": query}, stream_mode=["messages", "custom"]
        ):
            if mode == "messages":
                chunk, metadata = payload
                if chunk.content and metadata.get("langgraph_node") == "generator":
                    print(chunk.content, end="", flush=True)
            elif mode == "custom":
                print(json.loads(payload)["citation"])


if __name__ == "__main__":
    settings = Settings()
    client = weaviate.connect_to_local()

    logger.debug("Getting components")
    llm, llm_json_mode, retriever, db, web_search_tool = build_comp(client, settings)
    logger.debug("Finished loading components.")
    agents = {
        "router": RouterAgent(llm_json=llm_json_mode, verbose=True),
        "retriever": RetrievalAgent(llm, llm_json_mode, retriever, verbose=True),
        "sql": SQLAgent(llm, llm_json_mode, db, verbose=True),
        "web_search": WebSearchAgent(llm, web_search_tool, verbose=True),
        "generator": LLM(llm, verbose=True),
    }
    logger.debug("Finished loading Agents.")

    main_stream(agents)
