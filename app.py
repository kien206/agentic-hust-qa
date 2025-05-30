import json
import logging
import os

import streamlit as st
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
        # reset=True
    )

    retriever = get_retriever(vectorstore=vectorstore, k=settings.agent.top_k)
    web_search_tool = get_websearch(k=settings.websearch.search_depth)

    # Set up or connect to existing lecturer database
    if os.path.exists(lecturer_data_path):
        reload = True
        if os.path.exists("lecturers.db"):
            reload = False
        _, db = initialize_database(lecturer_data_path, db_path, reload=reload)

    agents = {
        "router": RouterAgent(llm_json=llm_json_mode, verbose=True),
        "retriever": RetrievalAgent(llm, llm_json_mode, retriever, verbose=True),
        "sql": SQLAgent(llm, llm_json_mode, db, verbose=True),
        "web_search": WebSearchAgent(llm, web_search_tool, verbose=True),
        "generator": LLM(llm, verbose=True),
    }

    return agents


def create_streamlit_app(agents, config={}):
    """
    Creates a Streamlit app for the LangGraph chatbot.
    """
    st.set_page_config(
        page_title="HUST Chatbot",
        page_icon="💬",
        layout="wide",
        menu_items={
            "Get Help": "https://www.extremelycoolapp.com/help",
            "Report a bug": "https://www.extremelycoolapp.com/bug",
            "About": "# This is a header. This is an *extremely* cool app!",
        },
    )

    with st.sidebar:
        st.title("Chats created")

        # Add a button to clear the conversation
        if st.button("Clear"):
            st.session_state.conversation = []
            st.rerun()

    st.header("Hỏi về quy chế, quy định và giảng viên Bách Khoa.")

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Initialize pipeline
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = Graph(agents, verbose=True)

    # Display conversation history
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    user_query = st.chat_input(">:")

    # Process user query
    if user_query:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process with LangGraph pipeline
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream the response
            try:
                for mode, payload in st.session_state.pipeline.graph.stream(
                    {"question": user_query.lower()}, stream_mode=["messages", "custom"], config=config
                ):
                    if mode == "messages":
                        chunk, metadata = payload
                        if (
                            chunk.content
                            and metadata.get("langgraph_node") == "generator"
                        ):
                            full_response += chunk.content
                            # Show streaming response with cursor
                            message_placeholder.markdown(full_response + "▌")
                    elif mode == "custom":
                        full_response += "\n"
                        full_response += payload["citation"]
                        message_placeholder.markdown(full_response + "▌")

                logger.info(f"Full response: {full_response}")
                # Display the final response without cursor
                message_placeholder.markdown(full_response)

                # Add assistant response to conversation
                st.session_state.conversation.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    settings = Settings()
    client = weaviate.connect_to_local()

    agents = build_comp(client, settings)
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    create_streamlit_app(agents, config)
