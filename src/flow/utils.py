import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_weaviate import WeaviateVectorStore
from sqlalchemy import Column, String, Table, create_engine


def get_llm(model, format, **kwargs):
    return ChatOllama(model=model, format=format, **kwargs)


def get_embedding(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)


def split_doc(text_dir, **kwargs):
    print("__SPLITTING__")
    text_files = os.listdir(text_dir)
    # Load documents

    docs = [
        TextLoader(os.path.join(text_dir, file), encoding="utf-8").load()
        for file in text_files
    ]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1024, chunk_overlap=200, **kwargs
    )
    doc_splits = text_splitter.split_documents(docs_list)

    return doc_splits


def get_vectorstore(client, embedding_model, index_name, **kwargs):
    if not client.collections.exists(index_name):
        if "text_dir" in kwargs:
            doc_list = split_doc(kwargs["text_dir"])
            vectorstore = WeaviateVectorStore.from_documents(
                client=client,
                documents=doc_list,
                embedding=embedding_model,
                index_name=index_name,
                **kwargs
            )

            return vectorstore

    vectorstore = WeaviateVectorStore(
        client=client,
        text_key="text",
        embedding=embedding_model,
        index_name=index_name,
        **kwargs
    )
    return vectorstore


def get_retriever(vectorstore, **kwargs):
    retriever = vectorstore.as_retriever(**kwargs)

    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_websearch():
    from langchain_community.tools.tavily_search import TavilySearchResults

    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = "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"
    web_search_tool = TavilySearchResults(k=3)

    return web_search_tool


def get_table(table_name, metadata_obj):
    teacher_table = Table(
        table_name,
        metadata_obj,
        Column("name", String(16), primary_key=True),
        Column("subjects", String(16)),
    )

    return teacher_table


def get_sql_engine():
    return create_engine("sqlite:///:memory:", future=True)


def get_database(engine, metadata_obj):
    metadata_obj.create_all(engine)
    db = SQLDatabase(engine=engine, include_tables=["teacher"])

    return db
