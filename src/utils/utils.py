import os

from ..utils.chunking_utils import get_chunks_with_metadata, get_chunks_with_metadata_new
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_weaviate import WeaviateVectorStore
from sqlalchemy import Column, String, Table, create_engine, inspect


def get_llm(model, format, provider="ollama", **kwargs):
    if provider.lower() == "ollama":
        return ChatOllama(model=model, format=format, **kwargs)
    elif provider.lower() == "huggingface":
        hf = HuggingFacePipeline.from_model_id(
            model_id="gpt2",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 10},
        )

        return hf


def get_embedding(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)

def format_metadata(metadata):
    section_mapping = {
        0: "Chương",
        1: "Mục",
        2: "Điều"
    }

    source_mapping = {
        "QC": "Quy chế",
        "QĐ": "Quy định",
        "HD": "Hướng dẫn",
        "QtĐ": "Quyết định"
    }

    source = metadata.get("source").split(".md")[0]

    for k, v in source_mapping.items():
        if source.startswith(k):
            source = source.replace(k, v).upper()

    numbers = (metadata['chapter_number'], metadata['section_number'], metadata['article_number'])
    titles = (metadata['chapter_title'], metadata['section_title'], metadata['article_title'])
    
    formatted_metadata = f"{source}\n"
    for i in range(len(titles)):

        if titles[i] is not None:
            formatted_metadata += f"{section_mapping[i]} {numbers[i]}: {titles[i].upper().strip("# ")}\n\n"
    
    return formatted_metadata

def split_md(dir, **kwargs):
    doc_list = []
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        if file.startswith("HD"):
            chunks = get_chunks_with_metadata_new(content)
        else:
            chunks = get_chunks_with_metadata(content)

        for chunk in chunks:
            chunk_metadata = chunk["metadata"]
            chunk_metadata["source"] = file
            formatted_metadata = format_metadata(chunk_metadata)
            doc_list.append(Document(page_content=formatted_metadata + chunk["text"], metadata=chunk_metadata, **kwargs))

    return doc_list


def split_doc(text_dir, **kwargs):
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


def get_vectorstore(
    client, embedding_model, index_name, text_dir, reset=False, **kwargs
):
    if reset:
        client.collections.delete_all()
    if not client.collections.exists(index_name):
        doc_list = split_md(text_dir)
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


def get_websearch(k):
    from langchain_community.tools.tavily_search import TavilySearchResults

    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = "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"
    web_search_tool = TavilySearchResults(k=k)

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


def get_database(engine):
    """
    Create a LangChain SQLDatabase object for the given engine

    Args:
        engine: SQLAlchemy engine

    Returns:
        SQLDatabase: LangChain SQLDatabase object
    """
    # Get all table names
    inspector = inspect(engine)
    all_tables = inspector.get_table_names()

    # Create a SQLDatabase instance with all tables
    db = SQLDatabase(engine=engine, include_tables=all_tables)
    return db
