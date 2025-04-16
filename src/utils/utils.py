import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_ollama import ChatOllama
from sqlalchemy import Column, String, Table, create_engine, inspect

column_mapping = {
    "name": "Tên",
    "subjects": "Môn giảng dạy",
    "interested_field": "Lĩnh vực quan tâm",
    "introduction": "Giới thiệu",
    "email": "Email",
    "publications": "Một số công bố khoa học tiêu biểu",
    "research_field": "Lĩnh vực nghiên cứu",
    "title": "Chức vụ",
    "education_path": "Con đường học vấn",
    "projects": "Một số dự án đã tham gia",
    "awards": "Giải thưởng tiêu biểu",
    "url": "Nguồn"
}

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


def format_docs(docs):
    return "\n-------------------------------\n".join(doc.page_content for doc in docs)


def get_websearch(k):
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
    # Get all table names
    inspector = inspect(engine)
    all_tables = inspector.get_table_names()

    # Create a SQLDatabase instance with all tables
    db = SQLDatabase(engine=engine, include_tables=all_tables)
    return db


def format_rag_metadata(metadata) -> str:
    """
    Format metadata of 1 document into a string
    """
    source = metadata.get("source").split(".md")[0]

    for k, v in source_mapping.items():
        if source.startswith(k):
            source = source.replace(k, v).upper()

    numbers = (metadata['chapter_number'], metadata['section_number'], metadata['article_number'])
    titles = (metadata['chapter_title'], metadata['section_title'], metadata['article_title'])
    
    formatted_metadata = [f"{source}"]
    for i in range(len(titles)):

        if titles[i] is not None:
            formatted_metadata.append(f"{section_mapping[i]} {numbers[i]}")
    
    return ", ".join(formatted_metadata)


def format_sql_output(sql_output):
    """
    Format SQL query result into answer
    """
    response = f"Có {len(sql_output)} giảng viên được tìm thấy.\n"

    for output_dict in sql_output:
        for attribute, value in output_dict.items():
            if attribute == "COUNT(*)":
                count_response = f"Có {value} giảng viên được tìm thấy."
                return count_response
            
            if attribute == "id":
                pass # do nothing

            if ("/n" in value or "\n" in value) and len(value) > 0:
                fixed_value = value.replace("\n", "\n- ").replace('/n', '\n- ')
                response += f"{column_mapping[attribute]}: \n- {fixed_value}\n\n"
            elif len(value) == 0:
                response += f"{column_mapping[attribute]}: Không có thông tin\n\n"
            else:
                response += f"{column_mapping[attribute]}: {value}\n\n"    

        response += f"{'-'*40}\n"

    return response
