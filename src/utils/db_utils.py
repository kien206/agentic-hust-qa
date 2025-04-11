import json
import os

from langchain_community.utilities import SQLDatabase
from sqlalchemy import inspect, create_engine
from sqlalchemy.orm import sessionmaker

from src.database.schema import Lecturer, Base


def get_engine(db_path="sqlite:///lecturers.db"):
    engine = create_engine(db_path, future=True)
    Base.metadata.create_all(engine)
    return engine


def get_database(engine):
    # Get all table names
    inspector = inspect(engine)
    all_tables = inspector.get_table_names()

    # Create a SQLDatabase instance with all tables
    db = SQLDatabase(engine=engine, include_tables=all_tables)
    return db


def insert_lecturer_data(engine, lecturer_data):
    Session = sessionmaker(bind=engine)
    session = Session()

    def join_list(text):
        if isinstance(text, list):
            new_text = "/n".join(text)
            return new_text

        return text

    try:
        # Create new lecturer
        lecturer = Lecturer(
            name=join_list(lecturer_data.get("name", "")),
            title=join_list(lecturer_data.get("title", "")),
            introduction=join_list(lecturer_data.get("introduction", "")),
            url=join_list(lecturer_data.get("url", "")),
            email=join_list(lecturer_data.get("email", "")),
            publications=join_list(lecturer_data.get("notable_publication", "")),
            awards=join_list(lecturer_data.get("awards", "")),
            subjects=join_list(lecturer_data.get("teaching_subjects", "")),
            projects=join_list(lecturer_data.get("current_project", "")),
            research_field=join_list(lecturer_data.get("research_field", "")),
            interested_field=join_list(lecturer_data.get("interested_field", "")),
        )
        session.add(lecturer)
        session.flush()

        session.commit()
        return lecturer.id

    except Exception as e:
        session.rollback()
        raise e

    finally:
        session.close()


def batch_insert_lecturers(engine, lecturer_data_list):
    ids = []
    for lecturer_data in lecturer_data_list:
        lecturer_id = insert_lecturer_data(engine, lecturer_data)
        ids.append(lecturer_id)
    return ids


def load_lecturer_data(json_file_path):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both single object and list formats
    if isinstance(data, dict):
        if "lecturers" in data:  # If it's a container with a 'lecturers' key
            return data["lecturers"]
        else:  # If it's a single lecturer object
            return [data]
    elif isinstance(data, list):  # If it's already a list of lecturers
        return data
    else:
        raise ValueError("Unexpected data format in JSON file")
