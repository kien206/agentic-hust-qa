import os

from ..utils.db_utils import (
    batch_insert_lecturers,
    get_database,
    get_engine,
    load_lecturer_data,
)


def initialize_database(json_file_path, db_path="sqlite:///lecturers.db", reload=False):
    # Set up the database
    engine = get_engine(db_path)

    if reload:
        print("Ingesting")
        # Load lecturer data
        lecturer_data_list = load_lecturer_data(json_file_path)

        # Insert data into database
        batch_insert_lecturers(engine, lecturer_data_list)

    # Get SQLDatabase object for LangChain
    db = get_database(engine)

    return engine, db
