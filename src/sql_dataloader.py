import json
import os
from src.utils.db_utils import setup_database, batch_insert_lecturers, get_database

def load_lecturer_data(json_file_path):
    """
    Load lecturer data from a JSON file
    
    Args:
        json_file_path (str): Path to the JSON file containing lecturer data
        
    Returns:
        list: List of lecturer data dictionaries
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both single object and list formats
    if isinstance(data, dict):
        if 'lecturers' in data:  # If it's a container with a 'lecturers' key
            return data['lecturers']
        else:  # If it's a single lecturer object
            return [data]
    elif isinstance(data, list):  # If it's already a list of lecturers
        return data
    else:
        raise ValueError("Unexpected data format in JSON file")

def initialize_database(json_file_path, db_path="sqlite:///lecturers.db"):
    """
    Initialize the database with lecturer data from a JSON file
    
    Args:
        json_file_path (str): Path to the JSON file containing lecturer data
        db_path (str): SQLAlchemy compatible database URL
        
    Returns:
        tuple: (engine, db) - SQLAlchemy engine and LangChain SQLDatabase object
    """
    # Set up the database
    engine = setup_database(db_path)
    
    # Load lecturer data
    lecturer_data_list = load_lecturer_data(json_file_path)
    
    # Insert data into database
    batch_insert_lecturers(engine, lecturer_data_list)
    
    # Get SQLDatabase object for LangChain
    db = get_database(engine)
    
    return engine, db

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load lecturer data into the database')
    parser.add_argument('json_file', help='Path to the JSON file containing lecturer data')
    parser.add_argument('--db-path', default="sqlite:///lecturers.db", help='SQLAlchemy compatible database URL')
    
    args = parser.parse_args()
    
    try:
        engine, db = initialize_database(args.json_file, args.db_path)
        print(f"Successfully loaded lecturer data into {args.db_path}")
        print(f"Found tables: {db.get_usable_table_names()}")
    except Exception as e:
        print(f"Error: {e}")