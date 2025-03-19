from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import your models
from src.database.db_schema import (
    Base,
    Lecturer,
    LecturerAward,
    LecturerEducation,
    LecturerEmail,
    LecturerInterest,
    LecturerProject,
    LecturerPublication,
    LecturerResearch,
    LecturerSubject,
)


def setup_database(db_path="sqlite:///:memory:"):
    """
    Set up the database, create tables and return engine

    Args:
        db_path (str): SQLAlchemy compatible database URL

    Returns:
        Engine: SQLAlchemy engine object
    """
    engine = create_engine(db_path, future=True)
    Base.metadata.create_all(engine)
    return engine


def insert_lecturer_data(engine, lecturer_data):
    """
    Insert lecturer data into the database

    Args:
        engine: SQLAlchemy engine
        lecturer_data (dict): Dictionary with lecturer information

    Returns:
        int: ID of the inserted lecturer
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Create new lecturer
        lecturer = Lecturer(
            name=lecturer_data.get("name", "").lower(),
            title=lecturer_data.get("title", ""),
            introduction=lecturer_data.get("introduction", ""),
            url=lecturer_data.get("url", ""),
        )
        session.add(lecturer)
        session.flush()

        # Add emails
        for email in lecturer_data.get("email", []):
            lecturer_email = LecturerEmail(lecturer_id=lecturer.id, email=email)
            session.add(lecturer_email)

        # Add education
        for edu in lecturer_data.get("education_path", []):
            education = LecturerEducation(lecturer_id=lecturer.id, education=edu)
            session.add(education)

        # Add research fields
        for field in lecturer_data.get("research_field", []):
            research = LecturerResearch(lecturer_id=lecturer.id, research_field=field)
            session.add(research)

        # Add interests
        for interest in lecturer_data.get("interested_field", []):
            lecturer_interest = LecturerInterest(
                lecturer_id=lecturer.id, interest=interest
            )
            session.add(lecturer_interest)

        # Add publications
        for pub in lecturer_data.get("notable_publication", []):
            publication = LecturerPublication(lecturer_id=lecturer.id, publication=pub)
            session.add(publication)

        # Add awards
        for award in lecturer_data.get("awards", []):
            lecturer_award = LecturerAward(lecturer_id=lecturer.id, award=award)
            session.add(lecturer_award)

        # Add subjects
        for subj in lecturer_data.get("teaching_subjects", []):
            subject = LecturerSubject(lecturer_id=lecturer.id, subject=subj)
            session.add(subject)

        # Add projects
        for proj in lecturer_data.get("current_project", []):
            project = LecturerProject(lecturer_id=lecturer.id, project=proj)
            session.add(project)

        session.commit()
        return lecturer.id

    except Exception as e:
        session.rollback()
        raise e

    finally:
        session.close()


def batch_insert_lecturers(engine, lecturer_data_list):
    """
    Insert multiple lecturer records at once

    Args:
        engine: SQLAlchemy engine
        lecturer_data_list (list): List of dictionaries with lecturer information

    Returns:
        list: List of inserted lecturer IDs
    """
    ids = []
    for lecturer_data in lecturer_data_list:
        lecturer_id = insert_lecturer_data(engine, lecturer_data)
        ids.append(lecturer_id)
    return ids
