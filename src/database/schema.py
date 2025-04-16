from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Lecturer(Base):
    __tablename__ = "lecturers"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    title = Column(Text)
    education_path = Column(Text)
    introduction = Column(Text)
    email = Column(String(100))
    publications = Column(Text)
    awards = Column(Text)
    subjects = Column(Text)
    projects = Column(Text)
    research_field = Column(Text)
    interested_field = Column(Text)
    url = Column(String(255))
