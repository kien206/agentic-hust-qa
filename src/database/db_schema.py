from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Lecturer(Base):
    __tablename__ = "lecturers"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    title = Column(Text)
    introduction = Column(Text)
    url = Column(String(255))

    # Relationships
    emails = relationship(
        "LecturerEmail", back_populates="lecturer", cascade="all, delete-orphan"
    )
    education = relationship(
        "LecturerEducation", back_populates="lecturer", cascade="all, delete-orphan"
    )
    research_fields = relationship(
        "LecturerResearch", back_populates="lecturer", cascade="all, delete-orphan"
    )
    interested_fields = relationship(
        "LecturerInterest", back_populates="lecturer", cascade="all, delete-orphan"
    )
    publications = relationship(
        "LecturerPublication", back_populates="lecturer", cascade="all, delete-orphan"
    )
    awards = relationship(
        "LecturerAward", back_populates="lecturer", cascade="all, delete-orphan"
    )
    teaching_subjects = relationship(
        "LecturerSubject", back_populates="lecturer", cascade="all, delete-orphan"
    )
    projects = relationship(
        "LecturerProject", back_populates="lecturer", cascade="all, delete-orphan"
    )


class LecturerEmail(Base):
    __tablename__ = "lecturer_emails"

    id = Column(Integer, primary_key=True)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"), nullable=False)
    email = Column(String(100), nullable=False, index=True)

    lecturer = relationship("Lecturer", back_populates="emails")


class LecturerEducation(Base):
    __tablename__ = "lecturer_education"

    id = Column(Integer, primary_key=True)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"), nullable=False)
    education = Column(Text, nullable=False)

    lecturer = relationship("Lecturer", back_populates="education")


class LecturerResearch(Base):
    __tablename__ = "lecturer_research"

    id = Column(Integer, primary_key=True)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"), nullable=False)
    research_field = Column(Text, nullable=False)

    lecturer = relationship("Lecturer", back_populates="research_fields")


class LecturerInterest(Base):
    __tablename__ = "lecturer_interests"

    id = Column(Integer, primary_key=True)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"), nullable=False)
    interest = Column(Text, nullable=False)

    lecturer = relationship("Lecturer", back_populates="interested_fields")


class LecturerPublication(Base):
    __tablename__ = "lecturer_publications"

    id = Column(Integer, primary_key=True)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"), nullable=False)
    publication = Column(Text, nullable=False)

    lecturer = relationship("Lecturer", back_populates="publications")


class LecturerAward(Base):
    __tablename__ = "lecturer_awards"

    id = Column(Integer, primary_key=True)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"), nullable=False)
    award = Column(Text, nullable=False)

    lecturer = relationship("Lecturer", back_populates="awards")


class LecturerSubject(Base):
    __tablename__ = "lecturer_subjects"

    id = Column(Integer, primary_key=True)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"), nullable=False)
    subject = Column(Text, nullable=False)

    lecturer = relationship("Lecturer", back_populates="teaching_subjects")


class LecturerProject(Base):
    __tablename__ = "lecturer_projects"

    id = Column(Integer, primary_key=True)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"), nullable=False)
    project = Column(Text, nullable=False)

    lecturer = relationship("Lecturer", back_populates="projects")
