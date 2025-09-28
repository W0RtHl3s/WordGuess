from sqlalchemy import Column, UUID, VARCHAR, Text, Boolean, ForeignKey, text
from pgvector.sqlalchemy import Vector
from .. import Base


class WordsEn(Base):
    """Database english words schema
    """
    __tablename__ = 'words_en'

    id = Column(UUID, primary_key=True,
                server_default=text('gen_random_uuid()'))
    word = Column(VARCHAR(20), nullable=False, unique=True)
    embedding = Column(Vector(300))
