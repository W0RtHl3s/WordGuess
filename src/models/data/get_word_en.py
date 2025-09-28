from pydantic import BaseModel

class GetWordEn(BaseModel):
    """Get model for English words
    """
    word: str
    distance: float