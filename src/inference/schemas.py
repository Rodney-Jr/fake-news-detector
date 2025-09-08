from pydantic import BaseModel

class NewsItem(BaseModel):
    title: str = ""
    text: str = ""

class Prediction(BaseModel):
    label: str
    proba: float | None = None
