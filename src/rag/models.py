from pydantic import BaseModel


class RAGSource(BaseModel):
    video_id: str
    content: str
    chunk_index: int
    relevance_score: float


class RAGResult(BaseModel):
    answer: str
    sources: list[RAGSource]
