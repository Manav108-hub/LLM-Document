from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    processing_time: float

class DocumentInfo(BaseModel):
    filename: str
    upload_time: str
    chunk_count: int
    status: str

class ChunkInfo(BaseModel):
    text: str
    filename: str
    chunk_id: int
    char_start: int
    char_end: int

class RetrievalResult(BaseModel):
    text: str
    score: float
    metadata: dict