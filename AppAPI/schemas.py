from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    user_email: str
    query: str

class QueryResponse(BaseModel):
    session_id: int
    response: Optional[str] = None
