from pydantic import BaseModel
from typing import Optional, Any, Dict


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    debug: Optional[bool] = False


class QueryResponse(BaseModel):
    answer: Any
    session_id: Optional[str] = None
    plan: Optional[Dict] = None
