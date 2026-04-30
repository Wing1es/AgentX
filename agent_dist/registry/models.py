from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import time as t


class Capabilities(BaseModel):
    tasks: List[str]
    input_types: List[str]
    requires: List[str] = []
    provides: List[str] = []
    compliance: List[str] = []
    input_schema: Optional[Dict[str, Any]] = None

class AgentRegistration(BaseModel):
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    tags: List[str] = []
    capabilities: Capabilities


class AgentRecord(AgentRegistration):
    last_heartbeat: float

    def is_alive(self, ttl: int) -> bool:
        return (t.time() - self.last_heartbeat) <= ttl


class DeleteRequest(BaseModel):
    names: List[str] = []
    delete_all: bool = False
