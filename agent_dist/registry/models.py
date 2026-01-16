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
    intent_group: str
    capability_cluster: str
    version: str = "1.0.0"
    description: Optional[str] = ""
    tags: List[str] = []
    capabilities: Capabilities


class AgentRecord(AgentRegistration):
    last_heartbeat: float

    def is_alive(self, ttl: int) -> bool:
        return (t.time() - self.last_heartbeat) <= ttl


class DeleteRequest(BaseModel):
    names: List[str] = []
    delete_all: bool = False

class IntentCreate(BaseModel):
    name: str
    description: str

class CapabilityCreate(BaseModel):
    name: str
    description: str

class IntentBulkCreate(BaseModel):
    intents: List[IntentCreate]

class CapabilityBulkItem(BaseModel):
    intent: str
    name: str
    description: str

class CapabilityBulkCreate(BaseModel):
    capabilities: List[CapabilityBulkItem]

class CapabilityDef(BaseModel):
    name: str
    description: str

class IntentSchema(BaseModel):
    description: str
    capabilities: List[CapabilityDef]

class RegistrySchemaBulk(BaseModel):
    intents: Dict[str, IntentSchema]
