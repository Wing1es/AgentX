from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional, Any
from time import time
from agent_dist.registry.models import AgentRecord


class Capabilities(BaseModel):
    tasks: List[str]
    input_types: List[str]

    requires: List[str] = Field(default_factory=list)
    provides: List[str] = Field(default_factory=list)

    compliance: List[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class MultiRouteDecision(BaseModel):
    mode: str
    scope: Optional[List[Dict[str, Any]]] = None


class ExecutionPlan(BaseModel):
    mode: str
    steps: List[str]
    active_agents: Optional[List[Dict[str, Any]]] = None
