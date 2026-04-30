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


class PlanStep(BaseModel):
    """A single step in the execution plan, bound to a specific agent."""
    agent_name: str                # Which agent to call
    agent_url: str = ""            # Agent's endpoint URL
    description: str               # What this step should accomplish
    requires: List[str] = Field(default_factory=list)   # Input fields needed from state
    provides: List[str] = Field(default_factory=list)    # Output fields this step produces
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0


class MultiRouteDecision(BaseModel):
    mode: str
    scope: List[Dict[str, Any]] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    mode: str
    steps: List[PlanStep]
    active_agents: Optional[List[Dict[str, Any]]] = None

    # ReAct specific fields
    max_iterations: int = 10
    current_iteration: int = 0
    scratchpad: List[str] = Field(default_factory=list) 
    final_answer: Optional[str] = None
