from agent_dist.registry.client import RegistryClient
from agent_dist.routing.router import SemanticRouter
from agent_dist.routing.planner import PlannerAgent
from agent_dist.routing.executor import Executor
from agent_dist.llm import get_llm
from .config import REGISTRY_URL

registry = RegistryClient(REGISTRY_URL)
llm = get_llm()

router = SemanticRouter(llm, registry)
planner = PlannerAgent()
executor = Executor(llm, registry)
