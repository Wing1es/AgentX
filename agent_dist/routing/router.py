import logging
import hashlib
import time
from typing import Dict, List
from agent_dist.registry.client import RegistryClient
from .models import MultiRouteDecision
from .prompts import ShouldUseAgentsPrompt

logger = logging.getLogger("orchestrator.router")

class SemanticRouter:
    def __init__(self, llm, registry: RegistryClient, threshold: float = 0.3):
        self.llm = llm
        self.registry = registry
        self.threshold = threshold
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl: int = 300 
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        _, ts = self._cache[key]
        return (time.time() - ts) < self._cache_ttl
    
    def invalidate_cache(self):
        self._cache.clear()
        logger.info('Router: cache invalidated')

    def route(self, query: str, history: List[Dict] = None) -> MultiRouteDecision:
        cache_key = self._hash(query)
        if self._is_cache_valid(cache_key):
            decision, _ = self._cache[cache_key]
            return decision

        # 1. Gatekeeping
        if not self._should_use_agents(query):
            decision = MultiRouteDecision(mode="llm_only")
            self._cache[cache_key] = (decision, time.time())
            return decision
    
        # 2. Semantic Search
        
        # Search for relevant agents
        scoped_agents = self.registry.search_agents(query, limit=5, threshold=self.threshold)
        
        if not scoped_agents:
            # Fallback: if no agents found, what should we do?
            # Option A: Try llm_only
            # Option B: Return all agents (dangerous if too many)
            # Option C: Return empty scope (planner might fail or LLM will say I can't do it)
            logger.info(f"Router: No agents found for query via vector search (threshold={self.threshold})")
            decision = MultiRouteDecision(mode="llm_only")
        else:
            logger.info(f"Router: Found {len(scoped_agents)} agents via vector search")
            decision = MultiRouteDecision(mode="react", scope=scoped_agents)

        self._cache[cache_key] = (decision, time.time())
        return decision

    def _hash(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def _should_use_agents(self, query: str) -> bool:
        try:
            prompt = ShouldUseAgentsPrompt.format(query=query)
            out = self.llm.invoke(prompt)
            return out.content.strip().lower() == "true"
        except Exception as e:
            # Fallback to True on error to be safe
            logger.warning(f"Router Warning: LLM gatekeeping failed: {e}")
            return True