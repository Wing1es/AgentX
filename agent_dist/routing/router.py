import hashlib
from typing import Dict, List
from agent_dist.registry.client import RegistryClient
from .models import MultiRouteDecision
from .prompts import IntentClassifierPrompt, CapabilityClassifierPrompt, ShouldUseAgentsPrompt

class HierarchicalRouter:
    def __init__(self, llm, registry: RegistryClient):
        self.llm = llm
        self.registry = registry
        self._cache: Dict[str, MultiRouteDecision] = {}

    def route(self, query: str, history: List[Dict] = None) -> MultiRouteDecision:
        cache_key = self._hash(query)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Gatekeeping
        if not self._should_use_agents(query):
            decision = MultiRouteDecision(mode="llm_only")
            self._cache[cache_key] = decision
            return decision

        # 2. Level 1: Intent Routing
        intents_map = self.registry.list_intents()
        intents_desc = "\n".join(
            [f"- {name}: {data['description']}" for name, data in intents_map.items()]
        )

        prompt = IntentClassifierPrompt.format(
            intents_desc=intents_desc,
            query=query
        )
        response = self.llm.invoke(prompt)
        target_intent = response.content.strip()

        all_agents = self.registry.list_agents()
        scoped_agents = all_agents

        if target_intent in intents_map:
            caps_map = self.registry.list_capabilities(target_intent)

            if len(caps_map) == 1:
                target_cap = list(caps_map.keys())[0]
            elif len(caps_map) > 1:
                caps_desc = "\n".join(
                    [f"- {name}: {desc}" for name, desc in caps_map.items()]
                )
                cap_prompt = CapabilityClassifierPrompt.format(
                    intent=target_intent,
                    caps_desc=caps_desc,
                    query=query
                )
                cap_resp = self.llm.invoke(cap_prompt)
                target_cap = cap_resp.content.strip()
            else:
                target_cap = "ALL"

            # 4. Filter Agents
            if target_cap and target_cap != "ALL" and target_cap in caps_map:
                scoped_agents = [
                    a for a in all_agents 
                    if a['intent_group'] == target_intent 
                    and a['capability_cluster'] == target_cap
                ]
            else:
                scoped_agents = [
                    a for a in all_agents 
                    if a['intent_group'] == target_intent
                ]

            decision = MultiRouteDecision(mode="react", scope=scoped_agents)
        else:
            decision = MultiRouteDecision(mode="react", scope=all_agents)

        self._cache[cache_key] = decision
        return decision

    def _hash(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def _should_use_agents(self, query: str) -> bool:
        prompt = ShouldUseAgentsPrompt.format(query=query)
        out = self.llm.invoke(prompt)
        return out.content.strip().lower() == "true"