import re
import logging
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from agent_dist.llm import get_llm
from .models import MultiRouteDecision, ExecutionPlan, PlanStep

logger = logging.getLogger("orchestrator.planner")

PLANNER_TEMPLATE = """
You are a Planning Agent. Break down the user's query into ordered steps.

AVAILABLE AGENTS:
{agents_desc}

USER QUERY:
{query}

INSTRUCTIONS:
1. Decide which agent(s) to call and in what order.
2. Respect contracts: if Agent B requires "diagnosis_code", schedule Agent A (which provides it) FIRST.
3. Output a numbered list. Each line MUST start with the agent name in square brackets.

FORMAT (strict):
1. [AgentName] What this step does
2. [AgentName] What this step does

EXAMPLE:
1. [Cardiologist] Diagnose the patient's chest pain
2. [Billing] Process payment using the diagnosis code from step 1
3. [Pharmacy] Fill the prescription using prescription_id and payment_confirmation

RULES:
- ONLY use agent names from the AVAILABLE AGENTS list.
- If the query needs just one agent, output one step.
- Output ONLY the numbered list. No explanation.

PLAN:
"""

class PlannerAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = PromptTemplate(
            template=PLANNER_TEMPLATE,
            input_variables=["agents_desc", "query"]
        )
        self._all_agents: List[Dict] = []  # Full registry for dependency expansion

    def set_all_agents(self, agents: List[Dict]):
        """Set the full agent list for contract-aware scope expansion."""
        self._all_agents = agents

    def plan(self, decision: MultiRouteDecision, query: str) -> ExecutionPlan:
        if not decision.scope or (decision.mode not in ["react", "plan"]):
            return ExecutionPlan(mode=decision.mode, steps=[], active_agents=decision.scope)

        # ── Expand scope: pull in agents that provide required fields (Recursive) ──
        expanded = list(decision.scope)
        expanded_names = {a["name"] for a in expanded}
        
        # Build global provides index from all known agents
        all_provides: Dict[str, Dict] = {}  # field -> agent
        for a in self._all_agents:
            caps = a if isinstance(a, dict) else a
            c = caps.get("capabilities", {})
            for field in c.get("provides", []):
                all_provides[field] = caps

        changed = True
        while changed:
            changed = False
            # Current pool of provided items
            scope_provides = set()
            for a in expanded:
                for field in a.get("capabilities", {}).get("provides", []):
                    scope_provides.add(field)

            # Find missing dependencies
            for a in list(expanded):
                caps = a.get("capabilities", {})
                for req_field in caps.get("requires", []):
                    if req_field not in scope_provides and req_field in all_provides:
                        provider = all_provides[req_field]
                        if provider["name"] not in expanded_names:
                            logger.info(f"Expanding scope: adding '{provider['name']}' (provides '{req_field}' for {a['name']})")
                            expanded.append(provider)
                            expanded_names.add(provider["name"])
                            changed = True

        # Forward expansion: include agents whose requires are NOW satisfiable
        MAX_SCOPE_SIZE = 15

        changed = True
        while changed:
            changed = False
            if len(expanded) >= MAX_SCOPE_SIZE:
                logger.warning('Scope expansion halted at limit')
                break

            scope_provides = set()
            for a in expanded:
                for field in a.get('capabilities', {}).get('provides', []):
                    scope_provides.add(field)

            # Only pull in agents needed to satisfy EXISTING scope requires
            # NOT any agent that happens to be satisfiable (prevents cascades)
            needed_fields = set()
            for a in expanded:
                for req in a.get('capabilities', {}).get('requires', []):
                    if req not in scope_provides:
                        needed_fields.add(req)

            for a in self._all_agents:
                if a['name'] in expanded_names:
                    continue
                provs = set(a.get('capabilities', {}).get('provides', []))
                if provs & needed_fields:  # only add if filling a gap
                    expanded.append(a)
                    expanded_names.add(a['name'])
                    changed = True
                    if len(expanded) >= MAX_SCOPE_SIZE:
                        break

        # Build agent map for lookup
        agent_map = {a["name"]: a for a in expanded}
        
        # Describe agents with full contract info
        agents_desc = []
        for a in expanded:
            caps = a.get("capabilities", {})
            requires = caps.get("requires", [])
            provides = caps.get("provides", [])
            desc = f"- {a['name']}: {a['description']} | requires: {requires} | provides: {provides}"
            agents_desc.append(desc)
        agents_desc_str = "\n".join(agents_desc)
        
        # Ask LLM to plan
        chain = self.prompt | self.llm
        result = chain.invoke({"agents_desc": agents_desc_str, "query": query})
        content = result.content if hasattr(result, "content") else str(result)
        
        logger.info(f"Planner raw output:\n{content}")
        
        # Parse steps: extract [AgentName] from each numbered line
        steps = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            
            # Extract [AgentName]
            match = re.search(r'\[([^\]]+)\]', line)
            if not match:
                continue
            
            agent_name = match.group(1).strip()
            
            # Fuzzy match if exact name not found
            if agent_name not in agent_map:
                matched = None
                for known in agent_map:
                    if known.lower() == agent_name.lower():
                        matched = known
                        break
                    if known.lower() in agent_name.lower() or agent_name.lower() in known.lower():
                        matched = known
                        break
                if matched:
                    agent_name = matched
                else:
                    logger.warning(f"Planner: unknown agent '{agent_name}', skipping")
                    continue
            
            agent = agent_map[agent_name]
            caps = agent.get("capabilities", {})
            
            # Extract description (everything after [AgentName])
            desc_part = line.split("]", 1)[-1].strip()
            if desc_part.startswith("."):
                desc_part = desc_part[1:].strip()
            
            steps.append(PlanStep(
                agent_name=agent_name,
                agent_url=agent.get("url", ""),
                description=desc_part or f"Call {agent_name}",
                requires=caps.get("requires", []),
                provides=caps.get("provides", []),
            ))
        
        # Deduplicate: remove consecutive duplicate agents
        deduped = []
        last_agent = None
        for s in steps:
            if s.agent_name != last_agent:
                deduped.append(s)
                last_agent = s.agent_name
        steps = deduped
        
        # Fallback: if parsing failed, create one step per agent
        if not steps:
            logger.warning("Planner: parsing failed, defaulting to all agents in order")
            for a in expanded:
                caps = a.get("capabilities", {})
                steps.append(PlanStep(
                    agent_name=a["name"],
                    agent_url=a.get("url", ""),
                    description=f"Call {a['name']}: {a['description']}",
                    requires=caps.get("requires", []),
                    provides=caps.get("provides", []),
                ))

        logger.info(f"Plan: {[f'{s.agent_name}({s.requires}->{s.provides})' for s in steps]}")
        
        return ExecutionPlan(
            mode=decision.mode,
            steps=steps,
            active_agents=decision.scope
        )