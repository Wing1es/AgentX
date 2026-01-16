import asyncio
import json
import httpx
import time
from typing import Dict, Any, List, Literal, AsyncGenerator
from agent_dist.registry.client import RegistryClient
from .models import ExecutionPlan
import logging

logger = logging.getLogger("orchestrator.executor")


class Executor:
    def __init__(self, llm, registry: RegistryClient):
        self.llm = llm
        self.registry = registry

    def _normalize_agent(self, agent) -> Dict[str, Any]:
        return agent if isinstance(agent, dict) else agent.model_dump()

    async def execute(self, plan: ExecutionPlan, user_query: str, history: List[Dict] = None) -> AsyncGenerator[Dict[str, Any], None]:
        context: Dict[str, Any] = {
            "ctx.user.query": user_query,
            "ctx.sys.trace": [],
            "ctx.sys.history": history or []
        }

        # Directly call ReAct execution
        async for event in self._execute_react(plan, context):
            yield event

    async def _execute_react(self, plan: ExecutionPlan, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        from .prompts import ReActSystemPrompt
        
        user_query = context["ctx.user.query"]

        if plan.active_agents is not None:
            agents = plan.active_agents
            logger.info(f"Executor running with restricted scope: {[a['name'] for a in agents]}")
        else:
            agents = self.registry.list_agents()
        
        agent_map = {a["name"]: a for a in agents}
        
        tools_desc = []
        for a in agents:
            caps = a.get("capabilities", {})

            # 1. Extract Inputs (Requires)
            requires = caps.get("requires", [])
            req_clean = [r.split(".")[-1] for r in requires]

            # 2. Extract Outputs (Provides)
            provides = caps.get("provides", [])
            prov_clean = [p.split(".")[-1] for p in provides]

            # 3. Format Agent Description
            desc = (
                f"- {a['name']}({', '.join(req_clean)}) "
                f"-> [{', '.join(prov_clean)}]: {a['description']}"
            )
            tools_desc.append(desc)
        
        tools_desc = "\n".join(tools_desc)
        
        # Format history into a string
        chat_history_str = ""
        user_confirmed = False
        if context.get("ctx.sys.history"):
            for h in context["ctx.sys.history"]:
                role = h.get("role", "unknown")
                content = h.get("content", "")
                chat_history_str += f"{role.capitalize()}: {content}\n"
                
                # Simple heuristic for confirmation in history
                if role == "user" and any(w in content.lower() for w in ["yes", "proceed", "confirm", "sure", "ok", "go ahead"]):
                    user_confirmed = True
        
        # Internal scratchpad history for the current ReAct loop
        scratchpad = ""
        
        max_steps = 10
        for i in range(max_steps):
            step_start_time = time.time()
            prompt = ReActSystemPrompt.format(
                goal=user_query,
                chat_history=chat_history_str,
                tools=tools_desc,
                history=scratchpad
            )
            
            # YIELD STREAM: Thought Process Starting
            yield {"type": "thought", "content": f"Step {i+1}: Thinking..."}
            
            result = self.llm.invoke(prompt)
            content = result.content if hasattr(result, "content") else str(result)
            
            logger.info(f"Step {i+1} Thought: {content[:100]}...")
            yield {"type": "thought", "content": content} # Stream full thought
            
            # Simple parsing of multiple lines
            lines = content.strip().split("\n")
            action_line = next((l for l in lines if l.startswith("Action:")), None)
            final_line = next((l for l in lines if l.startswith("Final Answer:")), None)
            
            scratchpad += f"\nStep {i+1}:\n{content}\n"
            
            if final_line:
                answer = final_line.replace("Final Answer:", "").strip()
                logger.info(f"ReAct Final Answer: {answer}")
                yield {"type": "final", "content": answer}
                return 
            
            if action_line:
                # Parse "Action: ToolName(json)"
                try:
                    part = action_line.replace("Action:", "").strip()
                    logger.info(f"ReAct Action: {part}")
                    agent_name = part.split("(")[0].strip()
                    args_str = part[len(agent_name):].strip().strip("()")
                    
                    if not args_str:
                        payload = {}
                    else:
                        payload = json.loads(args_str)
                    
                    # HITL CHECK
                    # Dynamic check based on Agent Registry metadata
                    agent_obj = agent_map.get(agent_name, {})
                    requires_confirmation = agent_obj.get("requires_confirmation", False)
                    # Also check inside capability dict if not found at top level (flexibility)
                    if not requires_confirmation:
                        requires_confirmation = agent_obj.get("capabilities", {}).get("requires_confirmation", False)

                    if requires_confirmation and not user_confirmed:
                        # STOP! Ask for permission.
                        obs = "Observation: SYSTEM ALERT: This action is sensitive. You MUST ask the user for confirmation before proceeding."
                        yield {"type": "observation", "content": obs}
                        yield {"type": "thought", "content": "Action blocked by Safety System. Asking user for confirmation."}
                        # We don't return here, we let the loop continue so the agent sees the observation and generates a Final Answer asking the user.
                    
                    elif agent_name not in agent_map:
                        obs = f"Observation: Error: Agent '{agent_name}' not found."
                        logger.warning(f"ReAct Agent Not Found: {agent_name}")
                        yield {"type": "observation", "content": obs}
                    else:
                        agent = agent_map[agent_name]
                        yield {"type": "action", "agent": agent_name, "input": payload}
                        
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            logger.info(f"Calling Agent: {agent['url']}")
                            
                            agent_start = time.time()
                            resp = await client.post(agent["url"], json=payload)
                            
                            if resp.status_code >= 400:
                                obs = f"Observation: Error {resp.status_code}: {resp.text}"
                                logger.error(f"Agent Error: {resp.status_code}")
                            else:
                                obs = f"Observation: {json.dumps(resp.json())}"
                                logger.info(f"Agent Response: {obs[:100]}...")
                                yield {"type": "observation", "content": obs}
                                
                    scratchpad += f"\n{obs}\n"
                    
                except Exception as e:
                    scratchpad += f"\nObservation: Error parsing/executing action: {str(e)}\n"
                    logger.error(f"ReAct Action Error: {e}")
                    yield {"type": "observation", "content": f"Error: {str(e)}"}
            else:
                 scratchpad += "\nObservation: No action found. Waiting for next thought.\n"
                 logger.info("No action found in step.")
                 yield {"type": "thought", "content": "No action found. Continuing..."}

        yield {"type": "final", "content": "Goal not reached max steps."}