import json
import httpx
import time
import asyncio
import re
from typing import Dict, Any, List, AsyncGenerator
from agent_dist.registry.client import RegistryClient
from .models import ExecutionPlan, PlanStep
from .prompts import ReactPrompt
from langchain_core.prompts import PromptTemplate
import logging

logger = logging.getLogger("orchestrator.executor")

SUMMARIZE_PROMPT = """
You are a helpful assistant. The user asked: "{query}"

The following agents were called and produced these results:
{trace_summary}

Accumulated data:
{state}

Write a clear, helpful response to the user based ONLY on the data above.
Do NOT add information that wasn't returned by the agents.
"""

class Executor:
    def __init__(self, llm, registry: RegistryClient):
        self.llm = llm
        self.registry = registry

    async def execute(self, plan: ExecutionPlan, user_query: str, history: List[Dict] = None) -> AsyncGenerator[Dict[str, Any], None]:
        context: Dict[str, Any] = {
            "ctx.user.query": user_query,
            "ctx.sys.history": history or []
        }

        if plan.mode == "react":
            # ══════════════════════════════════════════
            #  RE-ACT MODE: Loop "Thought -> Action -> Observation"
            # ══════════════════════════════════════════
            logger.info("Executing in TRUE ReAct mode")
            async for event in self._execute_react(plan, user_query):
                yield event
        elif plan.steps:
            # ══════════════════════════════════════════
            #  PLAN-THEN-EXECUTE: Execute fixed plan
            # ══════════════════════════════════════════
            logger.info("Executing in Plan-then-Execute mode")
            async for event in self._execute_plan_dynamic(plan, context):
                yield event
        else:
            # LLM-ONLY MODE: Direct answer
            logger.info("Executing in LLM-Only mode")
            result = self.llm.invoke(user_query)
            content = result.content if hasattr(result, "content") else str(result)
            yield {"type": "final", "content": content, "state": {}}

    async def _execute_react(self, plan: ExecutionPlan, user_query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        True ReAct Loop: Thought -> Action -> Observation -> Thought...
        """
        trace = [] # Raw text history for prompt
        actions_log = [] # Structured log for final summary
        state: Dict[str, Any] = {}
        MAX_ITERATIONS = 25
        
        # Build tool description string
        agents_desc = []
        agent_map = {}
        if plan.active_agents:
            for a in plan.active_agents:
                # Handle both dict and object
                aname = a.get("name") if isinstance(a, dict) else a.name
                adesc = a.get("description") if isinstance(a, dict) else a.description
                aurl = a.get("url") if isinstance(a, dict) else a.url
                
                # Capabilities
                item = a if isinstance(a, dict) else a.model_dump()
                caps = item.get("capabilities", {})
                reqs = caps.get("requires", [])
                provs = caps.get("provides", [])
                
                agent_map[aname] = item
                agents_desc.append(f"- {aname}: {adesc}\n  Inputs: {reqs}\n  Outputs: {provs}")
        
        tools_desc_str = "\n".join(agents_desc)

        yield {"type": "plan", "content": [], "mode": "react", "description": "Starting ReAct Loop..."}

        for i in range(MAX_ITERATIONS):
            # 1. THINK
            history_str = "\n".join(trace)
            prompt = ReactPrompt.format(tools_desc=tools_desc_str, query=user_query, history=history_str)
            
            logger.info(f"ReAct Iteration {i+1}...")
            
            try:
                res = self.llm.invoke(prompt)
                llm_out = res.content if hasattr(res, "content") else str(res)
            except Exception as e:
                yield {"type": "error", "content": f"LLM error: {e}"}
                break
            
            # Log raw thought for history context
            trace.append(f"Iteration {i+1}: {llm_out}")

            # Parse Output (Thought/Action/Action Input)
            # LLM output example:
            # Thought: I need to call X first.
            # Action: AgentX
            # Action Input: {"key": "value"}
            
            lines = llm_out.strip().split("\n")
            thought = ""
            action = ""
            action_input = ""
            final_answer = ""
            
            # Simple line parsing
            for line in lines:
                if line.startswith("Thought:"):
                    thought = line.replace("Thought:", "").strip()
                elif line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
                elif line.startswith("Final Answer:"):
                    # Finding strict final answer
                    idx = llm_out.find("Final Answer:")
                    final_answer = llm_out[idx + len("Final Answer:"):].strip()
                    break

            if final_answer:
                yield {"type": "final", "content": final_answer, "state": state, "steps_completed": i+1}
                return

            if not action:
                # Sometimes LLM outputs just text if it's chatting or confused
                if "Action:" not in llm_out:
                     # Treat as final answer if it looks like one, or retry
                     # Check iteration count
                     if i > 5:
                         yield {"type": "final", "content": llm_out, "state": state}
                         return
                     trace.append(f"System: Invalid Output. You must output 'Action: [AgentName]' or 'Final Answer:'.")
                     continue

            payload = self._extract_json(llm_out)
            
            # Merge state into payload automatically for convenience
            # Creating a hybrid payload: explicit args override state
            merged_payload = {"query": user_query}
            merged_payload.update(state)
            merged_payload.update(payload)

            yield {"type": "step_start", "step": i+1, "agent": action, "description": thought or "Executing..."}
            
            # Check if valid agent
            if action not in agent_map:
                err = f"Action '{action}' is not in valid tools list."
                trace.append(f"Observation: Error: {err}")
                yield {"type": "step_failed", "step": i+1, "agent": action, "error": err}
                continue

            # 2. ACT
            agent_data = agent_map[action]
            url = agent_data.get("url", "")
            
            obs_text = ""
            try:
                t0 = time.time()
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, json=merged_payload)
                    latency = time.time() - t0
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # ── CONTRACT ENFORCEMENT (ReAct) ──
                        caps = agent_data.get("capabilities", {})
                        contracted_provides = caps.get("provides", [])
                        
                        if contracted_provides and isinstance(data, dict):
                            # Check: agent MUST return ALL contracted fields
                            missing = [k for k in contracted_provides if k not in data]
                            if missing:
                                obs_text = f"Observation: CONTRACT VIOLATION - {action} failed to provide: {missing}"
                                logger.warning(f"Contract violation: {action} missing provides: {missing}")
                                yield {"type": "step_failed", "step": i+1, "agent": action, "error": obs_text}
                                trace.append(obs_text)
                                continue
                            
                            # Only inject contracted fields into state (no uncontracted leakage)
                            for key in contracted_provides:
                                state[key] = data[key]
                        else:
                            state.update(data)
                        
                        obs_text = f"Observation: {json.dumps(data)}"
                        yield {"type": "step_done", "step": i+1, "agent": action, "result": data, "latency": latency}
                    else:
                        obs_text = f"Observation: HTTP Error {resp.status_code} - {resp.text}"
                        yield {"type": "step_failed", "step": i+1, "agent": action, "error": obs_text}
            except Exception as e:
                obs_text = f"Observation: Exception {str(e)}"
                yield {"type": "step_failed", "step": i+1, "agent": action, "error": obs_text}

            trace.append(obs_text)
            
        yield {"type": "final", "content": "Max iterations reached without final answer.", "state": state}
    
    
    # ══════════════════════════════════════════
    #  PLAN-THEN-EXECUTE (Legacy / Optimized)
    # ══════════════════════════════════════════
    async def _extract_missing(self, query, keys, history):
        prompt = (
            f'Extract the following fields from the User Query.\n'
            f'Fields needed: {keys}\n'
            f'User Query: "{query}"\n\n'
            'STRICT RULES:\n'
            '- Return ONLY a JSON object with keys from the list.\n'
            '- A value MUST appear verbatim in the query text.\n'
            '- If NOT in the query, DO NOT include that key.\n'
            '- Never infer or generate plausible values.\n\n'
            'JSON:'
        )
        try:
            res = self.llm.invoke(prompt)
            content = res.content if hasattr(res, 'content') else str(res)
            content = content.strip()
            start = content.find('{')
            end = content.rfind('}') + 1
            if start == -1:
                return {}
            extracted = json.loads(content[start:end])

            # ---- HALLUCINATION GUARD ---------------------------------
            # Reject any value that cannot be found in the original query
            query_lower = query.lower()
            validated = {}
            for k, v in extracted.items():
                if str(v).lower() in query_lower:
                    validated[k] = v
                else:
                    logger.warning(
                        f'Hallucination guard: rejected {k}={v!r} '
                        f'(not found in query)'
                    )
            return validated
        except Exception as e:
            logger.warning(f'Extraction failed for {keys}: {e}')
        return {}

    async def _run_single_step(self, step: PlanStep, user_query: str, state: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        """Executes a single step and returns a report. Safe to run in parallel tasks."""
        max_retries = 2
        step.status = "running"
        report = {
            "step_num": step_num,
            "agent": step.agent_name,
            "events": [],
            "new_state": {},
            "success": False
        }
        
        report["events"].append({"type": "step_start", "step": step_num, "agent": step.agent_name, "description": step.description})

        # Build payload
        payload = {"query": user_query}
        for field in step.requires:
            if field in state:
                payload[field] = state[field]

        # Retry Loop
        for attempt in range(1 + max_retries):
            try:
                t0 = time.time()
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(step.agent_url, json=payload)
                
                latency = time.time() - t0

                if resp.status_code >= 400:
                    error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    step.error = error_msg
                    
                    if attempt < max_retries:
                         report["events"].append({"type": "step_error", "step": step_num, "agent": step.agent_name, "error": error_msg, "attempt": attempt + 1})
                         continue
                    else:
                        step.status = "failed"
                        report["events"].append({"type": "step_failed", "step": step_num, "agent": step.agent_name, "error": error_msg})
                        break
                
                # Success
                resp_data = resp.json()
                step.result = resp_data
                
                # ── CONTRACT ENFORCEMENT (Plan Mode) ──
                if isinstance(resp_data, dict) and step.provides:
                    missing_provides = [k for k in step.provides if k not in resp_data]
                    if missing_provides:
                        error_msg = f"CONTRACT VIOLATION: {step.agent_name} did not provide: {missing_provides}"
                        logger.warning(error_msg)
                        step.status = "failed"
                        step.error = error_msg
                        report["events"].append({"type": "step_failed", "step": step_num, "agent": step.agent_name, 
                                                "error": error_msg})
                        break
                    
                    # Only inject contracted fields (strict isolation)
                    for key in step.provides:
                        report["new_state"][key] = resp_data[key]
                elif isinstance(resp_data, dict):
                    report["new_state"].update(resp_data)
                
                step.status = "completed"
                report["events"].append({"type": "step_done", "step": step_num, "agent": step.agent_name, 
                                        "result": resp_data, "latency": round(latency, 3)})
                report["success"] = True
                break

            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries:
                    report["events"].append({"type": "step_error", "step": step_num, "agent": step.agent_name, "error": error_msg, "attempt": attempt + 1})
                    continue
                else:
                    step.status = "failed"
                    report["events"].append({"type": "step_failed", "step": step_num, "agent": step.agent_name, "error": error_msg})
                    break
        
        return report

    async def _execute_plan_dynamic(self, plan: ExecutionPlan, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        user_query = context["ctx.user.query"]
        state: Dict[str, Any] = {}
        trace_summary = []
        
        remaining_indices = list(range(len(plan.steps)))
        running_tasks = {} # index -> Task
        
        # Initial Plan Event
        yield {"type": "plan", "content": [
            {"step": i+1, "agent": s.agent_name, "description": s.description}
            for i, s in enumerate(plan.steps)
        ]}

        while remaining_indices or running_tasks:
            
            # 1. Identify Ready Steps
            future_providers = set()
            for idx in remaining_indices:
                future_providers.update(plan.steps[idx].provides)
            for idx in running_tasks:
                future_providers.update(plan.steps[idx].provides)
                
            newly_started = []
            
            for idx in list(remaining_indices):
                step = plan.steps[idx]
                missing = [k for k in step.requires if k not in state]
                
                other_providers = set()
                for p_idx in remaining_indices:
                    if p_idx != idx: other_providers.update(plan.steps[p_idx].provides)
                for r_idx in running_tasks:
                     other_providers.update(plan.steps[r_idx].provides)
                
                can_start = True
                
                if missing:
                    blocked_keys = [k for k in missing if k in other_providers]
                    if blocked_keys:
                        can_start = False
                    else:
                        logger.info(f"Step {idx+1} ({step.agent_name}) missing {missing}. Attempting extraction...")
                        extracted = await self._extract_missing(user_query, missing, [])
                        if extracted:
                            state.update(extracted)
                            still_missing = [k for k in step.requires if k not in state]
                            if still_missing:
                                can_start = False
                                step.status = "failed"
                                failure_msg = f"Missing inputs: {still_missing}. Extraction failed."
                                step.error = failure_msg
                                yield {"type": "step_failed", "step": idx+1, "agent": step.agent_name, "error": failure_msg}
                                remaining_indices.remove(idx)
                                continue
                        else:
                            can_start = False
                            step.status = "failed"
                            failure_msg = f"Missing inputs: {missing}. Not provided by any agent."
                            step.error = failure_msg
                            yield {"type": "step_failed", "step": idx+1, "agent": step.agent_name, "error": failure_msg}
                            remaining_indices.remove(idx)
                            continue

                if can_start:
                    logger.info(f"Starting parallel step {idx+1}: {step.agent_name}")
                    remaining_indices.remove(idx)
                    task = asyncio.create_task(self._run_single_step(step, user_query, state.copy(), idx+1))
                    running_tasks[idx] = task
                    newly_started.append(idx)
            
            if remaining_indices and not running_tasks:
                msg = f"Deadlock! Remaining steps {remaining_indices} cannot start due to missing dependencies."
                logger.error(msg)
                yield {"type": "error", "content": msg}
                break

            if not running_tasks:
                break
                
            done, pending_set = await asyncio.wait(running_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            
            for idx, task in list(running_tasks.items()):
                if task in done:
                    report = task.result()
                    for event in report["events"]:
                        if event["type"] == "step_done":
                             event["state_keys"] = list(state.keys()) + list(report["new_state"].keys())
                        yield event
                    
                    if report["success"]:
                        state.update(report["new_state"])
                        trace_summary.append(f"Step {report['step_num']} [{report['agent']}]: {json.dumps(report.get('new_state', {}))}")
                    else:
                        trace_summary.append(f"Step {report['step_num']} [{report['agent']}]: FAILED")
                    
                    del running_tasks[idx]
            
        completed_steps = [s for s in plan.steps if s.status == "completed"]
        failed_steps = [s for s in plan.steps if s.status == "failed"]
        
        if completed_steps:
            summary_prompt = SUMMARIZE_PROMPT.format(
                query=user_query,
                trace_summary="\n".join(trace_summary),
                state=json.dumps(state, indent=2)
            )
            try:
                result = self.llm.invoke(summary_prompt)
                final_answer = result.content if hasattr(result, "content") else str(result)
            except Exception as e:
                final_answer = f"Completed {len(completed_steps)}/{len(plan.steps)} steps."
        else:
            final_answer = f"All steps failed."
        
        if failed_steps:
             final_answer += f"\n\n⚠️ {len(failed_steps)} step(s) failed."

        yield {"type": "final", "content": final_answer, "state": state, 
               "steps_completed": len(completed_steps), "steps_failed": len(failed_steps)}
    
    def _extract_json(self, text: str) -> dict:
    # Strategy 1: find JSON after 'Action Input:' label
        if 'Action Input:' in text:
            after = text.split('Action Input:', 1)[1].strip()
            if after.startswith('{'):
                depth = 0
                for i, ch in enumerate(after):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(after[:i+1])
                            except json.JSONDecodeError:
                                break

        # Strategy 2: scan entire text for balanced braces
        depth, start_idx = 0, None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start_idx is not None:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        start_idx = None

        logger.warning('JSON extraction failed; using empty payload')
        return {}