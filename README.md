# AgentX(agent_dist)

**Distributed Agentic Orchestration Framework**

`agent_dist` is a production-grade Python framework for building, registering, routing, and executing distributed AI agents using strict intent–capability contracts, hierarchical routing, and ReAct-style execution.

The framework is designed for **reliability, debuggability, and scalability** in complex multi-agent systems.

---

## Features

- Centralized agent registry with intent & capability schemas
- Strict agent contract validation
- Hierarchical intent → capability → agent routing
- ReAct-based multi-agent execution engine
- Heartbeat-based agent liveness
- SQLite-backed memory and trace storage
- CLI tools for trace inspection
- SDK for rapid agent development
- Supports OpenAI, Groq, and Ollama LLMs

---

## Installation

```bash
pip install agent_dist
```

### Optional provider dependencies:
```bash
pip install agent_dist[openai]
pip install agent_dist[groq]
pip install agent_dist[ollama]
```

# Running Core Services
The framework requires two core services to run: the Registry and the Orchestrator.

### 1. Start the Agent Registry
The registry manages intents, capabilities, agent registration, heartbeat tracking, and contract validation.
```bash
python -m agent_dist.registry.app
```

### 2. Start the Orchestrator
The orchestrator receives user queries, plans execution, coordinates agent calls, and manages memory.
```bash
python -m agent_dist.orchestrator.app
```

## Defining an Agent
Agents are created using the @agent decorator provided by the SDK. This handles automatic registration, input schema generation, and liveness checks.

```bash
from agent_dist.sdk import agent

@agent(
    url="http://localhost:9001/analyze",
    intent_group="general",
    capability_cluster="analysis",
    tasks=["analyze_text"],
    input_types=["json"],
    provides=["analysis.summary"]
)
def analyze(text: str):
    """
    Analyze input text and return a short summary.
    """
    return {"summary": text[:100]}

if __name__ == "__main__":
    analyze.serve()
```

#### SDK Capabilities

* Auto-Registration: Automatically registers the agent with the Registry upon startup.
* Schema Generation: Infers input schemas from Python type hints.
* Embedded Server: Launches a FastAPI server for the agent.
* Heartbeats: Sends automatic liveness checks to the Registry.

### Orchestrator API
Users interact exclusively with the Orchestrator via the /query endpoint.

Endpoint: `POST /query`

Request
```bash
JSON
{
  "query": "Analyze this document",
  "session_id": "optional-session-id",
  "debug": false
}
```

Response
```bash
JSON
{
  "answer": {
    "final_answer": "Analysis result",
    "trace": [],
    "total_duration_seconds": 0.92
  },
  "session_id": "generated-session-id",
  "plan": null
}
```
Note: If ```debug=true```, the response will include the detailed execution plan.

## Architecture & Concepts
![Agent Flow](images/agent_flow.png)

#### Routing System
* Routing is performed in multiple stages to ensure the correct agent is selected:
* Requirement Check: Determine whether agents are required for the query.
* Intent Classification: Classify the user's intent.
* Capability Selection: Select the appropriate capability cluster.
* Agent Filtering: Filter eligible agents from the registry.
* Routing decisions are cached using a hash of the query for performance.

#### Execution Engine (ReAct)
* The executor follows a ReAct-style loop:
* Thought Generation: The LLM reasons about the current state.
* Action Selection: The LLM selects an agent/tool to call.
* Observation: The output of the agent is fed back into the context.
* Iteration: Repeats until completion or the maximum step limit is reached.

#### Safety Controls
* Validation: Strict agent existence and capability validation.
* Filtering: Capability-based filtering prevents misuse.
* Step Limits: Maximum step enforcement to prevent infinite loops.
* Tracing: Every execution produces a structured trace.

## Memory & Tracing
The framework includes a built-in SQLite-backed memory system that handles conversation history per session and deterministic replay for debugging.

#### Trace CLI Tool
Inspect execution traces using the ```agent-trace``` CLI.

List Traces
```bash
agent-trace list
# With limit
agent-trace list --limit 20
```

Show a Specific Trace
```bash
agent-trace show <trace_id>
# Or using session ID
agent-trace show <session_uuid>
```
This prints the full execution trace in formatted JSON.

## Configuration
Configure the framework using environment variables.

```bash
# LLM Configuration
LLM_PROVIDER=ollama      # Options: ollama, openai, groq
LLM_MODEL=llama3
LLM_API_KEY=your_key
LLM_BASE_URL=http://localhost:11434

# System Configuration
AGENT_REGISTRY_URL=http://localhost:8000
STRICT_REGISTRY_VALIDATION=true
```

## Production Guidelines
1. Registry: Run a single registry instance with persistent storage.
2. Orchestrator: Scale orchestrators horizontally to handle load.
3. State: Keep individual agents stateless.
4. Validation: Always enable STRICT_REGISTRY_VALIDATION in production.
5. Monitoring: Inspect traces for all failure paths to improve agent reliability.

## License
MIT License
