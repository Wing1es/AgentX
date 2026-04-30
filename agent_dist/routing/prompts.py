from langchain_core.prompts import PromptTemplate

# =========================================================================
#  GATEKEEPER
# =========================================================================
SHOULD_USE_AGENTS_TEMPLATE = """
You are a decision-making system.
Determine if the following user query requires using external tools/agents to be answered, 
or if it is a general chat/greeting that you can answer yourself.

USER QUERY:
{query}

INSTRUCTIONS:
- Return "TRUE" if the query asks for specific data, actions, or domain knowledge (e.g., "price of wheat", "diagnose patient", "send email").
- Return "FALSE" if the query is a greeting, philosophy, general knowledge, or pure conversation (e.g., "hi", "how are you", "what is the meaning of life").
- Output ONLY "TRUE" or "FALSE".

DECISION:
"""

ShouldUseAgentsPrompt = PromptTemplate(
    template=SHOULD_USE_AGENTS_TEMPLATE,
    input_variables=["query"]
)

# =========================================================================
#  REACT LOOP
# =========================================================================
REACT_SYSTEM_PROMPT = """
You are a ReAct agent. You solve tasks by thinking, acting, and observing.

AVAILABLE TOOLS:
{tools_desc}

USER QUERY:
{query}

HISTORY:
{history}

INSTRUCTIONS:
1. Think about what to do next.
2. Select ONE tool to call.
3. Output MUST follow this strict format:

Thought: [your reasoning]
Action: [AgentName]
Action Input: {{ "key": "value" }}

(If you have the final answer)
Thought: I have the answer.
Final Answer: [your response to the user]

RULES:
- Use ONLY the tools listed above.
- The "Action Input" must be valid JSON matching the tool's requirements.
- If you need to pass data from a previous observation, include it in the JSON.
- Do not make up tools.
"""

ReactPrompt = PromptTemplate(
    template=REACT_SYSTEM_PROMPT,
    input_variables=["tools_desc", "query", "history"]
)