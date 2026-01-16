from langchain_core.prompts import PromptTemplate

ReActSystemPrompt = """
You are a Reasoning Agent (ReAct).

Goal: {goal}

Previous Conversation:
{chat_history}

Tools Available:
{tools}

Instructions:
1. "Thought": Analyze the current situation and previous history. Decide what to do next.
2. "Action": Call a tool if needed.
   Format: Action: ToolName(input_json)
   Example: Action: Search({{"query": "hospital hours"}})
3. "Observation": (Wait for the system to provide the result of the action).
4. "Final Answer": If you have enough information to solve context, stop.
   Format: Final Answer: <your response>

Constraints:
- Only use the tools provided.
- If you have the answer, output "Final Answer: ...".
- Loops are allowed if you need more info.

History:
{history}

Begin!
"""

INTENT_CLASSIFIER_TEMPLATE = """
You are a semantic router for an agent system.
Your goal is to route the user's query to the correct Intent Group.

AVAILABLE INTENT GROUPS:
{intents_desc}

USER QUERY:
{query}

INSTRUCTIONS:
1. Analyze the query.
2. Select the ONE intent group from the list above that best handles this query.
3. If the query is unclear or matches multiple, strictly output "UNKNOWN".
4. Output ONLY the name of the intent group. No markdown, no explanation.

### ROUTING EXAMPLES (CRITICAL):
Query: "I feel dizzy and have a sharp pain in my chest" -> Medical
Query: "Check the pulse of the database" -> System
Query: "What is the dosage for 500mg Paracetamol?" -> Medical
Query: "How is the health of the server?" -> System

### TASK:
Analyze the user query and pick the correct Intent.
Query: {query}
Target Intent:
"""

IntentClassifierPrompt = PromptTemplate(
    template=INTENT_CLASSIFIER_TEMPLATE,
    input_variables=["intents_desc", "query"]
)


CAPABILITY_CLASSIFIER_TEMPLATE = """
You are a routing assistant within the '{intent}' domain.
Your goal is to select the specific CAPABILITY required for the user's query.

AVAILABLE CAPABILITIES:
{caps_desc}

USER QUERY:
{query}

INSTRUCTIONS:
1. Select the ONE capability that best matches the query.
2. If the query requires multiple capabilities or is unclear, output "ALL".
3. Output ONLY the capability name (or "ALL").

TARGET CAPABILITY:
"""

CapabilityClassifierPrompt = PromptTemplate(
    template=CAPABILITY_CLASSIFIER_TEMPLATE,
    input_variables=["intent", "caps_desc", "query"]
)

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