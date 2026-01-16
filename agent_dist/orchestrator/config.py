import os

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8000")
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 7001))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
