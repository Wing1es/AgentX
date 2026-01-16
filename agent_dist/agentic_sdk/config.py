import os
from typing import Optional


AGENT_REGISTRY_URL: Optional[str] = os.getenv("AGENT_REGISTRY_URL")


HEARTBEAT_TTL: int = int(os.getenv("HEARTBEAT_TTL", "30"))

STRICT_REGISTRY_VALIDATION: bool = (
    os.getenv("STRICT_REGISTRY_VALIDATION", "false").lower() == "true"
)
