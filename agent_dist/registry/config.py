import os

TEST_LOCAL: bool = os.getenv("TEST_LOCAL", "true").lower() == "true"
REGISTRY_PORT: int = int(os.getenv("REGISTRY_PORT", "8000"))

HEARTBEAT_TTL: int = int(os.getenv("HEARTBEAT_TTL", "30"))
CLEANUP_INTERVAL: int = int(os.getenv("CLEANUP_INTERVAL", "10"))

REGISTRY_DB: str = os.getenv("REGISTRY_DB", "registry.db")
