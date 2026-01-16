import copy
import time
import socket
import uvicorn
import threading
import sqlite3
import json
from typing import Dict
from fastapi import FastAPI, HTTPException

from .models import (
    AgentRegistration,
    AgentRecord,
    DeleteRequest,
    IntentCreate,
    CapabilityCreate,
    IntentBulkCreate,
    CapabilityBulkCreate,
    RegistrySchemaBulk
)
from .schemas import DEFAULT_INPUT_TYPES, INTENTS as DEFAULT_INTENTS
from .config import (
    HEARTBEAT_TTL,
    CLEANUP_INTERVAL,
    REGISTRY_DB,
    REGISTRY_PORT,
    TEST_LOCAL,
)


class Registry:
    def __init__(self):
        self.app = FastAPI(title="Hospital Agent Registry")
        self._lock = threading.RLock()

        self.intents = copy.deepcopy(DEFAULT_INTENTS)
        self.input_types = set(DEFAULT_INPUT_TYPES)
        self.agents: Dict[str, AgentRecord] = {}

        self.db = sqlite3.connect(REGISTRY_DB, check_same_thread=False)
        self._init_db()
        self._load_from_db()

        self._setup_routes()
        self._start_cleanup_loop()

    def _init_db(self):
        c = self.db.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS intents (name TEXT PRIMARY KEY, description TEXT)")
        c.execute(
            "CREATE TABLE IF NOT EXISTS capabilities (intent TEXT, name TEXT, description TEXT, PRIMARY KEY(intent, name))"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS agents (name TEXT PRIMARY KEY, intent TEXT, capability TEXT, payload TEXT)"
        )
        self.db.commit()

    def _load_from_db(self):
        c = self.db.cursor()
        with self._lock:
            self.intents.clear()
            for name, desc in c.execute("SELECT name, description FROM intents"):
                self.intents[name] = {"description": desc, "capabilities": {}}

            for intent, name, desc in c.execute("SELECT intent, name, description FROM capabilities"):
                self.intents[intent]["capabilities"][name] = {"description": desc}

            self.agents.clear()
            for name, intent, capability, payload in c.execute(
                "SELECT name, intent, capability, payload FROM agents"
            ):
                data = json.loads(payload)
                self.agents[name] = AgentRecord(
                    **data,
                    last_heartbeat=time.time()
                )

    def _setup_routes(self):

        @self.app.get("/intents")
        async def list_intents():
            return {
                intent: {"description": data["description"]}
                for intent, data in self.intents.items()
            }

        @self.app.get("/intents/{intent}")
        async def get_intent(intent: str):
            if intent not in self.intents:
                raise HTTPException(404, "Intent not found")
            return self.intents[intent]

        @self.app.get("/intents/{intent}/capabilities")
        async def list_capabilities(intent: str):
            if intent not in self.intents:
                raise HTTPException(404, "Intent not found")
            return {
                cap: data["description"]
                for cap, data in self.intents[intent]["capabilities"].items()
            }

        @self.app.post("/intents")
        async def add_intent(payload: IntentCreate):
            with self._lock:
                if payload.name in self.intents:
                    raise HTTPException(409, "Intent already exists")
                self.intents[payload.name] = {
                    "description": payload.description,
                    "capabilities": {}
                }
                self.db.execute(
                    "INSERT INTO intents VALUES (?, ?)",
                    (payload.name, payload.description)
                )
                self.db.commit()
            return {"status": "added", "intent": payload.name}
        
        @self.app.post("/intents/bulk")
        async def add_intents_bulk(payload: IntentBulkCreate):
            created, skipped = [], []

            with self._lock:
                for intent in payload.intents:
                    if intent.name in self.intents:
                        skipped.append(intent.name)
                        continue

                    self.intents[intent.name] = {
                        "description": intent.description,
                        "capabilities": {}
                    }

                    self.db.execute(
                        "INSERT OR IGNORE INTO intents VALUES (?, ?)",
                        (intent.name, intent.description)
                    )

                    created.append(intent.name)

                self.db.commit()

            return {
                "created": created,
                "skipped_existing": skipped
            }

        @self.app.delete("/intents/{intent}")
        async def delete_intent(intent: str):
            with self._lock:
                if intent not in self.intents:
                    raise HTTPException(404, "Intent not found")
                if self._intent_in_use(intent):
                    raise HTTPException(400, "Intent is in use by agents")
                self.intents.pop(intent)
                self.db.execute("DELETE FROM capabilities WHERE intent=?", (intent,))
                self.db.execute("DELETE FROM intents WHERE name=?", (intent,))
                self.db.commit()
            return {"status": "deleted", "intent": intent}

        @self.app.post("/intents/{intent}/capabilities")
        async def add_capability(intent: str, payload: CapabilityCreate):
            with self._lock:
                if intent not in self.intents:
                    raise HTTPException(404, "Intent not found")
                caps = self.intents[intent]["capabilities"]
                if payload.name in caps:
                    raise HTTPException(409, "Capability already exists")
                caps[payload.name] = {"description": payload.description}
                self.db.execute(
                    "INSERT INTO capabilities VALUES (?, ?, ?)",
                    (intent, payload.name, payload.description)
                )
                self.db.commit()
            return {"status": "added", "capability": payload.name}
        
        @self.app.post("/capabilities/bulk")
        async def add_capabilities_bulk(payload: CapabilityBulkCreate):
            created, skipped, errors = [], [], []

            with self._lock:
                for item in payload.capabilities:
                    if item.intent not in self.intents:
                        errors.append(
                            {"capability": item.name, "error": "Invalid intent"}
                        )
                        continue

                    caps = self.intents[item.intent]["capabilities"]

                    if item.name in caps:
                        skipped.append(f"{item.intent}:{item.name}")
                        continue

                    caps[item.name] = {"description": item.description}

                    self.db.execute(
                        "INSERT OR IGNORE INTO capabilities VALUES (?, ?, ?)",
                        (item.intent, item.name, item.description)
                    )

                    created.append(f"{item.intent}:{item.name}")

                self.db.commit()

            return {
                "created": created,
                "skipped_existing": skipped,
                "errors": errors
            }

        @self.app.delete("/intents/{intent}/capabilities/{capability}")
        async def delete_capability(intent: str, capability: str):
            with self._lock:
                if intent not in self.intents:
                    raise HTTPException(404, "Intent not found")
                caps = self.intents[intent]["capabilities"]
                if capability not in caps:
                    raise HTTPException(404, "Capability not found")
                if self._capability_in_use(intent, capability):
                    raise HTTPException(400, "Capability is in use by agents")
                caps.pop(capability)
                self.db.execute(
                    "DELETE FROM capabilities WHERE intent=? AND name=?",
                    (intent, capability)
                )
                self.db.commit()
            return {"status": "deleted", "capability": capability}
            
        @self.app.post("/schema/bulk")
        async def add_full_schema(payload: RegistrySchemaBulk):
            created = {"intents": [], "capabilities": []}

            with self._lock:
                for intent_name, intent_def in payload.intents.items():
                    if intent_name not in self.intents:
                        self.intents[intent_name] = {
                            "description": intent_def.description,
                            "capabilities": {}
                        }
                        self.db.execute(
                            "INSERT OR IGNORE INTO intents VALUES (?, ?)",
                            (intent_name, intent_def.description)
                        )
                        created["intents"].append(intent_name)

                    for cap in intent_def.capabilities:
                        caps = self.intents[intent_name]["capabilities"]
                        if cap.name in caps:
                            continue

                        caps[cap.name] = {"description": cap.description}
                        self.db.execute(
                            "INSERT OR IGNORE INTO capabilities VALUES (?, ?, ?)",
                            (intent_name, cap.name, cap.description)
                        )
                        created["capabilities"].append(
                            f"{intent_name}:{cap.name}"
                        )

                self.db.commit()

            return created

        @self.app.post("/register")
        async def register_agent(agent: AgentRegistration, overwrite: bool = False):
            with self._lock:
                self._validate_agent(agent)
                if agent.name in self.agents and not overwrite:
                    raise HTTPException(409, "Agent already exists")
                record = AgentRecord(**agent.model_dump(), last_heartbeat=time.time())
                self.agents[agent.name] = record
                self.db.execute(
                    "INSERT OR REPLACE INTO agents VALUES (?, ?, ?, ?)",
                    (
                        agent.name,
                        agent.intent_group,
                        agent.capability_cluster,
                        json.dumps(agent.model_dump())
                    )
                )
                self.db.commit()
            return {"status": "registered", "agent": agent.name}

        @self.app.post("/agents/{name}/heartbeat")
        async def heartbeat(name: str):
            with self._lock:
                agent = self.agents.get(name)
                if not agent:
                    raise HTTPException(404, "Agent not found")
                agent.last_heartbeat = time.time()
            return {"status": "alive"}

        @self.app.get("/agents")
        async def list_agents():
            return list(self.agents.values())

        @self.app.get("/agents/{name}")
        async def get_agent(name: str):
            agent = self.agents.get(name)
            if not agent:
                raise HTTPException(404, "Agent not found")
            return agent

        @self.app.delete("/agents/{name}")
        async def delete_agent(name: str):
            with self._lock:
                if name not in self.agents:
                    raise HTTPException(404, "Agent not found")
                self.agents.pop(name)
                self.db.execute("DELETE FROM agents WHERE name=?", (name,))
                self.db.commit()
            return {"status": "deleted", "agent": name}

        @self.app.delete("/agents/by-intent/{intent}")
        async def delete_agents_by_intent(intent: str):
            with self._lock:
                if intent not in self.intents:
                    raise HTTPException(404, "Intent not found")
                to_delete = [
                    name for name, agent in self.agents.items()
                    if agent.intent_group == intent
                ]
                for name in to_delete:
                    self.agents.pop(name, None)
                    self.db.execute("DELETE FROM agents WHERE name=?", (name,))
                self.db.commit()
            if not to_delete:
                return {"status": "no_agents", "intent": intent}
            return {
                "status": "deleted",
                "intent": intent,
                "count": len(to_delete),
                "agents": to_delete
            }
        
        @self.app.get("/agents/contracts")
        async def list_agent_contracts():
            return {
                name: {
                    "requires": agent.capabilities.requires,
                    "provides": agent.capabilities.provides,
                }
                for name, agent in self.agents.items()
            }

        @self.app.get("/ping")
        async def ping():
            return {
                "status": "alive",
                "total_agents": len(self.agents),
                "agents": list(self.agents.keys())
            }

    def _intent_in_use(self, intent: str) -> bool:
        return any(agent.intent_group == intent for agent in self.agents.values())

    def _capability_in_use(self, intent: str, capability: str) -> bool:
        return any(
            agent.intent_group == intent and agent.capability_cluster == capability
            for agent in self.agents.values()
        )

    def _validate_agent(self, agent: AgentRegistration):
        if agent.intent_group not in self.intents:
            raise HTTPException(400, "Invalid intent_group")
        allowed_caps = self.intents[agent.intent_group]["capabilities"]
        if agent.capability_cluster not in allowed_caps:
            raise HTTPException(
                400,
                f"Capability '{agent.capability_cluster}' not allowed for intent '{agent.intent_group}'"
            )
        caps = agent.capabilities
        if not caps.tasks:
            raise HTTPException(400, "capabilities.tasks required")
        if not caps.input_types:
            raise HTTPException(400, "capabilities.input_types required")
        for itype in caps.input_types:
            if itype not in self.input_types:
                raise HTTPException(400, f"Invalid input_type: {itype}")
        
        if agent.tags and not all(isinstance(t, str) for t in agent.tags):
            raise HTTPException(400, "tags must be a list of strings")
        
        caps = agent.capabilities

        if not isinstance(caps.requires, list):
            raise HTTPException(400, "capabilities.requires must be a list")

        if not isinstance(caps.provides, list):
            raise HTTPException(400, "capabilities.provides must be a list")

        for k in caps.requires + caps.provides:
            if not isinstance(k, str):
                raise HTTPException(
                    400,
                    "capabilities.requires/provides must be list of strings"
                )

    def _start_cleanup_loop(self):
        def cleanup():
            while True:
                time.sleep(CLEANUP_INTERVAL)
                with self._lock:
                    dead = [
                        name for name, agent in self.agents.items()
                        if not agent.is_alive(HEARTBEAT_TTL)
                    ]
                    for name in dead:
                        self.agents.pop(name, None)
        threading.Thread(target=cleanup, daemon=True).start()

    @staticmethod
    def get_local_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip

    def run(self, test_local: bool = True, port: int = 8000):
        host = "127.0.0.1" if test_local else self.get_local_ip()
        print(f"\nRegistry running at: http://{host}:{port}\n")
        uvicorn.run(self.app, host=host, port=port)

def run():
    Registry().run(test_local=TEST_LOCAL, port=REGISTRY_PORT)

# For uvicorn
registry_service = Registry()
app = registry_service.app


if __name__ == "__main__":
    run()