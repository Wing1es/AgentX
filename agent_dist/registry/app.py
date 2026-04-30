import copy
import time
import socket
import uvicorn
import threading
import sqlite3
import json
from typing import Dict
from fastapi import FastAPI, HTTPException, Query
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .models import (
    AgentRegistration,
    AgentRecord,
    DeleteRequest
)
from .schemas import DEFAULT_INPUT_TYPES
from .config import (
    HEARTBEAT_TTL,
    CLEANUP_INTERVAL,
    REGISTRY_DB,
    REGISTRY_PORT,
    TEST_LOCAL,
)


class Registry:
    def __init__(self, db_path: str = None, model_name: str = 'all-MiniLM-L6-v2'):
        self.app = FastAPI(title="Hospital Agent Registry")
        self._lock = threading.RLock()

        self.input_types = set(DEFAULT_INPUT_TYPES)
        self.agents: Dict[str, AgentRecord] = {}
        self.agent_vectors: Dict[str, np.ndarray] = {}

        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"Loaded embedding model: {model_name}")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.model = None
        else:
            print("SentenceTransformer not installed. Vector search disabled.")
            self.model = None

        self.db_path = db_path or REGISTRY_DB
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        self._load_from_db()

        self._setup_routes()
        self._start_cleanup_loop()

    def _init_db(self):
        c = self.db.cursor()

        try:
            c.execute("SELECT embedding FROM agents LIMIT 1")
        except sqlite3.OperationalError:
            pass

        c.execute(
            "CREATE TABLE IF NOT EXISTS agents (name TEXT PRIMARY KEY, payload TEXT, embedding BLOB)"
        )
        
        try:
            c.execute("ALTER TABLE agents ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass
            
        self.db.commit()

    def _load_from_db(self):
        c = self.db.cursor()
        with self._lock:
            self.agents.clear()
            self.agent_vectors.clear()
            for name, payload, embedding_blob in c.execute(
                "SELECT name, payload, embedding FROM agents"
            ):
                data = json.loads(payload)
                self.agents[name] = AgentRecord(
                    **data,
                    last_heartbeat=time.time()
                )
                if embedding_blob:
                    self.agent_vectors[name] = np.frombuffer(embedding_blob, dtype=np.float32)


    async def register_agent(self, agent: AgentRegistration, overwrite: bool = False):
        with self._lock:
            self._validate_agent(agent)
            if agent.name in self.agents and not overwrite:
                raise HTTPException(409, "Agent already exists")
            
            record = AgentRecord(**agent.model_dump(), last_heartbeat=time.time())
            self.agents[agent.name] = record
            
            # Compute embedding
            embedding_blob = None
            if self.model:
                text_to_embed = f"{agent.name} {agent.description} {' '.join(agent.tags)}"
                embedding = self.model.encode(text_to_embed)
                self.agent_vectors[agent.name] = embedding
                embedding_blob = embedding.tobytes()

            self.db.execute(
                "INSERT OR REPLACE INTO agents (name, payload, embedding) VALUES (?, ?, ?)",
                (
                    agent.name,
                    json.dumps(agent.model_dump()),
                    embedding_blob
                )
            )
            self.db.commit()
        return {"status": "registered", "agent": agent.name}

    async def search_agents(self, query: str, limit: int = 5, threshold: float = 0.3):
        # 1. Vector Search
        if self.model:
            q_vec = self.model.encode(query)
            results = []
            with self._lock:
                for name, vec in self.agent_vectors.items():
                    if vec is None or name not in self.agents:
                        continue
                    # Cosine similarity
                    sim = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
                    if sim >= threshold:
                        results.append((sim, self.agents[name]))
            results.sort(key=lambda x: x[0], reverse=True)
            if results:
                return [agent for _, agent in results[:limit]]

        # 2. Keyword Fallback (if no vector model OR no vector match)
        STOP_WORDS = {"a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
                       "at", "by", "for", "with", "about", "to", "from", "up", "down",
                       "in", "out", "on", "off", "over", "under", "is", "are", "was",
                       "were", "be", "been", "being", "do", "does", "did", "will", "would",
                       "shall", "should", "may", "might", "must", "can", "could",
                       "i", "my", "me", "we", "you", "he", "she", "it", "they",
                       "this", "that", "these", "those", "there", "here",
                       "who", "what", "which", "how", "where", "when", "why",
                       "not", "no", "nor", "so", "too", "very", "just", "also",
                       "need", "have", "has", "had", "get", "got"}
        
        print(f"Registry: Falling back to keyword search for '{query}'")
        query_terms = [t.strip(".,?!") for t in query.lower().split() if t.strip(".,?!") not in STOP_WORDS]
        if not query_terms:
            return []

        scores = []
        with self._lock:
            for name, agent in self.agents.items():
                score = 0
                searchable = f"{name} {agent.description} {' '.join(agent.tags)}".lower()
                
                for term in query_terms:
                    if term in searchable:
                        score += 1
                
                if score > 0:
                    scores.append((score, agent))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [agent for _, agent in scores[:limit]]

    def _setup_routes(self):


        @self.app.post("/register")
        async def register_agent_route(agent: AgentRegistration, overwrite: bool = False):
            return await self.register_agent(agent, overwrite)
        
        @self.app.get("/search")
        async def search_agents_route(query: str, limit: int = 5, threshold: float = 0.3):
            return await self.search_agents(query, limit, threshold)

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

    def _validate_agent(self, agent: AgentRegistration):
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