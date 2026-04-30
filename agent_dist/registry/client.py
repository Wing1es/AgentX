import requests
from typing import Dict, List, Any

class RegistryClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()



    def list_agents(self) -> List[Dict[str, Any]]:
        return self.session.get(f"{self.base_url}/agents").json()
    
    
    def search_agents(self, query: str, limit: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Semantic search for agents.
        Endpoint: GET /search
        """
        try:
            return self.session.get(
                f"{self.base_url}/search",
                params={"query": query, "limit": limit, "threshold": threshold}
            ).json()
        except requests.exceptions.RequestException as e:
            print(f"Registry Warning: Search failed: {e}")
            return []

    def get_agent(self, name: str) -> Dict[str, Any]:
        try:
            return self.session.get(f"{self.base_url}/agents/{name}").json()
        except:
            agents = self.list_agents()
            for a in agents:
                if a["name"] == name:
                    return a
            raise ValueError(f"Agent '{name}' not found in registry")