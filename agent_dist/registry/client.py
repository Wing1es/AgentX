import requests
from typing import Dict, List, Any

class RegistryClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def list_intents(self) -> Dict[str, Any]:
        """
        Used by Router Level 1.
        Endpoint: GET /intents
        Returns: {'Medical': {'description': '...'}, ...}
        """
        return self.session.get(f"{self.base_url}/intents").json()

    def list_capabilities(self, intent: str) -> Dict[str, str]:
        """
        Used by Router Level 2.
        Endpoint: GET /intents/{intent}/capabilities
        Returns: {'diagnosis': 'description...', ...}
        """
        url = f"{self.base_url}/intents/{intent}/capabilities"
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Registry Warning: Could not fetch capabilities for '{intent}': {e}")
            return {}

    def list_agents(self) -> List[Dict[str, Any]]:
        return self.session.get(f"{self.base_url}/agents").json()
    
    def get_agent(self, name: str) -> Dict[str, Any]:
        try:
            return self.session.get(f"{self.base_url}/agents/{name}").json()
        except:
            agents = self.list_agents()
            for a in agents:
                if a["name"] == name:
                    return a
            raise ValueError(f"Agent '{name}' not found in registry")
            agents = self.list_agents()
            for a in agents:
                if a["name"] == name:
                    return a
            raise ValueError(f"Agent '{name}' not found in registry")