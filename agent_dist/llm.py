import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm():
    """
    Factory function to initialize the LLM based on environment variables.
    Supported Providers: 'ollama', 'groq', 'openai'.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    if provider == "groq":
        from langchain_groq import ChatGroq
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
        
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            api_key=api_key
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=api_key,
            base_url=base_url
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or "llama3:8b",
            base_url=base_url or "http://localhost:11434"
        )
    
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")