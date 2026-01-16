import os
import inspect
import threading
import time
import requests
import functools
import uvicorn
from fastapi import FastAPI, Request
from urllib.parse import urlparse
from typing import get_origin, get_args, Union, List, Dict, Any, Optional

from .config import (
    AGENT_REGISTRY_URL,
    HEARTBEAT_TTL,
    STRICT_REGISTRY_VALIDATION,
)


class _SDKConfig:
    registry_url: Optional[str] = None
    heartbeat_interval: int = HEARTBEAT_TTL
    strict_validation: bool = STRICT_REGISTRY_VALIDATION


_CONFIG = _SDKConfig()


def configure(
    *,
    registry_url: Optional[str] = None,
    heartbeat_interval: Optional[int] = None,
    strict_validation: Optional[bool] = None
):
    if registry_url:
        _CONFIG.registry_url = registry_url
    if heartbeat_interval:
        _CONFIG.heartbeat_interval = heartbeat_interval
    if strict_validation is not None:
        _CONFIG.strict_validation = strict_validation


def _resolve_registry_url(override: Optional[str] = None) -> str:
    if override:
        return override
    if AGENT_REGISTRY_URL:
        return AGENT_REGISTRY_URL
    if _CONFIG.registry_url:
        return _CONFIG.registry_url
    return "http://localhost:8000"


def list_intents(registry_url: Optional[str] = None) -> Dict[str, Dict]:
    url = _resolve_registry_url(registry_url)
    return requests.get(f"{url}/intents", timeout=5).json()


def list_capabilities(intent: str, registry_url: Optional[str] = None) -> Dict[str, Dict]:
    url = _resolve_registry_url(registry_url)
    return requests.get(f"{url}/intents/{intent}/capabilities", timeout=5).json()


def _validate_against_registry(intent: str, capability: str, registry_url: str):
    intents = list_intents(registry_url)
    if intent not in intents:
        raise ValueError(f"Unknown intent_group: {intent}")
    caps = list_capabilities(intent, registry_url)
    if capability not in caps:
        raise ValueError(
            f"Capability '{capability}' not allowed for intent '{intent}'"
        )


def _pytype_to_json_schema(annotation):
    if annotation is inspect._empty:
        raise ValueError("Missing type annotation")

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)][0]
        schema = _pytype_to_json_schema(non_none)
        return {**schema, "nullable": True}

    if origin in (list, List):
        if not args:
            return {"type": "array"}
        return {
            "type": "array",
            "items": _pytype_to_json_schema(args[0]),
        }

    if origin in (dict, Dict):
        return {"type": "object"}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list:
        return {"type": "array"}
    if annotation is dict:
        return {"type": "object"}

    if annotation is Any:
        return {"type": "object"}

    raise ValueError(f"Unsupported type annotation: {annotation}")


def _build_input_schema(func):
    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        if param.annotation is inspect._empty:
            raise ValueError(
                f"Parameter '{name}' in agent '{func.__name__}' must have a type annotation"
            )

        schema = _pytype_to_json_schema(param.annotation)
        properties[name] = schema

        if param.default is inspect._empty:
            required.append(name)

    if not properties:
        raise ValueError(
            f"Agent '{func.__name__}' must define at least one typed parameter"
        )

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _start_heartbeat(agent_name: str, registry_url: str, interval: int):
    def beat():
        while True:
            try:
                requests.post(
                    f"{registry_url}/agents/{agent_name}/heartbeat",
                    timeout=2,
                )
            except Exception:
                pass
            time.sleep(interval)

    threading.Thread(target=beat, daemon=True).start()


def agent(
    *,
    url: str,
    intent_group: str,
    capability_cluster: str,
    tasks: List[str],
    input_types: List[str],
    requires: Optional[List[str]] = None,
    provides: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    output_types: Optional[List[str]] = None,
    compliance: Optional[List[str]] = None,
    version: str = "1.0.0",
    expose_schema: bool = False,
    registry_url: Optional[str] = None,
    overwrite: bool = False,
):
    def decorator(func):
        description = (func.__doc__ or "").strip()
        resolved_registry = _resolve_registry_url(registry_url)

        if _CONFIG.strict_validation:
            _validate_against_registry(
                intent_group,
                capability_cluster,
                resolved_registry,
            )

        input_schema = _build_input_schema(func)

        if tags is not None and not all(isinstance(t, str) for t in tags):
            raise ValueError("tags must be a list of strings")

        payload = {
            "name": func.__name__,
            "description": description,
            "tags": tags or [],
            "url": url,
            "intent_group": intent_group,
            "capability_cluster": capability_cluster,
            "version": version,
            "capabilities": {
                "tasks": tasks,
                "input_types": input_types,
                "requires": requires or [],
                "provides": provides or [],
                "compliance": compliance or [],
                **({"input_schema": input_schema} if expose_schema else {}),
            },
        }

        try:
            response = requests.post(
                f"{resolved_registry}/register",
                json=payload,
                params={"overwrite": overwrite},
                timeout=5,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 409:
                raise RuntimeError(e.response.text) from None
        except Exception as e:
            raise RuntimeError(str(e)) from None

        _start_heartbeat(
            func.__name__,
            resolved_registry,
            _CONFIG.heartbeat_interval,
        )

        def serve():
            """
            Starts a FastAPI server for this agent.
            """
            # Extract port from URL
            parsed = urlparse(url)
            port = parsed.port or 8000
            host = parsed.hostname or "0.0.0.0"
            
            app = FastAPI(title=func.__name__, version=version)

            @app.post(parsed.path or "/")
            async def handle_request(request: Request):
                if "json" in input_types:
                    body = await request.json()
                    try:
                        return func(**body)
                    except Exception as e:
                        return {
                            "error": str(e),
                            "received_payload": body,
                        }
                elif "text" in input_types:
                    text = await request.body()
                    return func(text.decode())
                else:
                    return func()

            print(f"Starting agent '{func.__name__}' on {host}:{port}")
            uvicorn.run(app, host=host, port=port)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.serve = serve
        return wrapper

    return decorator
