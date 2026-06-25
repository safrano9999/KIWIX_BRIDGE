"""OpenAI-compatible LiteLLM proxy helpers for KIWIX_BRIDGE."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from python_header import get, get_int  # noqa: E402

_THINKING_BUDGETS = {"low": 1024, "medium": 5000, "high": 16000}
_REASONING_EFFORTS = {"low": "low", "medium": "medium", "high": "high"}
_SKIP_MODEL_PARTS = (
    "embed",
    "embedding",
    "tts",
    "dall-e",
    "whisper",
    "moderat",
    "audio",
    "realtime",
    "image",
    "search",
    "computer",
    "live",
)


def _clean_url(value: str) -> str:
    return value.strip().rstrip("/")


def litellm_base_url() -> str:
    host = _clean_url(get("LITELLM_URL", "http://127.0.0.1"))
    port = get_int("LITELLM_PORT", 4000)
    if host.endswith("/v1"):
        return host
    if re.search(r":\d+$", host):
        return f"{host}/v1"
    return f"{host}:{port}/v1"


def litellm_api_key() -> str:
    return get("LITELLM_API_KEY", "not-needed") or "not-needed"


def client(timeout: float = 60.0) -> OpenAI:
    return OpenAI(base_url=litellm_base_url(), api_key=litellm_api_key(), timeout=timeout)


def _is_chat_model(model: str) -> bool:
    lowered = model.lower()
    return bool(model.strip()) and not any(part in lowered for part in _SKIP_MODEL_PARTS)


def list_models() -> List[str]:
    try:
        response = client(timeout=10.0).models.list()
        models = [item.id for item in response.data if _is_chat_model(item.id)]
        return sorted(dict.fromkeys(models))
    except Exception:
        return []


def build_model_registry() -> Dict[str, List[str]]:
    registry: Dict[str, List[str]] = {}
    for model in list_models():
        provider = model.split("/", 1)[0] if "/" in model else "litellm"
        registry.setdefault(provider, []).append(model)
    return {provider: sorted(models) for provider, models in sorted(registry.items())}


def _model_base(model: str) -> str:
    return model.split("/")[-1].lower()


def _is_claude(model: str) -> bool:
    return "claude" in _model_base(model)


def _is_openai_reasoning(model: str) -> bool:
    return bool(re.match(r"^o[1-9]", _model_base(model)))


def chat_params(
    model: str,
    temperature: float | None = None,
    thinking: str = "off",
    max_tokens: int | None = None,
) -> dict:
    params: dict = {"model": model}
    extra_body: dict = {}

    budget = _THINKING_BUDGETS.get(thinking)
    if budget:
        if _is_claude(model):
            extra_body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            params["temperature"] = 1.0
        elif _is_openai_reasoning(model):
            extra_body["reasoning_effort"] = _REASONING_EFFORTS[thinking]
            if temperature is not None:
                params["temperature"] = temperature
        elif temperature is not None:
            params["temperature"] = temperature
    elif temperature is not None:
        params["temperature"] = temperature

    if max_tokens:
        params["max_tokens"] = max_tokens
    if extra_body:
        params["extra_body"] = extra_body
    return params


def chat_completion(**kwargs):
    return client().chat.completions.create(**kwargs)
