"""
KIWIX_BRIDGE - Interactive CLI chat with local Wikipedia tool
Uses LiteLLM (provider/model from .env) + Kiwix function calling.
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional

import litellm
from dotenv import load_dotenv
from kiwix_tool import wikipedia_lookup

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("kiwix_bridge")

# ── Env loading ────────────────────────────────────────────────────────────────

_DIR = Path(__file__).parent

def _load_dotenv(filename: str):
    """Load .filename from own dir first, then sibling CLAWBRIDGE dir."""
    own  = _DIR / filename
    boss = _DIR.parent / "CLAWBRIDGE" / filename
    if own.exists():
        load_dotenv(own, override=False)
    elif boss.exists():
        load_dotenv(boss, override=False)

_load_dotenv(".env")

# ── Provider/model detection (same pattern as TELEGRAM-AI-BOT) ────────────────

_ENV_TO_PROVIDER = {"google": "gemini"}

_NON_LLM_SUFFIXES = frozenset([
    "BRAVE_API_KEY", "TAVILY_API_KEY", "SERPAPI_KEY", "SERPER_API_KEY",
    "DALLE_API_KEY", "STABILITY_API_KEY", "REPLICATE_API_KEY", "FAL_API_KEY",
    "WHISPER_API_KEY", "GEMINI_VOICE_KEY", "ELEVENLABS_API_KEY", "OPENAI_TTS_KEY",
])


def parse_env_providers() -> Dict[str, str]:
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    providers = {}
    for key, value in os.environ.items():
        if not key.endswith("_API_KEY") or not value or value in ("...", ""):
            continue
        if key in _NON_LLM_SUFFIXES:
            continue
        raw      = key.removesuffix("_API_KEY").lower()
        provider = _ENV_TO_PROVIDER.get(raw, raw)
        providers[provider] = value
    return providers


def get_models_for_provider(provider: str) -> List[str]:
    all_models = litellm.models_by_provider.get(provider, [])
    prefix     = f"{provider}/"
    bare       = [m[len(prefix):] if m.startswith(prefix) else m for m in all_models]
    known      = set(litellm.model_cost.keys())
    skip       = ("embed", "tts", "dall-e", "whisper", "moderat", "audio",
                  "realtime", "image", "search", "computer", "live")
    chat_models = [m for m in bare
                   if (f"{provider}/{m}" in known or m in known)
                   and not any(s in m.lower() for s in skip)]
    return sorted(chat_models)


def build_model_registry() -> Dict[str, List[str]]:
    registry = {}
    for provider, _ in parse_env_providers().items():
        models = get_models_for_provider(provider)
        if models:
            registry[provider] = models
    return registry


# ── Tool definition ────────────────────────────────────────────────────────────

WIKIPEDIA_TOOL = {
    "type": "function",
    "function": {
        "name": "wikipedia_lookup",
        "description": (
            "Nachschlagen in der lokalen offline Wikipedia (Kiwix). "
            "Verwende dieses Tool wenn du dir bei konkreten Fakten, Daten, Namen, "
            "Zahlen, Orten oder historischen Ereignissen nicht 100% sicher bist. "
            "Antworte direkt aus deinem Wissen wenn die Frage allgemeines Verständnis, "
            "Konzepte oder Reasoning erfordert."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Suchbegriff für Wikipedia (Artikelname oder Schlagwort)",
                },
                "lang": {
                    "type": "string",
                    "enum": ["de", "en"],
                    "description": "Sprache: 'de' für Deutsch (Standard), 'en' für Englisch",
                },
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent mit Zugriff auf eine lokale offline Wikipedia (Kiwix). "
    "Nutze das wikipedia_lookup Tool gezielt für faktische Fragen wo Genauigkeit wichtig ist. "
    "Für allgemeine Erklärungen, Konzepte und Reasoning antworte direkt. "
    "Wenn du Kiwix abgefragt hast, gib die Quelle kurz an."
)

# ── LLM call with agentic tool loop ───────────────────────────────────────────

def call_llm(model_key: str, messages: List[Dict]) -> str:
    """Run one turn: LLM → optional tool calls → final answer."""
    provider   = model_key.split("/")[0] if "/" in model_key else ""
    api_key    = parse_env_providers().get(provider, "")
    kwargs: Dict = {
        "model":    model_key,
        "messages": messages,
        "tools":    [WIKIPEDIA_TOOL],
        "timeout":  60,
    }
    if api_key:
        kwargs["api_key"] = api_key

    # Agentic loop
    while True:
        response = litellm.completion(**kwargs)
        msg      = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            return msg.content or ""

        # Append assistant message with tool_calls
        kwargs["messages"] = list(kwargs["messages"]) + [msg]

        # Execute each tool call
        for tc in tool_calls:
            import json
            args   = json.loads(tc.function.arguments)
            query  = args.get("query", "")
            lang   = args.get("lang", "de")
            print(f"  \033[2m[Kiwix → {query!r} ({lang})]\033[0m")
            result = wikipedia_lookup(query=query, lang=lang)
            kwargs["messages"].append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

# ── Model selection ────────────────────────────────────────────────────────────

def pick_model(registry: Dict[str, List[str]]) -> Optional[str]:
    """Let user pick provider and model interactively."""
    providers = sorted(registry.keys())
    if not providers:
        return None

    print("\nVerfügbare Provider:")
    for i, p in enumerate(providers, 1):
        print(f"  {i}) {p}  ({len(registry[p])} Modelle)")
    try:
        idx = int(input("Provider wählen [Nr]: ").strip()) - 1
        provider = providers[idx]
    except (ValueError, IndexError):
        print("Ungültige Auswahl.")
        return None

    models = registry[provider]
    print(f"\nModelle für {provider}:")
    for i, m in enumerate(models, 1):
        print(f"  {i:3}) {m}")
    try:
        idx   = int(input("Modell wählen [Nr]: ").strip()) - 1
        model = models[idx]
    except (ValueError, IndexError):
        print("Ungültige Auswahl.")
        return None

    return f"{provider}/{model}"

# ── Main chat loop ─────────────────────────────────────────────────────────────

def main():
    registry = build_model_registry()
    if not registry:
        print("Fehler: Keine LLM API Keys in .env gefunden.")
        print("Lege eine .env Datei an mit z.B. ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)

    model_key = pick_model(registry)
    if not model_key:
        sys.exit(1)

    print(f"\nModell: {model_key}")
    print("Wikipedia: lokal via Kiwix (https://127.0.0.1:450)")
    print("Chat gestartet — 'quit' oder Strg+C zum Beenden\n")

    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("Du: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTschüss!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Tschüss!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            answer = call_llm(model_key, messages)
            print(f"\nAssistent: {answer}\n")
            messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"\nFehler: {e}\n")
            messages.pop()  # don't keep failed user message in history


if __name__ == "__main__":
    main()
