#!/usr/bin/env python3
"""
KIWIX_BRIDGE - interactive CLI chat with local Wikipedia tool.
Uses the OpenAI-compatible LiteLLM proxy configured through config.conf/.env.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

_venv_site = Path(__file__).parent.parent / "venv" / "lib"
if _venv_site.exists():
    for _p in _venv_site.glob("python*/site-packages"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from python_header import get  # noqa: E402
from kiwix_tool import wikipedia_lookup  # noqa: E402
from llm_proxy import build_model_registry, chat_completion, chat_params  # noqa: E402

logging.basicConfig(level=logging.WARNING)


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


def _message_to_dict(message) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return dict(message)


def call_llm(model_key: str, messages: List[Dict]) -> str:
    """Run one turn: LLM -> optional tool calls -> final answer."""
    kwargs: Dict = {
        **chat_params(model_key),
        "messages": messages,
        "tools": [WIKIPEDIA_TOOL],
    }

    while True:
        response = chat_completion(**kwargs)
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            return msg.content or ""

        kwargs["messages"] = list(kwargs["messages"]) + [_message_to_dict(msg)]

        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            query = args.get("query", "")
            lang = args.get("lang", "de")
            print(f"  \033[2m[Kiwix -> {query!r} ({lang})]\033[0m")
            result = wikipedia_lookup(query=query, lang=lang)
            kwargs["messages"].append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


def pick_model(registry: Dict[str, List[str]]) -> Optional[str]:
    providers = sorted(registry.keys())
    if not providers:
        return None

    print("\nVerfügbare Provider:")
    for i, provider in enumerate(providers, 1):
        print(f"  {i}) {provider}  ({len(registry[provider])} Modelle)")
    try:
        provider = providers[int(input("Provider wählen [Nr]: ").strip()) - 1]
    except (ValueError, IndexError):
        print("Ungültige Auswahl.")
        return None

    models = registry[provider]
    print(f"\nModelle für {provider}:")
    for i, model in enumerate(models, 1):
        print(f"  {i:3}) {model}")
    try:
        return models[int(input("Modell wählen [Nr]: ").strip()) - 1]
    except (ValueError, IndexError):
        print("Ungültige Auswahl.")
        return None


def main():
    registry = build_model_registry()
    if not registry:
        print("Fehler: Keine Modelle vom LiteLLM Proxy verfügbar.")
        print("Setze LITELLM_API_KEY in .env und LITELLM_URL/LITELLM_PORT in config.conf.")
        sys.exit(1)

    model_key = pick_model(registry)
    if not model_key:
        sys.exit(1)

    print(f"\nModell: {model_key}")
    print(f"Wikipedia: lokal via Kiwix ({get('KIWIX_URL', 'https://127.0.0.1:450')})")
    print("Chat gestartet - 'quit' oder Strg+C zum Beenden\n")

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
        except Exception as exc:
            print(f"\nFehler: {exc}\n")
            messages.pop()


if __name__ == "__main__":
    main()
