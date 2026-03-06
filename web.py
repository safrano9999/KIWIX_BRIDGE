#!/usr/bin/env python3
"""
KIWIX_BRIDGE web.py — Q&A with local Wikipedia (Kiwix) + LiteLLM
Always fetches Wikipedia context first (RAG), then streams LLM answer.
Works with small/offline models (no function calling needed).
Run: python web.py
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import List, Dict

# ── venv sys.path trick ──────────────────────────────────────────────────────
_venv_site = Path(__file__).parent / "venv" / "lib"
if _venv_site.exists():
    for _p in _venv_site.glob("python*/site-packages"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string
import litellm
from dotenv import load_dotenv
from kiwix_tool import build_context

logging.basicConfig(level=logging.WARNING)

# ── Env loading ──────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent

def _load_dotenv(filename: str):
    own  = _DIR / filename
    boss = _DIR.parent / "CLAWBRIDGE" / filename
    if own.exists():
        load_dotenv(own, override=False)
    elif boss.exists():
        load_dotenv(boss, override=False)

_load_dotenv(".env")

# ── Provider / model registry ────────────────────────────────────────────────
_ENV_TO_PROVIDER = {"google": "gemini"}
_NON_LLM_SUFFIXES = frozenset([
    "BRAVE_API_KEY", "TAVILY_API_KEY", "SERPAPI_KEY", "SERPER_API_KEY",
    "DALLE_API_KEY", "STABILITY_API_KEY", "REPLICATE_API_KEY", "FAL_API_KEY",
    "WHISPER_API_KEY", "GEMINI_VOICE_KEY", "ELEVENLABS_API_KEY", "OPENAI_TTS_KEY",
])

def parse_env_providers() -> Dict[str, str]:
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    out = {}
    for key, value in os.environ.items():
        if not key.endswith("_API_KEY") or not value or value in ("...", ""):
            continue
        if key in _NON_LLM_SUFFIXES:
            continue
        raw      = key.removesuffix("_API_KEY").lower()
        provider = _ENV_TO_PROVIDER.get(raw, raw)
        out[provider] = value
    return out

def _litellm_models_for(provider: str) -> List[str]:
    all_m  = litellm.models_by_provider.get(provider, [])
    prefix = f"{provider}/"
    bare   = [m[len(prefix):] if m.startswith(prefix) else m for m in all_m]
    known  = set(litellm.model_cost.keys())
    skip   = ("embed", "tts", "dall-e", "whisper", "moderat", "audio",
              "realtime", "image", "search", "computer", "live")
    return sorted(m for m in bare
                  if (f"{provider}/{m}" in known or m in known)
                  and not any(s in m.lower() for s in skip))

def _get_ollama_models() -> List[str]:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        return sorted(m["name"] for m in resp.json().get("models", []))
    except Exception:
        return []

def _get_kilocode_models(api_key: str) -> List[str]:
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.kilo.ai/api/gateway/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = []
        for m in (data.get("data", data) if isinstance(data, dict) else data):
            mid = m.get("id", "") if isinstance(m, dict) else str(m)
            if mid:
                models.append(mid)
        skip = ("embed", "tts", "image", "whisper", "moderat", "audio")
        return sorted(m for m in models if not any(s in m.lower() for s in skip))
    except Exception:
        return []

def build_model_registry() -> Dict[str, List[str]]:
    registry = {}

    # API key providers
    for provider, api_key in parse_env_providers().items():
        if provider == "kilocode":
            models = _get_kilocode_models(api_key)
        else:
            models = _litellm_models_for(provider)
        if models:
            registry[provider] = models

    # Ollama (local, no key needed)
    ollama_models = _get_ollama_models()
    if ollama_models:
        registry["ollama"] = ollama_models

    return registry

# ── System prompt (RAG style — works with any model size) ────────────────────
def build_system_prompt(has_context: bool) -> str:
    if has_context:
        return (
            "Du bist ein präziser Fakten-Assistent. Du bekommst Wikipedia-Artikel als Kontext "
            "und beantwortest Fragen ausschließlich darauf basierend.\n\n"
            "REGELN:\n"
            "- Stütze dich NUR auf die bereitgestellten Artikel — keine eigenen Behauptungen\n"
            "- Wenn die Artikel die Frage nicht beantworten können, sage das klar\n"
            "- Antworte in der Sprache der gestellten Frage\n"
            "- Fasse dich präzise und klar — keine Füllsätze"
        )
    else:
        return (
            "Du bist ein hilfreicher Assistent. "
            "Antworte präzise. Bei unsicheren Fakten weise darauf hin dass du dir nicht sicher bist."
        )

# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kiwix Bridge</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Courier New', monospace; background: #0d0d0d; color: #e0e0e0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  /* Top bar */
  #topbar {
    background: #111; border-bottom: 1px solid #2a2a2a;
    padding: 10px 16px; display: flex; align-items: center; gap: 12px; flex-shrink: 0;
  }
  #topbar-title { font-size: 13px; color: #7eb8f7; letter-spacing: 2px; text-transform: uppercase; margin-right: 8px; white-space: nowrap; }

  select {
    background: #1a1a1a; color: #ccc; border: 1px solid #333;
    font-family: 'Courier New', monospace; font-size: 12px;
    padding: 5px 8px; border-radius: 3px; cursor: pointer;
  }
  select:focus { outline: none; border-color: #7eb8f7; }
  #provider-select { min-width: 130px; }
  #model-select    { min-width: 220px; flex: 1; max-width: 380px; }

  .lang-btn {
    background: #1a1a1a; color: #555; border: 1px solid #2a2a2a;
    font-family: 'Courier New', monospace; font-size: 11px; padding: 5px 10px;
    border-radius: 3px; cursor: pointer; transition: all 0.15s;
  }
  .lang-btn.active { background: #1e2a3a; color: #7eb8f7; border-color: #2a5298; }

  #kiwix-indicator {
    margin-left: auto; font-size: 10px; color: #333; display: flex; align-items: center; gap: 5px;
    white-space: nowrap;
  }
  #kiwix-dot { width: 6px; height: 6px; border-radius: 50%; background: #2a2a2a; display: inline-block; }
  #kiwix-dot.on { background: #5cb85c; box-shadow: 0 0 5px #5cb85c; }

  /* Results */
  #results { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 28px; }

  .qa-block { display: flex; flex-direction: column; gap: 10px; }

  .q-row { display: flex; gap: 10px; align-items: flex-start; }
  .q-label { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; padding-top: 2px; min-width: 18px; }
  .q-text  { color: #a8c7e8; font-size: 13px; line-height: 1.6; }

  .a-row { display: flex; gap: 10px; align-items: flex-start; }
  .a-label { font-size: 10px; color: #444; text-transform: uppercase; letter-spacing: 1px; padding-top: 2px; min-width: 18px; }
  .a-body  { flex: 1; }
  .a-text  { color: #e0e0e0; font-size: 13px; line-height: 1.7; white-space: pre-wrap; }

  .citations {
    margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  }
  .cite-label { font-size: 10px; color: #3a3a3a; }
  .cite-link {
    font-size: 11px; color: #3d6b6b; text-decoration: none;
    border: 1px solid #1e3a3a; padding: 2px 8px; border-radius: 2px;
    transition: all 0.15s;
  }
  .cite-link:hover { color: #5cc8c8; border-color: #2a5a5a; background: #0d1e1e; }

  .copy-btn {
    margin-left: auto; background: none; color: #333; border: 1px solid #222;
    font-family: 'Courier New', monospace; font-size: 11px; padding: 2px 10px;
    border-radius: 2px; cursor: pointer; white-space: nowrap; transition: all 0.15s;
  }
  .copy-btn:hover { color: #7eb8f7; border-color: #2a5298; }
  .copy-btn.copied { color: #5cb85c; border-color: #3a6b3a; }

  .no-kiwix { font-size: 11px; color: #2a4040; font-style: italic; margin-top: 6px; }

  .searching { color: #3d5a3d; font-size: 11px; font-style: italic; }
  @keyframes blink { 0%,100%{opacity:0.3} 50%{opacity:1} }
  .cursor { animation: blink 0.8s infinite; }

  /* Input bar */
  #inputbar {
    padding: 12px 16px; border-top: 1px solid #1e1e1e;
    background: #0d0d0d; display: flex; gap: 10px; flex-shrink: 0;
  }
  #input {
    flex: 1; background: #141414; color: #e0e0e0;
    border: 1px solid #2a2a2a; border-radius: 4px;
    padding: 10px 12px; font-family: 'Courier New', monospace; font-size: 13px;
    resize: none; height: 44px; line-height: 1.4;
  }
  #input:focus { outline: none; border-color: #7eb8f7; }
  #ask-btn {
    padding: 0 20px; background: #1e3a5a; color: #7eb8f7;
    border: 1px solid #2a5298; border-radius: 4px; cursor: pointer;
    font-family: 'Courier New', monospace; font-size: 13px;
  }
  #ask-btn:hover { background: #2a5298; }
  #ask-btn:disabled { opacity: 0.35; cursor: default; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0d0d0d; }
  ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
</style>
</head>
<body>

<div id="topbar">
  <span id="topbar-title">KIWIX BRIDGE</span>
  <select id="provider-select" onchange="onProviderChange()">
    <option value="">Lade...</option>
  </select>
  <select id="model-select">
    <option value="">— Provider wählen —</option>
  </select>
  <button class="lang-btn active" id="btn-de" onclick="setLang('de')">DE</button>
  <button class="lang-btn"        id="btn-en" onclick="setLang('en')">EN</button>
  <div id="kiwix-indicator">
    <span id="kiwix-dot"></span> Kiwix offline
  </div>
</div>

<div id="results"></div>

<div id="inputbar">
  <textarea id="input" placeholder="Frage stellen..." rows="1"></textarea>
  <button id="ask-btn" onclick="ask()">Fragen</button>
</div>

<script>
let registry = {};
let lang = 'de';

async function loadModels() {
  const r = await fetch('/api/models');
  registry = await r.json();
  const psel = document.getElementById('provider-select');
  psel.innerHTML = '';
  const providers = Object.keys(registry).sort();
  if (!providers.length) {
    psel.innerHTML = '<option>Keine API Keys</option>';
    return;
  }
  for (const p of providers) {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p + ' (' + registry[p].length + ')';
    psel.appendChild(opt);
  }
  onProviderChange();
}

function onProviderChange() {
  const provider = document.getElementById('provider-select').value;
  const msel = document.getElementById('model-select');
  msel.innerHTML = '';
  for (const m of (registry[provider] || [])) {
    const opt = document.createElement('option');
    opt.value = provider + '/' + m;
    opt.textContent = m;
    msel.appendChild(opt);
  }
}

function setLang(l) {
  lang = l;
  document.getElementById('btn-de').classList.toggle('active', l === 'de');
  document.getElementById('btn-en').classList.toggle('active', l === 'en');
}

function getModel() { return document.getElementById('model-select').value; }

function flashKiwix() {
  const dot = document.getElementById('kiwix-dot');
  dot.classList.add('on');
  setTimeout(() => dot.classList.remove('on'), 2500);
}

function copyText(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ Kopiert';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = '📋 Kopieren'; btn.classList.remove('copied'); }, 2000);
  });
}

async function ask() {
  const input = document.getElementById('input');
  const question = input.value.trim();
  if (!question) return;
  const model = getModel();
  if (!model) { alert('Bitte erst ein Modell wählen.'); return; }

  input.value = '';
  input.style.height = '44px';
  document.getElementById('ask-btn').disabled = true;

  const resultsEl = document.getElementById('results');

  // Question row
  const qBlock = document.createElement('div');
  qBlock.className = 'qa-block';

  const qRow = document.createElement('div');
  qRow.className = 'q-row';
  qRow.innerHTML = '<span class="q-label">F</span>';
  const qText = document.createElement('span');
  qText.className = 'q-text';
  qText.textContent = question;
  qRow.appendChild(qText);
  qBlock.appendChild(qRow);

  // Answer row
  const aRow = document.createElement('div');
  aRow.className = 'a-row';
  aRow.innerHTML = '<span class="a-label">A</span>';
  const aBody = document.createElement('div');
  aBody.className = 'a-body';
  const aText = document.createElement('div');
  aText.className = 'a-text';
  aText.innerHTML = '<span class="searching">↳ überlege...</span>';
  aBody.appendChild(aText);
  aRow.appendChild(aBody);
  qBlock.appendChild(aRow);

  resultsEl.appendChild(qBlock);
  resultsEl.scrollTop = resultsEl.scrollHeight;

  let fullText = '';
  let citations = [];

  try {
    const resp = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question, model, lang })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let started = false;

    while (true) {
      const { done, value } = reader.read ? await reader.read() : { done: true };
      if (done) break;
      for (const line of decoder.decode(value).split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);
        if (raw === '[DONE]') continue;
        let obj;
        try { obj = JSON.parse(raw); } catch { continue; }

        if (obj.type === 'decision') {
          if (obj.result === 'direct') {
            aText.innerHTML = '<span class="searching">↳ aus eigenem Wissen<span class="cursor"> ▋</span></span>';
          } else {
            aText.innerHTML = '<span class="searching">↳ schlage nach...</span>';
          }
        } else if (obj.type === 'citations') {
          citations = obj.items;
          if (citations.length) flashKiwix();
          const terms = (obj.terms || []).slice(0, 3).join(' · ');
          aText.innerHTML = '<span class="searching">↳ ' + terms + '...</span>';
        } else if (obj.type === 'token') {
          if (!started) { aText.textContent = ''; started = true; }
          fullText += obj.text;
          aText.textContent = fullText;
          resultsEl.scrollTop = resultsEl.scrollHeight;
        } else if (obj.type === 'no_results') {
          const terms = (obj.terms || []).slice(0, 3).join(' · ');
          const note = document.createElement('div');
          note.className = 'no-kiwix';
          note.textContent = '↳ Kiwix: keine Treffer für "' + terms + '" — Antwort aus Modellwissen';
          aBody.insertBefore(note, aText);
          aText.innerHTML = '<span class="cursor">▋</span>';
        } else if (obj.type === 'error') {
          aText.textContent = 'Fehler: ' + obj.text;
          aText.style.color = '#d9534f';
        }
      }
    }

    // Citations + copy button
    if (fullText) {
      const citRow = document.createElement('div');
      citRow.className = 'citations';
      if (citations.length) {
        const label = document.createElement('span');
        label.className = 'cite-label';
        label.textContent = 'Quellen:';
        citRow.appendChild(label);
        for (const c of citations) {
          const a = document.createElement('a');
          a.className = 'cite-link';
          a.href = c.url;
          a.target = '_blank';
          a.textContent = c.title;
          citRow.appendChild(a);
        }
      }
      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-btn';
      copyBtn.textContent = '📋 Kopieren';
      const copyText_ = fullText + (citations.length
        ? '\n\nQuellen: ' + citations.map(c => c.title + ' (' + c.url + ')').join(', ')
        : '');
      copyBtn.onclick = () => copyText(copyText_, copyBtn);
      citRow.appendChild(copyBtn);
      aBody.appendChild(citRow);
    }

  } catch(e) {
    aText.textContent = 'Verbindungsfehler: ' + e;
    aText.style.color = '#d9534f';
  }

  document.getElementById('ask-btn').disabled = false;
  input.focus();
}

// Enter → send, Shift+Enter → newline
document.getElementById('input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
});
document.getElementById('input').addEventListener('input', function() {
  this.style.height = '44px';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

loadModels();
</script>
</body>
</html>
"""

# ── Question classifier (no LLM needed — pure heuristics) ───────────────────
#
# Two question types:
#   DIRECT  — logic, math, concepts, definitions, "how does X work"
#   LOOKUP  — facts, numbers, dates, names, places, statistics, biographies
#
# Strategy: check strong LOOKUP signals first, then DIRECT signals,
# then fall back on proper noun detection.

import re as _re

_LOOKUP_PATTERNS = [
    # Quantity / statistics
    r'\bwie viele?\b', r'\bwie (groß|hoch|alt|weit|lang|breit|schwer|tief)\b',
    r'\bwie viel (kostet?|verdient?|wiegt?)\b',
    r'\beinwohner\b', r'\bbevölkerung\b', r'\bfläche\b', r'\bhöhe\b',
    r'\bbip\b', r'\bbruttoinlandsprodukt\b',
    # Time / dates
    r'\bwann\b', r'\bin welchem jahr\b', r'\bseit wann\b', r'\bab wann\b',
    r'\bgegründet\b', r'\bgeboren\b', r'\bgestorben\b', r'\berfunden\b',
    r'\burgründung\b', r'\bentdeckt\b',
    # People / places
    r'\bwer (ist|war|sind|waren|hat|hatte|gründete?|erfand|entdeckte?)\b',
    r'\bwo (liegt|befindet|ist|war|steht|lebt)\b',
    r'\bhauptstadt\b', r'\bregierung\b', r'\bpräsident\b', r'\bkanzler\b',
    r'\bminister\b',
    # History / biography
    r'\bgeschichte\b', r'\bbiographie?\b', r'\blebenslauf\b',
    r'\bherkunft\b', r'\bursprung\b',
    r'\bwurde .{0,20} (gebaut|gegründet|erfunden|entdeckt)\b',
    # English equivalents
    r'\bhow (many|much|old|tall|high|far|long|wide|heavy)\b',
    r'\bwhen (was|is|did|were)\b', r'\bsince when\b',
    r'\bwho (is|was|are|were|invented|founded|discovered|created|built)\b',
    r'\bwhere (is|was|are|were|does|did)\b',
    r'\bpopulation\b', r'\bcapital (city|of)\b', r'\bpresident\b',
    r'\bborn\b', r'\bdied\b', r'\bfounded\b', r'\binvented\b', r'\bdiscovered\b',
    r'\bhistory of\b', r'\bbiography\b',
]

_DIRECT_PATTERNS = [
    # Pure math expression
    r'^[\d\s\+\-\*\/\(\)\.\,\^%=]+$',
    r'\bberechne?\b', r'\brechne?\b', r'\bwas (ergibt|ist) \d',
    r'\b\d+\s*[\+\-\*\/]\s*\d+\b',
    # Definitions / concepts
    r'\bwas bedeutet\b',
    r'\berkl[äa]r(e|ung)?\b', r'\bdefinition\b',
    r'\bwas ist der unterschied\b', r'\bunterschied zwischen\b',
    r'\bwie funktioniert\b', r'\bwie arbeitet\b', r'\bwie (macht|geht) man\b',
    r'\bwarum\b', r'\bwieso\b', r'\bweshalb\b',
    # Programming / tech concepts
    r'\bprogrammier\b', r'\bcode\b', r'\balgorithmus\b',
    # English
    r'\bexplain\b', r'\bwhat does .+ mean\b',
    r'\bhow does\b', r'\bhow (do|can) (i|you|we)\b',
    r'\bwhat.s the difference\b', r'\bwhy (is|are|does|do|did)\b',
    r'\bdefine\b',
]

_LOOKUP_RE     = _re.compile('|'.join(_LOOKUP_PATTERNS), _re.IGNORECASE)
_DIRECT_RE     = _re.compile('|'.join(_DIRECT_PATTERNS), _re.IGNORECASE)
# German nouns are capitalized — mid-sentence caps = proper noun = factual topic
_PROPER_NOUN_RE = _re.compile(r'(?<=[a-züäöß\s])\b([A-ZÜÄÖ][a-züäöß]{3,})\b')
_NON_NOUNS      = frozenset([
    "Ich", "Du", "Er", "Sie", "Wir", "Ihr", "Bitte", "Danke",
    "Was", "Wer", "Wie", "Wo", "Wann", "Warum", "Welche", "Welcher",
    "Ist", "Sind", "Hat", "Haben", "Wird", "Können", "Sollte",
])


def classify_question(question: str) -> str:
    """
    Returns 'lookup' or 'direct' — instant, no LLM call.

    Logic:
      1. Strong LOOKUP signals (wann/wer/wie viele/einwohner/...) → lookup
      2. Strong DIRECT signals (math, warum, erkläre, wie funktioniert) → direct
      3. German proper nouns detected mid-sentence → likely factual → lookup
      4. Default → lookup (safer: rather check than hallucinate)
    """
    q = question.strip()

    if _LOOKUP_RE.search(q):
        return "lookup"

    if _DIRECT_RE.search(q):
        return "direct"

    # German proper nouns mid-sentence → factual question
    nouns = [n for n in _PROPER_NOUN_RE.findall(q) if n not in _NON_NOUNS]
    if nouns:
        return "lookup"

    return "lookup"  # default: when in doubt, check


def _build_llm_kwargs(model: str, api_key: str) -> Dict:
    """Build base litellm kwargs for a given model."""
    kwargs: Dict = {"model": model, "timeout": 60}
    if api_key:
        kwargs["api_key"] = api_key
    if model.startswith("ollama/"):
        kwargs["api_base"] = "http://localhost:11434"
    if model.startswith("kilocode/"):
        kwargs["model"]    = f"openai/{model.split('/', 1)[1]}"
        kwargs["api_base"] = "https://api.kilo.ai/api/gateway/"
        kwargs["api_key"]  = os.getenv("KILOCODE_API_KEY", api_key)
    return kwargs


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/models")
def api_models():
    return jsonify(build_model_registry())


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data     = request.get_json()
    question = data.get("question", "").strip()
    model    = data.get("model", "").strip()
    lang     = data.get("lang", "de")

    if not question or not model:
        return jsonify({"error": "missing question or model"}), 400

    provider    = model.split("/")[0] if "/" in model else ""
    api_key     = parse_env_providers().get(provider, "")
    llm_kwargs  = _build_llm_kwargs(model, api_key)

    def generate():
        # ── Step 1: Classify question (instant, no LLM call) ────────────────
        decision = classify_question(question)
        yield f"data: {json.dumps({'type': 'decision', 'result': decision})}\n\n"

        # ── Step 2: Kiwix lookup (only if needed) ───────────────────────────
        kiwix = {"found": False, "context": "", "citations": [], "terms": []}
        if decision == "lookup":
            kiwix = build_context(question, lang=lang, n_articles=2)
            if kiwix["found"]:
                yield f"data: {json.dumps({'type': 'citations', 'items': kiwix['citations'], 'terms': kiwix.get('terms', [])})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'no_results', 'terms': kiwix.get('terms', [])})}\n\n"

        # ── Step 3: Build prompt and stream answer ───────────────────────────
        system = build_system_prompt(kiwix["found"])

        if kiwix["found"]:
            user_content = (
                f"Wikipedia-Kontext:\n\n{kiwix['context']}\n\n"
                f"---\n\nFrage: {question}"
            )
        else:
            user_content = question

        stream_kwargs = {
            **llm_kwargs,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            "stream": True,
        }

        try:
            for chunk in litellm.completion(**stream_kwargs):
                token = chunk.choices[0].delta.content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    PORT = int(os.environ.get("KIWIX_BRIDGE_PORT", 7710))
    HOST = os.environ.get("KIWIX_BRIDGE_HOST", "127.0.0.1")
    reg  = build_model_registry()
    print(f"[KIWIX_BRIDGE] http://{HOST}:{PORT}")
    print(f"[KIWIX_BRIDGE] Kiwix: https://127.0.0.1:450")
    print(f"[KIWIX_BRIDGE] Provider: {', '.join(reg.keys()) or 'keine — .env prüfen'}")
    app.run(host=HOST, port=PORT, debug=False)
