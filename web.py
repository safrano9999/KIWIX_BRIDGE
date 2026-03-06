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
from pathlib import Path
from typing import List, Dict

# ── venv sys.path trick — must come before any third-party imports ────────────
_venv_site = Path(__file__).parent / "venv" / "lib"
if _venv_site.exists():
    for _p in _venv_site.glob("python*/site-packages"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import requests
from flask import Flask, request, jsonify, Response, stream_with_context, render_template_string
import litellm
from dotenv import load_dotenv
import re
from kiwix_tool import fetch_articles

logging.basicConfig(level=logging.WARNING)

# ── SKILLS.md parser ─────────────────────────────────────────────────────────

def _load_skills(path: Path) -> Dict[str, str]:
    """
    Parse SKILLS.md — sections start with '## skill_name', content follows.
    Returns {"skill_name": "content string", ...}
    """
    skills: Dict[str, str] = {}
    if not path.exists():
        return skills
    current_key  = None
    current_lines: list = []
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            if current_key:
                skills[current_key] = "\n".join(current_lines).strip()
            current_key   = line[3:].strip()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
    if current_key:
        skills[current_key] = "\n".join(current_lines).strip()
    return skills

SKILLS = _load_skills(Path(__file__).parent / "SKILLS.md")

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
    key = "system_with_context" if has_context else "system_no_context"
    return SKILLS.get(key, "You are a helpful assistant.")

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
    margin-left: auto; font-size: 10px; color: #333; display: flex; align-items: center; gap: 5px; white-space: nowrap;
  }
  #kiwix-dot { width: 6px; height: 6px; border-radius: 50%; background: #2a2a2a; display: inline-block; }
  #kiwix-dot.on { background: #5cb85c; box-shadow: 0 0 5px #5cb85c; }

  /* Body: chat left + wiki panel right */
  #body { display: flex; flex: 1; overflow: hidden; }

  /* Left: chat */
  #chat-col { display: flex; flex-direction: column; flex: 1; overflow: hidden; min-width: 0; }
  #results { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex; flex-direction: column; gap: 28px; }

  .qa-block { display: flex; flex-direction: column; gap: 8px; }

  .q-row { display: flex; gap: 10px; align-items: flex-start; }
  .q-label { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; padding-top: 2px; min-width: 18px; }
  .q-text  { color: #a8c7e8; font-size: 13px; line-height: 1.6; }

  .a-row { display: flex; gap: 10px; align-items: flex-start; }
  .a-label { font-size: 10px; color: #444; text-transform: uppercase; letter-spacing: 1px; padding-top: 2px; min-width: 18px; }
  .a-body  { flex: 1; min-width: 0; }
  .a-text  { color: #e0e0e0; font-size: 13px; line-height: 1.7; white-space: pre-wrap; }

  /* Keywords row */
  .kw-row { display: flex; flex-wrap: wrap; gap: 5px; margin-left: 28px; }
  .kw-chip {
    font-size: 10px; color: #3a5a3a; border: 1px solid #1e3a1e;
    padding: 1px 7px; border-radius: 10px; background: #0d1a0d;
  }
  .model-tag {
    font-size: 10px; color: #2a3a4a; margin-left: 28px; margin-top: 2px;
  }

  /* Citations */
  .citations { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .cite-label { font-size: 10px; color: #3a3a3a; }
  .cite-link {
    font-size: 11px; color: #3d6b6b; text-decoration: none;
    border: 1px solid #1e3a3a; padding: 2px 8px; border-radius: 2px; cursor: pointer;
    transition: all 0.15s;
  }
  .cite-link:hover { color: #5cc8c8; border-color: #2a5a5a; background: #0d1e1e; }
  .cite-link.active { color: #7eb8f7; border-color: #2a5298; background: #0d1a2a; }

  .copy-btn {
    margin-left: auto; background: none; color: #333; border: 1px solid #222;
    font-family: 'Courier New', monospace; font-size: 11px; padding: 2px 10px;
    border-radius: 2px; cursor: pointer; white-space: nowrap; transition: all 0.15s;
  }
  .copy-btn:hover { color: #7eb8f7; border-color: #2a5298; }
  .copy-btn.copied { color: #5cb85c; border-color: #3a6b3a; }

  .no-kiwix { font-size: 11px; color: #2a4040; font-style: italic; margin-top: 4px; }
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

  /* Right: wiki panel */
  #wiki-panel {
    width: 0; flex-shrink: 0; border-left: none;
    display: flex; flex-direction: column; overflow: hidden;
    transition: width 0.25s ease, border-color 0.25s;
    background: #111;
  }
  #wiki-panel.open { width: 42%; border-left: 1px solid #2a2a2a; }

  #wiki-bar {
    padding: 8px 12px; border-bottom: 1px solid #222;
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
    background: #0d0d0d;
  }
  #wiki-title { font-size: 11px; color: #555; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #wiki-close {
    background: none; border: none; color: #444; cursor: pointer;
    font-size: 16px; line-height: 1; padding: 0 2px;
  }
  #wiki-close:hover { color: #888; }
  #wiki-frame { flex: 1; border: none; background: #fff; }

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

<div id="body">
  <div id="chat-col">
    <div id="results"></div>
    <div id="inputbar">
      <textarea id="input" placeholder="Frage stellen..." rows="1"></textarea>
      <button id="ask-btn" onclick="ask()">Fragen</button>
    </div>
  </div>

  <div id="wiki-panel">
    <div id="wiki-bar">
      <span id="wiki-title">—</span>
      <button id="wiki-close" onclick="closeWiki()">✕</button>
    </div>
    <iframe id="wiki-frame" src="about:blank"></iframe>
  </div>
</div>

<script>
let registry = {};
let lang = 'de';
let activeLink = null;

async function loadModels() {
  const r = await fetch('/api/models');
  registry = await r.json();
  const psel = document.getElementById('provider-select');
  psel.innerHTML = '';
  const providers = Object.keys(registry).sort();
  if (!providers.length) { psel.innerHTML = '<option>Keine API Keys</option>'; return; }
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

function openWiki(url, title, linkEl) {
  document.getElementById('wiki-frame').src = url;
  document.getElementById('wiki-title').textContent = title;
  document.getElementById('wiki-panel').classList.add('open');
  if (activeLink) activeLink.classList.remove('active');
  activeLink = linkEl;
  if (linkEl) linkEl.classList.add('active');
}

function closeWiki() {
  document.getElementById('wiki-panel').classList.remove('open');
  document.getElementById('wiki-frame').src = 'about:blank';
  if (activeLink) { activeLink.classList.remove('active'); activeLink = null; }
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

  const qBlock = document.createElement('div');
  qBlock.className = 'qa-block';

  // Question
  const qRow = document.createElement('div');
  qRow.className = 'q-row';
  qRow.innerHTML = '<span class="q-label">F</span>';
  const qText = document.createElement('span');
  qText.className = 'q-text';
  qText.textContent = question;
  qRow.appendChild(qText);
  qBlock.appendChild(qRow);

  // Model tag
  const modelTag = document.createElement('div');
  modelTag.className = 'model-tag';
  modelTag.textContent = model;
  qBlock.appendChild(modelTag);

  // Keywords row (filled later)
  const kwRow = document.createElement('div');
  kwRow.className = 'kw-row';
  qBlock.appendChild(kwRow);

  // Answer
  const aRow = document.createElement('div');
  aRow.className = 'a-row';
  aRow.innerHTML = '<span class="a-label">A</span>';
  const aBody = document.createElement('div');
  aBody.className = 'a-body';
  const aText = document.createElement('div');
  aText.className = 'a-text';
  aText.innerHTML = '<span class="searching">↳ suche Keywords...</span>';
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
      const { done, value } = await reader.read();
      if (done) break;
      for (const line of decoder.decode(value).split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);
        if (raw === '[DONE]') continue;
        let obj;
        try { obj = JSON.parse(raw); } catch { continue; }

        if (obj.type === 'status') {
          aText.innerHTML = '<span class="searching">' + obj.text + '</span>';

        } else if (obj.type === 'keywords') {
          // Show keywords as chips below the question
          kwRow.innerHTML = '';
          for (const kw of (obj.items || [])) {
            const chip = document.createElement('span');
            chip.className = 'kw-chip';
            chip.textContent = kw;
            kwRow.appendChild(chip);
          }
          aText.innerHTML = '<span class="searching">↳ durchsuche Wikipedia...</span>';

        } else if (obj.type === 'citations') {
          citations = obj.items;
          if (citations.length) flashKiwix();
          aText.innerHTML = '<span class="searching">↳ formuliere Antwort...<span class="cursor"> ▋</span></span>';

        } else if (obj.type === 'token') {
          if (!started) { aText.textContent = ''; started = true; }
          fullText += obj.text;
          aText.textContent = fullText;
          resultsEl.scrollTop = resultsEl.scrollHeight;

        } else if (obj.type === 'no_results') {
          const note = document.createElement('div');
          note.className = 'no-kiwix';
          note.textContent = '↳ Kiwix: keine Treffer — Antwort aus Modellwissen';
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
          a.textContent = c.title;
          a.onclick = (e) => { e.preventDefault(); openWiki(c.url, c.title, a); };
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
        # ── Step 1: LLM extracts 3-5 Wikipedia search keywords ──────────────
        yield f"data: {json.dumps({'type': 'status', 'text': '↳ suche Keywords...'})}\n\n"

        keywords = []
        try:
            kw_resp = litellm.completion(
                **{
                    **llm_kwargs,
                    "messages": [{
                        "role": "user",
                        "content": SKILLS.get("keyword_extraction", "Name 3 Wikipedia articles for: {question}").replace("{question}", question)
                    }],
                    "max_tokens": 80,
                    "stream": False,
                }
            )
            raw = (kw_resp.choices[0].message.content or "").strip()
            # Extract JSON array even if model adds surrounding text
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                keywords = json.loads(m.group())
        except Exception:
            pass

        if not keywords:
            # Fallback: use question itself as single keyword
            keywords = [question]

        yield f"data: {json.dumps({'type': 'keywords', 'items': keywords})}\n\n"

        # ── Step 2: Fetch Kiwix articles for all keywords ───────────────────
        yield f"data: {json.dumps({'type': 'status', 'text': '↳ durchsuche Wikipedia...'})}\n\n"

        kiwix = fetch_articles(keywords, lang=lang, n_articles=3)

        if kiwix["found"]:
            yield f"data: {json.dumps({'type': 'citations', 'items': kiwix['citations']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'no_results'})}\n\n"

        # ── Step 3: Stream final answer ──────────────────────────────────────
        if kiwix["found"]:
            system = build_system_prompt(has_context=True)
            user_content = (
                f"Wikipedia-Kontext:\n\n{kiwix['context']}\n\n"
                f"---\n\nFrage: {question}"
            )
        else:
            system = build_system_prompt(has_context=False)
            user_content = question

        try:
            for chunk in litellm.completion(**{
                **llm_kwargs,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_content},
                ],
                "stream": True,
            }):
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
