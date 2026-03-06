# 🦝 KIWIX BRIDGE

> **Wikipedia's facts + AI's intelligence — fully offline, brutally accurate.**

Even the smallest local models like `qwen3.5:0.8b` can answer complex factual questions with precision — because they don't have to *know* the answer, they just have to *read* it. KIWIX BRIDGE fetches the right Wikipedia articles first, then lets the AI reason over them. No hallucinations. Just facts. 🎯

---

## 🦙 100% Offline with Ollama

Run everything locally — no cloud, no API keys, no data leaving your machine:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (tiny but capable with RAG)
ollama pull qwen3.5:0.8b    # 0.8 GB — fits anywhere
ollama pull mistral          # 4 GB  — great quality
ollama pull deepseek-r1      # with native <think> reasoning
```

KIWIX BRIDGE **auto-detects Ollama** at startup — no config needed. Just start Ollama and it appears in the provider dropdown instantly. The `Native Think` toggle in Settings captures `<think>` reasoning output from models like Qwen3 and DeepSeek-R1. 🧠

---

## 📖 What is Kiwix?

[Kiwix](https://www.kiwix.org/) is an **offline Wikipedia reader** — it downloads the entire Wikipedia (all languages, all articles) and serves it locally as a fast HTTP server. No internet required. No rate limits. No censorship.

KIWIX BRIDGE connects to your local Kiwix instance and uses it as a **knowledge retrieval engine**:

```
Your Question
     │
     ▼
🤖 AI extracts 3 Wikipedia article titles
     │
     ▼
📚 Kiwix fetches those articles (offline, instant)
     │
     ▼
🧠 AI reads the articles and answers your question
     │
     ▼
✅ Precise answer + clickable Wikipedia citations
```

This is **RAG (Retrieval-Augmented Generation)** — but with your own local Wikipedia, no cloud, no API costs for retrieval.

---

## ✨ Why it works even with small models

A tiny model running on your laptop doesn't need to memorize all of Wikipedia. It just needs to:
1. Know what to search for *(easy)*
2. Read 3 articles and extract the answer *(easy)*

This means **`qwen3.5:0.8b`, `mistral:7b`, `llama3.2:3b`** and similar small Ollama models become genuinely useful for factual Q&A — grounded in real Wikipedia data, not hallucinations. 🔥

---

## 🚀 Installation

### 1. Prerequisites

- **Kiwix** running locally at `https://127.0.0.1:450/` with a Wikipedia ZIM file
  - Download Kiwix: [kiwix.org/en/download](https://www.kiwix.org/en/download/)
  - Download Wikipedia ZIMs: [library.kiwix.org](https://library.kiwix.org/)
- **Python 3.9+**
- At least one of: API keys for cloud providers, or Ollama running locally

### 2. Clone & setup

```bash
git clone https://github.com/safrano9999/KIWIX_BRIDGE.git
cd KIWIX_BRIDGE
python3 bin/setup.py
```

This creates a local `venv/` and installs all dependencies.

### 3. Configure Kiwix URL

Edit **`kiwix.conf`** — adapt the URL to your Kiwix server:

```ini
KIWIX_URL=https://127.0.0.1:450
```

### 4. Configure AI providers

```bash
cp .env.example .env
nano .env
```

### 5. Run

```bash
python web.py
```

Open **http://127.0.0.1:7710** in your browser.

---

## 🔑 Configuration (`.env`)

```env
# Cloud providers — add whichever you have
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...

# Kilocode gateway (access 100+ models with one key)
KILOCODE_API_KEY=...

# Ollama runs without a key — just start it locally
# OLLAMA_API_KEY=   ← not needed
```

All providers are **auto-detected** from your `.env` — no config file, just add the keys you have.

---

## 🤖 Supported Providers

| Provider | How |
|---|---|
| 🟣 **Anthropic** (Claude) | `ANTHROPIC_API_KEY` — supports extended thinking |
| 🟢 **OpenAI** | `OPENAI_API_KEY` |
| 🔵 **Google** (Gemini) | `GEMINI_API_KEY` |
| 🟠 **Groq** | `GROQ_API_KEY` — very fast inference |
| 🦙 **Ollama** | Auto-detected at `localhost:11434` — free, local, private |
| 🔶 **Kilocode** | `KILOCODE_API_KEY` — gateway to 100+ models (GPT, Claude, Gemini, ...) |

All providers go through **[LiteLLM](https://github.com/BerriAI/litellm)** — one unified interface for everything.

---

## 🎛️ Features

- 🔍 **Always fetches Wikipedia first** — AI never answers from memory alone
- 📎 **Clickable citations** — open the Wikipedia article in a side panel
- 🏷️ **Keyword chips** — see exactly what KIWIX BRIDGE searched for
- 🧠 **Thinking mode** — for Claude (extended thinking), OpenAI o-series (reasoning effort), and native `<think>` models like Qwen3, DeepSeek-R1
- ⚙️ **Per-query settings** — temperature, thinking depth, max tokens
- 🌐 **DE / EN** — search German or English Wikipedia
- 📋 **Copy button** — copy answer + citations in one click

---

## 🏗️ Tech Stack

- **Flask** — lightweight Python web server
- **LiteLLM** — unified API for all LLM providers
- **Kiwix HTTP API** — local Wikipedia search & article fetch
- **BeautifulSoup** — HTML → clean article text
- **SSE streaming** — real-time token streaming in the browser
