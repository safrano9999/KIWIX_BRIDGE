# 🦝 KIWIX BRIDGE
![Example Image](KIWIX_BRIDGE.png)

> **Wikipedia's facts + AI's intelligence — fully offline, brutally accurate.**

Even the smallest local models can answer complex factual questions with precision — because they don't have to *know* the answer, they just have to *read* it. KIWIX BRIDGE fetches the right Wikipedia articles first, then lets the AI reason over them. No hallucinations. Just facts. 🎯

---

## 🦙 Zero config local + optional cloud

**No API key needed to get started.** Install [Ollama](https://ollama.com), pull any model, and KIWIX BRIDGE auto-discovers it at startup — all local models appear in the provider dropdown automatically. No config, no keys, no internet. 🏠

For cloud providers, simply add their API key to `.env` — KIWIX BRIDGE will automatically discover all available models for that provider and populate them in the dropdown. Every provider you add shows up instantly.

Everything goes through **[LiteLLM](https://github.com/BerriAI/litellm)** — a universal adapter that makes every model, local or cloud, speak the same interface. [Kilocode](https://kilo.ai) is integrated on top to further expand the available model roster beyond LiteLLM's built-in providers. 🔌

The `Native Think` toggle in Settings captures `<think>` reasoning output from thinking-capable models. 🧠

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

This means even small Ollama models become genuinely useful for factual Q&A — grounded in real Wikipedia data, not hallucinations. 🔥

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

Edit **`kiwix.conf`** and adapt `KIWIX_URL` to your Kiwix server.

### 4. Configure AI providers

Copy `.env.example` to `.env` and add your API keys.

### 5. Run

```bash
python web.py
```

Open **http://127.0.0.1:7710** in your browser.

---

## 🔑 Configuration (`.env`)

Add API keys for the cloud providers you want to use. All providers are **auto-detected** — no extra config, just add the keys you have. Ollama needs no key at all.

---

## 🤖 Supported Providers

| Provider | Key in `.env` | Notes |
|---|---|---|
| 🦙 **Ollama** | *(none — auto-detected)* | Local, free, private |
| 🟣 **Anthropic** (Claude) | `ANTHROPIC_API_KEY` | Extended thinking support |
| 🟢 **OpenAI** | `OPENAI_API_KEY` | |
| 🔵 **Google** (Gemini) | `GEMINI_API_KEY` | |
| 🟠 **Groq** | `GROQ_API_KEY` | Very fast inference |
| 🔶 **Kilocode** | `KILOCODE_API_KEY` | Gateway to 100+ models |

All models auto-populate in the dropdown per provider. Everything goes through **[LiteLLM](https://github.com/BerriAI/litellm)**.

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
