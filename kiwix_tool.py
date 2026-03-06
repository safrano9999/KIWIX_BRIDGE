"""
KIWIX_BRIDGE kiwix_tool.py — Local Kiwix/Wikipedia lookup

Given a list of keywords (from LLM), searches Kiwix and returns
article intros + citation info for RAG injection.
"""

import os
import re
import urllib.parse
import requests
import urllib3
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _read_conf(path: Path) -> dict:
    """Parse a simple KEY=value config file."""
    conf = {}
    if not path.exists():
        return conf
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            conf[k.strip()] = v.strip()
    return conf


_CONF = _read_conf(Path(__file__).parent / "kiwix.conf")
KIWIX_CONF = _CONF  # exposed for web.py

# Adapt KIWIX_URL in kiwix.conf to match your Kiwix server
KIWIX_URL  = _CONF.get("KIWIX_URL", "https://127.0.0.1:450")
VERIFY_SSL = False


def _discover_books() -> List[str]:
    """Discover all ZIM books available in Kiwix — any type, any language."""
    try:
        resp = requests.get(f"{KIWIX_URL}/nojs", verify=VERIFY_SSL, timeout=5)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        seen  = []
        for a in soup.find_all("a", href=True):
            m = re.match(r"^/content/([^/?#\s]+)", a["href"])
            if m:
                name = m.group(1).rstrip("/")
                if name not in seen:
                    seen.append(name)
        return sorted(seen)
    except Exception:
        return []


BOOKS: List[str] = _discover_books()  # exposed for web.py (/api/books)


def _search(keyword: str, book: str, max_results: int = 5) -> List[Dict]:
    """Search Kiwix for one keyword in the given ZIM book. Returns [{title, path}]."""
    if not book:
        return []
    try:
        resp = requests.get(
            f"{KIWIX_URL}/search",
            params={"books.name": book, "pattern": keyword},
            verify=VERIFY_SSL,
            timeout=5,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    out  = []
    for a in soup.select(f"a[href*='/content/{book}']")[:max_results]:
        title = a.get_text(strip=True)
        path  = a["href"]
        if title and path:
            out.append({"title": title, "path": path})
    return out


def _base_forms(keyword: str) -> List[str]:
    """
    Generate candidate base/uninflected forms by stripping common
    German and English inflectional suffixes.
    Only applies to single words; multi-word phrases are returned as-is.
    """
    forms = [keyword]
    if " " in keyword:
        return forms
    for suffix in ("es", "s", "en", "em", "er", "e"):
        if keyword.lower().endswith(suffix) and len(keyword) - len(suffix) >= 3:
            base = keyword[:-len(suffix)]
            if base not in forms:
                forms.append(base)
    return forms


def _direct_lookup(keyword: str, book: str) -> Optional[Dict]:
    """
    Try to fetch an article by exact title via /content/{book}/A/{title}.
    Also tries uninflected base forms (e.g. 'Wiens' → 'Wien').
    Returns {title, path, score} or None.
    """
    if not book:
        return None
    for form in _base_forms(keyword):
        title = form.strip().replace(" ", "_")
        title = title[0].upper() + title[1:] if title else title
        path = f"/content/{book}/A/{urllib.parse.quote(title, safe='')}"
        try:
            resp = requests.get(f"{KIWIX_URL}{path}", verify=VERIFY_SSL, timeout=5)
            if resp.status_code == 200 and len(resp.text) > 500:
                return {"title": form, "path": path, "score": 150}
        except requests.RequestException:
            pass
    return None


def _score(title: str, keyword: str) -> int:
    """Rank results: exact match > starts with > contains > word overlap."""
    t = title.lower()
    k = keyword.lower()
    if t == k:                               return 100
    if t.startswith(k):                      return 70
    if k in t:                               return 50
    if all(w in t for w in k.split()):       return 30
    overlap = len(set(t.split()) & set(k.split()))
    return overlap * 8


def _fetch_intro(path: str, max_paragraphs: int = 4) -> str:
    """Fetch article and return intro text."""
    try:
        resp = requests.get(f"{KIWIX_URL}{path}", verify=VERIFY_SSL, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup.select("table, .infobox, .navbox, sup, .reference, .toc, .mw-editsection"):
        tag.decompose()

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 80:
            paragraphs.append(text)
        if len(paragraphs) >= max_paragraphs:
            break
    return "\n\n".join(paragraphs)


def fetch_articles(keywords: List[str], book: str, n_articles: int = 3) -> Dict:
    """
    Search Kiwix for each keyword in the given ZIM book, score and deduplicate results,
    fetch top N article intros.

    Returns:
      {
        "found":     bool,
        "context":   str,   # formatted for LLM prompt
        "citations": [{"title": str, "url": str}],
      }
    """
    seen: Dict[str, Dict] = {}  # path → best scored result

    for keyword in keywords:
        # 1. Try direct article title fetch first (score 150 — beats all search results)
        direct = _direct_lookup(keyword, book=book)
        if direct:
            seen[direct["path"]] = direct

        # 2. Fulltext search as fallback / complement
        for r in _search(keyword, book=book):
            path  = r["path"]
            score = _score(r["title"], keyword)
            if path not in seen or score > seen[path]["score"]:
                seen[path] = {**r, "score": score}

    if not seen:
        return {"found": False, "context": "", "citations": []}

    top = sorted(seen.values(), key=lambda x: -x["score"])

    articles  = []
    citations = []
    for c in top:
        if len(articles) >= n_articles:
            break
        text = _fetch_intro(c["path"])
        if not text:
            continue
        articles.append(f"[Kiwix: {c['title']}]\n{text}")
        citations.append({"title": c["title"], "url": f"{KIWIX_URL}{c['path']}"})

    if not articles:
        return {"found": False, "context": "", "citations": []}

    return {
        "found":     True,
        "context":   "\n\n---\n\n".join(articles),
        "citations": citations,
    }
