"""
KIWIX_BRIDGE kiwix_tool.py — Local Kiwix/Wikipedia lookup

Search strategy (multi-step):
  1. Extract key terms from the question (German nouns = capitalized, English = noun phrases)
  2. Search Kiwix for each key term separately
  3. Rank results: exact title match > partial match > full-text match
  4. Fetch intro paragraphs of top N unique articles
  5. Fallback: search with full question text if term extraction yields nothing
"""

import re
import requests
import urllib3
from bs4 import BeautifulSoup
from typing import List, Dict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

KIWIX_URL  = "https://127.0.0.1:450"
VERIFY_SSL = False

BOOKS = {
    "de": "wikipedia_de_all",
    "en": "wikipedia_en_all",
}

# Words that are capitalized in German but are NOT nouns worth searching for
_DE_STOPWORDS = frozenset([
    "Was", "Wer", "Wie", "Wo", "Wann", "Warum", "Welche", "Welcher", "Welches",
    "Womit", "Wodurch", "Wofür", "Woher", "Wohin", "Wessen",
    "Ist", "Sind", "War", "Waren", "Hat", "Haben", "Hatte", "Hatten",
    "Wird", "Werden", "Wurde", "Wurden", "Kann", "Können",
    "Der", "Die", "Das", "Ein", "Eine", "Einen", "Einem", "Eines",
    "Ich", "Du", "Er", "Sie", "Es", "Wir", "Ihr",
    "Mir", "Dir", "Ihm", "Uns", "Euch",
    "Bitte", "Danke", "Ja", "Nein",
])


def _extract_search_terms(question: str, lang: str) -> List[str]:
    """
    Extract meaningful search terms from a question.

    German: nouns are capitalized → grab capitalized words (excluding sentence-start
            and stopwords). Also grab quoted phrases and multi-word proper nouns.
    English: grab proper nouns (consecutive capitalized words) + quoted phrases.

    Always also includes the full question as last-resort fallback term.
    Returns list of terms, most specific first.
    """
    terms = []

    # 1. Quoted phrases are always highest priority
    quoted = re.findall(r'"([^"]+)"', question)
    terms.extend(quoted)

    # 2. Multi-word capitalized sequences (proper nouns / compound nouns)
    #    e.g. "Berliner Mauer", "Wolfgang Amadeus Mozart", "European Union"
    multi = re.findall(r'\b([A-ZÜÄÖ][a-züäöß]+(?:\s+[A-ZÜÄÖ][a-züäöß]+)+)\b', question)
    for m in multi:
        if m not in terms:
            terms.append(m)

    if lang == "de":
        # 3. Single capitalized German nouns (after removing sentence-start word)
        words = re.findall(r'\b([A-ZÜÄÖ][a-züäöß]{2,})\b', question)
        # Skip the very first word (capitalized due to sentence start)
        first_word = question.split()[0].rstrip("?!.,") if question.split() else ""
        for w in words:
            if w == first_word:
                continue
            if w in _DE_STOPWORDS:
                continue
            if w not in terms:
                terms.append(w)
    else:
        # 4. English: grab capitalized single words (likely proper nouns)
        words = re.findall(r'\b([A-Z][a-z]{2,})\b', question)
        first_word = question.split()[0].rstrip("?!.,") if question.split() else ""
        common_en = frozenset(["What", "Who", "How", "Where", "When", "Why", "Which",
                               "The", "This", "That", "These", "Those", "Is", "Are",
                               "Was", "Were", "Has", "Have", "Had", "Can", "Could",
                               "Will", "Would", "Should", "Please", "Does", "Did"])
        for w in words:
            if w == first_word or w in common_en:
                continue
            if w not in terms:
                terms.append(w)

    # 5. Fallback: full question (Kiwix full-text search handles this OK)
    clean = re.sub(r'[?!]', '', question).strip()
    if clean not in terms:
        terms.append(clean)

    return terms


def _search(query: str, lang: str, max_results: int = 5) -> List[Dict]:
    """Search Kiwix for query, return [{title, path, score}]."""
    book = BOOKS.get(lang, BOOKS["de"])
    try:
        resp = requests.get(
            f"{KIWIX_URL}/search",
            params={"books.name": book, "pattern": query},
            verify=VERIFY_SSL,
            timeout=5,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return []

    soup    = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.select("a[href*='/A/']")[:max_results]:
        title = a.get_text(strip=True)
        path  = a["href"]
        if title and path:
            results.append({"title": title, "path": path})
    return results


def _score_result(result: Dict, search_term: str, question: str) -> int:
    """
    Score a search result by relevance.
    Higher = better match.
    """
    title = result["title"].lower()
    term  = search_term.lower()
    q     = question.lower()
    score = 0

    # Exact title match is best
    if title == term:
        score += 100
    # Title starts with term
    elif title.startswith(term):
        score += 60
    # Term is fully contained in title
    elif term in title:
        score += 40
    # All words of term appear in title
    elif all(w in title for w in term.split()):
        score += 25
    # Title words appear in question
    title_words = set(title.split())
    q_words     = set(re.findall(r'\b\w{3,}\b', q))
    overlap     = len(title_words & q_words)
    score      += overlap * 5

    return score


def _fetch_article(path: str, max_paragraphs: int = 4) -> str:
    """Fetch article intro text (first N meaningful paragraphs)."""
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


def build_context(question: str, lang: str = "de", n_articles: int = 2) -> Dict:
    """
    Main entry point. Always fetches from Kiwix.

    Strategy:
      - Extract key terms from question
      - Search each term, collect + score all results
      - Pick top N unique articles by score
      - Fetch their intro paragraphs
      - Fallback to other language if nothing found

    Returns:
      {
        "context":   str,              # formatted text for LLM prompt injection
        "citations": [{"title", "url"}],
        "found":     bool,
        "terms":     [str],            # what was searched (for UI debug)
      }
    """
    terms = _extract_search_terms(question, lang)

    def _collect(lang_: str) -> List[Dict]:
        seen_paths = {}
        for term in terms:
            for r in _search(term, lang=lang_):
                path = r["path"]
                s    = _score_result(r, term, question)
                if path not in seen_paths or s > seen_paths[path]["score"]:
                    seen_paths[path] = {**r, "score": s}
        return sorted(seen_paths.values(), key=lambda x: -x["score"])

    candidates = _collect(lang)

    # Fallback: try other language
    if not candidates:
        other      = "en" if lang == "de" else "de"
        candidates = _collect(other)
        if candidates:
            lang = other

    if not candidates:
        return {"context": "", "citations": [], "found": False, "terms": terms}

    # Fetch top N articles
    articles   = []
    citations  = []
    for c in candidates:
        if len(articles) >= n_articles:
            break
        text = _fetch_article(c["path"])
        if not text:
            continue
        articles.append(f"[Artikel: {c['title']}]\n{text}")
        citations.append({
            "title": c["title"],
            "url":   f"{KIWIX_URL}{c['path']}",
        })

    if not articles:
        return {"context": "", "citations": [], "found": False, "terms": terms}

    return {
        "context":   "\n\n---\n\n".join(articles),
        "citations": citations,
        "found":     True,
        "terms":     terms,
    }
