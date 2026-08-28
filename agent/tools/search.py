from __future__ import annotations
import html
import re
import urllib.parse
import urllib.request
from typing import Optional
from agent.state import Result, SearchResult


def clean_html(raw: str) -> str:
    """Strip HTML tags and unescape entities."""
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def clean_url(url: str) -> str:
    """Extract direct target from redirect wrappers and fix protocol-relative URLs."""
    url = url.strip()
    if "uddg=" in url:
        m = re.search(r"uddg=([^&]+)", url)
        if m:
            return urllib.parse.unquote(m.group(1))
    return "https:" + url if url.startswith("//") else url


class Searcher:
    """Web search provider using DuckDuckGo HTML with offline mock fallback."""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def search(
        self,
        query: str,
        limit: int = 5,
        max_results: Optional[int] = None,
    ) -> list[Result]:
        n = max_results if max_results is not None else limit
        if self.use_mock:
            return self._mock(query, n)
        try:
            return self._fetch_ddg(query, n) or self._mock(query, n)
        except Exception:
            return self._mock(query, n)

    def _fetch_ddg(self, query: str, limit: int) -> list[Result]:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")

        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', body, re.DOTALL)
        titles = re.findall(r'<h2 class="result__title">.*?<a[^>]*>(.*?)</a>', body, re.DOTALL)
        urls = re.findall(r'<a class="result__url"[^>]*href="([^"]+)"', body, re.DOTALL)

        results: list[Result] = []
        for t, s, u in zip(titles[:limit], snippets[:limit], urls[:limit]):
            results.append(
                Result(
                    title=clean_html(t),
                    snippet=clean_html(s),
                    url=clean_url(u),
                    query=query,
                )
            )
        return results

    def _mock(self, query: str, limit: int = 5) -> list[Result]:
        q = urllib.parse.quote(query)
        mock_data = [
            Result(
                title=f"Comprehensive Overview: {query.title()}",
                snippet=f"Key facts, architecture, and current state regarding '{query}'. Covers recent developments and foundational concepts.",
                url=f"https://example.org/research/{q}",
                query=query,
            ),
            Result(
                title=f"Analysis & Key Perspectives on {query.title()}",
                snippet=f"Detailed analytical breakdown of challenges, opportunities, and future outlook for '{query}'.",
                url=f"https://example.org/analysis/{q}",
                query=query,
            ),
        ]
        return mock_data[:limit]

    # Aliases
    clean_html = staticmethod(clean_html)
    clean_url = staticmethod(clean_url)
    _mock_search = _mock
    _duckduckgo_search = _fetch_ddg


# Backward compatibility alias
WebSearcher = Searcher
