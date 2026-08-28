import urllib.request
import urllib.parse
import re
import html
from typing import List
from agent.state import SearchResult


class WebSearcher:
    """Performs web searches using DuckDuckGo HTML or local mock results."""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if self.use_mock:
            return self._mock_search(query, max_results)
        
        try:
            results = self._duckduckgo_search(query, max_results)
            if not results:
                return self._mock_search(query, max_results)
            return results
        except Exception:
            return self._mock_search(query, max_results)

    def _duckduckgo_search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            )
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")

        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', content, re.DOTALL)
        titles = re.findall(r'<h2 class="result__title">.*?<a[^>]*>(.*?)</a>', content, re.DOTALL)
        urls = re.findall(r'<a class="result__url"[^>]*href="([^"]+)"', content, re.DOTALL)

        results: List[SearchResult] = []
        limit = min(len(titles), len(snippets), max_results)
        for i in range(limit):
            clean_title = html.unescape(re.sub(r"<[^>]+>", "", titles[i]).strip())
            clean_snippet = html.unescape(re.sub(r"<[^>]+>", "", snippets[i]).strip())
            raw_url = urls[i].strip() if i < len(urls) else ""
            
            # Extract clean redirect target if present
            if "uddg=" in raw_url:
                match = re.search(r"uddg=([^&]+)", raw_url)
                if match:
                    raw_url = urllib.parse.unquote(match.group(1))
            elif raw_url.startswith("//"):
                raw_url = "https:" + raw_url
            
            results.append(SearchResult(
                title=clean_title,
                snippet=clean_snippet,
                url=raw_url,
                query=query
            ))

        return results

    def _mock_search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        return [
            SearchResult(
                title=f"Comprehensive Overview: {query.title()}",
                snippet=f"Key facts, architecture, and current state regarding '{query}'. Covers recent developments and foundational concepts.",
                url=f"https://example.org/research/{urllib.parse.quote(query)}",
                query=query
            ),
            SearchResult(
                title=f"Analysis & Key Perspectives on {query.title()}",
                snippet=f"Detailed analytical breakdown of challenges, opportunities, and future outlook for '{query}'.",
                url=f"https://example.org/analysis/{urllib.parse.quote(query)}",
                query=query
            )
        ]
