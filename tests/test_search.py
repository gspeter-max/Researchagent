"""Hermetic, table-driven unit tests for Searcher, sanitization, and fault tolerance."""

import unittest
from unittest.mock import patch
from agent.tools.search import Searcher, WebSearcher, clean_html, clean_url


class TestSearcher(unittest.TestCase):
    """Tests search retrieval boundaries, parsing sanitization, and fault injection fallbacks."""

    def test_search_result_limit_boundary_value_analysis(self):
        """Tests that limit slicing conforms strictly to requested boundaries [0, 1, 2, 5]."""
        # Arrange
        searcher = Searcher(use_mock=True)
        query = "Reinforcement Learning"
        limits = [0, 1, 2, 5]

        for limit in limits:
            with self.subTest(limit=limit):
                # Act
                results = searcher.search(query, limit=limit)

                # Assert
                self.assertEqual(len(results), min(limit, 2))  # Mock has 2 entries
                for r in results:
                    self.assertEqual(r.query, query)
                    self.assertTrue(len(r.title) > 0)
                    self.assertTrue(len(r.snippet) > 0)

    def test_clean_html_equivalence_partitions(self):
        """Table-driven test of HTML entity decoding and tag stripping vectors."""
        # Arrange: Matrix of (raw_html, expected_clean)
        vectors = [
            ("<b>Hello</b> &amp; <i>world</i>!", "Hello & world!"),
            ("<a href='http://x.com'>Link text</a>", "Link text"),
            ("&lt;script&gt;alert(1)&lt;/script&gt;", "<script>alert(1)</script>"),
            ("   Whitespace padding   ", "Whitespace padding"),
            ("", ""),
        ]

        for raw_html, expected_clean in vectors:
            with self.subTest(raw=raw_html):
                # Act & Assert
                self.assertEqual(clean_html(raw_html), expected_clean)
                self.assertEqual(Searcher.clean_html(raw_html), expected_clean)

    def test_clean_url_canonicalization_vectors(self):
        """Table-driven test of URL redirect stripping and protocol normalization."""
        # Arrange: Matrix of (raw_url, expected_canonical)
        vectors = [
            ("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=1", "https://example.com/page"),
            ("//example.com/test", "https://example.com/test"),
            ("https://nature.com/articles/123", "https://nature.com/articles/123"),
            ("   https://arxiv.org/abs/2301.00001   ", "https://arxiv.org/abs/2301.00001"),
        ]

        for raw_url, expected_canonical in vectors:
            with self.subTest(raw_url=raw_url):
                # Act & Assert
                self.assertEqual(clean_url(raw_url), expected_canonical)
                self.assertEqual(Searcher.clean_url(raw_url), expected_canonical)

    def test_fault_injection_network_failure_fallback(self):
        """Fault injection: Simulates HTTP network timeout to verify graceful fallback to mock results."""
        # Arrange
        searcher = Searcher(use_mock=False)
        query = "Fault Injection Test"

        # Act: Force _fetch_ddg to raise network exception
        with patch.object(searcher, "_fetch_ddg", side_effect=TimeoutError("Connection timed out")):
            results = searcher.search(query, limit=2)

        # Assert: Gracefully recovered and returned mock findings
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].query, query)

    def test_backward_compatibility_aliases(self):
        # Assert
        self.assertIs(WebSearcher, Searcher)
        searcher = WebSearcher(use_mock=True)
        results = searcher.search("Quantum", max_results=1)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
