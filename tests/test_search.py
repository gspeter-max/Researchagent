import unittest
from agent.tools.search import Searcher, WebSearcher, clean_html, clean_url


class TestSearcher(unittest.TestCase):
    def test_alias(self):
        self.assertIs(WebSearcher, Searcher)

    def test_mock_search_limit(self):
        searcher = Searcher(use_mock=True)
        results = searcher.search("Autonomous Agents", limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].query, "Autonomous Agents")
        self.assertTrue(len(results[0].snippet) > 0)

    def test_legacy_max_results_param(self):
        searcher = WebSearcher(use_mock=True)
        results = searcher.search("Autonomous Agents", max_results=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.query == "Autonomous Agents" for r in results))

    def test_clean_html(self):
        raw = "<b>Hello</b> &amp; <i>world</i>!"
        self.assertEqual(clean_html(raw), "Hello & world!")
        self.assertEqual(Searcher.clean_html(raw), "Hello & world!")

    def test_clean_url(self):
        redirect_url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=1"
        self.assertEqual(clean_url(redirect_url), "https://example.com/page")
        self.assertEqual(Searcher.clean_url(redirect_url), "https://example.com/page")

        proto_rel = "//example.com/test"
        self.assertEqual(clean_url(proto_rel), "https://example.com/test")


if __name__ == "__main__":
    unittest.main()
