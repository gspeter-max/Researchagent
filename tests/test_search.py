import unittest
from agent.tools.search import WebSearcher


class TestWebSearcher(unittest.TestCase):
    def test_mock_search(self):
        searcher = WebSearcher(use_mock=True)
        results = searcher.search("Autonomous Agents", max_results=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.query == "Autonomous Agents" for r in results))
        self.assertTrue(all(len(r.snippet) > 0 for r in results))


if __name__ == "__main__":
    unittest.main()
