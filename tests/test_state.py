import unittest
from agent.state import ResearchState, SearchResult, EvaluationResult, AgentStatus


class TestResearchState(unittest.TestCase):
    def test_state_initialization(self):
        state = ResearchState(topic="AI Safety")
        self.assertEqual(state.topic, "AI Safety")
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.max_iterations, 3)
        self.assertEqual(state.status, AgentStatus.INITIALIZED)
        self.assertEqual(len(state.findings), 0)
        self.assertIsNone(state.evaluation)

    def test_add_log(self):
        state = ResearchState(topic="Test Topic")
        state.add_log("Log message 1")
        self.assertIn("Log message 1", state.logs)

    def test_search_result(self):
        res = SearchResult(title="Test", snippet="Sample snippet", url="https://example.com", query="q")
        self.assertEqual(res.title, "Test")
        self.assertEqual(res.url, "https://example.com")


if __name__ == "__main__":
    unittest.main()
