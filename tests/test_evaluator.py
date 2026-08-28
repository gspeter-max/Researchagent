import unittest
from agent.state import ResearchState, SearchResult
from agent.llm import LLMClient
from agent.nodes.evaluator import EvaluatorNode


class TestEvaluatorNode(unittest.TestCase):
    def setUp(self):
        self.llm = LLMClient(provider="heuristic")
        self.evaluator = EvaluatorNode(self.llm)

    def test_evaluator_insufficient_on_few_findings(self):
        state = ResearchState(topic="Autonomous Drones", max_iterations=3)
        state.iteration = 1
        state.findings = [
            SearchResult(title="Drone 1", snippet="Snippet 1", url="https://example.com/1", query="drones")
        ]
        updated = self.evaluator.run(state)
        self.assertIsNotNone(updated.evaluation)
        self.assertFalse(updated.evaluation.is_sufficient)

    def test_evaluator_sufficient_on_multiple_findings(self):
        state = ResearchState(topic="Autonomous Drones", max_iterations=3)
        state.iteration = 2
        state.findings = [
            SearchResult(title="Drone Navigation", snippet="GPS-denied navigation", url="https://example.com/1", query="drones"),
            SearchResult(title="Drone Hardware", snippet="Rotor design and battery", url="https://example.com/2", query="drones"),
            SearchResult(title="Drone Regulations", snippet="FAA compliance benchmarks", url="https://example.com/3", query="drones"),
        ]
        updated = self.evaluator.run(state)
        self.assertIsNotNone(updated.evaluation)
        self.assertTrue(updated.evaluation.is_sufficient)
        self.assertGreaterEqual(updated.evaluation.score, 0.75)


if __name__ == "__main__":
    unittest.main()
