import unittest
from agent.state import ResearchState, SearchResult
from agent.llm import LLMClient
from agent.tools.search import WebSearcher
from agent.nodes import (
    Researcher,
    ResearchNode,
    Evaluator,
    EvaluatorNode,
    Synthesizer,
    SynthesizerNode,
)


class TestNodes(unittest.TestCase):
    def setUp(self):
        self.llm = LLMClient(provider="heuristic")
        self.searcher = WebSearcher(use_mock=True)
        self.evaluator = Evaluator(self.llm)
        self.researcher = Researcher(self.llm, self.searcher)
        self.synthesizer = Synthesizer(self.llm)

    def test_node_aliases(self):
        self.assertIs(ResearchNode, Researcher)
        self.assertIs(EvaluatorNode, Evaluator)
        self.assertIs(SynthesizerNode, Synthesizer)

    def test_researcher_queries_and_run(self):
        state = ResearchState(topic="Autonomous Systems", max_iterations=3)
        queries = self.researcher.queries(state)
        self.assertIsInstance(queries, list)
        self.assertGreater(len(queries), 0)

        updated = self.researcher.run(state)
        self.assertEqual(updated.iteration, 1)
        self.assertGreater(len(updated.findings), 0)
        self.assertGreater(len(updated.search_queries), 0)

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

    def test_synthesizer_run(self):
        state = ResearchState(topic="Quantum Encryption", max_iterations=3)
        state.findings = [
            SearchResult(title="QKD Protocols", snippet="BB84 and QKD key distribution", url="https://example.com/qkd", query="quantum"),
        ]
        updated = self.synthesizer.run(state)
        self.assertIsNotNone(updated.draft_report)
        self.assertGreater(len(updated.draft_report), 0)


if __name__ == "__main__":
    unittest.main()

