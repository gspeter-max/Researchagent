"""Hermetic, table-driven unit tests for Researcher, Evaluator, and Synthesizer nodes."""

import unittest
from unittest.mock import patch
from agent.state import State, Result, Score
from agent.llm import Client, ParseError
from agent.tools.search import Searcher
from agent.nodes import (
    Researcher,
    ResearchNode,
    Evaluator,
    EvaluatorNode,
    Synthesizer,
    SynthesizerNode,
)


class TestAgentNodes(unittest.TestCase):
    """Tests Agent Nodes, state invariants, query deduplication, and fault tolerance."""

    def setUp(self):
        self.llm = Client(provider="heuristic")
        self.searcher = Searcher(use_mock=True)
        self.evaluator = Evaluator(self.llm, sufficiency_threshold=0.75)
        self.researcher = Researcher(self.llm, self.searcher)
        self.synthesizer = Synthesizer(self.llm)

    def test_researcher_query_deduplication_invariant(self):
        """Invariant test: Researcher must never emit queries that were already searched."""
        # Arrange
        state = State(
            topic="Neurotechnology",
            search_queries=["neurotechnology overview and core principles"],
        )

        # Act
        qs = self.researcher.queries(state)

        # Assert: Deduplication preserves novelty
        self.assertNotIn("neurotechnology overview and core principles", qs)
        self.assertTrue(len(qs) > 0)

    def test_evaluator_boundary_value_analysis(self):
        """Table-driven test verifying sufficiency gating across findings counts."""
        # Arrange: Matrix of (findings_count, iteration, expected_sufficient)
        vectors = [
            (1, 1, False),  # 1 finding in cycle 1 -> Insufficient
            (3, 2, True),   # 3 findings in cycle 2 -> Sufficient (score >= 0.75)
        ]

        for findings_count, iteration, expected_sufficient in vectors:
            with self.subTest(findings_count=findings_count, iteration=iteration):
                # Arrange
                state = State(topic="Quantum Cryptography", iteration=iteration)
                state.findings = [
                    Result(title=f"Source {i}", snippet=f"Snippet {i}", url=f"http://test.org/{i}", query="quantum")
                    for i in range(findings_count)
                ]

                # Act
                updated = self.evaluator.run(state)

                # Assert
                self.assertIsNotNone(updated.evaluation)
                self.assertEqual(updated.evaluation.is_sufficient, expected_sufficient)

    def test_fault_injection_evaluator_llm_json_failure_recovery(self):
        """Fault injection: Evaluator must gracefully fallback to partial score on LLM JSON parse failure."""
        # Arrange
        state = State(topic="Edge AI", iteration=1)
        state.findings = [Result(title="Edge 1", snippet="Snippet", url="http://e.com/1", query="edge")]

        # Act: Force LLM generate_json to raise ParseError
        with patch.object(self.llm, "generate_json", side_effect=ParseError("Malformed JSON", raw="bad")):
            updated = self.evaluator.run(state)

        # Assert: Gracefully handled, logged warning, assigned fallback score
        self.assertIsNotNone(updated.evaluation)
        self.assertEqual(updated.evaluation.score, 0.5)
        self.assertFalse(updated.evaluation.is_sufficient)
        self.assertIn("Could not parse structured verification result", updated.evaluation.critique)

    def test_synthesizer_synthesis_and_fallback(self):
        """Tests draft report generation with source citations."""
        # Arrange
        state = State(topic="Autonomous Systems")
        state.findings = [
            Result(title="Robotics Overview", snippet="Autonomous robots in logistics", url="https://example.com/robotics", query="robotics")
        ]

        # Act
        updated = self.synthesizer.run(state)

        # Assert
        self.assertTrue(len(updated.draft_report) > 0)
        self.assertIn("Executive Summary", updated.draft_report)

    def test_node_aliases_backward_compatibility(self):
        # Assert
        self.assertIs(ResearchNode, Researcher)
        self.assertIs(EvaluatorNode, Evaluator)
        self.assertIs(SynthesizerNode, Synthesizer)


if __name__ == "__main__":
    unittest.main()

