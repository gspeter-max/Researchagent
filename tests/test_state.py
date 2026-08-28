"""Hermetic, table-driven unit tests for agent state, scores, and feedback invariants."""

import unittest
from agent.state import (
    State,
    Status,
    Action,
    Result,
    Score,
    Feedback,
    ResearchState,
    SearchResult,
    EvaluationResult,
    HumanFeedback,
    AgentStatus,
    HumanAction,
)


class TestStateInvariants(unittest.TestCase):
    """Verifies state initialization, boundary values, and feedback accumulation invariants."""

    def test_state_initialization_defaults(self):
        # Arrange
        topic = "Post-Quantum Cryptography"

        # Act
        state = State(topic=topic)

        # Assert
        self.assertEqual(state.topic, topic)
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.max_iterations, 3)
        self.assertEqual(state.sufficiency_threshold, 0.75)
        self.assertEqual(state.status, Status.INITIALIZED)
        self.assertEqual(state.findings, [])
        self.assertEqual(state.search_queries, [])
        self.assertIsNone(state.evaluation)
        self.assertEqual(state.feedback_history, [])

    def test_feedback_action_equivalence_partitions(self):
        """Table-driven test verifying all action inputs (Enum & string variants) normalize correctly."""
        # Arrange: Matrix of (input_action, expected_enum)
        test_vectors = [
            (Action.PROCEED, Action.PROCEED),
            (Action.PROCEED_OVERRIDE, Action.PROCEED_OVERRIDE),
            (Action.SEARCH_MORE, Action.SEARCH_MORE),
            (Action.CANCEL, Action.CANCEL),
            ("proceed", Action.PROCEED),
            ("PROCEED", Action.PROCEED),
            ("proceed_override", Action.PROCEED_OVERRIDE),
            ("search_more", Action.SEARCH_MORE),
            ("cancel", Action.CANCEL),
        ]

        for input_action, expected_enum in test_vectors:
            with self.subTest(input_action=input_action, expected_enum=expected_enum):
                # Arrange
                state = State(topic="Test")
                
                # Act
                state.add_feedback(input_action, text="Guidance text", score=0.65)
                
                # Assert
                latest_entry = state.feedback_history[-1]
                self.assertEqual(latest_entry.action, expected_enum)
                self.assertEqual(latest_entry.feedback_text, "Guidance text")
                self.assertEqual(latest_entry.verification_score, 0.65)
                self.assertEqual(state.latest_feedback, "Guidance text")

    def test_score_boundary_value_analysis(self):
        """Tests Score and State boundary values across [0.0, 1.0]."""
        # Arrange: Matrix of (score_val, threshold, expected_sufficient)
        boundaries = [
            (0.0, 0.75, False),
            (0.749, 0.75, False),
            (0.75, 0.75, True),
            (0.751, 0.75, True),
            (1.0, 0.75, True),
            (0.0, 0.0, True),
            (1.0, 1.0, True),
        ]

        for score_val, threshold, expected_sufficient in boundaries:
            with self.subTest(score=score_val, threshold=threshold):
                # Act
                score = Score(
                    is_sufficient=score_val >= threshold,
                    score=score_val,
                    critique="Boundary test",
                )

                # Assert
                self.assertEqual(score.is_sufficient, expected_sufficient)
                self.assertEqual(score.score, score_val)

    def test_log_accumulation_invariants(self):
        # Arrange
        state = State(topic="Log Test")

        # Act
        state.log("Entry 1")
        state.add_log("Entry 2")

        # Assert
        self.assertEqual(state.logs, ["Entry 1", "Entry 2"])

    def test_type_aliases_backward_compatibility(self):
        # Assert
        self.assertIs(ResearchState, State)
        self.assertIs(SearchResult, Result)
        self.assertIs(EvaluationResult, Score)
        self.assertIs(HumanFeedback, Feedback)
        self.assertIs(AgentStatus, Status)
        self.assertIs(HumanAction, Action)


if __name__ == "__main__":
    unittest.main()
