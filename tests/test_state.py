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


class TestState(unittest.TestCase):
    def test_state_init(self):
        state = State(topic="AI Safety")
        self.assertEqual(state.topic, "AI Safety")
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.max_iterations, 3)
        self.assertEqual(state.status, Status.INITIALIZED)
        self.assertEqual(len(state.findings), 0)
        self.assertIsNone(state.evaluation)

    def test_log_methods(self):
        state = State(topic="Test")
        state.log("Terse log")
        state.add_log("Legacy log")
        self.assertEqual(state.logs, ["Terse log", "Legacy log"])

    def test_add_feedback(self):
        state = State(topic="Test")
        state.iteration = 1
        state.add_feedback(Action.SEARCH_MORE, text="More details", score=0.6)
        self.assertEqual(state.latest_feedback, "More details")
        self.assertEqual(len(state.feedback_history), 1)
        self.assertEqual(state.feedback_history[0].action, Action.SEARCH_MORE)
        self.assertEqual(state.feedback_history[0].verification_score, 0.6)

        # String action support
        state.add_feedback("proceed", text="Good enough")
        self.assertEqual(state.feedback_history[1].action, Action.PROCEED)

    def test_dataclasses_and_aliases(self):
        self.assertIs(ResearchState, State)
        self.assertIs(SearchResult, Result)
        self.assertIs(EvaluationResult, Score)
        self.assertIs(HumanFeedback, Feedback)
        self.assertIs(AgentStatus, Status)
        self.assertIs(HumanAction, Action)

        res = Result(title="Test", snippet="Snippet", url="https://example.com", query="q")
        self.assertEqual(res.title, "Test")
        self.assertEqual(res.url, "https://example.com")

        score = Score(is_sufficient=True, score=0.9, critique="Solid", missing_aspects=[], suggested_queries=[])
        self.assertTrue(score.is_sufficient)
        self.assertEqual(score.score, 0.9)


if __name__ == "__main__":
    unittest.main()
