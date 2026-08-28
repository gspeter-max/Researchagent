import unittest
from agent.state import AgentStatus, ResearchState
from agent.llm import LLMClient
from agent.tools.search import WebSearcher
from agent.workflow import ResearchReviewWorkflow


class TestResearchReviewWorkflow(unittest.TestCase):
    def setUp(self):
        self.llm = LLMClient(provider="heuristic")
        self.searcher = WebSearcher(use_mock=True)

    def test_workflow_initial_cancel(self):
        workflow = ResearchReviewWorkflow(
            llm=self.llm,
            searcher=self.searcher,
            max_search_iterations=2
        )

        def mock_initial_cancel(topic, state):
            return "CANCEL", None

        final_state = workflow.run(
            topic="Quantum Computing",
            initial_guidance_callback=mock_initial_cancel
        )

        self.assertEqual(final_state.status, AgentStatus.CANCELLED)
        self.assertEqual(final_state.iteration, 0)
        self.assertEqual(len(final_state.findings), 0)

    def test_workflow_initial_guidance_and_completion(self):
        workflow = ResearchReviewWorkflow(
            llm=self.llm,
            searcher=self.searcher,
            max_search_iterations=3,
            sufficiency_threshold=0.70
        )

        def mock_initial_guidance(topic, state):
            return "PROCEED", "Focus on practical benchmarks and hardware"

        final_state = workflow.run(
            topic="Neuromorphic Computing",
            initial_guidance_callback=mock_initial_guidance
        )

        self.assertEqual(final_state.status, AgentStatus.COMPLETED)
        self.assertTrue(len(final_state.draft_report) > 0)
        self.assertTrue(len(final_state.findings) > 0)
        self.assertIn("practical benchmarks", final_state.latest_feedback.lower())

    def test_workflow_low_score_human_feedback_loop(self):
        workflow = ResearchReviewWorkflow(
            llm=self.llm,
            searcher=self.searcher,
            max_search_iterations=3,
            sufficiency_threshold=0.90  # High threshold to trigger HITL
        )

        feedback_calls = []

        def mock_verification_feedback(state):
            feedback_calls.append(state.iteration)
            if state.iteration == 1:
                return "SEARCH_MORE", "Focus specifically on production latency metrics"
            else:
                return "PROCEED", None

        final_state = workflow.run(
            topic="Edge AI Acceleration",
            verification_feedback_callback=mock_verification_feedback
        )

        self.assertEqual(final_state.status, AgentStatus.COMPLETED)
        self.assertTrue(len(feedback_calls) >= 1)
        self.assertTrue(len(final_state.feedback_history) >= 1)
        self.assertIn("production latency metrics", final_state.feedback_history[0].feedback_text)

    def test_workflow_human_override_on_low_score(self):
        workflow = ResearchReviewWorkflow(
            llm=self.llm,
            searcher=self.searcher,
            max_search_iterations=3,
            sufficiency_threshold=0.99  # Impossible to meet automatically
        )

        def mock_verification_override(state):
            return "PROCEED", None

        final_state = workflow.run(
            topic="Synthetic Biology",
            verification_feedback_callback=mock_verification_override
        )

        self.assertEqual(final_state.status, AgentStatus.COMPLETED)
        self.assertEqual(final_state.iteration, 1)
        self.assertEqual(final_state.feedback_history[-1].action, "PROCEED_OVERRIDE")
        self.assertTrue(len(final_state.draft_report) > 0)


if __name__ == "__main__":
    unittest.main()
