import unittest
from agent.state import AgentStatus, ResearchState, HumanAction
from agent.llm import LLMClient
from agent.tools.search import WebSearcher
from agent.workflow import Workflow, ResearchReviewWorkflow, Directive, WorkflowActionDirective
from cli import score_bar, render_score_bar, print_step


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.llm = LLMClient(provider="heuristic")
        self.searcher = WebSearcher(use_mock=True)

    def test_workflow_direct_completion(self):
        """When verification score is sufficient, completes autonomously without human review."""
        workflow = Workflow(
            llm=self.llm,
            searcher=self.searcher,
            max_iterations=3,
            threshold=0.70,
        )

        review_called = False

        def mock_review(state):
            nonlocal review_called
            review_called = True
            return HumanAction.PROCEED, None

        final_state = workflow.run(
            topic="Neuromorphic Computing",
            on_review=mock_review,
        )

        self.assertEqual(final_state.status, AgentStatus.COMPLETED)
        self.assertFalse(review_called)  # Score was sufficient, no human prompt needed
        self.assertTrue(len(final_state.draft_report) > 0)
        self.assertTrue(len(final_state.findings) > 0)

    def test_workflow_verification_cancel(self):
        """When verification score is below threshold and human cancels, terminates cleanly."""
        workflow = Workflow(
            llm=self.llm,
            searcher=self.searcher,
            max_iterations=3,
            threshold=0.95,
        )

        def mock_cancel(state):
            return HumanAction.CANCEL, "Stop research"

        final_state = workflow.run(
            topic="Quantum Computing",
            on_review=mock_cancel,
        )

        self.assertEqual(final_state.status, AgentStatus.CANCELLED)
        self.assertEqual(final_state.iteration, 1)
        self.assertEqual(final_state.feedback_history[-1].action, HumanAction.CANCEL)

    def test_workflow_low_score_human_feedback_loop(self):
        """When verification score is below threshold, user provides feedback and workflow loops back."""
        workflow = Workflow(
            llm=self.llm,
            searcher=self.searcher,
            max_iterations=3,
            threshold=0.90,  # High threshold to trigger HITL
        )

        feedback_calls = []

        def mock_review(state):
            feedback_calls.append(state.iteration)
            if state.iteration == 1:
                return HumanAction.SEARCH_MORE, "Focus specifically on production latency metrics"
            else:
                return HumanAction.PROCEED, None

        final_state = workflow.run(
            topic="Edge AI Acceleration",
            on_review=mock_review,
        )

        self.assertEqual(final_state.status, AgentStatus.COMPLETED)
        self.assertTrue(len(feedback_calls) >= 1)
        self.assertTrue(len(final_state.feedback_history) >= 1)
        self.assertIn("production latency metrics", final_state.feedback_history[0].feedback_text)

    def test_workflow_human_override_on_low_score(self):
        """When verification score is below threshold, user can force proceed to synthesis."""
        workflow = Workflow(
            llm=self.llm,
            searcher=self.searcher,
            max_iterations=3,
            threshold=0.99,  # Impossible to meet automatically
        )

        def mock_override(state):
            return HumanAction.PROCEED, None

        final_state = workflow.run(
            topic="Synthetic Biology",
            on_review=mock_override,
        )

        self.assertEqual(final_state.status, AgentStatus.COMPLETED)
        self.assertEqual(final_state.iteration, 1)
        self.assertEqual(final_state.feedback_history[-1].action, HumanAction.PROCEED_OVERRIDE)
        self.assertTrue(len(final_state.draft_report) > 0)

    def test_backward_compatibility_aliases(self):
        """Ensures legacy aliases and constructor arguments still work identically."""
        self.assertIs(ResearchReviewWorkflow, Workflow)
        self.assertIs(WorkflowActionDirective, Directive)

        workflow = ResearchReviewWorkflow(
            llm=self.llm,
            searcher=self.searcher,
            max_search_iterations=2,
            sufficiency_threshold=0.99,
        )

        self.assertEqual(workflow.max_iterations, 2)
        self.assertEqual(workflow.max_search_iterations, 2)
        self.assertEqual(workflow.threshold, 0.99)
        self.assertEqual(workflow.sufficiency_threshold, 0.99)

        steps = []
        final_state = workflow.run(
            topic="Legacy API Test",
            verification_feedback_callback=lambda s: ("PROCEED", None),
            on_step_callback=lambda msg, s: steps.append(msg),
        )
        self.assertEqual(final_state.status, AgentStatus.COMPLETED)
        self.assertTrue(len(steps) > 0)

    def test_cli_helpers(self):
        """Tests cli helper functions."""
        bar = score_bar(0.8, 0.75)
        self.assertIn("0.80", bar)
        self.assertEqual(render_score_bar(0.8, 0.75), bar)


if __name__ == "__main__":
    unittest.main()
