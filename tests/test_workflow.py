"""Hermetic, table-driven unit tests for Workflow lifecycle, threshold gating, and HITL routing."""

import unittest
from agent.state import Status, State, Action
from agent.llm import Client
from agent.tools.search import Searcher
from agent.workflow import Workflow, ResearchReviewWorkflow, Directive, WorkflowActionDirective
from cli import score_bar, render_score_bar, print_step


class TestWorkflowLifecycle(unittest.TestCase):
    """Tests Workflow state transitions, boundary gating, and action dispatch routing."""

    def setUp(self):
        self.llm = Client(provider="heuristic")
        self.searcher = Searcher(use_mock=True)

    def test_workflow_direct_completion_when_sufficient(self):
        """When score meets threshold, routes directly to synthesis without human prompt."""
        # Arrange
        workflow = Workflow(
            llm=self.llm,
            searcher=self.searcher,
            max_iterations=3,
            threshold=0.70,
        )
        human_prompted = False

        def mock_review(state):
            nonlocal human_prompted
            human_prompted = True
            return Action.PROCEED, None

        # Act
        final_state = workflow.run(
            topic="Neuromorphic Computing",
            on_review=mock_review,
        )

        # Assert: Finished successfully with 0 human prompts
        self.assertEqual(final_state.status, Status.COMPLETED)
        self.assertFalse(human_prompted)
        self.assertTrue(len(final_state.draft_report) > 0)
        self.assertGreater(len(final_state.findings), 0)

    def test_workflow_action_dispatch_table_driven_vectors(self):
        """Table-driven test of human feedback actions (CANCEL, PROCEED_OVERRIDE, SEARCH_MORE)."""
        # Arrange: Matrix of (action, guidance_text, expected_final_status, expected_iterations)
        vectors = [
            (Action.CANCEL, "Stop research", Status.CANCELLED, 1),
            (Action.PROCEED_OVERRIDE, "Force report now", Status.COMPLETED, 1),
        ]

        for action, text, expected_status, expected_iter in vectors:
            with self.subTest(action=action):
                # Arrange: Set impossible threshold (0.99) to force HITL prompt
                wf = Workflow(llm=self.llm, searcher=self.searcher, threshold=0.99)

                # Act
                res_state = wf.run(
                    topic="Synthetic Biology",
                    on_review=lambda s: (action, text),
                )

                # Assert
                self.assertEqual(res_state.status, expected_status)
                self.assertEqual(res_state.iteration, expected_iter)
                self.assertEqual(res_state.feedback_history[-1].action, action)

    def test_workflow_low_score_human_feedback_loop(self):
        """Tests iterative feedback loop where user guides the next search cycle."""
        # Arrange: High threshold to trigger HITL in cycle 1
        workflow = Workflow(
            llm=self.llm,
            searcher=self.searcher,
            max_iterations=3,
            threshold=0.90,
        )
        feedback_rounds = []

        def mock_review(state):
            feedback_rounds.append(state.iteration)
            if state.iteration == 1:
                return Action.SEARCH_MORE, "Focus on latency metrics"
            return Action.PROCEED, None

        # Act
        final_state = workflow.run(
            topic="Edge AI Acceleration",
            on_review=mock_review,
        )

        # Assert: Looped and incorporated user feedback
        self.assertEqual(final_state.status, Status.COMPLETED)
        self.assertTrue(len(feedback_rounds) >= 1)
        self.assertIn("Focus on latency metrics", final_state.feedback_history[0].feedback_text)

    def test_cli_score_bar_boundary_value_analysis(self):
        """Tests CLI score bar formatting across threshold boundaries."""
        # Arrange: Matrix of (score, threshold, should_contain_green)
        bars = [
            (0.80, 0.75, "0.80"),
            (0.50, 0.75, "0.50"),
            (0.00, 0.75, "0.00"),
            (1.00, 0.75, "1.00"),
        ]

        for score, threshold, substr in bars:
            with self.subTest(score=score, threshold=threshold):
                # Act
                bar = score_bar(score, threshold)

                # Assert
                self.assertIn(substr, bar)
                self.assertEqual(render_score_bar(score, threshold), bar)

    def test_backward_compatibility_aliases(self):
        # Assert
        self.assertIs(ResearchReviewWorkflow, Workflow)
        self.assertIs(WorkflowActionDirective, Directive)

        wf = ResearchReviewWorkflow(
            llm=self.llm,
            searcher=self.searcher,
            max_search_iterations=2,
            sufficiency_threshold=0.95,
        )
        self.assertEqual(wf.max_iterations, 2)
        self.assertEqual(wf.threshold, 0.95)


if __name__ == "__main__":
    unittest.main()
