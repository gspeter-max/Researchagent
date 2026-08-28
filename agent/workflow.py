from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, Union
from agent.state import ResearchState, AgentStatus, HumanAction, EvaluationResult
from agent.llm import LLMClient
from agent.tools.search import WebSearcher
from agent.nodes.researcher import ResearchNode
from agent.nodes.evaluator import EvaluatorNode
from agent.nodes.synthesizer import SynthesizerNode


# Callback signatures for Human-in-the-Loop (HITL)
# Initial: (topic, state) -> (action: HumanAction | str, guidance_text: Optional[str])
InitialGuidanceCallback = Callable[[str, ResearchState], Tuple[Union[HumanAction, str], Optional[str]]]

# Verification: (state) -> (action: HumanAction | str, feedback_text: Optional[str])
VerificationFeedbackCallback = Callable[[ResearchState], Tuple[Union[HumanAction, str], Optional[str]]]

# Step notification: (message, state) -> None
StepCallback = Callable[[str, ResearchState], None]


@dataclass
class WorkflowActionDirective:
    """Action outcome directive returned by action handlers."""
    terminate: bool = False
    break_loop: bool = False


class ResearchReviewWorkflow:
    """
    Loop-Engineered Research & Review Workflow.
    
    Employs Dictionary-based Action Dispatch to eliminate fragile if/else ladders.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        searcher: Optional[WebSearcher] = None,
        max_search_iterations: int = 3,
        sufficiency_threshold: float = 0.75,
    ):
        self.llm = llm or LLMClient()
        self.searcher = searcher or WebSearcher()
        self.max_search_iterations = max_search_iterations
        self.sufficiency_threshold = sufficiency_threshold

        self.research_node = ResearchNode(self.llm, self.searcher)
        self.evaluator_node = EvaluatorNode(self.llm, sufficiency_threshold=self.sufficiency_threshold)
        self.synthesizer_node = SynthesizerNode(self.llm)

        # Action Handler Dispatch Tables
        self._initial_action_handlers: Dict[
            HumanAction,
            Callable[[ResearchState, Optional[str], Callable[[str], None]], WorkflowActionDirective]
        ] = {
            HumanAction.CANCEL: self._handle_initial_cancel,
            HumanAction.PROCEED: self._handle_initial_proceed,
            HumanAction.PROCEED_OVERRIDE: self._handle_initial_proceed,
            HumanAction.SEARCH_MORE: self._handle_initial_proceed,
        }

        self._verification_action_handlers: Dict[
            HumanAction,
            Callable[[ResearchState, Optional[str], EvaluationResult, Callable[[str], None]], WorkflowActionDirective]
        ] = {
            HumanAction.CANCEL: self._handle_verification_cancel,
            HumanAction.PROCEED: self._handle_verification_override,
            HumanAction.PROCEED_OVERRIDE: self._handle_verification_override,
            HumanAction.SEARCH_MORE: self._handle_verification_search_more,
        }

    # -----------------------------------------------------------------------
    # Action Dispatch Handlers
    # -----------------------------------------------------------------------

    def _handle_initial_cancel(
        self,
        state: ResearchState,
        guidance: Optional[str],
        notify: Callable[[str], None]
    ) -> WorkflowActionDirective:
        state.status = AgentStatus.CANCELLED
        state.add_feedback(HumanAction.CANCEL, "Cancelled by user before research began.")
        notify("Workflow cancelled by user at initial alignment.")
        return WorkflowActionDirective(terminate=True)

    def _handle_initial_proceed(
        self,
        state: ResearchState,
        guidance: Optional[str],
        notify: Callable[[str], None]
    ) -> WorkflowActionDirective:
        if guidance:
            state.add_feedback(HumanAction.PROCEED, guidance)
            notify(f"Initial guidance applied: '{guidance}'")
        else:
            state.add_feedback(HumanAction.PROCEED, None)
        return WorkflowActionDirective()

    def _handle_verification_cancel(
        self,
        state: ResearchState,
        feedback_text: Optional[str],
        eval_res: EvaluationResult,
        notify: Callable[[str], None]
    ) -> WorkflowActionDirective:
        state.status = AgentStatus.CANCELLED
        state.add_feedback(
            HumanAction.CANCEL,
            feedback_text or "Cancelled by human during review.",
            eval_res.score
        )
        notify("Workflow cancelled by user during verification review.")
        return WorkflowActionDirective(terminate=True)

    def _handle_verification_override(
        self,
        state: ResearchState,
        feedback_text: Optional[str],
        eval_res: EvaluationResult,
        notify: Callable[[str], None]
    ) -> WorkflowActionDirective:
        state.add_feedback(HumanAction.PROCEED_OVERRIDE, feedback_text, eval_res.score)
        notify("Human chose to proceed with current findings. Routing to LLM Synthesizer.")
        return WorkflowActionDirective(break_loop=True)

    def _handle_verification_search_more(
        self,
        state: ResearchState,
        feedback_text: Optional[str],
        eval_res: EvaluationResult,
        notify: Callable[[str], None]
    ) -> WorkflowActionDirective:
        state.add_feedback(HumanAction.SEARCH_MORE, feedback_text, eval_res.score)
        notify(
            f"Human guidance recorded: '{feedback_text or 'Auto-refine missing aspects'}'. "
            f"Passing context to query generator for next iteration..."
        )
        return WorkflowActionDirective()

    def _parse_action(self, action_val: Any, default: HumanAction) -> HumanAction:
        if isinstance(action_val, HumanAction):
            return action_val
        if isinstance(action_val, str):
            try:
                return HumanAction(action_val.upper().strip())
            except ValueError:
                pass
        return default

    # -----------------------------------------------------------------------
    # Main Workflow Execution
    # -----------------------------------------------------------------------

    def run(
        self,
        topic: str,
        initial_guidance_callback: Optional[InitialGuidanceCallback] = None,
        verification_feedback_callback: Optional[VerificationFeedbackCallback] = None,
        on_step_callback: Optional[StepCallback] = None,
    ) -> ResearchState:
        """Executes the loop-engineered research, verification, and synthesis lifecycle."""
        state = ResearchState(
            topic=topic,
            max_iterations=self.max_search_iterations,
            sufficiency_threshold=self.sufficiency_threshold
        )

        def notify(msg: str):
            if on_step_callback:
                on_step_callback(msg, state)

        # -------------------------------------------------------------
        # Step 1: Initial Query & Human Alignment Checkpoint
        # -------------------------------------------------------------
        state.status = AgentStatus.AWAITING_INITIAL_GUIDANCE
        if initial_guidance_callback:
            raw_action, guidance = initial_guidance_callback(topic, state)
            action = self._parse_action(raw_action, default=HumanAction.PROCEED)
            handler = self._initial_action_handlers.get(action, self._handle_initial_proceed)
            directive = handler(state, guidance, notify)
            if directive.terminate:
                return state

        # -------------------------------------------------------------
        # Step 2: Self-Correcting Research & Verification Loop
        # -------------------------------------------------------------
        while state.iteration < state.max_iterations:
            notify(f"=== Research Cycle #{state.iteration + 1} ===")

            # 2a. Query Generation & Search Retrieval
            state = self.research_node.run(state)
            notify(f"Research cycle completed. Total unique findings collected: {len(state.findings)}")

            # 2b. Quality Verification
            state = self.evaluator_node.run(state)
            eval_res = state.evaluation

            notify(
                f"Verification Score: {eval_res.score:.2f} / Threshold: {self.sufficiency_threshold:.2f} "
                f"({'✓ SUFFICIENT' if eval_res.is_sufficient else '✗ BELOW THRESHOLD'})"
            )
            notify(f"Critique: {eval_res.critique}")

            # 2c. Threshold Routing
            if eval_res.is_sufficient:
                notify("✓ Quality threshold satisfied! Routing directly to LLM Synthesizer.")
                break

            # If below threshold and max iterations reached
            if state.iteration >= state.max_iterations:
                notify(f"Reached maximum loop iterations ({state.max_iterations}). Routing to synthesis.")
                break

            # 2d. Human-in-the-Loop Gate for Low Scores
            state.status = AgentStatus.AWAITING_VERIFICATION_FEEDBACK
            notify("--- Human-in-the-Loop Review Triggered (Score below threshold) ---")

            if verification_feedback_callback:
                raw_action, feedback_text = verification_feedback_callback(state)
                action = self._parse_action(raw_action, default=HumanAction.SEARCH_MORE)
            else:
                action, feedback_text = HumanAction.SEARCH_MORE, None

            handler = self._verification_action_handlers.get(action, self._handle_verification_search_more)
            directive = handler(state, feedback_text, eval_res, notify)

            if directive.terminate:
                return state
            if directive.break_loop:
                break

        # -------------------------------------------------------------
        # Step 3: LLM Synthesis (Final Answer Generation)
        # -------------------------------------------------------------
        notify("Generating final comprehensive intelligence report...")
        state = self.synthesizer_node.run(state)
        state.status = AgentStatus.COMPLETED
        notify("🎉 Synthesis complete! Final report ready.")
        return state
