from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any, Union
from agent.state import ResearchState, AgentStatus, HumanAction, EvaluationResult
from agent.llm import LLMClient
from agent.tools.search import WebSearcher
from agent.nodes.researcher import ResearchNode
from agent.nodes.evaluator import EvaluatorNode
from agent.nodes.synthesizer import SynthesizerNode

ReviewCallback = Callable[[ResearchState], Tuple[Union[HumanAction, str], Optional[str]]]
StepCallback = Callable[[str, ResearchState], None]

# Backward compatibility alias
VerificationFeedbackCallback = ReviewCallback


@dataclass
class Directive:
    """Action outcome directive returned by action handlers."""
    terminate: bool = False
    break_loop: bool = False


# Backward compatibility alias
WorkflowActionDirective = Directive


class Workflow:
    """Loop-engineered autonomous research & review workflow."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        searcher: Optional[WebSearcher] = None,
        max_iterations: int = 3,
        threshold: float = 0.75,
        *,
        max_search_iterations: Optional[int] = None,
        sufficiency_threshold: Optional[float] = None,
    ):
        self.llm = llm or LLMClient()
        self.searcher = searcher or WebSearcher()
        self.max_iterations = max_search_iterations if max_search_iterations is not None else max_iterations
        self.threshold = sufficiency_threshold if sufficiency_threshold is not None else threshold

        # Backward compatibility aliases
        self.max_search_iterations = self.max_iterations
        self.sufficiency_threshold = self.threshold

        self.researcher = ResearchNode(self.llm, self.searcher)
        self.evaluator = EvaluatorNode(self.llm, sufficiency_threshold=self.threshold)
        self.synthesizer = SynthesizerNode(self.llm)

        # Backward compatibility node aliases
        self.research_node = self.researcher
        self.evaluator_node = self.evaluator
        self.synthesizer_node = self.synthesizer

        self._handlers: Dict[
            HumanAction,
            Callable[[ResearchState, Optional[str], EvaluationResult, Callable[[str], None]], Directive]
        ] = {
            HumanAction.CANCEL: self._cancel,
            HumanAction.PROCEED: self._override,
            HumanAction.PROCEED_OVERRIDE: self._override,
            HumanAction.SEARCH_MORE: self._retry,
        }
        self._verification_action_handlers = self._handlers

    def _cancel(
        self,
        state: ResearchState,
        text: Optional[str],
        res: EvaluationResult,
        notify: Callable[[str], None],
    ) -> Directive:
        state.status = AgentStatus.CANCELLED
        state.add_feedback(HumanAction.CANCEL, text or "Cancelled by user.", res.score)
        notify("Workflow cancelled by user during review.")
        return Directive(terminate=True)

    def _override(
        self,
        state: ResearchState,
        text: Optional[str],
        res: EvaluationResult,
        notify: Callable[[str], None],
    ) -> Directive:
        state.add_feedback(HumanAction.PROCEED_OVERRIDE, text, res.score)
        notify("Proceeding with current findings. Routing to synthesizer.")
        return Directive(break_loop=True)

    def _retry(
        self,
        state: ResearchState,
        text: Optional[str],
        res: EvaluationResult,
        notify: Callable[[str], None],
    ) -> Directive:
        state.add_feedback(HumanAction.SEARCH_MORE, text, res.score)
        notify(f"Human guidance recorded: '{text or 'Auto-refine missing aspects'}'. Querying next iteration...")
        return Directive()

    # Aliases for action handlers
    _search_more = _retry
    _handle_verification_cancel = _cancel
    _handle_verification_override = _override
    _handle_verification_search_more = _retry

    def _parse(self, val: Any, default: HumanAction = HumanAction.SEARCH_MORE) -> HumanAction:
        if isinstance(val, HumanAction):
            return val
        if isinstance(val, str):
            try:
                return HumanAction(val.upper().strip())
            except ValueError:
                pass
        return default

    # Backward compatibility alias
    _parse_action = _parse

    def run(
        self,
        topic: str,
        on_review: Optional[ReviewCallback] = None,
        on_step: Optional[StepCallback] = None,
        *,
        verification_feedback_callback: Optional[ReviewCallback] = None,
        on_step_callback: Optional[StepCallback] = None,
    ) -> ResearchState:
        """Executes the autonomous research, verification, and synthesis lifecycle."""
        on_review = on_review or verification_feedback_callback
        on_step = on_step or on_step_callback

        state = ResearchState(
            topic=topic,
            max_iterations=self.max_iterations,
            sufficiency_threshold=self.threshold,
        )

        def notify(msg: str) -> None:
            if on_step:
                on_step(msg, state)

        while state.iteration < state.max_iterations:
            notify(f"=== Research Cycle #{state.iteration + 1} ===")

            state = self.researcher.run(state)
            notify(f"Research cycle completed. Total unique findings collected: {len(state.findings)}")

            state = self.evaluator.run(state)
            res = state.evaluation

            notify(
                f"Verification Score: {res.score:.2f} / Threshold: {self.threshold:.2f} "
                f"({'✓ SUFFICIENT' if res.is_sufficient else '✗ BELOW THRESHOLD'})"
            )
            notify(f"Critique: {res.critique}")

            if res.is_sufficient:
                notify("✓ Quality threshold satisfied! Routing directly to LLM Synthesizer.")
                break

            if state.iteration >= state.max_iterations:
                notify(f"Reached maximum loop iterations ({state.max_iterations}). Routing to synthesis.")
                break

            state.status = AgentStatus.AWAITING_VERIFICATION_FEEDBACK
            notify("--- Human-in-the-Loop Review Triggered (Score below threshold) ---")

            if on_review:
                raw_act, text = on_review(state)
                action = self._parse(raw_act, default=HumanAction.SEARCH_MORE)
            else:
                action, text = HumanAction.SEARCH_MORE, None

            handler = self._handlers.get(action, self._retry)
            directive = handler(state, text, res, notify)

            if directive.terminate:
                return state
            if directive.break_loop:
                break

        notify("Generating final comprehensive intelligence report...")
        state = self.synthesizer.run(state)
        state.status = AgentStatus.COMPLETED
        notify("🎉 Synthesis complete! Final report ready.")
        return state


# Backward compatibility alias
ResearchReviewWorkflow = Workflow
