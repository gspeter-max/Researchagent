from typing import Callable, Optional, Tuple
from agent.state import ResearchState, AgentStatus
from agent.llm import LLMClient
from agent.tools.search import WebSearcher
from agent.nodes.researcher import ResearchNode
from agent.nodes.evaluator import EvaluatorNode
from agent.nodes.synthesizer import SynthesizerNode


# Callback signatures for Human-in-the-Loop (HITL)
# Initial: (topic, state) -> (action: "PROCEED" | "CANCEL", guidance_text: Optional[str])
InitialGuidanceCallback = Callable[[str, ResearchState], Tuple[str, Optional[str]]]

# Verification: (state) -> (action: "PROCEED" | "SEARCH_MORE" | "CANCEL", feedback_text: Optional[str])
VerificationFeedbackCallback = Callable[[ResearchState], Tuple[str, Optional[str]]]

# Step notification: (message, state) -> None
StepCallback = Callable[[str, ResearchState], None]


class ResearchReviewWorkflow:
    """
    Loop-Engineered Research & Review Workflow.
    
    Flow:
    1. Initial Query & Human Alignment (Proceed / Cancel / Custom Focus)
    2. Query Formulation (userquery + state.findings + state.evaluation + state.feedback)
    3. Web Search & Information Retrieval
    4. Quality Verification & Scoring (0.0 to 1.0)
    5. Threshold-Based Routing:
       - Score >= Threshold -> LLM Synthesizer -> Final Deliverable
       - Score < Threshold  -> Human-in-the-Loop (Proceed / Cancel / Search More with Feedback)
                               -> Feedback recorded in state -> Loops back to Step 2
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
            action, guidance = initial_guidance_callback(topic, state)
            action = action.upper().strip() if action else "PROCEED"

            if action == "CANCEL":
                state.status = AgentStatus.CANCELLED
                state.add_feedback(action="CANCEL", text="Cancelled by user before research began.")
                notify("Workflow cancelled by user at initial alignment.")
                return state

            if guidance:
                state.add_feedback(action="PROCEED", text=guidance)
                notify(f"Initial guidance applied: '{guidance}'")
            else:
                state.add_feedback(action="PROCEED", text=None)

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
                action, feedback_text = verification_feedback_callback(state)
                action = action.upper().strip() if action else "SEARCH_MORE"
            else:
                # Default behavior when running non-interactively
                action, feedback_text = "SEARCH_MORE", None

            if action == "CANCEL":
                state.status = AgentStatus.CANCELLED
                state.add_feedback(action="CANCEL", text=feedback_text or "Cancelled by human during review.", score=eval_res.score)
                notify("Workflow cancelled by user during verification review.")
                return state

            elif action == "PROCEED":
                # Human overrides low score to synthesize now
                state.add_feedback(action="PROCEED_OVERRIDE", text=feedback_text, score=eval_res.score)
                notify("Human chose to proceed with current findings. Routing to LLM Synthesizer.")
                break

            else:
                # SEARCH_MORE: Incorporate feedback and loop back to query generation
                state.add_feedback(action="SEARCH_MORE", text=feedback_text, score=eval_res.score)
                notify(
                    f"Human guidance recorded: '{feedback_text or 'Auto-refine missing aspects'}'. "
                    f"Passing context to query generator for next iteration..."
                )

        # -------------------------------------------------------------
        # Step 3: LLM Synthesis (Final Answer Generation)
        # -------------------------------------------------------------
        notify("Generating final comprehensive intelligence report...")
        state = self.synthesizer_node.run(state)
        state.status = AgentStatus.COMPLETED
        notify("🎉 Synthesis complete! Final report ready.")
        return state
