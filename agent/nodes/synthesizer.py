from agent.state import ResearchState, AgentStatus
from agent.llm import LLMClient


class SynthesizerNode:
    """Synthesizes accumulated research findings into a comprehensive draft report."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        state.status = AgentStatus.SYNTHESIZING
        state.add_log("Synthesizing comprehensive research report...")

        sources_text = "\n".join(
            f"[{i+1}] {f.title}\n    Snippet: {f.snippet}\n    Source: {f.url}"
            for i, f in enumerate(state.findings)
        )

        feedback_note = f"\nHuman guidance incorporated: {state.latest_feedback}" if state.latest_feedback else ""

        prompt = (
            f"Topic: {state.topic}\n"
            f"{feedback_note}\n\n"
            f"Verified Research Sources ({len(state.findings)} total):\n{sources_text}\n\n"
            f"Write a thorough, structured, and insightful Research Report in Markdown.\n"
            f"Include:\n"
            f"1. Executive Summary\n"
            f"2. Core Concepts & Technical Breakdown\n"
            f"3. Recent Progress & Key Applications\n"
            f"4. Challenges, Trade-offs & Future Outlook\n"
            f"5. References & Source Links (with citations to the numbered sources [1], [2], etc.)\n\n"
            f"Make the report informative, clear, and well-organized."
        )

        draft = self.llm.generate(
            prompt,
            system="You are a senior lead researcher writing a definitive intelligence report."
        )

        # If fallback text or clean-up needed
        if not draft.strip():
            draft = (
                f"# Research Report: {state.topic}\n\n"
                f"## Executive Summary\n"
                f"Research conducted across {len(state.findings)} verified sources.\n\n"
                f"## Key Sources\n" + sources_text
            )

        state.draft_report = draft
        state.add_log("Draft report synthesized successfully.")
        return state
