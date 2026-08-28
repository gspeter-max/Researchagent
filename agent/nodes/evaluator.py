from typing import Dict, Any, List
from agent.state import ResearchState, EvaluationResult, AgentStatus
from agent.llm import LLMClient


class EvaluatorNode:
    """Evaluates the quality, depth, and sufficiency of accumulated research findings."""

    def __init__(self, llm: LLMClient, sufficiency_threshold: float = 0.75):
        self.llm = llm
        self.sufficiency_threshold = sufficiency_threshold

    def run(self, state: ResearchState) -> ResearchState:
        state.status = AgentStatus.VERIFYING
        state.add_log(f"Verifying research findings (Count: {len(state.findings)})...")

        # Build context of findings
        findings_summary = "\n".join(
            f"- [{i+1}] Title: {f.title}\n    Snippet: {f.snippet}\n    URL: {f.url}"
            for i, f in enumerate(state.findings[:12])
        )

        feedback_context = state.latest_feedback if state.latest_feedback else "None"

        prompt = (
            f"Topic: {state.topic}\n"
            f"Current Iteration: {state.iteration} of {state.max_iterations}\n"
            f"Human Guidance / Feedback: {feedback_context}\n\n"
            f"Current Verified Findings:\n{findings_summary}\n\n"
            f"Evaluate whether these findings provide a thorough, accurate, and multi-dimensional answer.\n"
            f"Verification Criteria:\n"
            f"1. Factual depth and technical relevance.\n"
            f"2. Coverage of core concepts, real-world examples, benchmarks, and challenges.\n"
            f"3. Alignment with human guidance (if provided).\n\n"
            f"Respond with JSON schema:\n"
            f"{{\n"
            f'  "score": float between 0.0 and 1.0,\n'
            f'  "critique": "concise explanation of score and data quality",\n'
            f'  "missing_aspects": ["missing aspect 1", "missing aspect 2"],\n'
            f'  "suggested_queries": ["query 1", "query 2"]\n'
            f"}}"
        )

        resp = self.llm.generate_json(
            prompt,
            system="You are a rigorous verification reviewer assessing research completeness against quality criteria."
        )

        score = float(resp.get("score", 0.5))
        critique = str(resp.get("critique", "Verification completed."))
        missing_aspects = resp.get("missing_aspects", [])
        suggested_queries = resp.get("suggested_queries", [])

        is_sufficient = score >= self.sufficiency_threshold

        eval_res = EvaluationResult(
            is_sufficient=is_sufficient,
            score=score,
            critique=critique,
            missing_aspects=missing_aspects if isinstance(missing_aspects, list) else [],
            suggested_queries=suggested_queries if isinstance(suggested_queries, list) else []
        )

        state.evaluation = eval_res
        state.evaluation_history.append(eval_res)

        state.add_log(
            f"Verification Result: Sufficient={eval_res.is_sufficient} (Score: {eval_res.score:.2f} / Threshold: {self.sufficiency_threshold:.2f}) | {eval_res.critique}"
        )
        return state
