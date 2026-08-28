from agent.state import ResearchState, EvaluationResult, AgentStatus
from agent.llm import LLMClient


class Evaluator:
    """Evaluates quality, depth, and sufficiency of research findings."""

    def __init__(self, llm: LLMClient, sufficiency_threshold: float = 0.75):
        self.llm = llm
        self.sufficiency_threshold = sufficiency_threshold

    def run(self, state: ResearchState) -> ResearchState:
        state.status = AgentStatus.VERIFYING
        state.add_log(f"Verifying research findings (Count: {len(state.findings)})...")

        summary = "\n".join(
            f"- [{i+1}] Title: {f.title}\n    Snippet: {f.snippet}\n    URL: {f.url}"
            for i, f in enumerate(state.findings[:12])
        )
        fb = state.latest_feedback or "None"

        prompt = (
            f"Topic: {state.topic}\n"
            f"Current Iteration: {state.iteration} of {state.max_iterations}\n"
            f"Human Guidance / Feedback: {fb}\n\n"
            f"Current Verified Findings:\n{summary}\n\n"
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

        try:
            resp = self.llm.generate_json(
                prompt,
                system="You are a rigorous verification reviewer assessing research completeness against quality criteria.",
                raise_on_error=True,
            )
            score = float(resp.get("score", 0.5))
            critique = str(resp.get("critique", "Verification completed."))
            missing = resp.get("missing_aspects", [])
            suggested = resp.get("suggested_queries", [])
        except Exception as e:
            state.add_log(f"Warning: Evaluator JSON parsing exception ({e}). Defaulting to partial score.")
            score = 0.5
            critique = "Could not parse structured verification result; triggering re-query loop."
            missing = ["Further technical details and real-world evidence"]
            suggested = []

        res = EvaluationResult(
            is_sufficient=score >= self.sufficiency_threshold,
            score=score,
            critique=critique,
            missing_aspects=missing if isinstance(missing, list) else [],
            suggested_queries=suggested if isinstance(suggested, list) else [],
        )

        state.evaluation = res
        state.evaluation_history.append(res)
        state.add_log(
            f"Verification Result: Sufficient={res.is_sufficient} "
            f"(Score: {res.score:.2f} / Threshold: {self.sufficiency_threshold:.2f}) | {res.critique}"
        )
        return state


# Backward-compatibility alias
EvaluatorNode = Evaluator
