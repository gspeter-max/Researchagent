from typing import List
from agent.state import ResearchState, SearchResult, AgentStatus
from agent.llm import LLMClient
from agent.tools.search import WebSearcher


class Researcher:
    """Formulates search queries and aggregates web findings."""

    def __init__(self, llm: LLMClient, searcher: WebSearcher):
        self.llm = llm
        self.searcher = searcher

    def run(self, state: ResearchState) -> ResearchState:
        state.status = AgentStatus.RESEARCHING
        state.iteration += 1
        state.add_log(f"Starting research cycle #{state.iteration} for: '{state.topic}'")

        qs = self.queries(state)
        state.search_queries.extend(qs)
        state.add_log(f"Generated {len(qs)} search queries: {qs}")

        seen = {f.url for f in state.findings if f.url}
        hits: List[SearchResult] = []

        for q in qs:
            for r in self.searcher.search(q, max_results=3):
                if r.url and r.url in seen:
                    continue
                if r.url:
                    seen.add(r.url)
                hits.append(r)

        state.findings.extend(hits)
        state.add_log(f"Aggregated {len(hits)} new findings (Total: {len(state.findings)}).")
        return state

    def queries(self, state: ResearchState) -> List[str]:
        past = ", ".join(f'"{q}"' for q in state.search_queries[-6:]) if state.search_queries else "None"
        ctx = [
            f"Topic: {state.topic}",
            f"Current Iteration: {state.iteration + 1}",
            f"Already Searched Queries: {past}",
        ]

        if state.latest_feedback:
            ctx.append(f"Human Guidance / Feedback: {state.latest_feedback}")

        ev = state.evaluation
        if ev:
            if ev.missing_aspects:
                ctx.append(f"Missing Aspects from Evaluation: {', '.join(ev.missing_aspects)}")
            if ev.critique:
                ctx.append(f"Evaluator Critique: {ev.critique}")

        prompt = (
            "\n".join(ctx) + "\n\n"
            "Generate 2-3 high-impact, distinct, and targeted web search queries to find the most relevant information.\n"
            "Do NOT repeat already searched queries.\n"
            "Format as JSON: {\"queries\": [\"query1\", \"query2\"]}"
        )

        try:
            resp = self.llm.generate_json(
                prompt,
                system="You are an expert research analyst designing precise web search strategies.",
                raise_on_error=True,
            )
            raw = resp.get("queries", [])
        except Exception as e:
            state.add_log(f"Warning: Query generation JSON parsing exception ({e}). Using fallback queries.")
            raw = []

        if not raw or not isinstance(raw, list):
            if state.latest_feedback:
                raw = [f"{state.topic} {state.latest_feedback}", f"{state.topic} detailed analysis"]
            else:
                raw = [f"{state.topic} overview and core principles", f"{state.topic} latest developments and analysis"]

        seen_q = set(state.search_queries)
        out = []
        for q in raw:
            text = str(q).strip()
            if text and text not in seen_q and text not in out:
                out.append(text)

        return out or [f"{state.topic} deep dive #{state.iteration + 1}"]

    generate_queries = queries


# Backward-compatibility alias
ResearchNode = Researcher
