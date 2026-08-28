from typing import List
from agent.state import ResearchState, SearchResult, AgentStatus
from agent.llm import LLMClient
from agent.tools.search import WebSearcher


class ResearchNode:
    """Generates targeted search queries and aggregates web search findings."""

    def __init__(self, llm: LLMClient, searcher: WebSearcher):
        self.llm = llm
        self.searcher = searcher

    def run(self, state: ResearchState) -> ResearchState:
        state.status = AgentStatus.RESEARCHING
        state.iteration += 1
        state.add_log(f"Starting research cycle #{state.iteration} for: '{state.topic}'")

        # 1. Formulate search queries
        queries = self.generate_queries(state)
        state.search_queries.extend(queries)
        state.add_log(f"Generated {len(queries)} search queries: {queries}")

        # 2. Execute searches and aggregate findings
        existing_urls = {f.url for f in state.findings if f.url}
        new_findings: List[SearchResult] = []

        for q in queries:
            results = self.searcher.search(q, max_results=3)
            for r in results:
                if r.url and r.url in existing_urls:
                    continue
                if r.url:
                    existing_urls.add(r.url)
                new_findings.append(r)

        state.findings.extend(new_findings)
        state.add_log(f"Aggregated {len(new_findings)} new findings (Total: {len(state.findings)}).")
        return state

    def generate_queries(self, state: ResearchState) -> List[str]:
        past_queries_str = ", ".join(f'"{q}"' for q in state.search_queries[-6:]) if state.search_queries else "None"
        
        # Build contextual query prompt
        context_parts = [
            f"Topic: {state.topic}",
            f"Current Iteration: {state.iteration + 1}",
            f"Already Searched Queries: {past_queries_str}"
        ]

        if state.latest_feedback:
            context_parts.append(f"Human Guidance / Feedback: {state.latest_feedback}")

        if state.evaluation:
            if state.evaluation.missing_aspects:
                context_parts.append(f"Missing Aspects from Evaluation: {', '.join(state.evaluation.missing_aspects)}")
            if state.evaluation.critique:
                context_parts.append(f"Evaluator Critique: {state.evaluation.critique}")

        prompt = (
            "\n".join(context_parts) + "\n\n"
            "Generate 2-3 high-impact, distinct, and targeted web search queries to find the most relevant information.\n"
            "Do NOT repeat already searched queries.\n"
            "Format as JSON: {\"queries\": [\"query1\", \"query2\"]}"
        )

        try:
            resp = self.llm.generate_json(
                prompt,
                system="You are an expert research analyst designing precise web search strategies.",
                raise_on_error=True
            )
            queries = resp.get("queries", [])
        except Exception as e:
            state.add_log(f"Warning: Query generation JSON parsing exception ({e}). Using fallback queries.")
            queries = []
        if not queries or not isinstance(queries, list):
            # Fallback queries
            if state.latest_feedback:
                queries = [
                    f"{state.topic} {state.latest_feedback}",
                    f"{state.topic} detailed analysis"
                ]
            else:
                queries = [
                    f"{state.topic} overview and core principles",
                    f"{state.topic} latest developments and analysis"
                ]

        cleaned_queries = []
        for q in queries:
            q_str = str(q).strip()
            if q_str and q_str not in state.search_queries and q_str not in cleaned_queries:
                cleaned_queries.append(q_str)

        return cleaned_queries or [f"{state.topic} deep dive #{state.iteration + 1}"]
