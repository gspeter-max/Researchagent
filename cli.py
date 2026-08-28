#!/usr/bin/env python3
import argparse
import sys
import os
from typing import Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agent.llm import LLMClient
from agent.tools.search import WebSearcher
from agent.workflow import ResearchReviewWorkflow
from agent.state import ResearchState


# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_step(msg: str, state: ResearchState):
    print(f"{CYAN}⚙ [AGENT]{RESET} {msg}")


def render_score_bar(score: float, threshold: float = 0.75) -> str:
    total_blocks = 20
    filled = int(score * total_blocks)
    thresh_pos = int(threshold * total_blocks)
    
    bar = ""
    for i in range(total_blocks):
        if i < filled:
            bar += "█"
        elif i == thresh_pos:
            bar += "|"
        else:
            bar += "░"
    
    color = GREEN if score >= threshold else YELLOW
    return f"{color}[{bar}] {score:.2f} (Threshold: {threshold:.2f}){RESET}"


def cli_verification_feedback_callback(state: ResearchState) -> Tuple[str, Optional[str]]:
    eval_res = state.evaluation
    score = eval_res.score if eval_res else 0.0
    thresh = state.sufficiency_threshold

    print("\n" + "=" * 65)
    print(f"{BOLD}{YELLOW}🔔 VERIFICATION CHECKPOINT (Score Below Threshold){RESET}")
    print("=" * 65)
    print(f"{BOLD}Research Cycle:{RESET} #{state.iteration} of {state.max_iterations}")
    print(f"{BOLD}Score Meter:{RESET}    {render_score_bar(score, thresh)}")
    print(f"{BOLD}Critique:{RESET}       {eval_res.critique if eval_res else 'N/A'}")
    
    if eval_res and eval_res.missing_aspects:
        print(f"{BOLD}Missing Gaps:{RESET}")
        for aspect in eval_res.missing_aspects:
            print(f"  • {aspect}")

    print(f"{BOLD}Findings Count:{RESET} {len(state.findings)} sources collected")
    print("-" * 65)
    print("  • Press [Enter] to auto-search missing aspects")
    print("  • Type specific feedback / direction to guide the next search")
    print("  • Type 'proceed' (or 'p') to force generate report now")
    print("  • Type 'cancel' (or 'q') to abort")
    print("-" * 65)

    try:
        user_input = input(f"{BOLD}Next Action / Feedback: {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        sys.exit(0)

    if user_input.lower() in ["cancel", "q", "quit", "exit"]:
        return "CANCEL", None
    elif user_input.lower() in ["proceed", "p", "force", "yes", "y"]:
        return "PROCEED", None
    elif user_input:
        return "SEARCH_MORE", user_input
    else:
        return "SEARCH_MORE", None


def main():
    parser = argparse.ArgumentParser(
        description="Research Review Agent: Loop-engineered autonomous research with verification gating."
    )
    parser.add_argument("--topic", "-t", type=str, help="Research topic to investigate")
    parser.add_argument("--provider", "-p", type=str, default=None, help="LLM Provider (anthropic, openai, gemini, heuristic)")
    parser.add_argument("--api-key", "-k", type=str, default=None, help="API key for LLM provider")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model name (e.g. claude-3-5-haiku-20241022)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save the final report markdown file")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max search iterations before draft synthesis")
    parser.add_argument("--threshold", type=float, default=0.75, help="Verification sufficiency threshold (0.0 to 1.0)")
    parser.add_argument("--mock-search", action="store_true", help="Use mock search results (offline mode)")

    args = parser.parse_args()

    topic = args.topic
    if not topic:
        try:
            topic = input(f"{BOLD}Enter research topic: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

    if not topic:
        print(f"{RED}Error: Topic cannot be empty.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{GREEN}🚀 Launching Loop-Engineered Research Agent for topic: '{topic}'{RESET}\n")

    llm = LLMClient(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model
    )
    searcher = WebSearcher(use_mock=args.mock_search)

    workflow = ResearchReviewWorkflow(
        llm=llm,
        searcher=searcher,
        max_search_iterations=args.max_iterations,
        sufficiency_threshold=args.threshold
    )

    final_state = workflow.run(
        topic=topic,
        verification_feedback_callback=cli_verification_feedback_callback,
        on_step_callback=print_step
    )

    if final_state.status.value == "CANCELLED":
        print(f"\n{YELLOW}🛑 Process terminated: Workflow was cancelled.{RESET}")
        sys.exit(0)

    print("\n" + "=" * 65)
    print(f"{BOLD}{GREEN}🎉 FINAL DELIVERABLE: INTELLIGENCE REPORT{RESET}")
    print("=" * 65 + "\n")
    print(final_state.draft_report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_state.draft_report)
        print(f"\n{GREEN}✔ Report successfully saved to: {args.output}{RESET}")


if __name__ == "__main__":
    main()
