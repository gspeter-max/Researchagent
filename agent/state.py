from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AgentStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    AWAITING_INITIAL_GUIDANCE = "AWAITING_INITIAL_GUIDANCE"
    GENERATING_QUERIES = "GENERATING_QUERIES"
    RESEARCHING = "RESEARCHING"
    VERIFYING = "VERIFYING"
    AWAITING_VERIFICATION_FEEDBACK = "AWAITING_VERIFICATION_FEEDBACK"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class HumanAction(str, Enum):
    PROCEED = "PROCEED"
    PROCEED_OVERRIDE = "PROCEED_OVERRIDE"
    SEARCH_MORE = "SEARCH_MORE"
    CANCEL = "CANCEL"


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    query: str


@dataclass
class EvaluationResult:
    is_sufficient: bool
    score: float  # 0.0 to 1.0
    critique: str
    missing_aspects: List[str] = field(default_factory=list)
    suggested_queries: List[str] = field(default_factory=list)


@dataclass
class HumanFeedback:
    iteration: int
    action: HumanAction
    feedback_text: Optional[str] = None
    verification_score: Optional[float] = None


@dataclass
class ResearchState:
    topic: str
    iteration: int = 0
    max_iterations: int = 3
    sufficiency_threshold: float = 0.75
    search_queries: List[str] = field(default_factory=list)
    findings: List[SearchResult] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    evaluation_history: List[EvaluationResult] = field(default_factory=list)
    draft_report: str = ""
    latest_feedback: Optional[str] = None
    feedback_history: List[HumanFeedback] = field(default_factory=list)
    status: AgentStatus = AgentStatus.INITIALIZED
    logs: List[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        self.logs.append(message)

    def add_feedback(
        self,
        action: "HumanAction | str",
        text: Optional[str] = None,
        score: Optional[float] = None
    ) -> None:
        if text:
            self.latest_feedback = text

        if isinstance(action, str):
            action_enum = HumanAction(action.upper())
        else:
            action_enum = action

        feedback_entry = HumanFeedback(
            iteration=self.iteration,
            action=action_enum,
            feedback_text=text,
            verification_score=score
        )
        self.feedback_history.append(feedback_entry)
