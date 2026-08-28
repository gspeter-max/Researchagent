from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(str, Enum):
    INITIALIZED = "INITIALIZED"
    GENERATING_QUERIES = "GENERATING_QUERIES"
    RESEARCHING = "RESEARCHING"
    VERIFYING = "VERIFYING"
    AWAITING_VERIFICATION_FEEDBACK = "AWAITING_VERIFICATION_FEEDBACK"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class Action(str, Enum):
    PROCEED = "PROCEED"
    PROCEED_OVERRIDE = "PROCEED_OVERRIDE"
    SEARCH_MORE = "SEARCH_MORE"
    CANCEL = "CANCEL"


@dataclass
class Result:
    title: str
    snippet: str
    url: str
    query: str


@dataclass
class Score:
    is_sufficient: bool
    score: float
    critique: str
    missing_aspects: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)


@dataclass
class Feedback:
    iteration: int
    action: Action
    feedback_text: Optional[str] = None
    verification_score: Optional[float] = None


@dataclass
class State:
    topic: str
    iteration: int = 0
    max_iterations: int = 3
    sufficiency_threshold: float = 0.75
    search_queries: list[str] = field(default_factory=list)
    findings: list[Result] = field(default_factory=list)
    evaluation: Optional[Score] = None
    evaluation_history: list[Score] = field(default_factory=list)
    draft_report: str = ""
    latest_feedback: Optional[str] = None
    feedback_history: list[Feedback] = field(default_factory=list)
    status: Status = Status.INITIALIZED
    logs: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def add_log(self, msg: str) -> None:
        self.log(msg)

    def add_feedback(
        self,
        action: Action | str,
        text: Optional[str] = None,
        score: Optional[float] = None,
    ) -> None:
        if text:
            self.latest_feedback = text
        act = Action(action.upper()) if isinstance(action, str) else action
        self.feedback_history.append(
            Feedback(
                iteration=self.iteration,
                action=act,
                feedback_text=text,
                verification_score=score,
            )
        )


# Backward compatibility aliases
SearchResult = Result
EvaluationResult = Score
HumanFeedback = Feedback
ResearchState = State
AgentStatus = Status
HumanAction = Action
