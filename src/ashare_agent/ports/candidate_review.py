from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, Sequence

from ashare_agent.domain.candidate_pool import Candidate


@dataclass(frozen=True)
class CandidateReviewAssessment:
    symbol: str
    score: float
    confidence: float
    thesis: str
    risk_flags: tuple[str, ...] = ()
    invalidators: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateReviewResult:
    assessments: tuple[CandidateReviewAssessment, ...]
    market_view: str
    meta: dict[str, Any] = field(default_factory=dict)


class CandidateReviewer(Protocol):
    name: str

    def review(
        self,
        candidates: Sequence[Candidate],
        *,
        as_of: datetime,
        data_session: str,
    ) -> CandidateReviewResult:
        """Review only the supplied point-in-time candidate set."""
