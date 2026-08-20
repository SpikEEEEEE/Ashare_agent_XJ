from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ashare_agent.adapters.portfolio_multi_agent import (
    OpenAIStructuredAgentClient,
)
from ashare_agent.core.config import Settings
from ashare_agent.core.json_safety import sanitize_json_value
from ashare_agent.domain.candidate_pool import Candidate
from ashare_agent.ports.candidate_review import (
    CandidateReviewAssessment,
    CandidateReviewer,
    CandidateReviewResult,
)


CompactText = Annotated[str, Field(min_length=1, max_length=240)]
RiskText = Annotated[str, Field(min_length=1, max_length=80)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class _Assessment(_StrictModel):
    symbol: str = Field(min_length=1, max_length=32)
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    thesis: CompactText
    risk_flags: list[RiskText] = Field(max_length=3)
    invalidators: list[CompactText] = Field(max_length=2)


class _ReviewOutput(_StrictModel):
    assessments: list[_Assessment] = Field(min_length=1, max_length=200)
    market_view: str = Field(min_length=1, max_length=800)


class OpenAICandidateReviewer(CandidateReviewer):
    """One-call, pool-constrained cross-sectional candidate reviewer."""

    name = "llm_cross_sectional_review"

    def __init__(
        self,
        settings: Settings,
        *,
        agent_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.agent_client = agent_client or OpenAIStructuredAgentClient(settings)

    @staticmethod
    def _payload_candidate(candidate: Candidate) -> dict[str, Any]:
        return {
            "symbol": candidate.provider_symbol,
            "name": candidate.name,
            "industry": candidate.industry,
            "quant_rank": candidate.rank,
            "quant_score": candidate.score,
            "reference_price": candidate.reference_price,
            "signals": sanitize_json_value(candidate.signals),
        }

    def review(
        self,
        candidates: Sequence[Candidate],
        *,
        as_of: datetime,
        data_session: str,
    ) -> CandidateReviewResult:
        if not candidates:
            raise ValueError("Candidate reviewer requires at least one candidate")
        symbols = [candidate.provider_symbol for candidate in candidates]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Candidate reviewer input contains duplicate symbols")

        output = self.agent_client.invoke(
            agent_name="candidate_pool_reviewer",
            system_prompt=(
                "You are an A-share cross-sectional stock-selection reviewer, not a "
                "trader. Compare every supplied candidate using only the supplied "
                "point-in-time quantitative evidence. Do not introduce symbols, use "
                "future information, recommend position sizes, or issue buy/sell "
                "instructions. Return exactly one assessment for every input symbol. "
                "Score relative candidate quality from 0 to 100. Confidence must fall "
                "when evidence is incomplete or conflicting. Thesis, risks, and "
                "invalidators must be concise and evidence-linked."
            ),
            payload={
                "market": "CN",
                "as_of": as_of.isoformat(),
                "data_session": data_session,
                "instruction": (
                    "Review and comparatively score the supplied pool. Preserve the "
                    "exact symbol set and return one row per symbol."
                ),
                "candidates": [
                    self._payload_candidate(candidate) for candidate in candidates
                ],
            },
            response_model=_ReviewOutput,
        )

        output_symbols = [item.symbol for item in output.assessments]
        if len(output_symbols) != len(set(output_symbols)):
            raise ValueError("Candidate review returned duplicate symbols")
        if set(output_symbols) != set(symbols):
            raise ValueError("Candidate review did not preserve the input symbol set")
        by_symbol = {item.symbol: item for item in output.assessments}
        assessments = tuple(
            CandidateReviewAssessment(
                symbol=symbol,
                score=float(by_symbol[symbol].score),
                confidence=float(by_symbol[symbol].confidence),
                thesis=by_symbol[symbol].thesis,
                risk_flags=tuple(by_symbol[symbol].risk_flags),
                invalidators=tuple(by_symbol[symbol].invalidators),
            )
            for symbol in symbols
        )
        meta_getter = getattr(self.agent_client, "meta", None)
        client_meta = meta_getter() if callable(meta_getter) else {}
        return CandidateReviewResult(
            assessments=assessments,
            market_view=output.market_view,
            meta={
                "reviewer": self.name,
                "model": self.settings.llm_model,
                "client": sanitize_json_value(client_meta),
            },
        )
