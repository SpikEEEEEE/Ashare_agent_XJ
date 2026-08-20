from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from ashare_agent.adapters.candidate_review import OpenAICandidateReviewer
from ashare_agent.domain.candidate_pool import Candidate
from ashare_agent.domain.instruments import InstrumentId
from ashare_agent.ports.candidate_review import (
    CandidateReviewAssessment,
    CandidateReviewResult,
)
from ashare_agent.selection.config import CandidateReviewConfig
from ashare_agent.selection.review import finalize_candidate_review

from .helpers import make_settings


def _candidate(
    code: str,
    rank: int,
    *,
    industry: str,
) -> Candidate:
    exchange = "XSHG" if code.startswith("6") else "XSHE"
    suffix = "SH" if exchange == "XSHG" else "SZ"
    return Candidate(
        instrument_id=InstrumentId(exchange, code),
        provider_symbol=f"{code}.{suffix}",
        rank=rank,
        score=1.0 - rank / 10.0,
        reason="quant",
        industry=industry,
        reference_price=10.0 + rank,
        signals={"momentum_20": 0.1 * rank},
    )


def test_review_blend_is_bounded_and_preserves_final_industry_cap():
    candidates = (
        _candidate("600001", 1, industry="A"),
        _candidate("600002", 2, industry="A"),
        _candidate("000003", 3, industry="B"),
        _candidate("000004", 4, industry="B"),
    )
    scores = [20.0, 20.0, 10.0, 100.0]
    review = CandidateReviewResult(
        assessments=tuple(
            CandidateReviewAssessment(
                symbol=candidate.provider_symbol,
                score=score,
                confidence=1.0,
                thesis=f"review {candidate.provider_symbol}",
                risk_flags=("risk",),
                invalidators=("invalidator",),
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ),
        market_view="cross-sectional review",
    )

    selected, diagnostics = finalize_candidate_review(
        candidates,
        review=review,
        config=CandidateReviewConfig(enabled=True, preselect_k=4, llm_weight=0.5),
        top_k=2,
        max_industry_fraction=0.5,
    )

    assert [candidate.provider_symbol for candidate in selected] == [
        "600001.SH",
        "000004.SZ",
    ]
    assert [candidate.rank for candidate in selected] == [1, 2]
    assert selected[1].signals["quant_rank"] == 4
    assert selected[1].signals["review_score"] == 100.0
    assert diagnostics["status"] == "applied"
    assert diagnostics["industry_cap"] == 1
    assert len(diagnostics["assessments"]) == 4


def test_review_failure_falls_back_to_quant_with_final_constraints():
    candidates = (
        _candidate("600001", 1, industry="A"),
        _candidate("600002", 2, industry="A"),
        _candidate("000003", 3, industry="B"),
    )

    selected, diagnostics = finalize_candidate_review(
        candidates,
        review=None,
        config=CandidateReviewConfig(enabled=True, preselect_k=3),
        top_k=2,
        max_industry_fraction=0.5,
        fallback_reason="TimeoutError",
    )

    assert [candidate.provider_symbol for candidate in selected] == [
        "600001.SH",
        "000003.SZ",
    ]
    assert all(candidate.reason == "quant_review_fallback" for candidate in selected)
    assert diagnostics["status"] == "fallback_quant"
    assert diagnostics["failure_reason"] == "TimeoutError"
    assert diagnostics["effective_llm_weight"] == 0.0


def test_openai_candidate_reviewer_requires_exact_symbol_coverage(tmp_path):
    candidates = (
        _candidate("600001", 1, industry="A"),
        _candidate("000003", 2, industry="B"),
    )

    class FakeAgentClient:
        def invoke(self, *, response_model, **kwargs):
            del kwargs
            return response_model.model_validate(
                {
                    "assessments": [
                        {
                            "symbol": candidate.provider_symbol,
                            "score": 60.0,
                            "confidence": 0.8,
                            "thesis": "evidence",
                            "risk_flags": [],
                            "invalidators": [],
                        }
                        for candidate in candidates
                    ],
                    "market_view": "neutral",
                }
            )

        def meta(self):
            return {"calls": 1}

    settings = make_settings(tmp_path)
    reviewer = OpenAICandidateReviewer(
        settings,
        agent_client=FakeAgentClient(),
    )
    result = reviewer.review(
        candidates,
        as_of=datetime(2026, 8, 18, 16, tzinfo=ZoneInfo("Asia/Shanghai")),
        data_session="2026-08-18",
    )

    assert [item.symbol for item in result.assessments] == [
        candidate.provider_symbol for candidate in candidates
    ]
    assert result.meta["client"]["calls"] == 1
    assert Decimal(str(result.assessments[0].confidence)) == Decimal("0.8")
