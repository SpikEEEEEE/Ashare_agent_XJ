from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Sequence

from ashare_agent.domain.candidate_pool import Candidate
from ashare_agent.ports.candidate_review import CandidateReviewResult
from ashare_agent.selection.config import CandidateReviewConfig


def _quant_quality(rank: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (count - 1)


def _constrained_top_k(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    max_industry_fraction: float,
) -> tuple[list[dict[str, Any]], int]:
    target_count = min(top_k, len(rows))
    industry_cap = max(1, math.ceil(top_k * max_industry_fraction))
    selected: list[dict[str, Any]] = []
    selected_symbols: set[str] = set()
    industry_counts: dict[str, int] = {}

    for row in rows:
        industry = str(row["candidate"].industry or "UNKNOWN")
        if industry_counts.get(industry, 0) >= industry_cap:
            continue
        selected.append(row)
        selected_symbols.add(row["candidate"].provider_symbol)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        row["selection_constraint"] = "industry_cap"
        if len(selected) >= target_count:
            return selected, industry_cap

    for row in rows:
        symbol = row["candidate"].provider_symbol
        if symbol in selected_symbols:
            continue
        selected.append(row)
        selected_symbols.add(symbol)
        row["selection_constraint"] = "industry_cap_relaxed"
        if len(selected) >= target_count:
            break
    return selected, industry_cap


def finalize_candidate_review(
    candidates: Sequence[Candidate],
    *,
    review: CandidateReviewResult | None,
    config: CandidateReviewConfig,
    top_k: int,
    max_industry_fraction: float,
    fallback_reason: str | None = None,
) -> tuple[tuple[Candidate, ...], dict[str, Any]]:
    """Blend bounded LLM evidence with quant rank and enforce final constraints."""

    if not candidates:
        raise ValueError("Candidate review requires a non-empty preselection")
    if top_k < 1:
        raise ValueError("Final candidate count must be positive")

    assessment_by_symbol = (
        {item.symbol: item for item in review.assessments}
        if review is not None
        else {}
    )
    symbols = {candidate.provider_symbol for candidate in candidates}
    if review is not None and set(assessment_by_symbol) != symbols:
        raise ValueError("Candidate review symbols do not match preselection")

    rows: list[dict[str, Any]] = []
    count = len(candidates)
    for candidate in candidates:
        quant_quality = _quant_quality(candidate.rank, count)
        assessment = assessment_by_symbol.get(candidate.provider_symbol)
        if assessment is None:
            llm_score = None
            confidence = None
            effective_llm_quality = quant_quality
            composite_score = quant_quality
        else:
            llm_score = float(assessment.score)
            confidence = float(assessment.confidence)
            raw_llm_quality = llm_score / 100.0
            # Low confidence shrinks the LLM opinion toward neutral instead of
            # allowing uncertain prose to overwhelm the trained model.
            effective_llm_quality = 0.5 + (raw_llm_quality - 0.5) * confidence
            composite_score = (
                (1.0 - config.llm_weight) * quant_quality
                + config.llm_weight * effective_llm_quality
            )
        rows.append(
            {
                "candidate": candidate,
                "quant_rank": candidate.rank,
                "quant_selection_score": candidate.score,
                "quant_quality": quant_quality,
                "llm_score": llm_score,
                "llm_confidence": confidence,
                "effective_llm_quality": effective_llm_quality,
                "composite_score": composite_score,
                "assessment": assessment,
                "selected": False,
                "selection_constraint": None,
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["composite_score"]),
            int(row["quant_rank"]),
            row["candidate"].provider_symbol,
        )
    )
    selected_rows, industry_cap = _constrained_top_k(
        rows,
        top_k=top_k,
        max_industry_fraction=max_industry_fraction,
    )
    selected_symbols = {
        row["candidate"].provider_symbol for row in selected_rows
    }
    for row in rows:
        row["selected"] = row["candidate"].provider_symbol in selected_symbols

    final_candidates: list[Candidate] = []
    for rank, row in enumerate(selected_rows, start=1):
        candidate = row["candidate"]
        assessment = row["assessment"]
        review_signals: dict[str, Any] = {
            **candidate.signals,
            "quant_rank": int(row["quant_rank"]),
            "quant_selection_score": float(row["quant_selection_score"]),
            "quant_quality": float(row["quant_quality"]),
            "review_composite_score": float(row["composite_score"]),
            "review_selection_constraint": row["selection_constraint"],
        }
        if assessment is not None:
            review_signals.update(
                {
                    "review_score": float(assessment.score),
                    "review_confidence": float(assessment.confidence),
                    "review_thesis": assessment.thesis,
                    "review_risk_flags": list(assessment.risk_flags),
                    "review_invalidators": list(assessment.invalidators),
                }
            )
        final_candidates.append(
            replace(
                candidate,
                rank=rank,
                score=float(row["composite_score"]),
                reason=(
                    "llm_cross_sectional_review"
                    if review is not None
                    else "quant_review_fallback"
                ),
                signals=review_signals,
            )
        )

    assessment_audit: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["quant_rank"])):
        candidate = row["candidate"]
        assessment = row["assessment"]
        assessment_audit.append(
            {
                "symbol": candidate.provider_symbol,
                "name": candidate.name,
                "industry": candidate.industry,
                "reference_price": candidate.reference_price,
                "quant_rank": int(row["quant_rank"]),
                "quant_selection_score": float(row["quant_selection_score"]),
                "quant_quality": float(row["quant_quality"]),
                "review_score": (
                    float(assessment.score) if assessment is not None else None
                ),
                "review_confidence": (
                    float(assessment.confidence)
                    if assessment is not None
                    else None
                ),
                "composite_score": float(row["composite_score"]),
                "selected": bool(row["selected"]),
                "selection_constraint": row["selection_constraint"],
                "thesis": assessment.thesis if assessment is not None else None,
                "risk_flags": (
                    list(assessment.risk_flags) if assessment is not None else []
                ),
                "invalidators": (
                    list(assessment.invalidators) if assessment is not None else []
                ),
                "signals": candidate.signals,
            }
        )

    diagnostics = {
        "status": "applied" if review is not None else "fallback_quant",
        "failure_reason": fallback_reason,
        "preselect_count": len(candidates),
        "final_count": len(final_candidates),
        "configured_llm_weight": config.llm_weight,
        "effective_llm_weight": config.llm_weight if review is not None else 0.0,
        "industry_cap": industry_cap,
        "market_view": review.market_view if review is not None else None,
        "review_meta": review.meta if review is not None else {},
        "assessments": assessment_audit,
    }
    return tuple(final_candidates), diagnostics
