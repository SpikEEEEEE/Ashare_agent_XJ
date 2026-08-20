from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

from ashare_agent.domain.candidate_evaluation import CandidatePoolEvaluation
from ashare_agent.domain.candidate_pool import CandidatePool


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else None


def _return_for_symbol(
    prices: dict[tuple[str, pd.Timestamp], float],
    symbol: str,
    base_session: pd.Timestamp,
    target_session: pd.Timestamp,
) -> float | None:
    base = prices.get((symbol, base_session))
    target = prices.get((symbol, target_session))
    if base is None or target is None or base <= 0 or target <= 0:
        return None
    return float(target / base - 1.0)


def _rank_ic(rows: list[dict[str, Any]], field: str, horizon: int) -> float | None:
    points = [
        (float(row[field]), float(row["returns"][str(horizon)]))
        for row in rows
        if _finite_float(row.get(field)) is not None
        and _finite_float(row.get("returns", {}).get(str(horizon))) is not None
    ]
    if len(points) < 3:
        return None
    left = pd.Series([point[0] for point in points]).rank(method="average")
    right = pd.Series([point[1] for point in points]).rank(method="average")
    correlation = left.corr(right)
    return _finite_float(correlation)


def evaluate_candidate_pool(
    pool: CandidatePool,
    frame: pd.DataFrame,
    *,
    data_cutoff: date,
    data_source: str,
    horizons: Iterable[int],
    evaluated_at: datetime,
) -> CandidatePoolEvaluation:
    """Evaluate a saved pool using only observations available at data_cutoff."""

    configured_horizons = tuple(sorted(set(int(item) for item in horizons)))
    if not configured_horizons or configured_horizons[0] < 1:
        raise ValueError("Outcome tracking requires positive horizons")
    required = {"date", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Outcome data is missing columns: {missing}")

    market = frame.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    if market["date"].isna().any():
        raise ValueError("Outcome data contains invalid dates")
    cutoff = pd.Timestamp(data_cutoff).normalize()
    market = market.loc[market["date"].le(cutoff)].copy()
    if market.empty:
        raise ValueError("Outcome data contains no observations at the cutoff")

    if "ts_code" in market.columns:
        market["_symbol"] = market["ts_code"].astype(str).str.strip().str.upper()
    elif "code" in market.columns:
        suffix_by_code = {
            candidate.instrument_id.local_code: candidate.provider_symbol
            for candidate in pool.candidates
        }
        market["_symbol"] = (
            market["code"].astype(str).str.strip().str.zfill(6).map(suffix_by_code)
        )
    else:
        raise ValueError("Outcome data requires ts_code or code")

    price_column = "adj_close" if "adj_close" in market.columns else "close"
    market[price_column] = pd.to_numeric(market[price_column], errors="coerce")
    market = market.loc[
        market["_symbol"].notna()
        & market[price_column].notna()
        & market[price_column].gt(0)
    ]
    market = market.drop_duplicates(["_symbol", "date"], keep="last")
    base_session = pd.Timestamp(pool.data_session).normalize()
    sessions = sorted(
        pd.Timestamp(item)
        for item in market.loc[market["date"].gt(base_session), "date"].unique()
    )
    available_horizons = tuple(
        horizon for horizon in configured_horizons if len(sessions) >= horizon
    )
    target_sessions = {
        horizon: sessions[horizon - 1] for horizon in available_horizons
    }
    prices = {
        (str(symbol), pd.Timestamp(session)): float(price)
        for symbol, session, price in market[
            ["_symbol", "date", price_column]
        ].itertuples(index=False, name=None)
    }

    benchmark_returns: dict[int, float | None] = {}
    symbols_by_session: dict[pd.Timestamp, set[str]] = {}
    for session, group in market.groupby("date", sort=False):
        symbols_by_session[pd.Timestamp(session)] = set(group["_symbol"].astype(str))
    for horizon, target_session in target_sessions.items():
        common = symbols_by_session.get(base_session, set()) & symbols_by_session.get(
            target_session, set()
        )
        benchmark_returns[horizon] = _mean(
            _return_for_symbol(
                prices,
                symbol,
                base_session,
                target_session,
            )
            for symbol in common
        )

    candidate_outcomes: list[dict[str, Any]] = []
    for candidate in pool.candidates:
        returns: dict[str, float | None] = {}
        excess_returns: dict[str, float | None] = {}
        for horizon, target_session in target_sessions.items():
            value = _return_for_symbol(
                prices,
                candidate.provider_symbol,
                base_session,
                target_session,
            )
            benchmark = benchmark_returns[horizon]
            returns[str(horizon)] = value
            excess_returns[str(horizon)] = (
                value - benchmark
                if value is not None and benchmark is not None
                else None
            )
        candidate_outcomes.append(
            {
                "symbol": candidate.provider_symbol,
                "rank": candidate.rank,
                "selection_score": candidate.score,
                "returns": returns,
                "excess_returns": excess_returns,
            }
        )

    summary: dict[str, Any] = {
        "return_basis": price_column,
        "benchmark": "equal_weight_available_market",
        "horizons": {},
    }
    for horizon, target_session in target_sessions.items():
        values = [
            _finite_float(row["returns"].get(str(horizon)))
            for row in candidate_outcomes
        ]
        finite = [value for value in values if value is not None]
        excess = [
            _finite_float(row["excess_returns"].get(str(horizon)))
            for row in candidate_outcomes
        ]
        summary["horizons"][str(horizon)] = {
            "target_session": target_session.date().isoformat(),
            "observed_candidates": len(finite),
            "mean_return": _mean(finite),
            "positive_rate": (
                float(sum(value > 0 for value in finite) / len(finite))
                if finite
                else None
            ),
            "mean_excess_return": _mean(excess),
            "benchmark_return": benchmark_returns[horizon],
        }

    review_diagnostics = pool.diagnostics.get("candidate_review")
    assessment_rows = (
        review_diagnostics.get("assessments", [])
        if isinstance(review_diagnostics, dict)
        else []
    )
    review_comparison: dict[str, Any] = {
        "available": False,
        "horizons": {},
    }
    if isinstance(assessment_rows, list) and assessment_rows:
        comparison_rows: list[dict[str, Any]] = []
        for raw in assessment_rows:
            if not isinstance(raw, dict):
                continue
            symbol = str(raw.get("symbol") or "").strip().upper()
            quant_rank = raw.get("quant_rank")
            if not symbol or not isinstance(quant_rank, int):
                continue
            returns = {
                str(horizon): _return_for_symbol(
                    prices,
                    symbol,
                    base_session,
                    target_session,
                )
                for horizon, target_session in target_sessions.items()
            }
            comparison_rows.append(
                {
                    "symbol": symbol,
                    "quant_rank": quant_rank,
                    "quant_quality": _finite_float(raw.get("quant_quality")),
                    "review_score": _finite_float(raw.get("review_score")),
                    "composite_score": _finite_float(raw.get("composite_score")),
                    "selected": raw.get("selected") is True,
                    "returns": returns,
                }
            )
        final_count = len(pool.candidates)
        quant_baseline = sorted(
            comparison_rows,
            key=lambda row: (row["quant_rank"], row["symbol"]),
        )[:final_count]
        reviewed = [row for row in comparison_rows if row["selected"]]
        review_comparison["available"] = bool(quant_baseline and reviewed)
        review_comparison["candidate_count"] = len(comparison_rows)
        for horizon in available_horizons:
            quant_mean = _mean(
                _finite_float(row["returns"].get(str(horizon)))
                for row in quant_baseline
            )
            reviewed_mean = _mean(
                _finite_float(row["returns"].get(str(horizon)))
                for row in reviewed
            )
            review_comparison["horizons"][str(horizon)] = {
                "quant_baseline_mean_return": quant_mean,
                "reviewed_mean_return": reviewed_mean,
                "review_uplift": (
                    reviewed_mean - quant_mean
                    if reviewed_mean is not None and quant_mean is not None
                    else None
                ),
                "quant_rank_ic": _rank_ic(
                    comparison_rows, "quant_quality", horizon
                ),
                "review_rank_ic": _rank_ic(
                    comparison_rows, "review_score", horizon
                ),
                "composite_rank_ic": _rank_ic(
                    comparison_rows, "composite_score", horizon
                ),
            }

    status = (
        "pending"
        if not available_horizons
        else (
            "complete"
            if available_horizons == configured_horizons
            else "partial"
        )
    )
    provisional = CandidatePoolEvaluation(
        evaluation_id="pending",
        pool_id=pool.pool_id,
        pool_digest=pool.content_digest,
        market=pool.market,
        data_session=pool.data_session,
        evaluated_at=evaluated_at,
        data_cutoff=data_cutoff.isoformat(),
        data_source=data_source,
        configured_horizons=configured_horizons,
        available_horizons=available_horizons,
        status=status,
        candidate_outcomes=tuple(candidate_outcomes),
        summary=summary,
        review_comparison=review_comparison,
    )
    return replace(
        provisional,
        evaluation_id=(
            f"eval_{pool.pool_id}_{data_cutoff.isoformat()}_"
            f"{provisional.content_digest[:12]}"
        ),
    )
