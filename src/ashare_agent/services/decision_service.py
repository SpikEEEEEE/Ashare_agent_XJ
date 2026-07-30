from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from ashare_agent.domain.candidate_pool import CandidatePool
from ashare_agent.domain.instruments import get_market_profile
from ashare_agent.domain.models import (
    DecisionInput,
    PortfolioSnapshot,
    Position,
    RawDecisionBundle,
)
from ashare_agent.core.json_safety import sanitize_json_value
from ashare_agent.ports.decision_engine import DecisionEngine
from ashare_agent.ports.market_data import MarketDataProvider
from ashare_agent.ports.risk_policy import RiskPolicy
from ashare_agent.ports.universe import CandidatePoolSelector
from ashare_agent.repositories.sqlite import SQLiteRepository


logger = logging.getLogger(__name__)
SAFE_EXCEPTION_TYPES = {
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "TimeoutError",
    "ConnectionError",
    "DataUnavailableError",
    "StaleDataError",
    "ProviderConfigurationError",
    "PortfolioAgentGraphError",
    "PortfolioAgentOutputError",
    "DecisionOutputError",
}


def _safe_exception_type(exc: Exception) -> str:
    name = type(exc).__name__
    return name if name in SAFE_EXCEPTION_TYPES else "Exception"


class DecisionService:
    def __init__(
        self,
        repository: SQLiteRepository,
        market_data: MarketDataProvider,
        decision_engine: DecisionEngine,
        risk_policy: RiskPolicy,
        universe_selector: CandidatePoolSelector | None = None,
    ) -> None:
        self.repository = repository
        self.market_data = market_data
        self.decision_engine = decision_engine
        self.risk_policy = risk_policy
        self.universe_selector = universe_selector

    @staticmethod
    def _portfolio(payload: dict[str, Any]) -> PortfolioSnapshot:
        positions = tuple(
            Position(
                symbol=str(item["symbol"]),
                shares=int(item["shares"]),
                available_shares=int(item.get("available_shares", item["shares"])),
                average_cost=Decimal(str(item["average_cost"])),
                holding_days=(
                    int(item["holding_days"])
                    if item.get("holding_days") is not None
                    else None
                ),
            )
            for item in payload.get("positions", [])
        )
        return PortfolioSnapshot(
            portfolio_id=payload["id"],
            version=int(payload["version"]),
            name=str(payload["name"]),
            cash=Decimal(str(payload["cash"])),
            positions=positions,
        )

    @staticmethod
    def _as_of(value: str, market_id: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        timezone = ZoneInfo(get_market_profile(market_id).timezone)
        return (
            parsed.replace(tzinfo=timezone)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone)
        )

    def _resolve_candidate_pool(
        self,
        run: dict[str, Any],
        as_of: datetime,
        data_date: date,
    ) -> CandidatePool | None:
        source = str(run.get("universe_source") or "static")
        if source not in {"selected", "fresh_selection"}:
            return None
        if self.universe_selector is None:
            raise RuntimeError("No candidate-pool selector is configured")
        if self.universe_selector.market != run["market_id"]:
            raise ValueError(
                "Candidate-pool selector market does not match the decision run"
            )
        pool = (
            None
            if source == "fresh_selection"
            else self.universe_selector.latest()
        )
        if pool is None or pool.data_session != data_date.isoformat():
            pool = self.universe_selector.select(
                as_of,
                data_cutoff=data_date,
                # "fresh" means recompute the model output. The data adapter
                # still performs its normal incremental tail refresh; a full
                # historical redownload is reserved for the explicit refresh API.
                force_refresh=False,
            )
        if pool.data_session != data_date.isoformat():
            raise ValueError(
                "Candidate pool data session does not match the decision cutoff"
            )
        self.repository.resolve_decision_run_universe(
            run["id"],
            universe_version=pool.content_digest[:16],
            universe=list(pool.symbols),
            candidate_pool_id=pool.pool_id,
        )
        return pool

    def run(self, run_id: str) -> None:
        run = self.repository.get_decision_run(run_id)
        if not run:
            return
        try:
            portfolio = self._portfolio(run["input"])
            market_id = str(run.get("market_id") or "CN").upper()
            if getattr(self.market_data, "market", market_id) != market_id:
                raise ValueError(
                    "Market data provider does not support the decision market"
                )
            if getattr(self.risk_policy, "market", market_id) != market_id:
                raise ValueError(
                    "Risk policy does not support the decision market"
                )
            as_of = self._as_of(run["as_of"], market_id)
            held_symbols = [position.symbol for position in portfolio.positions]
            if run["mode"] == "holdings_only" and not held_symbols:
                raise ValueError(
                    "holdings_only mode requires at least one position"
                )
            data_date = self.market_data.latest_completed_session(as_of)
            valid_for_session = self.market_data.next_session(data_date)
            candidate_pool: CandidatePool | None = None
            if (
                run["mode"] != "holdings_only"
                and run.get("universe_source")
                in {"selected", "fresh_selection"}
            ):
                self.repository.update_run_status(run_id, "selecting_universe")
                candidate_pool = self._resolve_candidate_pool(
                    run,
                    as_of,
                    data_date,
                )
                run = self.repository.get_decision_run(run_id) or run
            if run["mode"] == "holdings_only":
                symbols = list(dict.fromkeys(held_symbols))
            else:
                symbols = list(dict.fromkeys([*run["universe"], *held_symbols]))

            self.repository.update_run_status(run_id, "fetching_data")
            market = {}
            unavailable: dict[str, str] = {}
            data_quality_warnings: list[str] = []
            for symbol in symbols:
                try:
                    snapshot = self.market_data.load_symbol(
                        symbol,
                        data_date,
                        as_of,
                    )
                    market[symbol] = snapshot
                    data_quality_warnings.extend(
                        f"{symbol}: {warning}"
                        for warning in snapshot.data_quality_warnings
                    )
                except Exception as exc:
                    failure_type = _safe_exception_type(exc)
                    unavailable[symbol] = (
                        f"{failure_type}: market data unavailable"
                    )
                    logger.warning(
                        "Market data unavailable for %s (%s)",
                        symbol,
                        failure_type,
                    )

            decision_input = DecisionInput(
                run_id=run_id,
                portfolio=portfolio,
                mode=run["mode"],
                as_of=as_of,
                data_date=data_date,
                valid_for_session=valid_for_session,
                universe_version=run["universe_version"],
                symbols=tuple(symbols),
                market=market,
                unavailable_symbols=unavailable,
                market_id=market_id,
                data_provider=self.market_data.name,
                universe_source=str(run.get("universe_source") or "static"),
                candidate_pool_id=(
                    candidate_pool.pool_id if candidate_pool else None
                ),
                candidate_metadata=(
                    {
                        candidate.provider_symbol: candidate.to_dict()
                        for candidate in candidate_pool.candidates
                    }
                    if candidate_pool
                    else {}
                ),
            )

            try:
                bundle = self.decision_engine.decide(
                    decision_input,
                    on_stage=lambda stage: self.repository.update_run_status(
                        run_id, stage
                    ),
                )
            except Exception as exc:
                failure_type = _safe_exception_type(exc)
                logger.error(
                    "LLM decision failed for %s (%s)",
                    run_id,
                    failure_type,
                )
                bundle = RawDecisionBundle(
                    decisions={},
                    meta={
                        "calls": 0,
                        "provider_attempts": 0,
                        "engine": "failed_safe_hold",
                        "decision_quality": "failed",
                        "analysis_coverage": 0.0,
                        "stage_health": {
                            "decision_engine": {
                                "status": "failed",
                                "failure_category": failure_type,
                            }
                        },
                    },
                    warnings=(
                        "LLM decision failed; safe-hold fallback applied "
                        f"({failure_type})",
                    ),
                )

            safe_decisions = sanitize_json_value(bundle.decisions)
            safe_meta = sanitize_json_value(bundle.meta)
            bundle = RawDecisionBundle(
                decisions=safe_decisions if isinstance(safe_decisions, dict) else {},
                meta=safe_meta if isinstance(safe_meta, dict) else {},
                warnings=bundle.warnings,
            )

            if data_quality_warnings:
                bundle = RawDecisionBundle(
                    decisions=bundle.decisions,
                    meta=bundle.meta,
                    warnings=tuple([*bundle.warnings, *data_quality_warnings]),
                )

            self.repository.update_run_status(run_id, "validating")
            result = self.risk_policy.apply(decision_input, bundle)
            market_snapshot: dict[str, dict[str, Any]] = {}
            for symbol, snapshot in market.items():
                try:
                    close_series = snapshot.bars.get("close")
                    recent_closes = (
                        [
                            float(value)
                            for value in close_series.tail(7).tolist()
                        ]
                        if close_series is not None
                        else []
                    )
                except Exception:
                    recent_closes = []
                market_snapshot[symbol] = {
                    "market": snapshot.market_id,
                    "currency": snapshot.currency,
                    "provider": snapshot.provider,
                    "data_date": snapshot.data_date.isoformat(),
                    "reference_price": float(snapshot.reference_price),
                    "retrieved_at": snapshot.retrieved_at.isoformat()
                    if snapshot.retrieved_at
                    else None,
                    "recent_closes": recent_closes,
                    "news": [
                        {
                            "id": item.get("id"),
                            "title": item.get("title"),
                            "published_utc": item.get("published_utc"),
                            "api_source": item.get("api_source"),
                            "description": item.get("description"),
                        }
                        for item in snapshot.news
                    ],
                    "fundamentals": snapshot.fundamentals,
                    "data_quality_warnings": list(snapshot.data_quality_warnings),
                }
            result["market_snapshot"] = market_snapshot
            result["universe_provenance"] = (
                candidate_pool.to_dict()
                if candidate_pool is not None
                else {
                    "source": str(run.get("universe_source") or "static"),
                    "market": market_id,
                    "version": run["universe_version"],
                    "symbols": run["universe"],
                }
            )
            degraded = bool(
                unavailable
                or result.get("warnings")
                or result.get("decision_quality") != "healthy"
            )
            if market and int(bundle.meta.get("calls", 0) or 0) == 0:
                degraded = True
                result.setdefault("warnings", []).append(
                    "No successful decision-agent call was recorded"
                )
            self.repository.complete_decision_run(run_id, result, degraded=degraded)
        except Exception as exc:
            failure_type = _safe_exception_type(exc)
            logger.error(
                "Decision run %s failed (%s)",
                run_id,
                failure_type,
            )
            self.repository.fail_decision_run(
                run_id,
                code=failure_type.upper(),
                message=f"Decision run failed safely ({failure_type})",
            )
