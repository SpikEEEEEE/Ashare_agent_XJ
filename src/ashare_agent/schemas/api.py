from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ashare_agent.domain.instruments import get_market_profile


def _normalize_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._:-]{1,31}", symbol):
        raise ValueError("symbol must be a non-empty provider or canonical symbol")
    return symbol


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PositionPayload(StrictModel):
    symbol: str
    shares: int = Field(gt=0, le=2_000_000_000)
    available_shares: int | None = Field(
        default=None, ge=0, le=2_000_000_000
    )
    average_cost: Decimal = Field(ge=0, le=Decimal("1000000000"))
    holding_days: int | None = Field(default=None, ge=0, le=36500)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return _normalize_symbol(value)

    @model_validator(mode="after")
    def validate_available_shares(self) -> "PositionPayload":
        if self.available_shares is None:
            self.available_shares = self.shares
        if self.available_shares > self.shares:
            raise ValueError("available_shares cannot exceed shares")
        return self


class PortfolioWriteRequest(StrictModel):
    name: str = Field(default="Current portfolio", min_length=1, max_length=100)
    cash: Decimal = Field(ge=0, le=Decimal("1000000000000000"))
    positions: list[PositionPayload] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def reject_duplicate_symbols(self) -> "PortfolioWriteRequest":
        symbols = [position.symbol for position in self.positions]
        if len(symbols) != len(set(symbols)):
            raise ValueError("positions contain duplicate symbols")
        return self


class PortfolioResponse(StrictModel):
    id: str
    name: str
    cash: Decimal
    positions: list[PositionPayload]
    market_id: str
    version: int
    created_at: datetime
    updated_at: datetime


class DecisionRunCreateRequest(StrictModel):
    portfolio_id: str = Field(min_length=1, max_length=64)
    mode: Literal["holdings_only", "rebalance"] = "rebalance"
    as_of: datetime | None = None
    market: str = "CN"
    universe_source: Literal[
        "static",
        "selected",
        "fresh_selection",
    ] = "static"
    universe: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Optional point-in-time universe for this run. When omitted, the "
            "server-configured universe is used."
        ),
    )

    @model_validator(mode="after")
    def normalize_market_and_universe(self) -> "DecisionRunCreateRequest":
        market = self.market.strip().upper()
        profile = get_market_profile(market)
        self.market = market
        if self.universe is None:
            return self
        if self.universe_source != "static":
            raise ValueError(
                "universe and a dynamic universe_source cannot be used together"
            )
        symbols = [
            profile.normalize_provider_symbol(symbol)
            for symbol in self.universe
        ]
        if len(symbols) != len(set(symbols)):
            raise ValueError("universe contains duplicate symbols")
        self.universe = symbols
        return self


class DecisionRunResponse(StrictModel):
    id: str
    portfolio_id: str
    portfolio_version: int
    status: Literal[
        "pending",
        "selecting_universe",
        "fetching_data",
        "building_features",
        "calling_llm",
        "validating",
        "completed",
        "degraded",
        "failed",
    ]
    mode: Literal["holdings_only", "rebalance"]
    as_of: datetime
    market_id: str = "CN"
    universe_source: str = "static"
    universe_version: str
    universe: list[str]
    candidate_pool_id: str | None = None
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class UniverseResponse(StrictModel):
    version: str
    symbols: list[str]
    source: str = "static"
    market: str = "CN"
    candidate_pool_id: str | None = None
    data_session: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class CandidatePoolResponse(StrictModel):
    schema_version: str
    pool_id: str
    market: str
    created_at: datetime
    as_of: datetime
    data_session: str
    valid_for_session: str | None
    strategy_id: str
    strategy_version: str
    config_hash: str
    data_source: str
    data_provenance: dict[str, Any]
    parent_pool_id: str | None
    content_digest: str
    candidates: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class CandidatePoolEvaluationResponse(StrictModel):
    schema_version: str
    evaluation_id: str
    pool_id: str
    pool_digest: str
    market: str
    data_session: str
    evaluated_at: datetime
    data_cutoff: str
    data_source: str
    configured_horizons: list[int]
    available_horizons: list[int]
    status: Literal["pending", "partial", "complete"]
    candidate_outcomes: list[dict[str, Any]]
    summary: dict[str, Any]
    review_comparison: dict[str, Any]
    content_digest: str


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    service: str
