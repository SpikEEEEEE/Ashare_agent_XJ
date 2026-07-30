from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from ashare_agent.core.json_safety import sanitize_json_value
from ashare_agent.domain.instruments import InstrumentId, get_market_profile


CANDIDATE_POOL_SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class Candidate:
    instrument_id: InstrumentId
    provider_symbol: str
    rank: int
    score: float
    reason: str
    name: str | None = None
    industry: str | None = None
    reference_price: float | None = None
    signals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("Candidate rank must be positive")
        if not math.isfinite(float(self.score)):
            raise ValueError("Candidate score must be finite")
        if self.reference_price is not None and (
            not math.isfinite(float(self.reference_price))
            or float(self.reference_price) <= 0
        ):
            raise ValueError("Candidate reference price must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id.canonical,
            "provider_symbol": self.provider_symbol,
            "rank": self.rank,
            "score": self.score,
            "reason": self.reason,
            "name": self.name,
            "industry": self.industry,
            "reference_price": self.reference_price,
            "signals": sanitize_json_value(self.signals),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Candidate":
        return cls(
            instrument_id=InstrumentId.parse(str(payload["instrument_id"])),
            provider_symbol=str(payload["provider_symbol"]),
            rank=int(payload["rank"]),
            score=float(payload["score"]),
            reason=str(payload["reason"]),
            name=str(payload["name"]) if payload.get("name") is not None else None,
            industry=(
                str(payload["industry"])
                if payload.get("industry") is not None
                else None
            ),
            reference_price=(
                float(payload["reference_price"])
                if payload.get("reference_price") is not None
                else None
            ),
            signals=dict(payload.get("signals") or {}),
        )


@dataclass(frozen=True)
class CandidatePool:
    pool_id: str
    market: str
    as_of: datetime
    data_session: str
    valid_for_session: str | None
    strategy_id: str
    strategy_version: str
    config_hash: str
    data_source: str
    data_provenance: dict[str, Any]
    candidates: tuple[Candidate, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    parent_pool_id: str | None = None
    schema_version: str = CANDIDATE_POOL_SCHEMA_VERSION
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        market = self.market.strip().upper()
        profile = get_market_profile(market)
        object.__setattr__(self, "market", market)
        if self.as_of.tzinfo is None or self.created_at.tzinfo is None:
            raise ValueError("Candidate pool timestamps must be timezone-aware")
        date.fromisoformat(self.data_session)
        if self.valid_for_session is not None:
            date.fromisoformat(self.valid_for_session)
        if not self.pool_id.strip():
            raise ValueError("Candidate pool id cannot be empty")
        if not self.candidates:
            raise ValueError("Candidate pool cannot be empty")
        canonical_ids = [
            candidate.instrument_id.canonical for candidate in self.candidates
        ]
        symbols = [candidate.provider_symbol for candidate in self.candidates]
        ranks = [candidate.rank for candidate in self.candidates]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError("Candidate pool contains duplicate instruments")
        if len(symbols) != len(set(symbols)):
            raise ValueError("Candidate pool contains duplicate provider symbols")
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("Candidate ranks must be contiguous and ordered")
        for candidate in self.candidates:
            expected_symbol = profile.provider_symbol(candidate.instrument_id)
            if candidate.provider_symbol != expected_symbol:
                raise ValueError(
                    "Candidate provider symbol does not match its instrument id"
                )

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(candidate.provider_symbol for candidate in self.candidates)

    @property
    def content_digest(self) -> str:
        """Hash every persisted business field except the caller-facing pool id."""
        if self.schema_version == "1.0":
            # Compatibility with pools written before temporal lineage and
            # diagnostics became part of the immutable payload.
            digest_payload = {
                "schema_version": self.schema_version,
                "market": self.market,
                "data_session": self.data_session,
                "strategy_id": self.strategy_id,
                "strategy_version": self.strategy_version,
                "config_hash": self.config_hash,
                "data_source": self.data_source,
                "data_provenance": sanitize_json_value(
                    self.data_provenance
                ),
                "candidates": [
                    candidate.to_dict() for candidate in self.candidates
                ],
            }
        else:
            digest_payload = {
                "schema_version": self.schema_version,
                "market": self.market,
                "created_at": self.created_at.isoformat(),
                "as_of": self.as_of.isoformat(),
                "data_session": self.data_session,
                "valid_for_session": self.valid_for_session,
                "strategy_id": self.strategy_id,
                "strategy_version": self.strategy_version,
                "config_hash": self.config_hash,
                "data_source": self.data_source,
                "data_provenance": sanitize_json_value(
                    self.data_provenance
                ),
                "parent_pool_id": self.parent_pool_id,
                "candidates": [
                    candidate.to_dict() for candidate in self.candidates
                ],
                "diagnostics": sanitize_json_value(self.diagnostics),
            }
        encoded = json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pool_id": self.pool_id,
            "market": self.market,
            "created_at": self.created_at.isoformat(),
            "as_of": self.as_of.isoformat(),
            "data_session": self.data_session,
            "valid_for_session": self.valid_for_session,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "data_source": self.data_source,
            "data_provenance": sanitize_json_value(self.data_provenance),
            "parent_pool_id": self.parent_pool_id,
            "content_digest": self.content_digest,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "diagnostics": sanitize_json_value(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidatePool":
        return cls(
            schema_version=str(
                payload.get("schema_version") or CANDIDATE_POOL_SCHEMA_VERSION
            ),
            pool_id=str(payload["pool_id"]),
            market=str(payload["market"]),
            created_at=datetime.fromisoformat(
                str(payload.get("created_at") or payload["as_of"])
            ),
            as_of=datetime.fromisoformat(str(payload["as_of"])),
            data_session=str(payload["data_session"]),
            valid_for_session=(
                str(payload["valid_for_session"])
                if payload.get("valid_for_session") is not None
                else None
            ),
            strategy_id=str(payload["strategy_id"]),
            strategy_version=str(payload["strategy_version"]),
            config_hash=str(payload["config_hash"]),
            data_source=str(payload["data_source"]),
            data_provenance=dict(payload.get("data_provenance") or {}),
            parent_pool_id=(
                str(payload["parent_pool_id"])
                if payload.get("parent_pool_id") is not None
                else None
            ),
            candidates=tuple(
                Candidate.from_dict(item)
                for item in payload.get("candidates", [])
            ),
            diagnostics=dict(payload.get("diagnostics") or {}),
        )
