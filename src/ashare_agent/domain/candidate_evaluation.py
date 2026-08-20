from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ashare_agent.core.json_safety import sanitize_json_value


CANDIDATE_EVALUATION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CandidatePoolEvaluation:
    evaluation_id: str
    pool_id: str
    pool_digest: str
    market: str
    data_session: str
    evaluated_at: datetime
    data_cutoff: str
    data_source: str
    configured_horizons: tuple[int, ...]
    available_horizons: tuple[int, ...]
    status: str
    candidate_outcomes: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    review_comparison: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CANDIDATE_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.evaluation_id.strip() or not self.pool_id.strip():
            raise ValueError("Evaluation and pool ids cannot be empty")
        if not self.pool_digest.strip():
            raise ValueError("Candidate pool digest cannot be empty")
        date.fromisoformat(self.data_session)
        date.fromisoformat(self.data_cutoff)
        if self.evaluated_at.tzinfo is None:
            raise ValueError("Candidate evaluation timestamp must be timezone-aware")
        if self.status not in {"pending", "partial", "complete"}:
            raise ValueError("Unknown candidate evaluation status")
        if any(horizon < 1 for horizon in self.configured_horizons):
            raise ValueError("Evaluation horizons must be positive")
        if not set(self.available_horizons).issubset(self.configured_horizons):
            raise ValueError("Available horizons must be configured horizons")

    @property
    def content_digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "pool_id": self.pool_id,
            "pool_digest": self.pool_digest,
            "market": self.market,
            "data_session": self.data_session,
            "evaluated_at": self.evaluated_at.isoformat(),
            "data_cutoff": self.data_cutoff,
            "data_source": self.data_source,
            "configured_horizons": list(self.configured_horizons),
            "available_horizons": list(self.available_horizons),
            "status": self.status,
            "candidate_outcomes": sanitize_json_value(self.candidate_outcomes),
            "summary": sanitize_json_value(self.summary),
            "review_comparison": sanitize_json_value(self.review_comparison),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "pool_id": self.pool_id,
            "pool_digest": self.pool_digest,
            "market": self.market,
            "data_session": self.data_session,
            "evaluated_at": self.evaluated_at.isoformat(),
            "data_cutoff": self.data_cutoff,
            "data_source": self.data_source,
            "configured_horizons": list(self.configured_horizons),
            "available_horizons": list(self.available_horizons),
            "status": self.status,
            "candidate_outcomes": sanitize_json_value(self.candidate_outcomes),
            "summary": sanitize_json_value(self.summary),
            "review_comparison": sanitize_json_value(self.review_comparison),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidatePoolEvaluation":
        return cls(
            schema_version=str(
                payload.get("schema_version")
                or CANDIDATE_EVALUATION_SCHEMA_VERSION
            ),
            evaluation_id=str(payload["evaluation_id"]),
            pool_id=str(payload["pool_id"]),
            pool_digest=str(payload["pool_digest"]),
            market=str(payload["market"]),
            data_session=str(payload["data_session"]),
            evaluated_at=datetime.fromisoformat(str(payload["evaluated_at"])),
            data_cutoff=str(payload["data_cutoff"]),
            data_source=str(payload["data_source"]),
            configured_horizons=tuple(
                int(item) for item in payload.get("configured_horizons", [])
            ),
            available_horizons=tuple(
                int(item) for item in payload.get("available_horizons", [])
            ),
            status=str(payload["status"]),
            candidate_outcomes=tuple(
                dict(item) for item in payload.get("candidate_outcomes", [])
            ),
            summary=dict(payload.get("summary") or {}),
            review_comparison=dict(payload.get("review_comparison") or {}),
        )
