from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from ashare_agent.domain.candidate_pool import CandidatePool


class CandidatePoolSelector(Protocol):
    name: str
    market: str

    def select(
        self,
        as_of: datetime,
        *,
        data_cutoff: date,
        board_scope: str | None = None,
        force_refresh: bool = False,
    ) -> CandidatePool:
        """Build, persist, and return a point-in-time candidate pool."""

    def latest(self) -> CandidatePool | None:
        """Return the latest persisted pool without remote access."""


class CandidatePoolRepository(Protocol):
    def save(self, pool: CandidatePool) -> CandidatePool:
        """Persist an immutable pool and return the canonical stored object."""

    def get(self, pool_id: str) -> CandidatePool | None:
        """Load a candidate pool by id."""

    def latest(self, market: str) -> CandidatePool | None:
        """Load the most recently persisted pool for a market."""

    def previous(
        self,
        market: str,
        *,
        before_session: str,
        strategy_id: str,
        config_hash: str,
    ) -> CandidatePool | None:
        """Return the newest compatible pool strictly before one session."""

    def recent(
        self,
        market: str,
        *,
        before_or_equal_session: str,
        limit: int,
    ) -> tuple[CandidatePool, ...]:
        """Return recent pools at or before a completed market session."""


class UniverseSelectionError(RuntimeError):
    pass


class UniverseSelectionConfigurationError(UniverseSelectionError):
    pass
