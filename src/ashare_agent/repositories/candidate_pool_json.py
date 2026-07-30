from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None  # type: ignore[assignment]

from ashare_agent.domain.candidate_pool import CandidatePool


class JsonCandidatePoolRepository:
    """Filesystem repository with immutable pool files and atomic pointers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._thread_lock = threading.RLock()

    @staticmethod
    def _safe_component(value: str) -> str:
        normalized = value.strip()
        if not normalized or any(character not in "-_." and not character.isalnum() for character in normalized):
            raise ValueError(f"Unsafe repository key: {value!r}")
        return normalized

    def _pool_path(self, pool_id: str) -> Path:
        return self.root / "pools" / f"{self._safe_component(pool_id)}.json"

    def _latest_path(self, market: str) -> Path:
        return self.root / f"latest-{self._safe_component(market.upper())}.json"

    @staticmethod
    def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _write_guard(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".write.lock"
        with self._thread_lock, lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _recency_key(
        pool: CandidatePool,
    ) -> tuple[str, datetime, datetime, str]:
        return (
            pool.data_session,
            pool.as_of,
            pool.created_at,
            pool.pool_id,
        )

    def save(self, pool: CandidatePool) -> CandidatePool:
        with self._write_guard():
            pool_path = self._pool_path(pool.pool_id)
            payload = pool.to_dict()
            if pool_path.exists():
                stored = self.get(pool.pool_id)
                if stored is None:
                    raise ValueError(
                        f"Candidate pool {pool.pool_id!r} disappeared while saving"
                    )
                if stored.content_digest != payload["content_digest"]:
                    raise ValueError(
                        f"Candidate pool id collision for {pool.pool_id!r}"
                    )
            else:
                self._atomic_json_write(pool_path, payload)
                stored = pool

            latest = self.latest(pool.market)
            if latest is None or self._recency_key(stored) > self._recency_key(
                latest
            ):
                self._atomic_json_write(
                    self._latest_path(pool.market),
                    {
                        "schema_version": stored.schema_version,
                        "market": stored.market,
                        "pool_id": stored.pool_id,
                        "content_digest": stored.content_digest,
                    },
                )
            return stored

    def get(self, pool_id: str) -> CandidatePool | None:
        path = self._pool_path(pool_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        pool = CandidatePool.from_dict(payload)
        expected_digest = payload.get("content_digest")
        if pool.pool_id != pool_id.strip():
            raise ValueError(
                f"Candidate pool file key {pool_id!r} does not match its payload"
            )
        if (
            not isinstance(expected_digest, str)
            or not expected_digest
            or pool.content_digest != expected_digest
        ):
            raise ValueError(f"Candidate pool {pool_id!r} failed integrity check")
        return pool

    def latest(self, market: str) -> CandidatePool | None:
        pointer_path = self._latest_path(market)
        if not pointer_path.exists():
            return None
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if str(pointer.get("market", "")).upper() != market.strip().upper():
            raise ValueError("Candidate pool latest pointer has the wrong market")
        pool = self.get(str(pointer["pool_id"]))
        if pool is None:
            raise ValueError("Candidate pool latest pointer references a missing pool")
        if pointer.get("content_digest") != pool.content_digest:
            raise ValueError("Candidate pool latest pointer failed integrity check")
        return pool

    def previous(
        self,
        market: str,
        *,
        before_session: str,
        strategy_id: str,
        config_hash: str,
    ) -> CandidatePool | None:
        pools_dir = self.root / "pools"
        if not pools_dir.exists():
            return None
        matches: list[CandidatePool] = []
        for path in pools_dir.glob("*.json"):
            try:
                pool = self.get(path.stem)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            if pool is None or pool.pool_id != path.stem:
                continue
            if (
                pool.market == market.strip().upper()
                and pool.data_session < before_session
                and pool.strategy_id == strategy_id
                and pool.config_hash == config_hash
            ):
                matches.append(pool)
        return (
            max(matches, key=self._recency_key)
            if matches
            else None
        )
