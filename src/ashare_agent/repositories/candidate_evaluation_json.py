from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None  # type: ignore[assignment]

from ashare_agent.domain.candidate_evaluation import CandidatePoolEvaluation


class JsonCandidateEvaluationRepository:
    """Immutable evaluation snapshots with one atomic latest pointer per pool."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._thread_lock = threading.RLock()

    @staticmethod
    def _safe(value: str) -> str:
        normalized = value.strip()
        if not normalized or any(
            char not in "-_." and not char.isalnum() for char in normalized
        ):
            raise ValueError(f"Unsafe evaluation repository key: {value!r}")
        return normalized

    def _path(self, pool_id: str, evaluation_id: str) -> Path:
        return (
            self.root
            / self._safe(pool_id)
            / f"{self._safe(evaluation_id)}.json"
        )

    def _latest_path(self, pool_id: str) -> Path:
        return self.root / self._safe(pool_id) / "latest.json"

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
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
    def _guard(self) -> Iterator[None]:
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

    def save(self, evaluation: CandidatePoolEvaluation) -> CandidatePoolEvaluation:
        with self._guard():
            path = self._path(evaluation.pool_id, evaluation.evaluation_id)
            payload = evaluation.to_dict()
            if path.exists():
                stored = self.get(evaluation.pool_id, evaluation.evaluation_id)
                if stored is None or stored.content_digest != evaluation.content_digest:
                    raise ValueError("Candidate evaluation id collision")
            else:
                self._atomic_write(path, payload)
                stored = evaluation

            latest = self.latest(evaluation.pool_id)
            if latest is None or (
                stored.data_cutoff,
                stored.evaluated_at,
                stored.evaluation_id,
            ) > (
                latest.data_cutoff,
                latest.evaluated_at,
                latest.evaluation_id,
            ):
                self._atomic_write(
                    self._latest_path(evaluation.pool_id),
                    {
                        "pool_id": stored.pool_id,
                        "evaluation_id": stored.evaluation_id,
                        "content_digest": stored.content_digest,
                    },
                )
            return stored

    def get(
        self,
        pool_id: str,
        evaluation_id: str,
    ) -> CandidatePoolEvaluation | None:
        path = self._path(pool_id, evaluation_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        evaluation = CandidatePoolEvaluation.from_dict(payload)
        if evaluation.pool_id != pool_id or evaluation.evaluation_id != evaluation_id:
            raise ValueError("Candidate evaluation file key does not match payload")
        if payload.get("content_digest") != evaluation.content_digest:
            raise ValueError("Candidate evaluation failed integrity check")
        return evaluation

    def latest(self, pool_id: str) -> CandidatePoolEvaluation | None:
        path = self._latest_path(pool_id)
        if not path.exists():
            return None
        pointer = json.loads(path.read_text(encoding="utf-8"))
        if pointer.get("pool_id") != pool_id:
            raise ValueError("Candidate evaluation latest pointer has wrong pool")
        evaluation = self.get(pool_id, str(pointer["evaluation_id"]))
        if evaluation is None:
            raise ValueError("Candidate evaluation pointer references missing data")
        if pointer.get("content_digest") != evaluation.content_digest:
            raise ValueError("Candidate evaluation pointer failed integrity check")
        return evaluation
