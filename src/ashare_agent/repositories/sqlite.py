from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ashare_agent.core.json_safety import sanitize_json_value


TERMINAL_STATUSES = {"completed", "degraded", "failed"}


class IdempotencyConflictError(ValueError):
    """The same idempotency key was used for a different logical request."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(
        sanitize_json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


class SQLiteRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._write_lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS portfolios (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    cash TEXT NOT NULL,
                    positions_json TEXT NOT NULL,
                    market_id TEXT NOT NULL DEFAULT 'CN',
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS decision_runs (
                    id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    portfolio_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    market_id TEXT NOT NULL DEFAULT 'CN',
                    universe_source TEXT NOT NULL DEFAULT 'static',
                    universe_version TEXT NOT NULL,
                    universe_json TEXT NOT NULL,
                    candidate_pool_id TEXT,
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    idempotency_key TEXT UNIQUE,
                    request_fingerprint TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY(portfolio_id) REFERENCES portfolios(id)
                );
                CREATE INDEX IF NOT EXISTS idx_decision_runs_portfolio_created
                    ON decision_runs(portfolio_id, created_at DESC);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(decision_runs)"
                ).fetchall()
            }
            if "request_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE decision_runs ADD COLUMN request_fingerprint TEXT"
                )
            if "market_id" not in columns:
                connection.execute(
                    "ALTER TABLE decision_runs "
                    "ADD COLUMN market_id TEXT NOT NULL DEFAULT 'CN'"
                )
            if "universe_source" not in columns:
                connection.execute(
                    "ALTER TABLE decision_runs "
                    "ADD COLUMN universe_source TEXT NOT NULL DEFAULT 'static'"
                )
            if "candidate_pool_id" not in columns:
                connection.execute(
                    "ALTER TABLE decision_runs ADD COLUMN candidate_pool_id TEXT"
                )
            portfolio_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(portfolios)"
                ).fetchall()
            }
            if "market_id" not in portfolio_columns:
                connection.execute(
                    "ALTER TABLE portfolios "
                    "ADD COLUMN market_id TEXT NOT NULL DEFAULT 'CN'"
                )
            # In-process tasks are not durable. Make interrupted runs explicit
            # instead of leaving them in a permanently running state.
            completed_at = _now()
            connection.execute(
                """
                UPDATE decision_runs
                   SET status = 'failed',
                       error_code = 'PROCESS_RESTARTED',
                       error_message = 'The in-process task was interrupted by a service restart',
                       completed_at = ?
                 WHERE status NOT IN ('completed', 'degraded', 'failed')
                """,
                (completed_at,),
            )

    def ping(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except Exception:
            return False

    @staticmethod
    def _portfolio_row(row: sqlite3.Row) -> dict[str, Any]:
        columns = set(row.keys())
        return {
            "id": row["id"],
            "name": row["name"],
            "cash": row["cash"],
            "positions": json.loads(row["positions_json"]),
            "market_id": (
                row["market_id"] if "market_id" in columns else "CN"
            ),
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict[str, Any]:
        columns = set(row.keys())
        return {
            "id": row["id"],
            "portfolio_id": row["portfolio_id"],
            "portfolio_version": row["portfolio_version"],
            "status": row["status"],
            "mode": row["mode"],
            "as_of": row["as_of"],
            "market_id": row["market_id"] if "market_id" in columns else "CN",
            "universe_source": (
                row["universe_source"]
                if "universe_source" in columns
                else "static"
            ),
            "universe_version": row["universe_version"],
            "universe": json.loads(row["universe_json"]),
            "candidate_pool_id": (
                row["candidate_pool_id"]
                if "candidate_pool_id" in columns
                else None
            ),
            "input": json.loads(row["input_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }

    def create_portfolio(
        self,
        payload: dict[str, Any],
        *,
        market_id: str = "CN",
    ) -> dict[str, Any]:
        portfolio_id = f"pf_{uuid.uuid4().hex}"
        created_at = _now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO portfolios
                    (id, name, cash, positions_json, market_id,
                     version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    portfolio_id,
                    payload["name"],
                    str(payload["cash"]),
                    _json(payload.get("positions", [])),
                    market_id.strip().upper(),
                    created_at,
                    created_at,
                ),
            )
        return self.get_portfolio(portfolio_id)  # type: ignore[return-value]

    def get_portfolio(self, portfolio_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)
            ).fetchone()
        return self._portfolio_row(row) if row else None

    def update_portfolio(
        self,
        portfolio_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        updated_at = _now()
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE portfolios
                   SET name = ?, cash = ?, positions_json = ?,
                       version = version + 1, updated_at = ?
                 WHERE id = ?
                """,
                (
                    payload["name"],
                    str(payload["cash"]),
                    _json(payload.get("positions", [])),
                    updated_at,
                    portfolio_id,
                ),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_portfolio(portfolio_id)

    def create_decision_run(
        self,
        *,
        portfolio: dict[str, Any],
        mode: str,
        as_of: str,
        universe_version: str,
        universe: list[str],
        idempotency_key: str | None,
        request_fingerprint: str,
        market_id: str = "CN",
        universe_source: str = "static",
        candidate_pool_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        run_id = f"run_{uuid.uuid4().hex}"
        created_at = _now()
        with self._write_lock, self._connect() as connection:
            if idempotency_key:
                existing = connection.execute(
                    "SELECT * FROM decision_runs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    if existing["request_fingerprint"] != request_fingerprint:
                        raise IdempotencyConflictError(
                            "Idempotency-Key has already been used for a different request"
                        )
                    return self._run_row(existing), False
            connection.execute(
                """
                INSERT INTO decision_runs (
                    id, portfolio_id, portfolio_version, status, mode, as_of,
                    market_id, universe_source, universe_version, universe_json,
                    candidate_pool_id, input_json,
                    idempotency_key, request_fingerprint, created_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    portfolio["id"],
                    portfolio["version"],
                    mode,
                    as_of,
                    market_id,
                    universe_source,
                    universe_version,
                    _json(universe),
                    candidate_pool_id,
                    _json(portfolio),
                    idempotency_key,
                    request_fingerprint,
                    created_at,
                ),
            )
        return self.get_decision_run(run_id), True  # type: ignore[return-value]

    def resolve_decision_run_universe(
        self,
        run_id: str,
        *,
        universe_version: str,
        universe: list[str],
        candidate_pool_id: str | None,
    ) -> None:
        if not universe:
            raise ValueError("Resolved decision universe cannot be empty")
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE decision_runs
                   SET universe_version = ?,
                       universe_json = ?,
                       candidate_pool_id = ?
                 WHERE id = ?
                   AND status NOT IN ('completed', 'degraded', 'failed')
                """,
                (
                    universe_version,
                    _json(universe),
                    candidate_pool_id,
                    run_id,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    f"Decision run {run_id!r} cannot accept a resolved universe"
                )

    def get_decision_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM decision_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._run_row(row) if row else None

    def list_decision_runs(
        self,
        *,
        portfolio_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100))
        with self._connect() as connection:
            if portfolio_id is None:
                rows = connection.execute(
                    """
                    SELECT *
                      FROM decision_runs
                     ORDER BY created_at DESC
                     LIMIT ?
                    """,
                    (bounded_limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                      FROM decision_runs
                     WHERE portfolio_id = ?
                     ORDER BY created_at DESC
                     LIMIT ?
                    """,
                    (portfolio_id, bounded_limit),
                ).fetchall()
        return [self._run_row(row) for row in rows]

    def update_run_status(self, run_id: str, status: str) -> None:
        with self._write_lock, self._connect() as connection:
            if status in {"selecting_universe", "fetching_data"}:
                connection.execute(
                    """
                    UPDATE decision_runs
                       SET status = ?, started_at = COALESCE(started_at, ?)
                     WHERE id = ?
                       AND status NOT IN ('completed', 'degraded', 'failed')
                    """,
                    (status, _now(), run_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE decision_runs
                       SET status = ?
                     WHERE id = ?
                       AND status NOT IN ('completed', 'degraded', 'failed')
                    """,
                    (status, run_id),
                )

    def complete_decision_run(
        self,
        run_id: str,
        result: dict[str, Any],
        degraded: bool,
    ) -> None:
        status = "degraded" if degraded else "completed"
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE decision_runs
                   SET status = ?, result_json = ?, completed_at = ?,
                       error_code = NULL, error_message = NULL
                 WHERE id = ?
                   AND status NOT IN ('completed', 'degraded', 'failed')
                """,
                (status, _json(result), _now(), run_id),
            )

    def fail_decision_run(self, run_id: str, code: str, message: str) -> None:
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE decision_runs
                   SET status = 'failed', error_code = ?, error_message = ?, completed_at = ?
                 WHERE id = ?
                   AND status NOT IN ('completed', 'degraded', 'failed')
                """,
                (code, message[:2000], _now(), run_id),
            )
