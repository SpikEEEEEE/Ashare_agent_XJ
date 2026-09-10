from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ashare_agent.repositories.sqlite import SQLiteRepository
from .helpers import make_settings


def _run(repository: SQLiteRepository):
    portfolio = repository.create_portfolio(
        {"name": "test", "cash": "1000", "positions": []}
    )
    run, _ = repository.create_decision_run(
        portfolio=portfolio,
        mode="rebalance",
        as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")).isoformat(),
        universe_version="test",
        universe=["600519.SH"],
        idempotency_key=None,
        request_fingerprint="repository-terminal-test",
    )
    return run


def test_terminal_run_cannot_be_resurrected_by_an_orphan_worker(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    run = _run(repository)
    repository.fail_decision_run(run["id"], "PROCESS_RESTARTED", "restart")

    repository.update_run_status(run["id"], "calling_llm")
    repository.complete_decision_run(run["id"], {"unsafe": True}, degraded=False)

    persisted = repository.get_decision_run(run["id"])
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert persisted["error_code"] == "PROCESS_RESTARTED"
    assert persisted["result"] is None


def test_timeout_cannot_overwrite_a_completed_run(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    run = _run(repository)
    repository.complete_decision_run(run["id"], {"ok": True}, degraded=False)

    repository.fail_decision_run(run["id"], "TASK_TIMEOUT", "late watchdog")

    persisted = repository.get_decision_run(run["id"])
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert persisted["result"] == {"ok": True}


def test_rollforward_does_not_overwrite_a_newer_portfolio_version(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    run = _run(repository)
    updated = repository.update_portfolio(
        run["portfolio_id"],
        {"name": "manual edit", "cash": "900", "positions": []},
    )
    assert updated is not None
    assert updated["version"] == 2

    repository.complete_decision_run(
        run["id"],
        {"ok": True},
        degraded=False,
        portfolio_update={
            "name": "stale prediction",
            "cash": "800",
            "positions": [],
        },
        rollforward_audit={"cash_after": "800"},
    )

    persisted = repository.get_decision_run(run["id"])
    current = repository.get_latest_portfolio(market_id="CN")
    assert persisted is not None
    assert current is not None
    assert persisted["status"] == "completed"
    assert (
        persisted["result"]["portfolio_rollforward"]["status"]
        == "skipped_version_conflict"
    )
    assert current["name"] == "manual edit"
    assert current["cash"] == "900"
    assert current["version"] == 2
