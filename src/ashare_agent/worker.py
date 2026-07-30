from __future__ import annotations

import argparse
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path

from ashare_agent.adapters.decision_engine_factory import build_decision_engine
from ashare_agent.adapters.market_data_factory import build_market_data_provider
from ashare_agent.adapters.risk_policy_factory import build_risk_policy
from ashare_agent.adapters.universe_factory import build_universe_selector
from ashare_agent.core.config import Settings
from ashare_agent.repositories.sqlite import SQLiteRepository
from ashare_agent.services.decision_service import DecisionService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one persisted decision run")
    parser.add_argument("run_id")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--database-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()
    settings = replace(
        Settings.from_env(project_root),
        database_path=Path(args.database_path).resolve(),
    )
    repository = SQLiteRepository(settings.database_path)
    universe_selector = build_universe_selector(settings)
    service = DecisionService(
        repository=repository,
        market_data=build_market_data_provider(settings),
        decision_engine=build_decision_engine(settings),
        risk_policy=build_risk_policy(settings),
        universe_selector=universe_selector,
    )
    # The parent normally enforces the hard deadline. This self-destruct timer
    # also bounds an orphan worker if the API process is killed without cleanup.
    orphan_timeout = threading.Timer(
        settings.decision_task_timeout_seconds + 10,
        lambda: os._exit(124),
    )
    orphan_timeout.daemon = True
    orphan_timeout.start()
    try:
        service.run(args.run_id)
    finally:
        orphan_timeout.cancel()
    run = repository.get_decision_run(args.run_id)
    return 0 if run and run["status"] in {"completed", "degraded", "failed"} else 1


if __name__ == "__main__":
    sys.exit(main())
