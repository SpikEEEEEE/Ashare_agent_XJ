from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ashare_agent.adapters.market_data_factory import build_market_data_provider
from ashare_agent.adapters.universe_factory import build_universe_selector
from ashare_agent.core.config import Settings
from ashare_agent.domain.instruments import get_market_profile
from ashare_agent.repositories.candidate_evaluation_json import (
    JsonCandidateEvaluationRepository,
)


_PROJECT_ROOT_ENV = "ASHARE_AGENT_PROJECT_ROOT"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ashare-agent",
        description="Stock selection and AI portfolio advisory service",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root containing .env and config/",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    select = commands.add_parser(
        "select-pool",
        help="Refresh and persist a candidate pool without starting the API",
    )
    select.add_argument("--as-of")
    select.add_argument("--force-refresh", action="store_true")
    evaluation = commands.add_parser(
        "pool-evaluation",
        help="Show the latest persisted T+N evaluation for a candidate pool",
    )
    evaluation.add_argument(
        "--pool-id",
        help="Defaults to the latest candidate pool for the active market",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve() if args.project_root else None
    if args.command == "serve":
        import uvicorn

        settings = Settings.from_env(project_root)
        previous_project_root = os.environ.get(_PROJECT_ROOT_ENV)
        os.environ[_PROJECT_ROOT_ENV] = str(settings.project_root)
        try:
            uvicorn.run(
                "ashare_agent.main:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
            )
        finally:
            if previous_project_root is None:
                os.environ.pop(_PROJECT_ROOT_ENV, None)
            else:
                os.environ[_PROJECT_ROOT_ENV] = previous_project_root
        return 0

    settings = Settings.from_env(project_root)
    if args.command == "pool-evaluation":
        pool_id = args.pool_id
        if not pool_id:
            latest_pool = build_universe_selector(settings).latest()
            if latest_pool is None:
                raise RuntimeError("No candidate pool exists yet")
            pool_id = latest_pool.pool_id
        evaluation = JsonCandidateEvaluationRepository(
            settings.candidate_pool_path / "outcomes"
        ).latest(pool_id)
        if evaluation is None:
            raise RuntimeError(
                "No completed outcome horizon exists for this candidate pool"
            )
        print(
            json.dumps(
                evaluation.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
        )
        return 0
    timezone = ZoneInfo(get_market_profile(settings.active_market).timezone)
    as_of = (
        datetime.fromisoformat(args.as_of)
        if args.as_of
        else datetime.now(timezone)
    )
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone)
    else:
        as_of = as_of.astimezone(timezone)
    selector = build_universe_selector(settings)
    data_cutoff = build_market_data_provider(
        settings
    ).latest_completed_session(as_of)
    pool = selector.select(
        as_of,
        data_cutoff=data_cutoff,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(pool.to_dict(), ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
