from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from ashare_agent.adapters.backtest_feed_factory import build_historical_feed
from ashare_agent.adapters.decision_engine_factory import build_decision_engine
from ashare_agent.adapters.risk_policy_factory import build_risk_policy
from ashare_agent.adapters.universe_factory import build_universe_selector
from ashare_agent.core.config import Settings
from ashare_agent.domain.instruments import get_market_profile

from .cache import BacktestDecisionCache
from .engine import PortfolioBacktester
from .models import BacktestConfig


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must use YYYY-MM-DD"
        ) from exc


def _decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Expected a decimal number"
        ) from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError("Number must be finite")
    return parsed


def _symbols(value: str, market: str) -> tuple[str, ...]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("At least one symbol is required")
    profile = get_market_profile(market)
    return tuple(
        dict.fromkeys(
            profile.normalize_provider_symbol(symbol)
            for symbol in raw
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the A-share portfolio decision engine with "
            "point-in-time data and next-session-open execution"
        )
    )
    parser.add_argument("--start", required=True, type=_date)
    parser.add_argument("--end", required=True, type=_date)
    parser.add_argument(
        "--initial-cash",
        type=_decimal,
        default=Decimal("1000000"),
    )
    parser.add_argument(
        "--decision-frequency",
        choices=("daily", "weekly", "monthly"),
        default=None,
        help="How often to run the portfolio decision engine (default: monthly)",
    )
    parser.add_argument(
        "--selection-frequency",
        choices=("once", "daily", "weekly", "monthly"),
        default="once",
        help=(
            "How often to rebuild the candidate pool; 'once' keeps the "
            "configured/static universe"
        ),
    )
    parser.add_argument(
        "--rebalance",
        choices=("daily", "weekly", "monthly"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=24,
        help="Hard guard against unexpectedly expensive LLM backtests",
    )
    parser.add_argument(
        "--engine",
        choices=("single_llm", "portfolio_multi_agent"),
        default=None,
        help="Defaults to DECISION_ENGINE from the environment",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols; defaults to the configured universe",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/backtests/<run_id>",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Use only already cached Tushare inputs",
    )
    parser.add_argument(
        "--no-decision-cache",
        action="store_true",
    )
    parser.add_argument(
        "--no-initial-rebalance",
        action="store_true",
    )
    parser.add_argument(
        "--reuse-sale-proceeds",
        action="store_true",
        help="Allow same-open sale proceeds to fund buys",
    )
    parser.add_argument(
        "--commission-rate",
        type=_decimal,
        default=Decimal("0.0003"),
    )
    parser.add_argument(
        "--minimum-commission",
        type=_decimal,
        default=Decimal("5"),
    )
    parser.add_argument(
        "--slippage-bps",
        type=_decimal,
        default=Decimal("5"),
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.rebalance is not None and args.decision_frequency is not None:
        parser.error(
            "--rebalance is deprecated and cannot be combined with "
            "--decision-frequency"
        )
    decision_frequency = (
        args.decision_frequency or args.rebalance or "monthly"
    )
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else None
    )
    settings = Settings.from_env(project_root)
    effective_engine = args.engine or settings.decision_engine_mode
    settings = replace(
        settings,
        decision_engine_mode=effective_engine,
        data_mode=(
            "offline_only" if args.offline_only else settings.data_mode
        ),
    )
    if settings.data_mode == "auto" and not settings.tushare_token:
        parser.error(
            "TUSHARE_TOKEN is required unless --offline-only has complete cache"
        )
    if not settings.llm_api_key:
        parser.error(
            "LLM_API_KEY or OPENAI_API_KEY is required for historical decisions"
        )

    dynamic_selection = args.selection_frequency != "once"
    if dynamic_selection and args.symbols is not None:
        parser.error(
            "--symbols cannot be combined with dynamic --selection-frequency; "
            "use --selection-frequency once for a fixed custom universe"
        )
    if dynamic_selection:
        universe = ()
        universe_version = (
            f"dynamic_{settings.universe_selector}_"
            f"{args.selection_frequency}"
        )
    elif args.symbols is None:
        universe_version, loaded_symbols = settings.load_universe()
        universe = tuple(loaded_symbols)
    else:
        try:
            universe = _symbols(args.symbols, settings.active_market)
        except ValueError as exc:
            parser.error(str(exc))
        universe_version = "cli_symbols_v1"

    config = BacktestConfig(
        start=args.start,
        end=args.end,
        initial_cash=args.initial_cash,
        decision_frequency=decision_frequency,
        selection_frequency=args.selection_frequency,
        initial_rebalance=not args.no_initial_rebalance,
        max_decisions=args.max_decisions,
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
        slippage_bps=args.slippage_bps,
        reuse_sale_proceeds=args.reuse_sale_proceeds,
    )
    feed = build_historical_feed(settings)
    decision_cache = (
        None
        if args.no_decision_cache
        else BacktestDecisionCache(
            settings.cache_path / "backtest_decisions",
            settings,
        )
    )
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=feed,
        decision_engine=build_decision_engine(settings),
        risk_policy=build_risk_policy(settings),
        decision_cache=decision_cache,
        universe_selector=(
            build_universe_selector(settings)
            if dynamic_selection
            else None
        ),
    )
    result = backtester.run(
        config=config,
        universe=universe,
        universe_version=universe_version,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else settings.project_root / "data" / "backtests" / result.run_id
    )
    result.write(output_dir)
    print(
        json.dumps(
            {
                **result.summary(),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
