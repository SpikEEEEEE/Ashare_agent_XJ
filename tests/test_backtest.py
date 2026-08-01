from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from ashare_agent.backtest.cache import BacktestDecisionCache, decision_quality
from ashare_agent.backtest.engine import BacktestAccount, PortfolioBacktester
from ashare_agent.backtest.models import BacktestConfig
from ashare_agent.domain.candidate_pool import Candidate, CandidatePool
from ashare_agent.domain.instruments import InstrumentId
from ashare_agent.domain.models import RawDecisionBundle, SymbolMarketSnapshot
from ashare_agent.domain.risk import AShareRiskPolicy

from .helpers import make_settings


SYMBOL = "600519.SH"
SESSIONS = (
    date(2025, 1, 2),
    date(2025, 1, 3),
    date(2025, 1, 31),
    date(2025, 2, 3),
    date(2025, 2, 28),
    date(2025, 3, 3),
)


class FakeHistoricalFeed:
    name = "fake_history"

    def __init__(self) -> None:
        self.snapshot_dates: list[date] = []
        self.prices = {
            session: {
                "open": Decimal(str(10 + index * 0.4)),
                "close": Decimal(str(10.2 + index * 0.4)),
            }
            for index, session in enumerate(SESSIONS)
        }

    def prepare(self, symbols, start, end):
        assert symbols == (SYMBOL,)
        assert start == SESSIONS[0]
        assert end == SESSIONS[-1]
        return SESSIONS

    def sessions(self, start, end):
        assert start == SESSIONS[0]
        assert end == SESSIONS[-1]
        return SESSIONS

    def snapshot(
        self,
        symbol: str,
        data_date: date,
        as_of: datetime,
    ) -> SymbolMarketSnapshot:
        assert symbol == SYMBOL
        assert as_of.date() == data_date
        self.snapshot_dates.append(data_date)
        eligible = [
            session for session in SESSIONS if session <= data_date
        ]
        bars = pd.DataFrame(
            {
                "date": eligible,
                "open": [
                    float(self.prices[item]["open"])
                    for item in eligible
                ],
                "high": [
                    float(self.prices[item]["close"]) + 0.1
                    for item in eligible
                ],
                "low": [
                    float(self.prices[item]["open"]) - 0.1
                    for item in eligible
                ],
                "close": [
                    float(self.prices[item]["close"])
                    for item in eligible
                ],
                "volume": [100000.0] * len(eligible),
                "vwap": [
                    float(self.prices[item]["close"])
                    for item in eligible
                ],
            }
        )
        return SymbolMarketSnapshot(
            symbol=symbol,
            data_date=data_date,
            reference_price=self.prices[data_date]["close"],
            bars=bars,
            fundamentals={"pe_ttm": 20.0},
            news=(),
            data_quality_warnings=(),
        )

    def price(
        self,
        symbol: str,
        session: date,
        field: str,
    ) -> Decimal | None:
        assert symbol == SYMBOL
        return self.prices.get(session, {}).get(field)


class FiftyPercentDecisionEngine:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, decision_input, on_stage=None) -> RawDecisionBundle:
        self.calls += 1
        snapshot = decision_input.market[SYMBOL]
        position = decision_input.portfolio.position_map().get(SYMBOL)
        current = float(snapshot.reference_price) * (
            position.shares if position else 0
        )
        total_assets = float(decision_input.portfolio.cash) + current
        target = total_assets * 0.5
        if abs(target - current) <= 0.01:
            action = "hold"
        elif target > current:
            action = "increase"
        elif target <= 0:
            action = "close"
        else:
            action = "decrease"
        return RawDecisionBundle(
            decisions={
                SYMBOL: {
                    "action": action,
                    "target_cash_amount": target,
                    "confidence": 0.9,
                    "reasons": ["scripted fifty-percent allocation"],
                }
            },
            meta={"calls": 1, "engine": "scripted"},
        )


class AuditedDecisionEngine(FiftyPercentDecisionEngine):
    def __init__(self, quality: str = "healthy") -> None:
        super().__init__()
        self.quality = quality

    def decide(self, decision_input, on_stage=None) -> RawDecisionBundle:
        bundle = super().decide(decision_input, on_stage=on_stage)
        return RawDecisionBundle(
            decisions=bundle.decisions,
            meta={
                "calls": 4,
                "provider_attempts": 4,
                "validated_outputs": 2,
                "output_repair_attempts": 1,
                "engine": "audited",
                "decision_quality": self.quality,
                "analysis_coverage": (
                    1.0 if self.quality == "healthy" else 0.65
                ),
                "stage_health": {
                    "analysts": {
                        "status": self.quality,
                        "successful": 2,
                    }
                },
            },
        )


class FailingDecisionEngine:
    def decide(self, decision_input, on_stage=None) -> RawDecisionBundle:
        raise RuntimeError("scripted provider failure")


SECOND_SYMBOL = "300750.SZ"


class DynamicHistoricalFeed:
    name = "dynamic_fake_history"

    def __init__(self) -> None:
        self.prepared_symbols: tuple[str, ...] = ()
        self.prices = {
            symbol: {
                session: {
                    "open": Decimal(str(base + index * 0.2)),
                    "close": Decimal(str(base + 0.1 + index * 0.2)),
                }
                for index, session in enumerate(SESSIONS)
            }
            for symbol, base in ((SYMBOL, 10), (SECOND_SYMBOL, 20))
        }

    def sessions(self, start, end):
        assert start == SESSIONS[0]
        assert end == SESSIONS[-1]
        return SESSIONS

    def prepare(self, symbols, start, end):
        self.prepared_symbols = symbols
        return self.sessions(start, end)

    def snapshot(self, symbol, data_date, as_of):
        eligible = [session for session in SESSIONS if session <= data_date]
        prices = self.prices[symbol]
        bars = pd.DataFrame(
            {
                "date": eligible,
                "open": [float(prices[item]["open"]) for item in eligible],
                "high": [float(prices[item]["close"]) + 0.1 for item in eligible],
                "low": [float(prices[item]["open"]) - 0.1 for item in eligible],
                "close": [float(prices[item]["close"]) for item in eligible],
                "volume": [100000.0] * len(eligible),
                "vwap": [float(prices[item]["close"]) for item in eligible],
            }
        )
        return SymbolMarketSnapshot(
            symbol=symbol,
            data_date=data_date,
            reference_price=prices[data_date]["close"],
            bars=bars,
        )

    def price(self, symbol, session, field):
        return self.prices.get(symbol, {}).get(session, {}).get(field)


class ScheduledCandidateSelector:
    name = "scheduled_test_selector"
    market = "CN"

    def __init__(self) -> None:
        self.calls: list[date] = []

    def select(self, as_of, *, data_cutoff, force_refresh=False):
        assert as_of.date() == data_cutoff
        assert force_refresh is False
        self.calls.append(data_cutoff)
        symbol = SYMBOL if len(self.calls) % 2 else SECOND_SYMBOL
        mic = "XSHG" if symbol.endswith(".SH") else "XSHE"
        local_code = symbol.split(".", 1)[0]
        return CandidatePool(
            pool_id=f"pool_{data_cutoff:%Y%m%d}_{local_code}",
            market="CN",
            as_of=as_of,
            data_session=data_cutoff.isoformat(),
            valid_for_session=None,
            strategy_id=self.name,
            strategy_version="test-v1",
            config_hash="test-config",
            data_source="fake",
            data_provenance={},
            candidates=(
                Candidate(
                    instrument_id=InstrumentId(mic, local_code),
                    provider_symbol=symbol,
                    rank=1,
                    score=1.0,
                    reason="scheduled test candidate",
                ),
            ),
            created_at=as_of.astimezone(timezone.utc),
        )

    def latest(self):
        return None


class RecordingHoldDecisionEngine:
    def __init__(self) -> None:
        self.inputs = []

    def decide(self, decision_input, on_stage=None):
        self.inputs.append(decision_input)
        positions = decision_input.portfolio.position_map()
        return RawDecisionBundle(
            decisions={
                symbol: {
                    "action": "hold",
                    "target_cash_amount": float(
                        snapshot.reference_price
                        * (positions[symbol].shares if symbol in positions else 0)
                    ),
                    "confidence": 0.9,
                    "reasons": ["scheduled hold"],
                }
                for symbol, snapshot in decision_input.market.items()
            },
            meta={"calls": 1, "engine": "recording_hold"},
        )


def test_backtest_uses_next_open_and_writes_auditable_results(tmp_path):
    settings = make_settings(tmp_path)
    feed = FakeHistoricalFeed()
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=feed,
        decision_engine=FiftyPercentDecisionEngine(),
        risk_policy=AShareRiskPolicy(settings),
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        initial_cash=Decimal("100000"),
        decision_frequency="monthly",
    )

    result = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )

    assert feed.snapshot_dates == [
        SESSIONS[0],
        SESSIONS[2],
        SESSIONS[4],
    ]
    assert result.trades
    assert result.trades[0].decision_session == SESSIONS[0]
    assert result.trades[0].session == SESSIONS[1]
    assert result.trades[0].side == "buy"
    assert result.trades[0].execution_price > result.trades[0].reference_open
    assert result.metrics["decision_count"] == 3
    assert result.metrics["llm_provider_calls"] == 3
    assert result.metrics["healthy_decision_count"] == 3
    assert result.metrics["result_quality_status"] == "valid"
    assert result.metrics["total_return"] > 0
    assert result.metrics["equal_weight_universe_return"] > 0
    assert result.metrics["total_fees"] > 0

    output = result.write(tmp_path / "result")
    assert (output / "summary.json").exists()
    assert (output / "equity_curve.csv").exists()
    assert (output / "trades.csv").exists()
    assert (output / "decisions.json").exists()


def test_rebalance_schedule_never_decides_without_a_next_session():
    monthly = PortfolioBacktester.decision_sessions(
        SESSIONS,
        "monthly",
        True,
    )
    daily = PortfolioBacktester.decision_sessions(
        SESSIONS,
        "daily",
        False,
    )

    assert monthly == {SESSIONS[0], SESSIONS[2], SESSIONS[4]}
    assert daily == set(SESSIONS[:-1])
    assert SESSIONS[-1] not in monthly
    assert SESSIONS[-1] not in daily


def test_daily_decisions_reuse_monthly_candidate_pool(tmp_path):
    settings = make_settings(tmp_path)
    feed = DynamicHistoricalFeed()
    selector = ScheduledCandidateSelector()
    engine = RecordingHoldDecisionEngine()
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=feed,
        decision_engine=engine,
        risk_policy=AShareRiskPolicy(settings),
        universe_selector=selector,
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        initial_cash=Decimal("100000"),
        decision_frequency="daily",
        selection_frequency="monthly",
        max_decisions=10,
    )

    result = backtester.run(
        config=config,
        universe=(),
        universe_version="dynamic_test_monthly",
    )

    assert selector.calls == [SESSIONS[0], SESSIONS[2], SESSIONS[4]]
    assert feed.prepared_symbols == (SYMBOL, SECOND_SYMBOL)
    assert [item.symbols for item in engine.inputs] == [
        (SYMBOL,),
        (SYMBOL,),
        (SECOND_SYMBOL,),
        (SECOND_SYMBOL,),
        (SYMBOL,),
    ]
    assert result.metrics["decision_count"] == 5
    assert result.metrics["selection_count"] == 3
    assert result.config.to_dict()["decision_frequency"] == "daily"
    assert result.config.to_dict()["selection_frequency"] == "monthly"
    assert result.decisions[1]["selection_session"] == SESSIONS[0].isoformat()
    assert result.decisions[2]["selection_session"] == SESSIONS[2].isoformat()
    assert result.decisions[2]["candidate_pool_id"].startswith("pool_")
    assert not any("fixed universe" in warning for warning in result.warnings)


def test_backtest_decision_cache_replays_without_new_model_calls(tmp_path):
    settings = make_settings(tmp_path)
    scripted_engine = FiftyPercentDecisionEngine()
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=scripted_engine,
        decision_cache=BacktestDecisionCache(
            tmp_path / "decision_cache",
            settings,
        ),
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        initial_cash=Decimal("100000"),
        decision_frequency="monthly",
    )

    first = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )
    second = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )

    assert scripted_engine.calls == 3
    assert first.metrics["decision_cache_hits"] == 0
    assert second.metrics["decision_cache_hits"] == 3
    assert second.metrics["llm_provider_calls"] == 0
    assert second.metrics["llm_provider_attempts"] == 0
    assert second.metrics["llm_cached_original_calls"] == 3
    assert second.metrics["llm_cached_original_provider_attempts"] == 3
    assert (
        second.metrics["llm_provider_attempts_represented_total"]
        == 3
    )
    assert second.metrics["healthy_decision_count"] == 3
    assert second.metrics["result_quality_status"] == "valid"
    assert second.decisions[0]["decision_quality"] == "healthy"
    assert second.decisions[0]["analysis_coverage"] == 1.0
    assert second.decisions[0]["stage_health"] == {}
    assert second.metrics["final_equity"] == first.metrics["final_equity"]


def test_backtest_cache_key_includes_provider_retry_and_quorum_settings(
    tmp_path,
):
    settings = make_settings(tmp_path)
    engine = FiftyPercentDecisionEngine()
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=engine,
    )
    backtest_account = BacktestAccount(cash=Decimal("100000"))
    decision_input = backtester._decision_input(
        run_id="stable-key-input",
        account=backtest_account,
        universe=(SYMBOL,),
        universe_version="test_v1",
        data_date=SESSIONS[0],
        execution_session=SESSIONS[1],
    )
    assert backtest_account.cash == Decimal("100000")

    variants = (
        replace(settings, llm_structured_output_mode="json_object"),
        replace(
            settings,
            multi_agent_output_retries=(
                settings.multi_agent_output_retries + 1
            ),
        ),
        replace(
            settings,
            multi_agent_semantic_retries=(
                settings.multi_agent_semantic_retries + 1
            ),
        ),
        replace(settings, multi_agent_min_analysts=3),
        replace(settings, multi_agent_min_risk_reviews=3),
    )
    baseline = BacktestDecisionCache(tmp_path / "baseline", settings).key(
        decision_input
    )
    variant_keys = {
        BacktestDecisionCache(tmp_path / "variant", item).key(
            decision_input
        )
        for item in variants
    }

    assert baseline not in variant_keys
    assert len(variant_keys) == len(variants)
    assert (
        BacktestDecisionCache(tmp_path / "mode", settings).key(
            replace(decision_input, mode="holdings_only")
        )
        != baseline
    )


def test_backtest_cache_key_does_not_crash_on_malformed_snapshot(
    tmp_path,
):
    settings = make_settings(tmp_path)
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=FiftyPercentDecisionEngine(),
    )
    decision_input = backtester._decision_input(
        run_id="malformed-key-input",
        account=BacktestAccount(cash=Decimal("100000")),
        universe=(SYMBOL,),
        universe_version="test_v1",
        data_date=SESSIONS[0],
        execution_session=SESSIONS[1],
    )
    snapshot = decision_input.market[SYMBOL]
    malformed = replace(
        decision_input,
        market={
            SYMBOL: replace(
                snapshot,
                bars=snapshot.bars.drop(columns=["date"]),
            )
        },
    )
    cache = BacktestDecisionCache(tmp_path / "decision_cache", settings)

    malformed_key = cache.key(malformed)

    assert len(malformed_key) == 64
    assert malformed_key != cache.key(decision_input)


def test_empty_or_malformed_quality_decisions_are_not_cacheable(tmp_path):
    cache = BacktestDecisionCache(
        tmp_path / "decision_cache",
        make_settings(tmp_path),
    )
    empty = RawDecisionBundle(
        decisions={},
        meta={"calls": 0, "decision_quality": "healthy"},
    )
    malformed = RawDecisionBundle(
        decisions={
            SYMBOL: {
                "action": "hold",
                "target_cash_amount": 0,
            }
        },
        meta={"decision_quality": "unexpected"},
    )

    assert decision_quality(empty) == "failed"
    assert decision_quality(malformed) == "failed"
    assert cache.write("empty", empty) is False
    assert cache.write("malformed", malformed) is False


def test_backtest_cache_rejects_legacy_envelope(tmp_path):
    settings = make_settings(tmp_path)
    cache = BacktestDecisionCache(tmp_path / "decision_cache", settings)
    legacy_key = "legacy"
    cache.root.mkdir(parents=True)
    (cache.root / f"{legacy_key}.json").write_text(
        json.dumps(
            {
                "decisions": {},
                "meta": {"calls": 1},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    assert cache.read(legacy_key) is None


def test_degraded_decisions_are_not_cached_and_invalidate_results(tmp_path):
    settings = make_settings(tmp_path)
    engine = AuditedDecisionEngine(quality="degraded")
    cache_root = tmp_path / "decision_cache"
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=engine,
        decision_cache=BacktestDecisionCache(cache_root, settings),
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        initial_cash=Decimal("100000"),
        decision_frequency="monthly",
    )

    first = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )
    second = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )

    assert engine.calls == 6
    assert not list(cache_root.glob("*.json"))
    assert first.metrics["degraded_decision_count"] == 3
    assert first.metrics["healthy_decision_count"] == 0
    assert first.metrics["degraded_decision_rate"] == 1.0
    assert first.metrics["result_quality_status"] == "invalid"
    assert first.metrics["llm_provider_attempts"] == 12
    assert first.metrics["llm_validated_outputs"] == 6
    assert first.metrics["llm_output_repair_attempts"] == 3
    assert first.metrics["buy_count"] == 0
    assert first.metrics["trade_count"] == 0
    assert second.metrics["decision_cache_hits"] == 0
    assert all(
        item["analysis_coverage"] == 0.65
        and item["stage_health"]["analysts"]["status"] == "degraded"
        for item in first.decisions
    )
    assert any(
        warning.startswith("BACKTEST_RESULT_INVALID:")
        for warning in first.warnings
    )


def test_cached_audit_metrics_separate_fresh_and_original_attempts(
    tmp_path,
):
    settings = make_settings(tmp_path)
    engine = AuditedDecisionEngine(quality="healthy")
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=engine,
        decision_cache=BacktestDecisionCache(
            tmp_path / "decision_cache",
            settings,
        ),
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        initial_cash=Decimal("100000"),
        decision_frequency="monthly",
    )

    first = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )
    second = backtester.run(
        config=config,
        universe=(SYMBOL,),
        universe_version="test_v1",
    )

    assert first.metrics["llm_provider_calls"] == 12
    assert first.metrics["llm_provider_attempts"] == 12
    assert first.metrics["llm_validated_outputs"] == 6
    assert first.metrics["llm_output_repair_attempts"] == 3
    assert second.metrics["llm_provider_calls"] == 0
    assert second.metrics["llm_provider_attempts"] == 0
    assert second.metrics["llm_cached_original_calls"] == 12
    assert (
        second.metrics["llm_cached_original_provider_attempts"]
        == 12
    )
    assert second.metrics["llm_cached_original_validated_outputs"] == 6
    assert (
        second.metrics["llm_cached_original_output_repair_attempts"]
        == 3
    )
    assert (
        second.metrics["llm_provider_attempts_represented_total"]
        == 12
    )


def test_failed_safe_holds_are_counted_and_mark_result_invalid(tmp_path):
    settings = make_settings(tmp_path)
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=FailingDecisionEngine(),
        decision_cache=BacktestDecisionCache(
            tmp_path / "decision_cache",
            settings,
        ),
    )

    result = backtester.run(
        config=BacktestConfig(
            start=SESSIONS[0],
            end=SESSIONS[-1],
            initial_cash=Decimal("100000"),
            decision_frequency="monthly",
        ),
        universe=(SYMBOL,),
        universe_version="test_v1",
    )

    assert result.metrics["failed_decision_count"] == 3
    assert result.metrics["failed_decision_rate"] == 1.0
    assert result.metrics["result_quality_status"] == "invalid"
    assert result.metrics["llm_provider_attempts"] == 0
    assert all(
        item["decision_quality"] == "failed"
        and item["analysis_coverage"] == 0.0
        and item["stage_health"]["decision_engine"] == "failed"
        for item in result.decisions
    )
    assert any(
        warning.startswith("BACKTEST_RESULT_INVALID:")
        for warning in result.warnings
    )
    assert "scripted provider failure" not in " ".join(result.warnings)


def test_backtest_decision_budget_blocks_accidental_expensive_run(tmp_path):
    settings = make_settings(tmp_path)
    backtester = PortfolioBacktester(
        settings=settings,
        data_feed=FakeHistoricalFeed(),
        decision_engine=FiftyPercentDecisionEngine(),
    )
    config = BacktestConfig(
        start=SESSIONS[0],
        end=SESSIONS[-1],
        decision_frequency="daily",
        max_decisions=2,
    )

    with pytest.raises(ValueError, match="exceeding"):
        backtester.run(
            config=config,
            universe=(SYMBOL,),
            universe_version="test_v1",
        )
