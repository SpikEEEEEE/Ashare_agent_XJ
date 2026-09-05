from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ashare_agent.domain.candidate_pool import Candidate, CandidatePool
from ashare_agent.domain.instruments import InstrumentId
from ashare_agent.domain.risk import AShareRiskPolicy
from ashare_agent.repositories.sqlite import SQLiteRepository
from ashare_agent.services.decision_service import DecisionService
from .helpers import FakeDecisionEngine, FakeMarketData, make_settings


class FakeCandidatePoolSelector:
    name = "fake_selector"
    market = "CN"

    def __init__(self, pool: CandidatePool) -> None:
        self.pool = pool
        self.select_calls: list[bool] = []

    def latest(self) -> CandidatePool | None:
        return self.pool

    def select(
        self,
        as_of: datetime,
        *,
        data_cutoff,
        board_scope=None,
        force_refresh: bool = False,
    ) -> CandidatePool:
        del as_of, data_cutoff, board_scope
        self.select_calls.append(force_refresh)
        return self.pool


def _candidate_pool(as_of: datetime) -> CandidatePool:
    return CandidatePool(
        pool_id="pool_cn_pipeline",
        market="CN",
        as_of=as_of,
        data_session="2026-07-21",
        valid_for_session="2026-07-22",
        strategy_id="fake_selector",
        strategy_version="v1",
        config_hash="fake_config",
        data_source="fake_cross_section",
        data_provenance={"normalization_schema": "selection-frame-v1"},
        candidates=(
            Candidate(
                instrument_id=InstrumentId("XSHG", "600519"),
                provider_symbol="600519.SH",
                rank=1,
                score=0.9,
                reason="ranked",
            ),
            Candidate(
                instrument_id=InstrumentId("XSHE", "300750"),
                provider_symbol="300750.SZ",
                rank=2,
                score=0.8,
                reason="ranked",
            ),
        ),
    )


def test_research_run_selects_then_advises_pool_union_holdings(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    as_of = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    portfolio = repository.create_portfolio(
        {
            "name": "dynamic universe",
            "cash": "10000",
            "positions": [
                {
                    "symbol": "000001.SZ",
                    "shares": 100,
                    "available_shares": 100,
                    "average_cost": "9",
                }
            ],
        }
    )
    pool = _candidate_pool(as_of)
    selector = FakeCandidatePoolSelector(pool)
    engine = FakeDecisionEngine()
    service = DecisionService(
        repository=repository,
        market_data=FakeMarketData(),
        decision_engine=engine,
        risk_policy=AShareRiskPolicy(settings),
        universe_selector=selector,
    )
    run, _ = repository.create_decision_run(
        portfolio=portfolio,
        mode="rebalance",
        as_of=as_of.isoformat(),
        universe_version="pending-fresh_selection",
        universe=[],
        idempotency_key=None,
        request_fingerprint="dynamic-pipeline-test",
        market_id="CN",
        universe_source="fresh_selection",
    )

    service.run(run["id"])

    completed = repository.get_decision_run(run["id"])
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["candidate_pool_id"] == pool.pool_id
    assert completed["universe"] == ["600519.SH", "300750.SZ"]
    # Recompute the selection while keeping the provider's normal incremental
    # refresh policy. A full history redownload is an explicit maintenance action.
    assert selector.select_calls == [False]
    assert engine.last_input.symbols == (
        "600519.SH",
        "300750.SZ",
        "000001.SZ",
    )
    assert engine.last_input.candidate_pool_id == pool.pool_id
    assert engine.last_input.candidate_metadata["600519.SH"]["rank"] == 1
    assert (
        completed["result"]["universe_provenance"]["content_digest"]
        == pool.content_digest
    )
