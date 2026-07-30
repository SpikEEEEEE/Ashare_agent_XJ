from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from ashare_agent.container import AppContainer
from ashare_agent.domain.candidate_pool import Candidate, CandidatePool
from ashare_agent.domain.instruments import InstrumentId
from ashare_agent.domain.risk import AShareRiskPolicy
from ashare_agent.main import create_app
from ashare_agent.repositories.sqlite import SQLiteRepository
from ashare_agent.services.decision_service import DecisionService
from ashare_agent.services.task_runner import DecisionTaskRunner
from .helpers import FakeDecisionEngine, FakeMarketData, make_settings


class ApiCandidateSelector:
    name = "api_fake"
    market = "CN"

    def __init__(self) -> None:
        self.pool = CandidatePool(
            pool_id="pool_cn_api",
            market="CN",
            as_of=datetime.now(tz=ZoneInfo("Asia/Shanghai")),
            data_session="2026-07-21",
            valid_for_session="2026-07-22",
            strategy_id="api_fake",
            strategy_version="v1",
            config_hash="api_config",
            data_source="fake",
            data_provenance={"schema": "v1"},
            candidates=(
                Candidate(
                    instrument_id=InstrumentId("XSHG", "600519"),
                    provider_symbol="600519.SH",
                    rank=1,
                    score=0.9,
                    reason="ranked",
                ),
            ),
        )

    def latest(self) -> CandidatePool:
        return self.pool

    def select(
        self,
        as_of,
        *,
        data_cutoff,
        force_refresh=False,
    ) -> CandidatePool:
        del as_of, data_cutoff, force_refresh
        return self.pool


def test_dynamic_decision_api_resolves_and_exposes_candidate_pool(tmp_path):
    settings = make_settings(tmp_path)
    repository = SQLiteRepository(settings.database_path)
    market = FakeMarketData()
    engine = FakeDecisionEngine()
    risk = AShareRiskPolicy(settings)
    selector = ApiCandidateSelector()
    service = DecisionService(
        repository,
        market,
        engine,
        risk,
        universe_selector=selector,
    )
    runner = DecisionTaskRunner(service, mode="inline", max_workers=1)
    container = AppContainer(
        settings=settings,
        repository=repository,
        market_data=market,
        decision_engine=engine,
        risk_policy=risk,
        decision_service=service,
        task_runner=runner,
        universe_selector=selector,
    )

    with TestClient(create_app(container)) as client:
        portfolio = client.post(
            "/api/v1/portfolios",
            json={"name": "dynamic", "cash": "50000", "positions": []},
        ).json()
        response = client.post(
            "/api/v1/decision-runs",
            json={
                "portfolio_id": portfolio["id"],
                "mode": "rebalance",
                "universe_source": "fresh_selection",
            },
        )

        assert response.status_code == 202
        run = response.json()
        assert run["status"] == "completed"
        assert run["universe_source"] == "fresh_selection"
        assert run["candidate_pool_id"] == selector.pool.pool_id
        assert run["universe"] == ["600519.SH"]

        selected = client.get("/api/v1/universe?source=selected")
        assert selected.status_code == 200
        assert selected.json()["candidate_pool_id"] == selector.pool.pool_id
        pool = client.get("/api/v1/candidate-pools/latest")
        assert pool.status_code == 200
        assert pool.json()["content_digest"] == selector.pool.content_digest
