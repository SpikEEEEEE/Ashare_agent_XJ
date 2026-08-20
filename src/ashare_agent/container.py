from __future__ import annotations

from dataclasses import dataclass

from ashare_agent.adapters.decision_engine_factory import build_decision_engine
from ashare_agent.adapters.market_data_factory import build_market_data_provider
from ashare_agent.adapters.risk_policy_factory import build_risk_policy
from ashare_agent.adapters.universe_factory import build_universe_selector
from ashare_agent.core.config import Settings
from ashare_agent.ports.decision_engine import DecisionEngine
from ashare_agent.ports.market_data import MarketDataProvider
from ashare_agent.ports.risk_policy import RiskPolicy
from ashare_agent.ports.universe import CandidatePoolSelector
from ashare_agent.repositories.sqlite import SQLiteRepository
from ashare_agent.repositories.candidate_evaluation_json import (
    JsonCandidateEvaluationRepository,
)
from ashare_agent.services.decision_service import DecisionService
from ashare_agent.services.task_runner import DecisionTaskRunner


@dataclass
class AppContainer:
    settings: Settings
    repository: SQLiteRepository
    market_data: MarketDataProvider
    decision_engine: DecisionEngine
    risk_policy: RiskPolicy
    decision_service: DecisionService
    task_runner: DecisionTaskRunner
    universe_selector: CandidatePoolSelector | None = None
    evaluation_repository: JsonCandidateEvaluationRepository | None = None

    @classmethod
    def build(cls, settings: Settings | None = None) -> "AppContainer":
        effective_settings = settings or Settings.from_env()
        repository = SQLiteRepository(effective_settings.database_path)
        market_data = build_market_data_provider(effective_settings)
        decision_engine = build_decision_engine(effective_settings)
        risk_policy = build_risk_policy(effective_settings)
        universe_selector = build_universe_selector(effective_settings)
        evaluation_repository = JsonCandidateEvaluationRepository(
            effective_settings.candidate_pool_path / "outcomes"
        )
        decision_service = DecisionService(
            repository,
            market_data,
            decision_engine,
            risk_policy,
            universe_selector=universe_selector,
        )
        task_runner = DecisionTaskRunner(
            decision_service,
            mode=effective_settings.execution_mode,
            max_workers=effective_settings.decision_workers,
            max_queue_size=effective_settings.decision_queue_capacity,
            settings=effective_settings,
            repository=repository,
            task_timeout_seconds=effective_settings.decision_task_timeout_seconds,
        )
        return cls(
            settings=effective_settings,
            repository=repository,
            market_data=market_data,
            decision_engine=decision_engine,
            risk_policy=risk_policy,
            decision_service=decision_service,
            task_runner=task_runner,
            universe_selector=universe_selector,
            evaluation_repository=evaluation_repository,
        )
