from __future__ import annotations

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ashare_agent.domain.candidate_pool import Candidate, CandidatePool
from ashare_agent.domain.instruments import InstrumentId
from ashare_agent.repositories.candidate_evaluation_json import (
    JsonCandidateEvaluationRepository,
)
from ashare_agent.selection.outcomes import evaluate_candidate_pool


def _pool() -> CandidatePool:
    assessments = [
        {
            "symbol": symbol,
            "quant_rank": rank,
            "quant_quality": 1.0 - (rank - 1) / 3,
            "review_score": score,
            "composite_score": score / 100,
            "selected": symbol in {"600001.SH", "000004.SZ"},
        }
        for rank, (symbol, score) in enumerate(
            [
                ("600001.SH", 70.0),
                ("600002.SH", 40.0),
                ("000003.SZ", 30.0),
                ("000004.SZ", 100.0),
            ],
            start=1,
        )
    ]
    return CandidatePool(
        pool_id="pool_cn_outcome_test",
        market="CN",
        as_of=datetime(2026, 1, 2, 16, tzinfo=ZoneInfo("Asia/Shanghai")),
        data_session="2026-01-02",
        valid_for_session=None,
        strategy_id="test",
        strategy_version="v1",
        config_hash="config",
        data_source="tushare",
        data_provenance={},
        candidates=(
            Candidate(
                instrument_id=InstrumentId("XSHG", "600001"),
                provider_symbol="600001.SH",
                rank=1,
                score=0.9,
                reason="reviewed",
                reference_price=10,
            ),
            Candidate(
                instrument_id=InstrumentId("XSHE", "000004"),
                provider_symbol="000004.SZ",
                rank=2,
                score=0.8,
                reason="reviewed",
                reference_price=10,
            ),
        ),
        diagnostics={
            "candidate_review": {
                "status": "applied",
                "final_count": 2,
                "assessments": assessments,
            }
        },
    )


def _market_frame() -> pd.DataFrame:
    sessions = pd.to_datetime(
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    )
    paths = {
        "600001.SH": [10.0, 10.5, 10.7, 11.0],
        "600002.SH": [10.0, 9.8, 9.7, 9.5],
        "000003.SZ": [10.0, 10.0, 10.1, 10.0],
        "000004.SZ": [10.0, 11.0, 11.5, 12.0],
    }
    rows = []
    for symbol, prices in paths.items():
        for session, price in zip(sessions, prices, strict=True):
            rows.append(
                {
                    "date": session,
                    "ts_code": symbol,
                    "close": price,
                    "adj_close": price,
                }
            )
    return pd.DataFrame(rows)


def test_candidate_pool_outcomes_measure_review_uplift_and_rank_ic(tmp_path):
    pool = _pool()
    evaluation = evaluate_candidate_pool(
        pool,
        _market_frame(),
        data_cutoff=date(2026, 1, 7),
        data_source="tushare",
        horizons=[1, 3, 5],
        evaluated_at=datetime(
            2026,
            1,
            7,
            16,
            tzinfo=ZoneInfo("Asia/Shanghai"),
        ),
    )

    assert evaluation.status == "partial"
    assert evaluation.available_horizons == (1, 3)
    assert evaluation.summary["return_basis"] == "adj_close"
    assert evaluation.summary["horizons"]["3"]["mean_return"] == pytest.approx(0.15)
    comparison = evaluation.review_comparison["horizons"]["3"]
    assert comparison["reviewed_mean_return"] == pytest.approx(0.15)
    assert comparison["quant_baseline_mean_return"] == pytest.approx(0.025)
    assert comparison["review_uplift"] > 0
    assert comparison["composite_rank_ic"] is not None

    repository = JsonCandidateEvaluationRepository(tmp_path / "outcomes")
    stored = repository.save(evaluation)
    assert repository.latest(pool.pool_id).to_dict() == stored.to_dict()

    path = (
        tmp_path
        / "outcomes"
        / pool.pool_id
        / f"{evaluation.evaluation_id}.json"
    )
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["data_cutoff"] = "2026-01-06"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        repository.get(pool.pool_id, evaluation.evaluation_id)
    except ValueError as exc:
        assert "integrity" in str(exc)
    else:  # pragma: no cover - makes the integrity expectation explicit.
        raise AssertionError("Tampered candidate evaluation was accepted")
