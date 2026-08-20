from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ashare_agent.domain.candidate_pool import Candidate, CandidatePool
from ashare_agent.domain.instruments import InstrumentId, get_market_profile
from ashare_agent.adapters.ml_universe_selector import MLUniverseSelector
from ashare_agent.adapters.selection_data import CsvSelectionDataProvider
from ashare_agent.ports.selection_data import SelectionDataset
from ashare_agent.ports.candidate_review import (
    CandidateReviewAssessment,
    CandidateReviewResult,
)
from ashare_agent.ports.universe import UniverseSelectionError
from ashare_agent.repositories.candidate_pool_json import (
    JsonCandidatePoolRepository,
)
from ashare_agent.selection.config import AppConfig
from ashare_agent.selection.data import load_market_data
from ashare_agent.selection.demo import generate_demo_data


def _pool() -> CandidatePool:
    return CandidatePool(
        pool_id="pool_cn_test",
        market="CN",
        as_of=datetime(2026, 7, 30, 16, tzinfo=ZoneInfo("Asia/Shanghai")),
        data_session="2026-07-30",
        valid_for_session="2026-07-31",
        strategy_id="test_selector",
        strategy_version="v1",
        config_hash="config123",
        data_source="fake",
        data_provenance={"schema": "canonical-v1"},
        candidates=(
            Candidate(
                instrument_id=InstrumentId("XSHG", "600519"),
                provider_symbol="600519.SH",
                rank=1,
                score=0.91,
                reason="top_rank",
            ),
            Candidate(
                instrument_id=InstrumentId("XSHE", "300750"),
                provider_symbol="300750.SZ",
                rank=2,
                score=0.83,
                reason="top_rank",
            ),
        ),
    )


def test_instrument_identity_preserves_market_specific_codes_and_lots():
    cn = get_market_profile("CN")
    hk = get_market_profile("HK")

    assert cn.normalize_provider_symbol("600519.sh") == "600519.SH"
    assert cn.instrument_from_parts("600519", exchange="SSE").canonical == (
        "XSHG:600519"
    )
    assert cn.instrument_from_parts("920001").canonical == "XBSE:920001"
    with pytest.raises(ValueError, match="Local code"):
        cn.instrument_from_parts(
            "600000",
            exchange="SZSE",
            provider_symbol="000001.SZ",
        )
    with pytest.raises(ValueError, match="Exchange"):
        cn.instrument_from_parts(
            "600519",
            exchange="SZSE",
            provider_symbol="600519.SH",
        )
    hk_instrument = hk.instrument_from_parts(
        "00700",
        provider_symbol="00700.HK",
    )
    assert hk_instrument.canonical == "XHKG:00700"
    assert hk.provider_symbol(hk_instrument) == "00700.HK"
    assert hk.board_lot(hk_instrument, {"board_lot": 100}) == 100
    with pytest.raises(ValueError, match="master data"):
        hk.board_lot(hk_instrument, {})


def test_csv_selection_adapter_rejects_unimplemented_markets(tmp_path):
    with pytest.raises(ValueError, match="supports CN only"):
        CsvSelectionDataProvider(
            AppConfig(),
            tmp_path / "market.csv",
            market="HK",
        )


def test_candidate_pool_repository_is_versioned_and_integrity_checked(tmp_path):
    repository = JsonCandidatePoolRepository(tmp_path / "candidate_pools")
    pool = _pool()

    repository.save(pool)

    loaded = repository.get(pool.pool_id)
    latest = repository.latest("CN")
    assert loaded is not None
    assert latest is not None
    assert loaded.to_dict() == pool.to_dict()
    assert latest.pool_id == pool.pool_id
    assert latest.symbols == ("600519.SH", "300750.SZ")

    pool_path = (
        tmp_path
        / "candidate_pools"
        / "pools"
        / f"{pool.pool_id}.json"
    )
    tampered = json.loads(pool_path.read_text(encoding="utf-8"))
    tampered["data_session"] = "2026-07-29"
    pool_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        repository.save(pool)

    forged = pool.to_dict()
    forged["pool_id"] = "pool_cn_forged"
    pool_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="file key"):
        repository.get(pool.pool_id)

    missing_digest = pool.to_dict()
    missing_digest.pop("content_digest")
    pool_path.write_text(json.dumps(missing_digest), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        repository.get(pool.pool_id)


def test_candidate_pool_digest_covers_temporal_lineage_and_diagnostics():
    pool = _pool()

    assert replace(
        pool,
        as_of=pool.as_of + timedelta(minutes=1),
    ).content_digest != pool.content_digest
    assert replace(
        pool,
        parent_pool_id="pool_cn_parent",
    ).content_digest != pool.content_digest
    assert replace(
        pool,
        diagnostics={"model_type": "fallback"},
    ).content_digest != pool.content_digest


def test_repository_latest_is_monotonic_and_previous_is_point_in_time(tmp_path):
    repository = JsonCandidatePoolRepository(tmp_path / "candidate_pools")
    base = _pool()
    older = replace(
        base,
        pool_id="pool_cn_older",
        as_of=base.as_of - timedelta(days=2),
        data_session="2026-07-28",
    )
    newest = replace(
        base,
        pool_id="pool_cn_newest",
        as_of=base.as_of + timedelta(days=1),
        data_session="2026-07-31",
    )
    incompatible = replace(
        base,
        pool_id="pool_cn_other_config",
        as_of=base.as_of - timedelta(days=1),
        data_session="2026-07-29",
        config_hash="other-config",
    )

    repository.save(newest)
    repository.save(older)
    repository.save(incompatible)

    assert repository.latest("CN").pool_id == newest.pool_id
    previous = repository.previous(
        "CN",
        before_session="2026-07-30",
        strategy_id=base.strategy_id,
        config_hash=base.config_hash,
    )
    assert previous is not None
    assert previous.pool_id == older.pool_id


def test_repository_promotes_a_later_revision_at_the_same_business_time(tmp_path):
    repository = JsonCandidatePoolRepository(tmp_path / "candidate_pools")
    original = _pool()
    revision = replace(
        original,
        pool_id="pool_cn_revision",
        created_at=original.created_at + timedelta(seconds=1),
        data_provenance={"schema": "canonical-v1", "revision": 2},
    )

    repository.save(original)
    repository.save(revision)

    assert repository.latest("CN").pool_id == revision.pool_id


def test_repository_reads_v1_pool_and_can_advance_its_latest_pointer(tmp_path):
    root = tmp_path / "candidate_pools"
    repository = JsonCandidatePoolRepository(root)
    legacy = replace(
        _pool(),
        pool_id="pool_cn_legacy",
        schema_version="1.0",
    )
    payload = legacy.to_dict()
    payload.pop("created_at")
    pools = root / "pools"
    pools.mkdir(parents=True)
    (pools / f"{legacy.pool_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (root / "latest-CN.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "market": "CN",
                "pool_id": legacy.pool_id,
                "content_digest": legacy.content_digest,
            }
        ),
        encoding="utf-8",
    )

    loaded = repository.latest("CN")
    assert loaded is not None
    assert loaded.pool_id == legacy.pool_id
    newer = replace(
        _pool(),
        pool_id="pool_cn_after_upgrade",
        data_session="2026-07-31",
        as_of=_pool().as_of + timedelta(days=1),
    )
    repository.save(newer)
    assert repository.latest("CN").pool_id == newer.pool_id


def test_ml_selector_publishes_a_candidate_pool_contract(tmp_path):
    data_path = generate_demo_data(
        tmp_path / "market.csv",
        stocks=40,
        days=360,
        seed=13,
    )
    config = AppConfig()
    config.universe.min_listing_days = 20
    config.universe.min_avg_amount = 0
    config.features.min_feature_history = 20
    config.model.train_lookback_days = 160
    config.model.min_train_days = 100
    config.model.min_train_rows = 1_000
    config.model.n_estimators = 20
    config.model.num_leaves = 7
    config.model.min_child_samples = 10
    config.model.n_jobs = 1
    config.selection.top_k = 5
    config.selection.max_industry_fraction = 0.4
    config.candidate_review.enabled = True
    config.candidate_review.preselect_k = 8
    frame = load_market_data(data_path, config)
    data_cutoff = frame["date"].max().date()

    class FrameProvider:
        name = "fake_other_source"
        market = "CN"

        def load(self, start, end, *, force_refresh=False):
            del start, end, force_refresh
            return SelectionDataset(
                frame=frame.copy(),
                provider=self.name,
                market=self.market,
                data_cutoff=data_cutoff,
                provenance={"normalization_schema": "selection-frame-v1"},
            )

    repository = JsonCandidatePoolRepository(tmp_path / "pools")

    class KeepQuantOrderReviewer:
        name = "fake_review"

        def review(self, candidates, *, as_of, data_session):
            del as_of, data_session
            return CandidateReviewResult(
                assessments=tuple(
                    CandidateReviewAssessment(
                        symbol=candidate.provider_symbol,
                        score=float(101 - candidate.rank),
                        confidence=1.0,
                        thesis="quant evidence remains consistent",
                    )
                    for candidate in candidates
                ),
                market_view="stable",
                meta={"calls": 1},
            )

    selector = MLUniverseSelector(
        market="CN",
        config=config,
        data_provider=FrameProvider(),
        repository=repository,
        artifacts_root=tmp_path / "artifacts",
        history_calendar_days=600,
        reviewer=KeepQuantOrderReviewer(),
    )
    previous_queries = []
    original_previous = repository.previous

    def tracked_previous(*args, **kwargs):
        previous_queries.append((args, kwargs))
        return original_previous(*args, **kwargs)

    repository.previous = tracked_previous
    as_of = datetime.combine(
        data_cutoff,
        time(16),
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )

    pool = selector.select(as_of, data_cutoff=data_cutoff)

    assert pool.data_source == "fake_other_source"
    assert pool.data_session == data_cutoff.isoformat()
    assert len(pool.candidates) == 5
    assert [candidate.rank for candidate in pool.candidates] == [1, 2, 3, 4, 5]
    assert all(symbol.endswith((".SH", ".SZ", ".BJ")) for symbol in pool.symbols)
    assert pool.pool_id.endswith(pool.content_digest[:16])
    assert pool.diagnostics["quant_preselected_stocks"] == 8
    assert pool.diagnostics["candidate_review"]["status"] == "applied"
    assert all(
        candidate.reason == "llm_cross_sectional_review"
        for candidate in pool.candidates
    )
    assert len(pool.data_provenance["data_fingerprint"]) == 64
    assert previous_queries[0][1]["before_session"] == data_cutoff.isoformat()
    assert repository.latest("CN").pool_id == pool.pool_id
    artifacts = tmp_path / "artifacts" / pool.pool_id
    assert len(pd.read_csv(artifacts / "candidates.csv")) == 5
    assert len(pd.read_csv(artifacts / "quant_preselection.csv")) == 8


def test_ml_selector_rejects_provider_cutoff_that_disagrees_with_frame(tmp_path):
    requested_cutoff = pd.Timestamp("2026-07-30").date()

    class MismatchedProvider:
        name = "mismatched"
        market = "CN"

        def load(self, start, end, *, force_refresh=False):
            del start, end, force_refresh
            return SelectionDataset(
                frame=pd.DataFrame(
                    {
                        "date": [pd.Timestamp("2026-07-29")],
                        "code": ["600519"],
                    }
                ),
                provider=self.name,
                market=self.market,
                data_cutoff=requested_cutoff,
            )

    selector = MLUniverseSelector(
        market="CN",
        config=AppConfig(),
        data_provider=MismatchedProvider(),
        repository=JsonCandidatePoolRepository(tmp_path / "pools"),
        artifacts_root=tmp_path / "artifacts",
        history_calendar_days=600,
    )

    with pytest.raises(UniverseSelectionError) as captured:
        selector.select(
            datetime(
                2026,
                7,
                30,
                16,
                tzinfo=ZoneInfo("Asia/Shanghai"),
            ),
            data_cutoff=requested_cutoff,
        )
    assert "latest observation" in str(captured.value.__cause__)
