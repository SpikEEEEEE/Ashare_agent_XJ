from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_agent import __version__
from ashare_agent.core.json_safety import sanitize_json_value
from ashare_agent.domain.candidate_pool import Candidate, CandidatePool
from ashare_agent.domain.instruments import MarketProfile, get_market_profile
from ashare_agent.ports.candidate_review import CandidateReviewer
from ashare_agent.ports.selection_data import SelectionDataProvider
from ashare_agent.ports.universe import (
    CandidatePoolRepository,
    UniverseSelectionError,
)
from ashare_agent.selection.config import AppConfig
from ashare_agent.selection.pipeline import (
    CandidateSelector,
    write_selection_result,
)
from ashare_agent.selection.outcomes import evaluate_candidate_pool
from ashare_agent.selection.review import finalize_candidate_review

STRATEGY_VERSION = "lightgbm-cross-sectional-v3"


class MLUniverseSelector:
    """Bridge the research selector to the versioned candidate-pool contract."""

    name = "lightgbm_cross_sectional"

    def __init__(
        self,
        *,
        market: str,
        config: AppConfig,
        data_provider: SelectionDataProvider,
        repository: CandidatePoolRepository,
        artifacts_root: Path,
        history_calendar_days: int,
        reviewer: CandidateReviewer | None = None,
        evaluation_repository: Any | None = None,
    ) -> None:
        self.market = market.upper()
        self.profile = get_market_profile(self.market)
        self.config = config
        self.data_provider = data_provider
        self.repository = repository
        self.artifacts_root = artifacts_root
        self.history_calendar_days = int(history_calendar_days)
        self.reviewer = reviewer
        self.evaluation_repository = evaluation_repository
        if self.history_calendar_days < 365:
            raise ValueError("Selection history must cover at least 365 calendar days")
        if data_provider.market.upper() != self.market:
            raise ValueError("Selection data provider and selector market differ")

    def latest(self) -> CandidatePool | None:
        return self.repository.latest(self.market)

    def _generated_feature_identity(self) -> dict[str, Any]:
        configured_path = self.config.features.generated_feature_path
        if not configured_path:
            return {"configured": False}
        path = Path(configured_path)
        if not path.exists():
            return {"configured": True, "sha256": None}
        return {
            "configured": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _config_hash(self) -> str:
        features = asdict(self.config.features)
        features["generated_feature_path"] = self._generated_feature_identity()
        strategy_config = {
            "application_version": __version__,
            "strategy_version": STRATEGY_VERSION,
            "dependency_versions": self._dependency_versions(),
            "history_calendar_days": self.history_calendar_days,
            "data": asdict(self.config.data),
            "universe": asdict(self.config.universe),
            "features": features,
            "model": asdict(self.config.model),
            "selection": asdict(self.config.selection),
            "candidate_review": asdict(self.config.candidate_review),
            "feature_screening": asdict(self.config.feature_screening),
        }
        encoded = json.dumps(
            strategy_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _dependency_versions() -> dict[str, str]:
        versions: dict[str, str] = {}
        for distribution in ("numpy", "pandas", "lightgbm"):
            try:
                versions[distribution] = version(distribution)
            except PackageNotFoundError:
                versions[distribution] = "unavailable"
        return versions

    @staticmethod
    def _data_fingerprint(frame: pd.DataFrame) -> str:
        columns = sorted(str(column) for column in frame.columns)
        order_by = [
            column
            for column in ("date", "code", "ts_code")
            if column in frame.columns
        ]
        canonical = (
            frame.sort_values(order_by, kind="stable")
            if order_by
            else frame
        )
        canonical = canonical.loc[:, columns].reset_index(drop=True)
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(columns, separators=(",", ":")).encode("utf-8")
        )
        hashes = pd.util.hash_pandas_object(
            canonical,
            index=False,
            categorize=True,
        )
        hasher.update(hashes.to_numpy(dtype="uint64", copy=False).tobytes())
        return hasher.hexdigest()

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _candidate(
        self,
        row: Any,
        rank: int,
    ) -> Candidate:
        payload = row._asdict()
        provider_symbol = self._optional_text(payload.get("ts_code"))
        instrument = self.profile.instrument_from_parts(
            str(payload["code"]),
            exchange=self._optional_text(payload.get("exchange")),
            provider_symbol=provider_symbol,
        )
        normalized_symbol = self.profile.provider_symbol(instrument)
        signals = {
            key: self._optional_float(payload.get(key))
            for key in (
                "global_rank",
                "alpha_percentile",
                "model_score",
                "avg_amount_liquidity",
                "volatility_20",
                "momentum_5",
                "momentum_20",
                "momentum_60",
                "amount_ratio_20",
                "price_position_20",
            )
            if key in payload
        }
        return Candidate(
            instrument_id=instrument,
            provider_symbol=normalized_symbol,
            rank=rank,
            score=float(payload["selection_score"]),
            reason=str(payload.get("selection_reason") or "selected"),
            name=self._optional_text(payload.get("name")),
            industry=self._optional_text(payload.get("industry")),
            reference_price=self._optional_float(payload.get("close")),
            signals={
                key: value for key, value in signals.items() if value is not None
            },
        )

    def _track_outcomes(
        self,
        *,
        frame: pd.DataFrame,
        data_cutoff: date,
        data_source: str,
        evaluated_at: datetime,
    ) -> dict[str, Any]:
        tracking = self.config.outcome_tracking
        if not tracking.enabled:
            return {"status": "disabled", "evaluated_pools": 0}
        if self.evaluation_repository is None:
            return {
                "status": "unavailable",
                "evaluated_pools": 0,
                "warnings": ["OUTCOME_REPOSITORY_UNAVAILABLE"],
            }
        recent = getattr(self.repository, "recent", None)
        if not callable(recent):
            return {
                "status": "unavailable",
                "evaluated_pools": 0,
                "warnings": ["CANDIDATE_POOL_RECENT_QUERY_UNAVAILABLE"],
            }

        try:
            pools = recent(
                self.market,
                before_or_equal_session=data_cutoff.isoformat(),
                limit=tracking.max_pools_per_run,
            )
        except Exception as exc:
            return {
                "status": "degraded",
                "evaluated_pools": 0,
                "failure_types": [type(exc).__name__],
            }
        saved = 0
        skipped = 0
        failures: list[str] = []
        completed_horizons: dict[str, list[int]] = {}
        for pool in pools:
            if pool.data_session >= data_cutoff.isoformat():
                skipped += 1
                continue
            try:
                previous = self.evaluation_repository.latest(pool.pool_id)
                if (
                    previous is not None
                    and previous.data_cutoff >= data_cutoff.isoformat()
                ):
                    skipped += 1
                    continue
                evaluation = evaluate_candidate_pool(
                    pool,
                    frame,
                    data_cutoff=data_cutoff,
                    data_source=data_source,
                    horizons=tracking.horizons,
                    evaluated_at=evaluated_at,
                )
                if not evaluation.available_horizons:
                    skipped += 1
                    continue
                self.evaluation_repository.save(evaluation)
                saved += 1
                completed_horizons[pool.pool_id] = list(
                    evaluation.available_horizons
                )
            except Exception as exc:
                failures.append(type(exc).__name__)
        return {
            "status": "ok" if not failures else "degraded",
            "evaluated_pools": saved,
            "skipped_pools": skipped,
            "completed_horizons": completed_horizons,
            "failure_types": failures,
        }

    @staticmethod
    def _write_final_candidate_artifact(
        *,
        result: Any,
        candidates: tuple[Candidate, ...],
        diagnostics: dict[str, Any],
        version_dir: Path,
    ) -> None:
        candidate_path = version_dir / "candidates.csv"
        quant_path = version_dir / "quant_preselection.csv"
        if candidate_path.exists():
            candidate_path.replace(quant_path)
        source = result.candidates.copy()
        if "code" not in source.columns:
            raise ValueError("Selection artifact is missing code")
        source["_local_code"] = source["code"].astype(str).str.strip().str.zfill(6)
        source = source.set_index("_local_code", drop=False)
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            local_code = candidate.instrument_id.local_code
            if local_code not in source.index:
                raise ValueError("Final candidate is absent from quant preselection")
            raw = source.loc[local_code]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[0]
            row = raw.to_dict()
            row.update(
                {
                    "candidate_position": candidate.rank,
                    "selection_score": candidate.score,
                    "selection_reason": candidate.reason,
                    "review_score": candidate.signals.get("review_score"),
                    "review_confidence": candidate.signals.get(
                        "review_confidence"
                    ),
                    "review_composite_score": candidate.signals.get(
                        "review_composite_score"
                    ),
                    "review_thesis": candidate.signals.get("review_thesis"),
                    "review_risk_flags": json.dumps(
                        candidate.signals.get("review_risk_flags", []),
                        ensure_ascii=False,
                    ),
                    "review_invalidators": json.dumps(
                        candidate.signals.get("review_invalidators", []),
                        ensure_ascii=False,
                    ),
                }
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv(candidate_path, index=False)
        diagnostics_path = version_dir / "selection_diagnostics.json"
        quant_diagnostics_path = version_dir / "quant_selection_diagnostics.json"
        if diagnostics_path.exists():
            diagnostics_path.replace(quant_diagnostics_path)
        diagnostics_path.write_text(
            json.dumps(
                sanitize_json_value(diagnostics),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def select(
        self,
        as_of: datetime,
        *,
        data_cutoff: date,
        force_refresh: bool = False,
    ) -> CandidatePool:
        end = data_cutoff
        if end > as_of.date():
            raise UniverseSelectionError(
                "Candidate selection cutoff cannot be after as_of"
            )
        start = end - timedelta(days=self.history_calendar_days)
        config_hash = self._config_hash()
        try:
            dataset = self.data_provider.load(
                start,
                end,
                force_refresh=force_refresh,
            )
            if dataset.market.strip().upper() != self.market:
                raise ValueError("Selection dataset market does not match selector")
            if dataset.data_cutoff != end:
                raise ValueError(
                    "Selection provider did not return the requested completed session"
                )
            if dataset.frame.empty or "date" not in dataset.frame.columns:
                raise ValueError("Selection dataset has no dated observations")
            dates = pd.to_datetime(dataset.frame["date"], errors="coerce")
            if dates.isna().any():
                raise ValueError("Selection dataset contains invalid dates")
            if dates.max().date() != dataset.data_cutoff:
                raise ValueError(
                    "Selection dataset cutoff does not match its latest observation"
                )
            previous = self.repository.previous(
                self.market,
                before_session=dataset.data_cutoff.isoformat(),
                strategy_id=self.name,
                config_hash=config_hash,
            )
            selector = CandidateSelector(self.config)
            prepared = selector.prepare(dataset.frame)
            previous_codes = (
                [
                    candidate.instrument_id.local_code
                    for candidate in previous.candidates
                ]
                if previous
                else None
            )
            result = selector.select_prepared(
                prepared,
                as_of=dataset.data_cutoff,
                previous_codes=previous_codes,
                top_k=(
                    self.config.candidate_review.preselect_k
                    if self.config.candidate_review.enabled
                    else None
                ),
            )
            if result.diagnostics.score_date != dataset.data_cutoff.isoformat():
                raise ValueError(
                    "Selection score date does not match the completed session"
                )
        except Exception as exc:
            raise UniverseSelectionError(
                f"Candidate selection failed safely ({type(exc).__name__})"
            ) from exc

        preselected_candidates = tuple(
            self._candidate(row, rank)
            for rank, row in enumerate(
                result.candidates.itertuples(index=False),
                start=1,
            )
        )
        diagnostics = asdict(result.diagnostics)
        outcome_diagnostics = self._track_outcomes(
            frame=dataset.frame,
            data_cutoff=dataset.data_cutoff,
            data_source=dataset.provider,
            evaluated_at=as_of,
        )
        review_diagnostics: dict[str, Any] = {"status": "disabled"}
        candidates = preselected_candidates
        if self.config.candidate_review.enabled:
            review_result = None
            fallback_reason: str | None = None
            try:
                if self.reviewer is None:
                    raise RuntimeError("Candidate reviewer is not configured")
                review_result = self.reviewer.review(
                    preselected_candidates,
                    as_of=as_of,
                    data_session=diagnostics["score_date"],
                )
            except Exception as exc:
                if self.config.candidate_review.failure_mode == "fail_closed":
                    raise UniverseSelectionError(
                        "Candidate review failed safely"
                    ) from exc
                fallback_reason = type(exc).__name__
            candidates, review_diagnostics = finalize_candidate_review(
                preselected_candidates,
                review=review_result,
                config=self.config.candidate_review,
                top_k=self.config.selection.top_k,
                max_industry_fraction=(
                    self.config.selection.max_industry_fraction
                ),
                fallback_reason=fallback_reason,
            )
            diagnostics["quant_preselected_stocks"] = len(
                preselected_candidates
            )
            diagnostics["selected_stocks"] = len(candidates)
        diagnostics["candidate_review"] = sanitize_json_value(
            review_diagnostics
        )
        diagnostics["outcome_tracking"] = sanitize_json_value(
            outcome_diagnostics
        )
        data_provenance = dict(
            sanitize_json_value(dataset.provenance) or {}
        )
        data_provenance.update(
            {
                "data_fingerprint": self._data_fingerprint(dataset.frame),
                "data_fingerprint_algorithm": (
                    f"pandas-hash-v1/pandas-{pd.__version__}"
                ),
                "data_start": dates.min().date().isoformat(),
                "data_cutoff": dataset.data_cutoff.isoformat(),
                "rows": len(dataset.frame),
            }
        )
        provisional = CandidatePool(
            pool_id="pending",
            market=self.market,
            as_of=as_of,
            data_session=diagnostics["score_date"],
            valid_for_session=None,
            strategy_id=self.name,
            strategy_version=(
                f"{STRATEGY_VERSION}:{diagnostics['model_type']}"
            ),
            config_hash=config_hash,
            data_source=dataset.provider,
            data_provenance=data_provenance,
            parent_pool_id=previous.pool_id if previous else None,
            candidates=candidates,
            diagnostics=diagnostics,
        )
        pool = replace(
            provisional,
            pool_id=(
                f"pool_{self.market.lower()}_{diagnostics['score_date']}_"
                f"{provisional.content_digest[:16]}"
            ),
        )
        version_dir = self.artifacts_root / pool.pool_id
        write_selection_result(result, version_dir)
        self._write_final_candidate_artifact(
            result=result,
            candidates=candidates,
            diagnostics=diagnostics,
            version_dir=version_dir,
        )
        (version_dir / "candidate_review.json").write_text(
            json.dumps(
                sanitize_json_value(review_diagnostics),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.repository.save(pool)
