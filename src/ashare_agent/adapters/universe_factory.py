from __future__ import annotations

from collections.abc import Callable

from ashare_agent.adapters.ml_universe_selector import MLUniverseSelector
from ashare_agent.adapters.selection_data import build_selection_data_provider
from ashare_agent.core.config import Settings
from ashare_agent.ports.universe import CandidatePoolSelector
from ashare_agent.repositories.candidate_pool_json import (
    JsonCandidatePoolRepository,
)
from ashare_agent.selection.config import load_config


UniverseSelectorBuilder = Callable[[Settings], CandidatePoolSelector]
_BUILDERS: dict[tuple[str, str], UniverseSelectorBuilder] = {}


def _build_lightgbm(settings: Settings) -> CandidatePoolSelector:
    selection_config = load_config(settings.selection_config_path)
    selection_config.tushare.cache_dir = str(
        settings.cache_path
        / "selection"
        / settings.selection_data_provider
        / settings.active_market.lower()
        / "v1"
    )
    provider = build_selection_data_provider(settings, selection_config)
    repository = JsonCandidatePoolRepository(settings.candidate_pool_path)
    return MLUniverseSelector(
        market=settings.active_market,
        config=selection_config,
        data_provider=provider,
        repository=repository,
        artifacts_root=settings.candidate_pool_path / "artifacts",
        history_calendar_days=selection_config.tushare.history_calendar_days,
    )


def register_universe_selector(
    selector: str,
    market: str,
    builder: UniverseSelectorBuilder,
    *,
    replace: bool = False,
) -> None:
    key = (selector.strip().lower(), market.strip().upper())
    if key in _BUILDERS and not replace:
        raise ValueError(f"Universe selector {key!r} is already registered")
    _BUILDERS[key] = builder


def build_universe_selector(settings: Settings) -> CandidatePoolSelector:
    key = (
        settings.universe_selector.strip().lower(),
        settings.active_market.strip().upper(),
    )
    try:
        builder = _BUILDERS[key]
    except KeyError as exc:
        available = ", ".join(
            f"{selector}/{market}" for selector, market in sorted(_BUILDERS)
        )
        raise ValueError(
            f"No universe selector for {key[0]}/{key[1]}; "
            f"available: {available}"
        ) from exc
    return builder(settings)


register_universe_selector("lightgbm", "CN", _build_lightgbm)
