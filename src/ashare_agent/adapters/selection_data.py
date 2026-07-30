from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_agent.core.config import Settings
from ashare_agent.ports.selection_data import SelectionDataProvider, SelectionDataset
from ashare_agent.selection.config import AppConfig
from ashare_agent.selection.data import load_market_data
from ashare_agent.selection.tushare_source import TushareDataSource


SelectionDataBuilder = Callable[[Settings, AppConfig], SelectionDataProvider]
_BUILDERS: dict[tuple[str, str], SelectionDataBuilder] = {}


class TushareSelectionDataProvider:
    name = "tushare"
    market = "CN"

    def __init__(
        self,
        config: AppConfig,
        token: str | None,
        *,
        allow_network: bool = True,
        timeout_seconds: int | None = None,
    ) -> None:
        self.config = config
        self.token = token
        self.allow_network = allow_network
        self.timeout_seconds = timeout_seconds

    def load(
        self,
        start: date,
        end: date,
        *,
        force_refresh: bool = False,
    ) -> SelectionDataset:
        source = TushareDataSource(
            self.config,
            token=self.token,
            allow_network=self.allow_network,
            timeout_seconds=self.timeout_seconds,
        )
        frame, stats = source.download(
            start,
            end,
            force_refresh=force_refresh,
        )
        cutoff = pd.Timestamp(frame["date"].max()).date()
        return SelectionDataset(
            frame=frame,
            provider=self.name,
            market=self.market,
            data_cutoff=cutoff,
            provenance={
                "provider": self.name,
                "normalization_schema": "selection-frame-v1",
                "data_quality_warnings": list(stats.warnings),
            },
            retrieval={"download": asdict(stats)},
        )


class CsvSelectionDataProvider:
    """Local canonical-data adapter useful for research and provider contracts."""

    name = "csv"

    def __init__(
        self,
        config: AppConfig,
        path: Path,
        *,
        market: str = "CN",
    ) -> None:
        self.config = config
        self.path = path
        self.market = market.upper()
        if self.market != "CN":
            raise ValueError(
                "The bundled CSV selection adapter currently supports CN only"
            )

    def load(
        self,
        start: date,
        end: date,
        *,
        force_refresh: bool = False,
    ) -> SelectionDataset:
        del force_refresh
        frame = load_market_data(self.path, self.config)
        mask = frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        frame = frame.loc[mask].copy()
        if frame.empty:
            raise ValueError(
                f"No CSV selection data between {start.isoformat()} and {end.isoformat()}"
            )
        return SelectionDataset(
            frame=frame,
            provider=self.name,
            market=self.market,
            data_cutoff=pd.Timestamp(frame["date"].max()).date(),
            provenance={
                "provider": self.name,
                "normalization_schema": "selection-frame-v1",
            },
            retrieval={"path": str(self.path)},
        )


def build_selection_data_provider(
    settings: Settings,
    selection_config: AppConfig,
) -> SelectionDataProvider:
    key = (
        settings.selection_data_provider.strip().lower(),
        settings.active_market.strip().upper(),
    )
    builder = _BUILDERS.get(key) or _BUILDERS.get((key[0], "*"))
    if builder is None:
        available = ", ".join(
            f"{provider}/{market}" for provider, market in sorted(_BUILDERS)
        )
        raise ValueError(
            f"No selection data adapter for {key[0]}/{key[1]}; "
            f"available: {available}"
        )
    return builder(settings, selection_config)


def register_selection_data_provider(
    provider: str,
    market: str,
    builder: SelectionDataBuilder,
    *,
    replace: bool = False,
) -> None:
    key = (provider.strip().lower(), market.strip().upper())
    if key in _BUILDERS and not replace:
        raise ValueError(f"Selection data adapter {key!r} is already registered")
    _BUILDERS[key] = builder


def _build_tushare(
    settings: Settings,
    selection_config: AppConfig,
) -> SelectionDataProvider:
    return TushareSelectionDataProvider(
        selection_config,
        settings.tushare_token,
        allow_network=settings.data_mode != "offline_only",
        timeout_seconds=settings.tushare_timeout_seconds,
    )


def _build_csv(
    settings: Settings,
    selection_config: AppConfig,
) -> SelectionDataProvider:
    if settings.selection_data_path is None:
        raise ValueError(
            "SELECTION_DATA_PATH is required when SELECTION_DATA_PROVIDER=csv"
        )
    return CsvSelectionDataProvider(
        selection_config,
        settings.selection_data_path,
        market=settings.active_market,
    )


register_selection_data_provider("tushare", "CN", _build_tushare)
register_selection_data_provider("csv", "CN", _build_csv)
