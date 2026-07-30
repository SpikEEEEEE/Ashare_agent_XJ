from __future__ import annotations

from collections.abc import Callable

from ashare_agent.adapters.market_data_factory import build_market_data_provider
from ashare_agent.backtest.data import HistoricalDataFeed, TushareHistoricalDataFeed
from ashare_agent.core.config import Settings


HistoricalFeedBuilder = Callable[[Settings], HistoricalDataFeed]
_BUILDERS: dict[tuple[str, str], HistoricalFeedBuilder] = {}


def register_historical_feed(
    provider: str,
    market: str,
    builder: HistoricalFeedBuilder,
    *,
    replace: bool = False,
) -> None:
    key = (provider.strip().lower(), market.strip().upper())
    if key in _BUILDERS and not replace:
        raise ValueError(f"Historical feed {key!r} is already registered")
    _BUILDERS[key] = builder


def build_historical_feed(settings: Settings) -> HistoricalDataFeed:
    key = (
        settings.market_data_provider.strip().lower(),
        settings.active_market.strip().upper(),
    )
    try:
        builder = _BUILDERS[key]
    except KeyError as exc:
        raise ValueError(f"No historical feed for {key[0]}/{key[1]}") from exc
    return builder(settings)


register_historical_feed(
    "tushare",
    "CN",
    lambda settings: TushareHistoricalDataFeed(
        build_market_data_provider(settings)
    ),
)
