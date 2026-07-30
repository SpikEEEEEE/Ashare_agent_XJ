from __future__ import annotations

from collections.abc import Callable

from ashare_agent.adapters.tushare import TushareMarketDataProvider
from ashare_agent.core.config import Settings
from ashare_agent.ports.market_data import MarketDataProvider


MarketDataBuilder = Callable[[Settings], MarketDataProvider]
_BUILDERS: dict[tuple[str, str], MarketDataBuilder] = {}


def register_market_data_provider(
    provider: str,
    market: str,
    builder: MarketDataBuilder,
    *,
    replace: bool = False,
) -> None:
    key = (provider.strip().lower(), market.strip().upper())
    if key in _BUILDERS and not replace:
        raise ValueError(
            f"Market data provider {key[0]!r} for {key[1]!r} is already registered"
        )
    _BUILDERS[key] = builder


def build_market_data_provider(settings: Settings) -> MarketDataProvider:
    key = (
        settings.market_data_provider.strip().lower(),
        settings.active_market.strip().upper(),
    )
    try:
        builder = _BUILDERS[key]
    except KeyError as exc:
        available = ", ".join(
            f"{provider}/{market}" for provider, market in sorted(_BUILDERS)
        )
        raise ValueError(
            f"No market data adapter for {key[0]}/{key[1]}; available: {available}"
        ) from exc
    return builder(settings)


register_market_data_provider(
    "tushare",
    "CN",
    lambda settings: TushareMarketDataProvider(settings),
)

