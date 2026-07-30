from __future__ import annotations

from collections.abc import Callable

from ashare_agent.core.config import Settings
from ashare_agent.domain.risk import AShareRiskPolicy
from ashare_agent.ports.risk_policy import RiskPolicy


RiskPolicyBuilder = Callable[[Settings], RiskPolicy]
_BUILDERS: dict[str, RiskPolicyBuilder] = {}


def register_risk_policy(
    market: str,
    builder: RiskPolicyBuilder,
    *,
    replace: bool = False,
) -> None:
    key = market.strip().upper()
    if key in _BUILDERS and not replace:
        raise ValueError(f"Risk policy for {key!r} is already registered")
    _BUILDERS[key] = builder


def build_risk_policy(settings: Settings) -> RiskPolicy:
    market = settings.active_market.strip().upper()
    try:
        builder = _BUILDERS[market]
    except KeyError as exc:
        raise ValueError(f"No deterministic risk policy for market {market!r}") from exc
    return builder(settings)


register_risk_policy("CN", lambda settings: AShareRiskPolicy(settings))

