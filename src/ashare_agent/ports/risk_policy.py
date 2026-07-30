from __future__ import annotations

from typing import Any, Protocol

from ashare_agent.domain.models import DecisionInput, RawDecisionBundle


class RiskPolicy(Protocol):
    market: str

    def apply(
        self,
        decision_input: DecisionInput,
        bundle: RawDecisionBundle,
    ) -> dict[str, Any]:
        """Apply deterministic market and portfolio constraints."""

