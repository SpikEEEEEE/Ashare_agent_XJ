from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


class PortfolioRollforwardError(ValueError):
    """A completed decision cannot be converted into a portfolio state."""


def _decimal(value: Any, *, field: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise PortfolioRollforwardError(f"{field} is not a valid decimal") from exc
    if not number.is_finite():
        raise PortfolioRollforwardError(f"{field} must be finite")
    return number


def _shares(value: Any, *, field: str) -> int:
    number = _decimal(value, field=field)
    if number < 0 or number != number.to_integral_value():
        raise PortfolioRollforwardError(
            f"{field} must be a non-negative integer"
        )
    return int(number)


def _decimal_text(value: Decimal) -> str:
    # Avoid scientific notation in persisted portfolio fields so the frontend
    # can round-trip the user's full cost precision without display rounding
    # becoming stored data.
    return format(value, "f")


def roll_forward_portfolio(
    portfolio: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply risk-controlled target quantities as simulated fills.

    Fills use the decision's reference price and intentionally omit brokerage
    fees and slippage: the online advisory settings only reserve a fee buffer
    and do not contain a complete execution-cost model. Newly purchased shares
    are unavailable for sale until the user advances them past A-share T+1.
    """

    cash_before = _decimal(portfolio.get("cash"), field="portfolio cash")
    if cash_before < 0:
        raise PortfolioRollforwardError("portfolio cash cannot be negative")

    existing_order: list[str] = []
    existing: dict[str, dict[str, Any]] = {}
    for item in portfolio.get("positions") or []:
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol or symbol in existing:
            raise PortfolioRollforwardError(
                "portfolio positions must have unique symbols"
            )
        shares = _shares(item.get("shares"), field=f"{symbol} shares")
        available = _shares(
            item.get("available_shares", shares),
            field=f"{symbol} available shares",
        )
        if available > shares:
            raise PortfolioRollforwardError(
                f"{symbol} available shares cannot exceed shares"
            )
        average_cost = _decimal(
            item.get("average_cost"),
            field=f"{symbol} average cost",
        )
        if average_cost < 0:
            raise PortfolioRollforwardError(
                f"{symbol} average cost cannot be negative"
            )
        existing_order.append(symbol)
        existing[symbol] = {
            "symbol": symbol,
            "shares": shares,
            "available_shares": available,
            "average_cost": average_cost,
            "holding_days": item.get("holding_days"),
        }

    decision_order: list[str] = []
    decision_map: dict[str, dict[str, Any]] = {}
    for item in decisions:
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol or symbol in decision_map:
            raise PortfolioRollforwardError(
                "decision rows must have unique symbols"
            )
        decision_order.append(symbol)
        decision_map[symbol] = item

    cash_after = cash_before
    positions_after: list[dict[str, Any]] = []
    for symbol in [
        *existing_order,
        *(item for item in decision_order if item not in existing),
    ]:
        before = existing.get(symbol)
        current_shares = int(before["shares"]) if before else 0
        decision = decision_map.get(symbol)
        if decision is not None and decision.get("current_shares") is not None:
            reported_current = _shares(
                decision.get("current_shares"),
                field=f"{symbol} reported current shares",
            )
            if reported_current != current_shares:
                raise PortfolioRollforwardError(
                    f"{symbol} decision does not match the input portfolio"
                )
        target_shares = (
            _shares(
                decision.get("target_shares"),
                field=f"{symbol} target shares",
            )
            if decision is not None
            else current_shares
        )
        delta = target_shares - current_shares

        reference_price: Decimal | None = None
        if decision is not None and decision.get("reference_price") is not None:
            reference_price = _decimal(
                decision.get("reference_price"),
                field=f"{symbol} reference price",
            )
            if reference_price <= 0:
                raise PortfolioRollforwardError(
                    f"{symbol} reference price must be positive"
                )
        if delta != 0 and reference_price is None:
            raise PortfolioRollforwardError(
                f"{symbol} changed quantity without a reference price"
            )

        if reference_price is not None:
            cash_after -= Decimal(delta) * reference_price

        if target_shares == 0:
            continue

        if delta > 0:
            old_cost = (
                Decimal(current_shares) * before["average_cost"]
                if before
                else Decimal("0")
            )
            average_cost = (
                old_cost + Decimal(delta) * reference_price
            ) / Decimal(target_shares)
            # Existing sellable shares remain sellable; today's simulated buy
            # is excluded to preserve the A-share T+1 constraint.
            available_shares = int(before["available_shares"]) if before else 0
        else:
            if before is None:
                raise PortfolioRollforwardError(
                    f"{symbol} target position has no cost basis"
                )
            average_cost = before["average_cost"]
            available_shares = min(
                int(before["available_shares"]),
                target_shares,
            )

        positions_after.append(
            {
                "symbol": symbol,
                "shares": target_shares,
                "available_shares": available_shares,
                "average_cost": _decimal_text(average_cost),
                "holding_days": (
                    before.get("holding_days") if before else 0
                ),
            }
        )

    if cash_after < 0:
        raise PortfolioRollforwardError(
            "simulated fills would make portfolio cash negative"
        )

    payload = {
        "name": str(portfolio.get("name") or "Current portfolio"),
        "cash": _decimal_text(cash_after),
        "positions": positions_after,
    }
    audit = {
        "pricing_assumption": "risk-controlled reference price",
        "fees_and_slippage_included": False,
        "new_buys_available_after_t_plus_one": True,
        "cash_before": _decimal_text(cash_before),
        "cash_after": _decimal_text(cash_after),
        "positions_after": positions_after,
    }
    return payload, audit
