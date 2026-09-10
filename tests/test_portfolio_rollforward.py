from __future__ import annotations

from ashare_agent.services.portfolio_rollforward import roll_forward_portfolio


def test_rollforward_reduces_and_closes_positions_at_reference_prices():
    payload, audit = roll_forward_portfolio(
        {
            "name": "sell test",
            "cash": "1000",
            "positions": [
                {
                    "symbol": "600519.SH",
                    "shares": 200,
                    "available_shares": 150,
                    "average_cost": "9.12345678",
                    "holding_days": 20,
                },
                {
                    "symbol": "300750.SZ",
                    "shares": 100,
                    "available_shares": 100,
                    "average_cost": "20.87654321",
                    "holding_days": 5,
                },
            ],
        },
        [
            {
                "symbol": "600519.SH",
                "current_shares": 200,
                "target_shares": 50,
                "reference_price": 10,
            },
            {
                "symbol": "300750.SZ",
                "current_shares": 100,
                "target_shares": 0,
                "reference_price": 30,
            },
        ],
    )

    assert payload["cash"] == "5500"
    assert payload["positions"] == [
        {
            "symbol": "600519.SH",
            "shares": 50,
            "available_shares": 50,
            "average_cost": "9.12345678",
            "holding_days": 20,
        }
    ]
    assert audit["cash_before"] == "1000"
    assert audit["cash_after"] == "5500"
