from __future__ import annotations

import pytest

from ashare_agent.frontend.api_client import AdvisorApi
from ashare_agent.frontend.validation import (
    InputValidationError,
    parse_positions,
    parse_universe,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:secret@localhost:8000",
        "http://localhost:8000?redirect=internal",
    ],
)
def test_api_client_rejects_unsafe_backend_origins(url):
    with pytest.raises(ValueError, match="HTTP"):
        AdvisorApi(url)


def test_parse_universe_normalizes_and_supports_multiple_separators():
    assert parse_universe("600519.sh\n300750.SZ, 430047.bj") == [
        "600519.SH",
        "300750.SZ",
        "430047.BJ",
    ]


def test_parse_universe_rejects_duplicates():
    with pytest.raises(InputValidationError, match="重复"):
        parse_universe("600519.SH, 600519.sh")


def test_parse_universe_uses_registered_market_identity():
    assert parse_universe("00700.hk,09988.HK", market="HK") == [
        "00700.HK",
        "09988.HK",
    ]


def test_parse_positions_ignores_empty_rows_and_defaults_available_shares():
    assert parse_positions(
        [
            {
                "symbol": "600519.sh",
                "shares": 100.0,
                "available_shares": None,
                "average_cost": 1500.25,
                "holding_days": 20.0,
            },
            {
                "symbol": "",
                "shares": None,
                "available_shares": None,
                "average_cost": None,
                "holding_days": None,
            },
        ]
    ) == [
        {
            "symbol": "600519.SH",
            "shares": 100,
            "available_shares": 100,
            "average_cost": "1500.25",
            "holding_days": 20,
        }
    ]


def test_parse_positions_rejects_invalid_t_plus_one_quantity():
    with pytest.raises(InputValidationError, match="可卖股数"):
        parse_positions(
            [
                {
                    "symbol": "000001.SZ",
                    "shares": 100,
                    "available_shares": 200,
                    "average_cost": 10,
                    "holding_days": 1,
                }
            ]
        )
