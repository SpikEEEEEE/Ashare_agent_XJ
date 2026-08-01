from __future__ import annotations

import unittest

import pandas as pd

from ashare_agent.selection.cli import _resolve_tushare_date_range
from ashare_agent.selection.config import AppConfig


class SelectionCliTest(unittest.TestCase):
    def test_tushare_start_date_defaults_to_configured_history(self) -> None:
        config = AppConfig()
        config.tushare.history_calendar_days = 550

        start, end = _resolve_tushare_date_range(
            None,
            "2023-06-01",
            config,
        )

        self.assertEqual(end, "20230601")
        self.assertEqual(
            start,
            (pd.Timestamp("2023-06-01") - pd.Timedelta(days=550)).strftime(
                "%Y%m%d"
            ),
        )

    def test_explicit_tushare_start_date_overrides_config(self) -> None:
        config = AppConfig()

        start, end = _resolve_tushare_date_range(
            "2023-01-01",
            "2023-06-01",
            config,
        )

        self.assertEqual((start, end), ("20230101", "20230601"))

    def test_tushare_start_date_must_not_follow_end_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be after"):
            _resolve_tushare_date_range(
                "2023-06-02",
                "2023-06-01",
                AppConfig(),
            )


if __name__ == "__main__":
    unittest.main()
