from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ashare_agent.selection.config import AppConfig
from ashare_agent.selection.pipeline import CandidateSelector
from ashare_agent.selection.tushare_source import (
    TushareDataSource,
    create_tushare_client,
)


class FakeTushareClient:
    def __init__(self, stocks: int = 25, days: int = 140):
        self.dates = pd.bdate_range("2025-01-02", periods=days)
        self.codes = [
            f"{index + 1:06d}.SZ" if index % 2 == 0 else f"{600000 + index:06d}.SH"
            for index in range(stocks)
        ]
        self.calls: Counter[str] = Counter()
        self.date_to_index = {
            date.strftime("%Y%m%d"): index for index, date in enumerate(self.dates)
        }

    def trade_cal(self, **kwargs: object) -> pd.DataFrame:
        self.calls["trade_cal"] += 1
        start = pd.Timestamp(str(kwargs["start_date"]))
        end = pd.Timestamp(str(kwargs["end_date"]))
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame(
            {
                "exchange": "SSE",
                "cal_date": dates.strftime("%Y%m%d"),
                "is_open": dates.weekday < 5,
                "pretrade_date": "",
            }
        )

    def stock_basic(self, **kwargs: object) -> pd.DataFrame:
        self.calls["stock_basic"] += 1
        if kwargs.get("list_status") != "L":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "ts_code": self.codes,
                "symbol": [code.split(".")[0] for code in self.codes],
                "name": [
                    "ST测试一" if index == 0 else f"测试{index}"
                    for index in range(len(self.codes))
                ],
                "area": "测试",
                "industry": [
                    f"行业{index % 5}" for index in range(len(self.codes))
                ],
                "market": "主板",
                "exchange": [
                    "SZSE" if code.endswith(".SZ") else "SSE" for code in self.codes
                ],
                "list_status": "L",
                "list_date": "20200101",
                "delist_date": pd.NA,
            }
        )

    def namechange(self, **_kwargs: object) -> pd.DataFrame:
        self.calls["namechange"] += 1
        start = self.dates[-20].strftime("%Y%m%d")
        return pd.DataFrame(
            {
                "ts_code": [self.codes[0]],
                "name": ["ST测试一"],
                "start_date": [start],
                "end_date": [pd.NA],
                "change_reason": ["ST"],
            }
        )

    def daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls["daily"] += 1
        trade_date = str(kwargs["trade_date"])
        day = self.date_to_index[trade_date]
        stock_index = np.arange(len(self.codes), dtype=float)
        close = 8.0 + stock_index * 0.15 + day * 0.003
        open_price = close * (1.0 - 0.001 * ((stock_index % 3) - 1))
        return pd.DataFrame(
            {
                "ts_code": self.codes,
                "trade_date": trade_date,
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.01,
                "low": np.minimum(open_price, close) * 0.99,
                "close": close,
                "pre_close": close - 0.003,
                "pct_chg": 0.03,
                "vol": 100_000.0 + stock_index * 1_000.0,
                "amount": 10_000.0 + stock_index * 100.0,
            }
        )

    def daily_basic(self, **kwargs: object) -> pd.DataFrame:
        self.calls["daily_basic"] += 1
        trade_date = str(kwargs["trade_date"])
        day = self.date_to_index[trade_date]
        limit_status = np.zeros(len(self.codes), dtype=int)
        if day == len(self.dates) - 1:
            limit_status[1] = 2
        return pd.DataFrame(
            {
                "ts_code": self.codes,
                "trade_date": trade_date,
                "turnover_rate": 1.0 + np.arange(len(self.codes)) * 0.01,
                "total_mv": 500_000.0 + np.arange(len(self.codes)) * 10_000.0,
                "circ_mv": 400_000.0 + np.arange(len(self.codes)) * 8_000.0,
                "limit_status": limit_status,
            }
        )

    def adj_factor(self, **kwargs: object) -> pd.DataFrame:
        self.calls["adj_factor"] += 1
        trade_date = str(kwargs["trade_date"])
        return pd.DataFrame(
            {
                "ts_code": self.codes,
                "trade_date": trade_date,
                "adj_factor": 1.0,
            }
        )


def tushare_test_config(cache_dir: Path) -> AppConfig:
    config = AppConfig()
    config.tushare.cache_dir = str(cache_dir)
    config.tushare.request_interval_seconds = 0.0
    config.tushare.max_retries = 1
    config.tushare.refresh_last_trading_days = 0
    config.universe.min_listing_days = 20
    config.universe.min_avg_amount = 0.0
    config.features.min_feature_history = 20
    config.model.train_lookback_days = 100
    config.model.min_train_days = 60
    config.model.min_train_rows = 1_000
    config.model.n_estimators = 20
    config.model.learning_rate = 0.1
    config.model.num_leaves = 7
    config.model.min_child_samples = 10
    config.model.subsample = 1.0
    config.model.colsample_bytree = 1.0
    config.model.n_jobs = 1
    config.selection.top_k = 5
    config.selection.max_industry_fraction = 0.4
    return config


class TushareSourceTest(unittest.TestCase):
    def test_client_timeout_is_passed_to_tushare_constructor(self) -> None:
        config = AppConfig()
        expected_client = object()
        fake_tushare = SimpleNamespace(
            set_token=Mock(),
            pro_api=Mock(return_value=expected_client),
        )

        with patch.dict(sys.modules, {"tushare": fake_tushare}):
            client = create_tushare_client(
                config,
                token="test-token",
                timeout_seconds=7,
            )

        self.assertIs(client, expected_client)
        fake_tushare.set_token.assert_called_once_with("test-token")
        fake_tushare.pro_api.assert_called_once_with(
            "test-token",
            timeout=7,
        )

    def test_download_cache_conversion_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeTushareClient()
            config = tushare_test_config(Path(temp_dir) / "cache")
            source = TushareDataSource(
                config, client=client, sleep=lambda _seconds: None
            )
            start = client.dates[0].strftime("%Y%m%d")
            end = client.dates[-1].strftime("%Y%m%d")

            market, stats = source.download(start, end)
            self.assertEqual(stats.trading_dates, len(client.dates))
            self.assertEqual(stats.output_rows, len(client.dates) * len(client.codes))
            self.assertEqual(market.iloc[0]["volume"], 100_000.0 * 100.0)
            self.assertEqual(market.iloc[0]["amount"], 10_000.0 * 1_000.0)
            self.assertEqual(market.iloc[0]["market_cap"], 500_000.0 * 10_000.0)
            self.assertAlmostEqual(market.iloc[0]["turnover_rate"], 0.01)
            self.assertTrue(market["industry"].eq("UNKNOWN").all())
            self.assertTrue(market["st_status_known"].all())

            first_call_counts = client.calls.copy()
            cached_market, cached_stats = source.download(start, end)
            self.assertEqual(client.calls, first_call_counts)
            self.assertEqual(len(cached_market), len(market))
            self.assertEqual(cached_stats.daily_cached, len(client.dates))
            cache_only_source = TushareDataSource(
                config,
                allow_network=False,
                sleep=lambda _seconds: None,
            )
            cache_only_market, _ = cache_only_source.download(start, end)
            self.assertEqual(len(cache_only_market), len(market))

            selector = CandidateSelector(config)
            result = selector.select_prepared(selector.prepare(market))
            self.assertEqual(len(result.candidates), config.selection.top_k)
            self.assertNotIn(
                client.codes[0].split(".")[0],
                set(result.candidates["code"].astype(str)),
            )
            self.assertNotIn(
                client.codes[1].split(".")[0],
                set(result.candidates["code"].astype(str)),
            )

    def test_offline_mode_fails_before_creating_a_remote_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = tushare_test_config(Path(temp_dir) / "cache")
            client = FakeTushareClient(days=2)
            source = TushareDataSource(
                config,
                client=client,
                allow_network=False,
                sleep=lambda _seconds: None,
            )

            with self.assertRaisesRegex(RuntimeError, "Offline mode"):
                source.download("20250102", "20250103")
            self.assertEqual(client.calls, Counter())

    def test_stock_master_snapshot_must_cover_requested_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            config = tushare_test_config(cache_dir)
            initial_client = FakeTushareClient(days=2)
            initial = TushareDataSource(
                config,
                client=initial_client,
                sleep=lambda _seconds: None,
            )
            initial.fetch_stock_basic()
            path = cache_dir / "master" / "stock_basic.csv"
            stale = pd.read_csv(path)
            stale[TushareDataSource.SNAPSHOT_COLUMN] = "2020-01-01"
            stale.to_csv(path, index=False)

            offline = TushareDataSource(
                config,
                allow_network=False,
                sleep=lambda _seconds: None,
            )
            with self.assertRaisesRegex(RuntimeError, "older"):
                offline.fetch_stock_basic(required_as_of="20250102")

            refreshed_client = FakeTushareClient(days=2)
            online = TushareDataSource(
                config,
                client=refreshed_client,
                sleep=lambda _seconds: None,
            )
            refreshed = online.fetch_stock_basic(
                required_as_of="20250102"
            )
            self.assertGreater(refreshed_client.calls["stock_basic"], 0)
            self.assertNotEqual(
                set(refreshed[TushareDataSource.SNAPSHOT_COLUMN]),
                {"2020-01-01"},
            )

    def test_middle_session_gap_fails_closed(self) -> None:
        class MissingMiddleClient(FakeTushareClient):
            def daily(self, **kwargs: object) -> pd.DataFrame:
                if str(kwargs["trade_date"]) == self.dates[1].strftime("%Y%m%d"):
                    return pd.DataFrame()
                return super().daily(**kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            client = MissingMiddleClient(days=3)
            config = tushare_test_config(Path(temp_dir) / "cache")
            source = TushareDataSource(
                config,
                client=client,
                sleep=lambda _seconds: None,
            )

            with self.assertRaisesRegex(RuntimeError, "partition is missing"):
                source.download(
                    client.dates[0].strftime("%Y%m%d"),
                    client.dates[-1].strftime("%Y%m%d"),
                )

    def test_partial_auxiliary_partitions_fail_closed(self) -> None:
        class PartialDailyBasicClient(FakeTushareClient):
            def daily_basic(self, **kwargs: object) -> pd.DataFrame:
                return super().daily_basic(**kwargs).iloc[:-1].copy()

        class PartialAdjFactorClient(FakeTushareClient):
            def adj_factor(self, **kwargs: object) -> pd.DataFrame:
                return super().adj_factor(**kwargs).iloc[:-1].copy()

        for client_type, endpoint in (
            (PartialDailyBasicClient, "daily_basic"),
            (PartialAdjFactorClient, "adj_factor"),
        ):
            with (
                self.subTest(endpoint=endpoint),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                client = client_type(days=2)
                config = tushare_test_config(Path(temp_dir) / "cache")
                source = TushareDataSource(
                    config,
                    client=client,
                    sleep=lambda _seconds: None,
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"{endpoint} partition is incomplete",
                ):
                    source.download(
                        client.dates[0].strftime("%Y%m%d"),
                        client.dates[-1].strftime("%Y%m%d"),
                    )

    def test_malformed_auxiliary_values_fail_closed(self) -> None:
        class MalformedAuxiliaryClient(FakeTushareClient):
            def __init__(
                self,
                endpoint: str,
                field: str,
                value: object,
            ):
                super().__init__(days=2)
                self.endpoint = endpoint
                self.field = field
                self.value = value

            def _malform(
                self,
                frame: pd.DataFrame,
                endpoint: str,
            ) -> pd.DataFrame:
                if self.endpoint == endpoint:
                    frame = frame.copy()
                    frame.loc[frame.index[0], self.field] = self.value
                return frame

            def daily_basic(self, **kwargs: object) -> pd.DataFrame:
                return self._malform(
                    super().daily_basic(**kwargs),
                    "daily_basic",
                )

            def adj_factor(self, **kwargs: object) -> pd.DataFrame:
                return self._malform(
                    super().adj_factor(**kwargs),
                    "adj_factor",
                )

        cases = (
            ("daily_basic", "turnover_rate", np.nan),
            ("daily_basic", "total_mv", np.inf),
            ("daily_basic", "limit_status", 7),
            ("adj_factor", "adj_factor", 0),
        )
        for endpoint, field, value in cases:
            with (
                self.subTest(endpoint=endpoint, field=field),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                client = MalformedAuxiliaryClient(endpoint, field, value)
                config = tushare_test_config(Path(temp_dir) / "cache")
                source = TushareDataSource(
                    config,
                    client=client,
                    sleep=lambda _seconds: None,
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"{endpoint} contains invalid {field}",
                ):
                    source.download(
                        client.dates[0].strftime("%Y%m%d"),
                        client.dates[-1].strftime("%Y%m%d"),
                    )

    def test_missing_name_history_marks_historical_st_status_unknown(self) -> None:
        class MissingHistoryClient(FakeTushareClient):
            def namechange(self, **_kwargs: object) -> pd.DataFrame:
                raise PermissionError("not entitled")

        with tempfile.TemporaryDirectory() as temp_dir:
            client = MissingHistoryClient(days=3)
            config = tushare_test_config(Path(temp_dir) / "cache")
            source = TushareDataSource(
                config,
                client=client,
                sleep=lambda _seconds: None,
            )

            market, stats = source.download(
                client.dates[0].strftime("%Y%m%d"),
                client.dates[-1].strftime("%Y%m%d"),
            )

            self.assertFalse(market["st_status_known"].any())
            self.assertIn("NAME_HISTORY_UNAVAILABLE", stats.warnings)


if __name__ == "__main__":
    unittest.main()
