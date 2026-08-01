from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_agent.selection.config import AppConfig, validate_config
from ashare_agent.selection.data import load_market_data
from ashare_agent.selection.demo import generate_demo_data
from ashare_agent.selection.features import build_features
from ashare_agent.selection.pipeline import CandidateSelector


def small_test_config() -> AppConfig:
    config = AppConfig()
    config.universe.min_listing_days = 20
    config.universe.min_avg_amount = 0.0
    config.features.min_feature_history = 20
    config.features.prediction_horizon = 5
    config.model.train_lookback_days = 160
    config.model.min_train_days = 100
    config.model.min_train_rows = 1_000
    config.model.n_estimators = 30
    config.model.learning_rate = 0.1
    config.model.num_leaves = 7
    config.model.min_child_samples = 10
    config.model.subsample = 1.0
    config.model.colsample_bytree = 1.0
    config.model.n_jobs = 1
    config.selection.top_k = 10
    config.selection.max_industry_fraction = 0.30
    config.backtest.rebalance_every_days = 5
    config.backtest.max_periods = 3
    return config


class CandidatePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_path = Path(cls.temp_dir.name) / "market.csv"
        generate_demo_data(cls.data_path, stocks=40, days=360, seed=11)
        cls.config = small_test_config()
        cls.market = load_market_data(cls.data_path, cls.config)
        cls.prepared = build_features(cls.market, cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_features_do_not_change_when_future_rows_are_removed(self) -> None:
        all_dates = sorted(self.market["date"].unique())
        cutoff = all_dates[-30]
        truncated_market = self.market.loc[self.market["date"].le(cutoff)].copy()
        truncated = build_features(truncated_market, self.config)

        full_row = self.prepared.frame.loc[
            self.prepared.frame["date"].eq(cutoff),
            ["code", *self.prepared.feature_columns],
        ].sort_values("code")
        truncated_row = truncated.frame.loc[
            truncated.frame["date"].eq(cutoff),
            ["code", *truncated.feature_columns],
        ].sort_values("code")
        self.assertEqual(list(full_row["code"]), list(truncated_row["code"]))
        np.testing.assert_allclose(
            full_row[self.prepared.feature_columns].to_numpy(float),
            truncated_row[truncated.feature_columns].to_numpy(float),
            equal_nan=True,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_selection_respects_filters_and_industry_cap(self) -> None:
        selector = CandidateSelector(self.config)
        result = selector.select_prepared(self.prepared)
        candidates = result.candidates
        self.assertEqual(len(candidates), self.config.selection.top_k)
        self.assertTrue(candidates["eligible"].all())
        self.assertFalse(candidates["is_st"].any())
        self.assertFalse(candidates["is_suspended"].any())
        self.assertFalse(candidates["is_limit_up"].any())
        self.assertEqual(
            result.diagnostics.model_type,
            "lightgbm_regression_l2",
        )
        self.assertEqual(
            result.diagnostics.trained_trees,
            self.config.model.n_estimators,
        )
        cap = int(
            np.ceil(
                self.config.selection.top_k
                * self.config.selection.max_industry_fraction
            )
        )
        self.assertLessEqual(int(candidates["industry"].value_counts().max()), cap)

    def test_walk_forward_backtest_runs(self) -> None:
        selector = CandidateSelector(self.config)
        result = selector.backtest_prepared(self.prepared)
        self.assertEqual(len(result.periods), 3)
        self.assertEqual(
            len(result.holdings), 3 * self.config.selection.top_k
        )
        self.assertIn("mean_spearman_ic", result.summary)

    def test_config_rejects_noncausal_feature_windows(self) -> None:
        for windows in ([-1, 5], [0, 5], [], [5, 5], [True, 5]):
            with self.subTest(windows=windows):
                config = AppConfig()
                config.features.momentum_windows = windows
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_config(config)

    def test_config_rejects_invalid_tushare_history_window(self) -> None:
        config = AppConfig()
        config.tushare.history_calendar_days = 364

        with self.assertRaisesRegex(ValueError, "at least 365"):
            validate_config(config)

    def test_config_rejects_invalid_daily_basic_coverage(self) -> None:
        for coverage in (0.0, 1.01):
            with self.subTest(coverage=coverage):
                config = AppConfig()
                config.tushare.daily_basic_min_coverage = coverage
                with self.assertRaisesRegex(ValueError, "must be in"):
                    validate_config(config)

    def test_csv_requires_explicit_eligibility_status_contract(self) -> None:
        source = pd.read_csv(self.data_path)
        status_columns = [
            "is_st",
            "st_status_known",
            "is_suspended",
            "is_limit_up",
            "is_limit_down",
        ]
        for column in status_columns:
            with self.subTest(column=column):
                path = Path(self.temp_dir.name) / f"missing_{column}.csv"
                source.drop(columns=[column]).to_csv(path, index=False)
                with self.assertRaisesRegex(
                    ValueError,
                    "Missing required eligibility status columns",
                ):
                    load_market_data(path, self.config)

    def test_csv_rejects_missing_or_non_boolean_eligibility_values(self) -> None:
        source = pd.read_csv(self.data_path)
        invalid_cases = [
            ("is_st", pd.NA, "cannot contain missing"),
            ("is_suspended", "", "cannot contain missing"),
            ("is_limit_up", "unknown", "Unrecognized boolean values"),
            ("is_limit_down", 2, "Unrecognized boolean values"),
        ]
        for index, (column, value, message) in enumerate(invalid_cases):
            with self.subTest(column=column, value=value):
                invalid = source.copy()
                invalid[column] = invalid[column].astype("object")
                invalid.loc[0, column] = value
                path = Path(self.temp_dir.name) / f"invalid_status_{index}.csv"
                invalid.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    load_market_data(path, self.config)


if __name__ == "__main__":
    unittest.main()
