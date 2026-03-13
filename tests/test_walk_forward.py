"""Tests for walk_forward module (windows, metrics, inner CV, and analyzer)."""

import logging
from typing import List

import numpy as np
import pytest

from src.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardResult,
    WalkForwardWindow,
    aggregate_oos_equity,
    calculate_adjusted_sharpe,
    calculate_daily_returns,
    calculate_relative_parameter_distance,
    print_wfa_summary,
    run_inner_cross_validation,
    split_is_data,
)


class TestAdjustedSharpe:
    def test_adjusted_sharpe_perfect_generalization(self):
        adjusted = calculate_adjusted_sharpe(train_sharpe=1.0, validate_sharpe=1.0)
        assert adjusted == pytest.approx(1.0)

    def test_adjusted_sharpe_with_overfitting(self):
        adjusted = calculate_adjusted_sharpe(train_sharpe=1.0, validate_sharpe=0.6)
        expected = 0.6 - 0.5 * 0.4  # 0.4
        assert adjusted == pytest.approx(expected)

    def test_adjusted_sharpe_negative(self):
        adjusted = calculate_adjusted_sharpe(train_sharpe=-0.5, validate_sharpe=-1.0)
        expected = -1.0 - 0.5 * 0.5  # -1.25
        assert adjusted == pytest.approx(expected)

    def test_adjusted_sharpe_validate_better_than_train(self):
        adjusted = calculate_adjusted_sharpe(train_sharpe=0.5, validate_sharpe=1.5)
        expected = 1.5 - 0.5 * 1.0  # 1.0
        assert adjusted == pytest.approx(expected)


class TestRelativeParameterDistance:
    def test_identical_parameters(self):
        distance = calculate_relative_parameter_distance((20, 50), (20, 50))
        assert distance == pytest.approx(0.0)

    def test_small_number_large_percentage_change(self):
        distance = calculate_relative_parameter_distance((10, 50), (15, 50))
        assert distance > 0.4  # large relative change in short

    def test_large_number_small_percentage_change(self):
        distance = calculate_relative_parameter_distance((20, 200), (20, 205))
        assert distance < 0.05  # small relative change in long only

    def test_difference_from_euclidean(self):
        euclidean = np.sqrt(5**2 + 5**2)
        relative = calculate_relative_parameter_distance((10, 200), (15, 205))
        assert euclidean > relative
        assert relative < 1.0

    def test_moderate_distance_and_symmetry(self):
        distance_forward = calculate_relative_parameter_distance((20, 50), (25, 62))
        distance_backward = calculate_relative_parameter_distance((25, 62), (20, 50))
        assert 0.2 < distance_forward < 0.6
        assert 0.2 < distance_backward < 0.6


class TestSplitIsData:
    def test_split_basic(self):
        data = [{"date": i} for i in range(10)]
        train, validate = split_is_data(data, train_ratio=0.7)
        assert len(train) == 7
        assert len(validate) == 3
        assert train[0]["date"] == 0
        assert validate[-1]["date"] == 9

    def test_split_preserves_order(self):
        data = [{"date": i} for i in range(100)]
        train, validate = split_is_data(data, train_ratio=0.7)
        assert train[-1]["date"] < validate[0]["date"]
        assert train[-1]["date"] + 1 == validate[0]["date"]

    def test_split_500_day_is_window(self):
        data = [{"date": i} for i in range(504)]
        train, validate = split_is_data(data, train_ratio=0.7)
        assert len(train) == 352
        assert len(validate) == 152

    def test_split_custom_ratio(self):
        data = [{"date": i} for i in range(100)]
        train, validate = split_is_data(data, train_ratio=0.8)
        assert len(train) == 80
        assert len(validate) == 20


class TestCalculateDailyReturns:
    def test_calculate_daily_returns_empty(self):
        with pytest.raises(ValueError):
            calculate_daily_returns([])

    def test_calculate_daily_returns_single(self):
        with pytest.raises(ValueError):
            calculate_daily_returns([100.0])

    def test_calculate_daily_returns_basic(self):
        prices = [100.0, 105.0, 110.0]
        returns = calculate_daily_returns(prices)
        assert returns == pytest.approx([0.05, 0.047619], rel=1e-5)


class TestAggregateOosEquity:
    def test_aggregate_oos_equity(self):
        res1 = WalkForwardResult(
            iteration_num=0,
            best_short_window=10,
            best_long_window=50,
            best_sharpe_is=1.0,
            best_sharpe_oos=0.8,
            oos_portfolio_values=[10000.0, 10100.0, 10200.0],
            oos_final_value=10200.0,
            oos_returns=[0.01, 0.0099],
        )
        res2 = WalkForwardResult(
            iteration_num=1,
            best_short_window=12,
            best_long_window=55,
            best_sharpe_is=0.9,
            best_sharpe_oos=0.7,
            oos_portfolio_values=[10200.0, 10300.0, 10400.0],
            oos_final_value=10400.0,
            oos_returns=[0.0098, 0.0097],
        )

        values, all_returns = aggregate_oos_equity(10000.0, [res1, res2])
        assert values[0] == pytest.approx(10000.0)
        assert values[-1] > 10000.0
        assert len(all_returns) == len(values) - 1


class TestPrintWFASummary:
    def test_print_wfa_summary(self, capsys):
        dummy_result = WalkForwardResult(
            iteration_num=0,
            best_short_window=10,
            best_long_window=50,
            best_sharpe_is=1.0,
            best_sharpe_oos=0.8,
            oos_portfolio_values=[10000.0, 10100.0],
            oos_final_value=10100.0,
            oos_returns=[0.01],
        )
        results_dict = {
            "iterations": [dummy_result],
            "final_value": 10100.0,
            "final_sharpe_oos": 1.0,
            "inner_cv_enabled": False,
        }
        print_wfa_summary(results_dict)
        out = capsys.readouterr().out
        assert "WALK-FORWARD ANALYSIS SUMMARY" in out


class TestWalkForwardAnalyzer:
    def test_walk_forward_analyzer_init_and_generate_windows(self):
        data = [{"Close": i} for i in range(800)]
        analyzer = WalkForwardAnalyzer(
            data=data,
            is_window_days=500,
            oos_window_days=200,
            step_size_days=200,
            warmup_days=50,
        )
        windows = analyzer.generate_windows()
        assert len(windows) > 0
        first = windows[0]
        assert isinstance(first, WalkForwardWindow)

    def test_walk_forward_analyzer_run_no_windows(self):
        data = [{"Close": i} for i in range(800)]
        analyzer = WalkForwardAnalyzer(
            data=data,
            is_window_days=500,
            oos_window_days=200,
            step_size_days=200,
            warmup_days=50,
        )
        analyzer.generate_windows = lambda: []
        with pytest.raises(ValueError, match="No valid Walk-Forward windows generated"):
            analyzer.run(lambda *_args, **_kwargs: None, lambda *_a, **_k: None)


def test_walk_forward_analyzer_stability_constraint(caplog):
    caplog.set_level(logging.INFO, logger="src.walk_forward")

    data = [{"Close": 100 + i} for i in range(1100)]
    analyzer = WalkForwardAnalyzer(
        data=data,
        is_window_days=500,
        oos_window_days=200,
        step_size_days=200,
        warmup_days=50,
    )

    def backtest_func_for_stability(data_subset, short, long, print_results=False):
        first_close = data_subset[0]["Close"]
        if first_close < 200:
            if (short, long) == (5, 20):
                return [100 + 2 * i for i in range(len(data_subset))]
            return [100 for _ in data_subset]
        if (short, long) == (6, 22):
            return [100 + 2 * i for i in range(len(data_subset))]
        return [100 for _ in data_subset]

    res = analyzer.run(
        grid_search_func=lambda *_args, **_kwargs: None,
        backtest_func=backtest_func_for_stability,
        initial_capital=10000.0,
        use_inner_cv=True,
        fast_range=range(5, 11),
        slow_range=range(20, 41),
    )

    assert res["inner_cv_enabled"] is True
    assert len(res["iterations"]) >= 2
    inner_cv_logs = [
        record for record in caplog.records if "Inner CV:" in record.getMessage()
    ]
    assert inner_cv_logs
