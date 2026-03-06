"""
tests/test_optimizer.py
----------------------
Unit tests for optimizer.py (run_grid_search, progress utilities).
"""

import pytest
from src.optimizer import run_grid_search, create_progress_bar, format_time, OptimizationResult


class TestProgressBar:
    """Tests for create_progress_bar function."""

    def test_progress_bar_full(self):
        """Assert that a full progress (1.0) shows all filled characters."""
        bar = create_progress_bar(1.0, width=10, char="=")
        assert "==========" in bar
        assert "100" in bar

    def test_progress_bar_empty(self):
        """Assert that empty progress (0.0) shows all dashes."""
        bar = create_progress_bar(0.0, width=10, char="=")
        assert "----------" in bar
        assert "0" in bar

    def test_progress_bar_half(self):
        """Assert that half progress (0.5) shows 50% filled."""
        bar = create_progress_bar(0.5, width=10, char="=")
        assert "====" in bar
        assert "-----" in bar
        assert "50" in bar

    def test_progress_bar_custom_char(self):
        """Assert that custom character is used in the progress bar."""
        bar = create_progress_bar(0.5, width=10, char="#")
        assert "#####" in bar

    def test_progress_bar_percentage_accuracy(self):
        """Assert that displayed percentage is accurate."""
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            bar = create_progress_bar(progress, width=10)
            expected_percent = int(progress * 100)
            assert f"{expected_percent:>3d}%" in bar


class TestFormatTime:
    """Tests for format_time function."""

    def test_format_time_seconds_only(self):
        """Assert that seconds less than 60 are formatted as 'XXs'."""
        assert format_time(30) == "30s"
        assert format_time(59) == "59s"

    def test_format_time_minutes(self):
        """Assert that times with minutes are formatted as 'XXmYYs'."""
        assert format_time(60) == "1m 00s"
        assert format_time(90) == "1m 30s"
        assert format_time(125) == "2m 05s"

    def test_format_time_hours(self):
        """Assert that times with hours are formatted as 'XhYYmZZs'."""
        assert format_time(3600) == "1h 00m 00s"
        assert format_time(3661) == "1h 01m 01s"
        assert format_time(7322) == "2h 02m 02s"

    def test_format_time_zero(self):
        """Assert that zero seconds returns '00s'."""
        assert format_time(0) == "00s"

    def test_format_time_negative(self):
        """Assert that negative seconds return 'N/A'."""
        assert format_time(-1) == "N/A"
        assert format_time(-100) == "N/A"


class TestRunGridSearch:
    """Tests for run_grid_search function."""

    def _mock_backtest(self, data, short_window, long_window, print_results=False):
        """Mock backtest function that returns dummy portfolio values."""
        # Return a simple portfolio value series that increases slightly
        base_value = 10000
        return [base_value * (1 + 0.001 * i) for i in range(len(data))]

    def _generate_dummy_data(self, n_records=100):
        """Generate dummy OHLCV data for testing."""
        return [
            {
                "Date": f"2025-01-{(i % 31) + 1:02d}",
                "Close": 100 + i * 0.5,
                "Open": 100 + i * 0.5,
                "High": 102 + i * 0.5,
                "Low": 98 + i * 0.5,
                "Volume": 1000000
            }
            for i in range(n_records)
        ]

    def test_grid_search_returns_optimization_result(self):
        """Assert that run_grid_search returns an OptimizationResult dict."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 15),
            slow_range=range(20, 30),
            backtest_func=self._mock_backtest,
            early_stopping=False  # Disable for testing
        )
        assert isinstance(result, dict)
        assert "short_window" in result
        assert "long_window" in result
        assert "sharpe_ratio" in result
        assert "final_value" in result
        assert "max_drawdown" in result

    def test_grid_search_respects_constraints(self):
        """Assert that returned parameters satisfy short_window < long_window."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 15),
            slow_range=range(20, 30),
            backtest_func=self._mock_backtest,
            early_stopping=False
        )
        assert result["short_window"] < result["long_window"]

    def test_grid_search_within_ranges(self):
        """Assert that returned parameters are within specified ranges."""
        data = self._generate_dummy_data(300)
        fast_range = range(5, 15)
        slow_range = range(20, 30)
        result = run_grid_search(
            data=data,
            fast_range=fast_range,
            slow_range=slow_range,
            backtest_func=self._mock_backtest,
            early_stopping=False
        )
        assert result["short_window"] in fast_range
        assert result["long_window"] in slow_range

    def test_grid_search_raises_on_insufficient_data(self):
        """Assert that ValueError is raised when data is too small."""
        data = self._generate_dummy_data(10)  # Only 10 records
        with pytest.raises(ValueError):
            run_grid_search(
                data=data,
                fast_range=range(5, 15),
                slow_range=range(20, 100),  # Requires 100 records minimum
                backtest_func=self._mock_backtest,
                early_stopping=False
            )

    def test_grid_search_raises_on_empty_ranges(self):
        """Assert that ValueError is raised for empty parameter ranges."""
        data = self._generate_dummy_data(300)
        with pytest.raises(ValueError):
            run_grid_search(
                data=data,
                fast_range=range(0, 0),  # Empty range
                slow_range=range(20, 30),
                backtest_func=self._mock_backtest,
                early_stopping=False
            )

    def test_grid_search_single_parameter_set(self):
        """Assert that grid search works with a single parameter combination."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(10, 11),  # Only one value: 10
            slow_range=range(30, 31),  # Only one value: 30
            backtest_func=self._mock_backtest,
            early_stopping=False
        )
        assert result["short_window"] == 10
        assert result["long_window"] == 30

    def test_grid_search_selects_best_sharpe(self):
        """Assert that grid search returns parameters with highest Sharpe ratio."""
        def mock_backtest_varying(data, short_window, long_window, print_results=False):
            """Mock that returns different sharpe ratios based on parameters."""
            base_value = 10000
            # Better sharpe for (10, 30) combination
            if short_window == 10 and long_window == 30:
                # Return increasing series (good sharpe)
                return [base_value * (1 + 0.01 * i) for i in range(len(data))]
            else:
                # Return flat series (poor sharpe)
                return [base_value] * len(data)

        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 15),
            slow_range=range(25, 35),
            backtest_func=mock_backtest_varying,
            early_stopping=False
        )
        # Should select the (10, 30) combination
        assert result["short_window"] == 10
        assert result["long_window"] == 30

    def test_grid_search_metric_values_are_reasonable(self):
        """Assert that returned metrics have reasonable values."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 15),
            slow_range=range(20, 30),
            backtest_func=self._mock_backtest,
            early_stopping=False
        )
        # Sharpe ratio should be a number
        assert isinstance(result["sharpe_ratio"], (int, float))
        # Final value should be positive
        assert result["final_value"] > 0
        # Max drawdown should be <= 0
        assert result["max_drawdown"] <= 0
