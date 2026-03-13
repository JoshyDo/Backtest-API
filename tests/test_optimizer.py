"""
tests/test_optimizer.py
----------------------
Unit tests for optimizer.py (run_grid_search, progress utilities).
"""

import pytest
from src.optimizer import (
    create_progress_bar,
    format_time,
    run_grid_search,
)
from src.cpp_optimizer import is_cpp_available


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
                "Volume": 1000000,
            }
            for i in range(n_records)
        ]

    def test_grid_search_returns_optimization_result(self):
        """Assert that run_grid_search returns both best_sharpe and best_returns."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 15),
            slow_range=range(20, 30),
            backtest_func=self._mock_backtest,
        )
        assert isinstance(result, dict)
        assert "best_sharpe" in result
        assert "best_returns" in result

        # Check best_sharpe structure
        assert "short_window" in result["best_sharpe"]
        assert "long_window" in result["best_sharpe"]
        assert "sharpe_ratio" in result["best_sharpe"]
        assert "final_value" in result["best_sharpe"]
        assert "max_drawdown" in result["best_sharpe"]

        # Check best_returns structure
        assert "short_window" in result["best_returns"]
        assert "long_window" in result["best_returns"]
        assert "sharpe_ratio" in result["best_returns"]
        assert "final_value" in result["best_returns"]
        assert "max_drawdown" in result["best_returns"]

    def test_grid_search_respects_constraints(self):
        """Assert that returned parameters satisfy short_window < long_window."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 15),
            slow_range=range(20, 30),
            backtest_func=self._mock_backtest,
        )
        assert (
            result["best_sharpe"]["short_window"] < result["best_sharpe"]["long_window"]
        )
        assert (
            result["best_returns"]["short_window"]
            < result["best_returns"]["long_window"]
        )

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
        )
        assert result["best_sharpe"]["short_window"] in fast_range
        assert result["best_sharpe"]["long_window"] in slow_range
        assert result["best_returns"]["short_window"] in fast_range
        assert result["best_returns"]["long_window"] in slow_range

    def test_grid_search_raises_on_insufficient_data(self):
        """Assert that ValueError is raised when data is too small."""
        data = self._generate_dummy_data(10)  # Only 10 records
        with pytest.raises(ValueError):
            run_grid_search(
                data=data,
                fast_range=range(5, 15),
                slow_range=range(20, 100),  # Requires 100 records minimum
                backtest_func=self._mock_backtest,
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
            )

    def test_grid_search_single_parameter_set(self):
        """Assert that grid search works with a single parameter combination."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(10, 11),  # Only one value: 10
            slow_range=range(30, 31),  # Only one value: 30
            backtest_func=self._mock_backtest,
        )
        assert result["best_sharpe"]["short_window"] == 10
        assert result["best_sharpe"]["long_window"] == 30
        assert result["best_returns"]["short_window"] == 10
        assert result["best_returns"]["long_window"] == 30

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
            slow_range=range(20, 40),  # Changed to include 30
            backtest_func=mock_backtest_varying,
        )
        # Should select the (10, 30) combination for best Sharpe
        assert result["best_sharpe"]["short_window"] == 10
        assert result["best_sharpe"]["long_window"] == 30

    def test_grid_search_metric_values_are_reasonable(self):
        """Assert that returned metrics have reasonable values."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 15),
            slow_range=range(20, 30),
            backtest_func=self._mock_backtest,
        )
        # Sharpe ratio should be a number
        assert isinstance(result["best_sharpe"]["sharpe_ratio"], (int, float))
        assert isinstance(result["best_returns"]["sharpe_ratio"], (int, float))
        # Final value should be positive
        assert result["best_sharpe"]["final_value"] > 0
        assert result["best_returns"]["final_value"] > 0
        # Max drawdown should be <= 0
        assert result["best_sharpe"]["max_drawdown"] <= 0
        assert result["best_returns"]["max_drawdown"] <= 0

    def test_grid_search_raises_on_invalid_portfolio_values(self):
        """Assert that TypeError is raised if backtest_func returns invalid values."""

        def mock_backtest_invalid(data, short_window, long_window, print_results=False):
            return "not a list"

        data = self._generate_dummy_data(300)
        with pytest.raises(
            TypeError, match="portfolio_values must be a list of floats/ints"
        ):
            run_grid_search(
                data=data,
                fast_range=range(5, 6),
                slow_range=range(20, 21),
                backtest_func=mock_backtest_invalid,
            )

    def test_grid_search_no_valid_results(self):
        """Assert that RuntimeError is raised when no results are found."""
        data = self._generate_dummy_data(300)

        # Ranges that don't produce any combined pairs
        with pytest.raises(
            RuntimeError, match="Grid search did not produce any valid results"
        ):
            # Mock a case where inner conditions prevent assigning best results (e.g. hack ranges)
            run_grid_search(
                data=data,
                fast_range=range(50, 51),
                slow_range=range(50, 51),  # short == long
                backtest_func=self._mock_backtest,
            )


class TestCppOptimizer:
    """Tests for C++ optimizer integration."""

    def test_cpp_available_check(self):
        """Assert that is_cpp_available() returns a boolean."""
        available = is_cpp_available()
        assert isinstance(available, bool)

    def test_cpp_grid_search_with_real_backtest(self):
        """Assert that C++ optimizer is used when available with run_backtest."""
        from main import run_backtest

        # Only test if C++ is available
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")

        # Generate dummy data
        data = [
            {
                "Date": f"2025-01-{(i % 31) + 1:02d}",
                "Close": 100 + i * 0.1,
                "Open": 100 + i * 0.1,
                "High": 102 + i * 0.1,
                "Low": 98 + i * 0.1,
                "Volume": 1000000,
            }
            for i in range(300)
        ]

        # Call grid search with the real run_backtest function
        result = run_grid_search(
            data=data,
            fast_range=range(5, 15),
            slow_range=range(20, 30),
            backtest_func=run_backtest,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "best_sharpe" in result
        assert "best_returns" in result
        assert result["best_sharpe"]["short_window"] >= 5
        assert result["best_sharpe"]["long_window"] >= 20


class TestOptimizerEdgeCases:
    """Tests for optimizer edge cases and error handling."""

    def test_import_error_fallback(self, monkeypatch):
        """Assert that CPP_AVAILABLE becomes False if import fails."""
        import sys

        # Hide src.cpp_optimizer from sys.modules to simulate missing module
        monkeypatch.setitem(sys.modules, "src.cpp_optimizer", None)

        # We need to reload the optimizer module to trigger the ImportError in the try/except block
        import importlib
        import src.optimizer

        try:
            # Re-import should hit the except ImportError
            importlib.reload(src.optimizer)
            assert not src.optimizer.CPP_AVAILABLE
        finally:
            # Restore state
            monkeypatch.delitem(sys.modules, "src.cpp_optimizer", raising=False)
            importlib.reload(src.optimizer)

    def _generate_dummy_data(self, n_records=100):
        """Generate dummy OHLCV data for testing."""
        return [
            {
                "Date": f"2025-01-{(i % 31) + 1:02d}",
                "Close": 100 + i * 0.5,
                "Open": 100 + i * 0.5,
                "High": 102 + i * 0.5,
                "Low": 98 + i * 0.5,
                "Volume": 1000000,
            }
            for i in range(n_records)
        ]

    def _mock_backtest(self, data, short_window, long_window, print_results=False):
        """Mock backtest function that returns dummy portfolio values."""
        base_value = 10000
        return [base_value * (1 + 0.001 * i) for i in range(len(data))]

    def test_grid_search_with_very_large_range(self):
        """Assert that grid search handles large parameter ranges."""
        data = self._generate_dummy_data(500)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 50),  # 45 combinations
            slow_range=range(50, 100),  # 50 combinations
            backtest_func=self._mock_backtest,
        )
        assert (
            result["best_sharpe"]["short_window"] < result["best_sharpe"]["long_window"]
        )

    def test_grid_search_with_minimal_ranges(self):
        """Assert that grid search works with minimal parameter ranges."""
        data = self._generate_dummy_data(100)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 6),  # Only 5
            slow_range=range(10, 11),  # Only 10
            backtest_func=self._mock_backtest,
        )
        assert result["best_sharpe"]["short_window"] == 5
        assert result["best_sharpe"]["long_window"] == 10

    def test_format_time_edge_cases(self):
        """Assert that format_time handles edge cases correctly."""
        # Test boundary values
        assert format_time(0) == "00s"
        assert format_time(1) == "01s"
        assert format_time(59) == "59s"
        assert format_time(60) == "1m 00s"
        assert format_time(3599) == "59m 59s"
        assert format_time(3600) == "1h 00m 00s"
        assert format_time(86399) == "23h 59m 59s"

    def test_create_progress_bar_edge_cases(self):
        """Assert that create_progress_bar handles edge cases."""
        # Test boundary values
        bar_0 = create_progress_bar(0.0)
        assert "0%" in bar_0
        assert "----" in bar_0

        bar_1 = create_progress_bar(1.0)
        assert "100%" in bar_1
        assert "====" in bar_1

        bar_mid = create_progress_bar(0.5)
        assert "50%" in bar_mid

    def test_grid_search_all_parameters_valid(self):
        """Assert that all returned parameters are valid."""
        data = self._generate_dummy_data(300)
        result = run_grid_search(
            data=data,
            fast_range=range(5, 20),
            slow_range=range(30, 50),
            backtest_func=self._mock_backtest,
        )

        # Check both best_sharpe and best_returns have valid metrics
        for key in ["best_sharpe", "best_returns"]:
            res = result[key]
            assert isinstance(res["short_window"], int)
            assert isinstance(res["long_window"], int)
            assert isinstance(res["sharpe_ratio"], (int, float))
            assert isinstance(res["final_value"], (int, float))
            assert isinstance(res["max_drawdown"], (int, float))

            # Check value ranges
            assert res["short_window"] >= 5
            assert res["long_window"] >= 30
            assert res["final_value"] > 0
            assert res["max_drawdown"] <= 0
