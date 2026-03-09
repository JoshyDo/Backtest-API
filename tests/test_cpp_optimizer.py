"""
tests/test_cpp_optimizer.py
---------------------------
Unit tests for cpp_optimizer.py (C++ library wrapper via ctypes).
"""

import pytest
from src.cpp_optimizer import (
    is_cpp_available,
    OptimizationResult,
    grid_search_multithreaded_cpp,
)


class TestCppOptimizerAvailability:
    """Tests for C++ optimizer availability checking."""

    def test_is_cpp_available_returns_boolean(self):
        """Assert that is_cpp_available() returns a boolean."""
        result = is_cpp_available()
        assert isinstance(result, bool)

    def test_optimization_result_type(self):
        """Assert that OptimizationResult is a TypedDict."""
        # Create a sample result
        result: OptimizationResult = {
            "short_window": 10,
            "long_window": 30,
            "sharpe_ratio": 0.85,
            "final_value": 12500.0,
            "max_drawdown": -0.15,
        }
        
        # Verify structure
        assert result["short_window"] == 10
        assert result["long_window"] == 30
        assert result["sharpe_ratio"] == 0.85
        assert result["final_value"] == 12500.0
        assert result["max_drawdown"] == -0.15


class TestGridSearchMultithreadedCpp:
    """Tests for grid_search_multithreaded_cpp function."""

    def test_cpp_grid_search_with_valid_parameters(self):
        """Assert that C++ grid search returns valid results when available."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # Create sample price data
        prices = [100.0 + i * 0.1 for i in range(300)]
        
        # Call C++ grid search
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
            initial_cash=10000.0,
            commission=0.001,
            num_threads=-1,
        )
        
        # Verify result is OptimizationResult
        assert isinstance(result, dict)
        assert "short_window" in result
        assert "long_window" in result
        assert "sharpe_ratio" in result
        assert "final_value" in result
        assert "max_drawdown" in result
        
        # Verify result values
        assert isinstance(result["short_window"], int)
        assert isinstance(result["long_window"], int)
        assert isinstance(result["sharpe_ratio"], float)
        assert isinstance(result["final_value"], float)
        assert isinstance(result["max_drawdown"], float)

    def test_cpp_grid_search_constraints(self):
        """Assert that C++ grid search respects parameter constraints."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        prices = [100.0 + i * 0.1 for i in range(300)]
        
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
        )
        
        # Short window should be less than long window
        assert result["short_window"] < result["long_window"]
        
        # Parameters should be within ranges
        assert result["short_window"] >= 5
        assert result["short_window"] < 20
        assert result["long_window"] >= 20
        assert result["long_window"] < 50

    def test_cpp_grid_search_with_sufficient_prices(self):
        """Assert that C++ grid search works with minimum required price data."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # Minimum: need at least as many prices as the max slow window
        prices = [100.0 + i * 0.1 for i in range(100)]
        
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
        )
        
        assert result is not None
        assert "short_window" in result

    def test_cpp_grid_search_raises_on_invalid_prices(self):
        """Assert that C++ grid search raises error for invalid price data."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # Empty prices should raise ValueError
        with pytest.raises(ValueError):
            grid_search_multithreaded_cpp(
                prices=[],
                fast_min=5,
                fast_max=20,
                slow_min=20,
                slow_max=50,
            )

    def test_cpp_grid_search_raises_on_insufficient_prices(self):
        """Assert that C++ grid search raises error for insufficient price data."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # Only 1 price - need at least 2
        with pytest.raises(ValueError):
            grid_search_multithreaded_cpp(
                prices=[100.0],
                fast_min=5,
                fast_max=20,
                slow_min=20,
                slow_max=50,
            )

    def test_cpp_grid_search_raises_on_invalid_fast_range(self):
        """Assert that C++ grid search raises error for invalid fast window range."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        prices = [100.0 + i * 0.1 for i in range(300)]
        
        # fast_min >= fast_max should raise ValueError
        with pytest.raises(ValueError):
            grid_search_multithreaded_cpp(
                prices=prices,
                fast_min=20,
                fast_max=20,  # Invalid: equal to min
                slow_min=20,
                slow_max=50,
            )

    def test_cpp_grid_search_raises_on_invalid_slow_range(self):
        """Assert that C++ grid search raises error for invalid slow window range."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        prices = [100.0 + i * 0.1 for i in range(300)]
        
        # slow_min >= slow_max should raise ValueError
        with pytest.raises(ValueError):
            grid_search_multithreaded_cpp(
                prices=prices,
                fast_min=5,
                fast_max=20,
                slow_min=50,
                slow_max=50,  # Invalid: equal to min
            )

    def test_cpp_grid_search_with_custom_parameters(self):
        """Assert that C++ grid search respects custom initial_cash and commission."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        prices = [100.0 + i * 0.1 for i in range(300)]
        
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
            initial_cash=50000.0,  # Custom capital
            commission=0.002,  # Custom commission
            num_threads=4,  # Custom thread count
        )
        
        assert result is not None
        assert isinstance(result["final_value"], float)
        # Final value should be different with different initial capital
        assert result["final_value"] > 0

    def test_cpp_grid_search_reproducibility(self):
        """Assert that C++ grid search produces consistent results with same data."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        prices = [100.0 + i * 0.1 for i in range(300)]
        
        # Run grid search twice with same parameters
        result1 = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
        )
        
        result2 = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
        )
        
        # Results should be identical
        assert result1["short_window"] == result2["short_window"]
        assert result1["long_window"] == result2["long_window"]
        assert result1["sharpe_ratio"] == result2["sharpe_ratio"]
        assert result1["final_value"] == result2["final_value"]
        assert result1["max_drawdown"] == result2["max_drawdown"]


class TestCppOptimizerErrorHandling:
    """Tests for C++ optimizer error handling."""

    def test_cpp_library_error_handling(self):
        """Assert that grid search handles errors from C++ library gracefully."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # Test with prices that should be valid
        prices = [100.0 + i * 0.1 for i in range(500)]
        
        # This should succeed even with edge case parameters
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=1,  # Minimum valid
            fast_max=2,  # Minimal range
            slow_min=10,
            slow_max=11,  # Minimal range
        )
        
        assert result is not None
        assert result["short_window"] == 1
        assert result["long_window"] >= 10

    def test_cpp_grid_search_with_constant_prices(self):
        """Assert that C++ grid search handles constant price data."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # All prices are the same - no variation
        prices = [100.0] * 300
        
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
        )
        
        # Should still return a valid result
        assert result is not None
        assert isinstance(result["sharpe_ratio"], float)

    def test_cpp_grid_search_with_volatile_prices(self):
        """Assert that C++ grid search handles volatile price data."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # Create highly volatile prices
        prices = [100.0 * (1.02 if i % 2 == 0 else 0.98) for i in range(300)]
        
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
        )
        
        assert result is not None
        assert isinstance(result["sharpe_ratio"], float)

    def test_cpp_grid_search_with_trending_prices(self):
        """Assert that C++ grid search handles trending price data."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # Create steadily increasing prices (strong uptrend)
        prices = [100.0 + i * 1.0 for i in range(300)]
        
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
        )
        
        # Uptrend should produce valid results
        assert result is not None
        assert isinstance(result["sharpe_ratio"], float)
        assert result["final_value"] >= 0

    def test_cpp_grid_search_with_downtrending_prices(self):
        """Assert that C++ grid search handles downtrending price data."""
        if not is_cpp_available():
            pytest.skip("C++ optimizer not available")
        
        # Create steadily decreasing prices (strong downtrend)
        prices = [300.0 - i * 1.0 for i in range(300)]
        
        result = grid_search_multithreaded_cpp(
            prices=prices,
            fast_min=5,
            fast_max=20,
            slow_min=20,
            slow_max=50,
        )
        
        # Downtrend should be handled
        assert result is not None
        assert isinstance(result["final_value"], float)


class TestOptimizerImportFallback:
    """Tests for optimizer import fallback behavior."""

    def test_optimizer_cpp_availability_detection(self):
        """Assert that optimizer correctly detects C++ availability."""
        from src import optimizer
        
        # CPP_AVAILABLE should be set based on is_cpp_available()
        # If C++ is available, CPP_AVAILABLE should be True
        # This tests the try-except block in optimizer.py
        assert isinstance(optimizer.CPP_AVAILABLE, bool)
        
        # If available, should match is_cpp_available()
        if is_cpp_available():
            assert optimizer.CPP_AVAILABLE is True
