"""
tests/test_metrics.py
---------------------
Unit tests for metrics.py (calculate_max_drawdown, calculate_sharpe_ratio).
"""

import pytest
from metrics import calculate_max_drawdown, calculate_sharpe_ratio


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def rising_values() -> list[float]:
    """
    Returns a monotonically increasing portfolio value series.
    Used to test the no-drawdown case and a positive Sharpe Ratio.
    """
    rising_list = [10.0, 1000.0, 1001.0, 2200.0, 3243.0, 3245.0, 9999.0, 10000.1, 100003.66]
    return rising_list


@pytest.fixture
def falling_values() -> list[float]:
    """
    Returns a monotonically decreasing portfolio value series.
    Used to test maximum possible drawdown and a negative Sharpe Ratio.
    """
    falling_list = [100000.00, 10000.0, 800.00, 750.00134, 14.0, 3.0]
    return falling_list


@pytest.fixture
def mixed_values() -> list[float]:
    """
    Returns a portfolio value series with a clear peak followed by a trough,
    then a partial recovery.
    Used to test that MDD correctly identifies the deepest peak-to-trough loss.
    Example shape: rise → peak → drop → small recovery.
    """
    # Peak is 10_000.0 at index 3, trough is 6_000.0 at index 5
    # MDD = (6_000 - 10_000) / 10_000 = -0.40
    return [8_000.0, 9_000.0, 9_500.0, 10_000.0, 7_500.0, 6_000.0, 6_800.0]


@pytest.fixture
def flat_values() -> list[float]:
    """
    Returns a portfolio value series where every value is identical.
    Used to test the edge case where std deviation is 0 (Sharpe → 0.0)
    and drawdown is 0.0.
    """
    return [5_000.0] * 10


# ── calculate_max_drawdown ─────────────────────────────────────────────────────

class TestCalculateMaxDrawdown:

    def test_returns_zero_for_empty_list(self):
        """
        Assert that calculate_max_drawdown([]) returns 0.0 without raising
        an exception, as documented in the docstring.
        """
        empty_list = []
        assert calculate_max_drawdown(empty_list) == 0.0

    def test_no_drawdown_on_rising_series(self, rising_values):
        """
        Assert that a monotonically increasing series produces a MDD of 0.0,
        because the value never falls below a previous peak.
        """
        assert calculate_max_drawdown(rising_values) == 0.0

    def test_correct_mdd_on_mixed_series(self, mixed_values):
        """
        Assert that the returned MDD matches the expected value calculated
        manually from the mixed_values fixture (peak-to-trough / peak).
        The result must be a negative float < 0.
        """
        assert calculate_max_drawdown(mixed_values) == pytest.approx(-0.40)

    def test_mdd_is_negative_or_zero(self, mixed_values):
        """
        Assert that the MDD is always <= 0.0 for any valid input,
        since a drawdown represents a loss relative to a peak.
        """
        assert calculate_max_drawdown(mixed_values) <= 0.0

    def test_raises_on_non_positive_values(self):
        """
        Assert that a ValueError is raised when portfolio_values contains
        a value <= 0, as documented in the Raises section of the docstring.
        """
        portfolio_values_negative = [1.0, 2.0, 3.0, -5.0, 100]
        with pytest.raises(ValueError):
            calculate_max_drawdown(portfolio_values_negative)

    def test_single_value_returns_zero(self):
        """
        Assert that a list containing a single value returns 0.0,
        since no peak-to-trough movement is possible.
        """
        assert calculate_max_drawdown([1.0]) == 0.0


# ── calculate_sharpe_ratio ────────────────────────────────────────────────────

class TestCalculateSharpeRatio:

    def test_returns_zero_when_std_is_zero(self, flat_values):
        """
        Assert that calculate_sharpe_ratio returns exactly 0.0 when all
        daily returns are identical (standard deviation = 0), to avoid
        a ZeroDivisionError.
        """
        assert calculate_sharpe_ratio(flat_values) == 0.0

    def test_positive_sharpe_on_rising_series(self, rising_values):
        """
        Assert that a steadily rising portfolio produces a Sharpe Ratio > 0,
        since mean excess return is positive.
        """
        assert calculate_sharpe_ratio(rising_values) > 0

    def test_negative_sharpe_on_falling_series(self, falling_values):
        """
        Assert that a steadily falling portfolio produces a Sharpe Ratio < 0,
        since mean excess return is negative.
        """
        assert calculate_sharpe_ratio(falling_values) < 0

    def test_raises_on_single_value(self):
        """
        Assert that a ValueError (or equivalent) is raised when fewer than
        2 portfolio values are provided, since at least one daily return
        is required for the calculation.
        """
        with pytest.raises(ValueError):
            calculate_sharpe_ratio([1.0])

    def test_risk_free_rate_reduces_sharpe(self, rising_values):
        """
        Assert that providing a positive risk_free_rate results in a lower
        Sharpe Ratio compared to risk_free_rate=0.0, because a higher
        hurdle rate reduces the excess return.
        """
        sharpe_no_rfr   = calculate_sharpe_ratio(rising_values, risk_free_rate=0)
        sharpe_high_rfr = calculate_sharpe_ratio(rising_values, risk_free_rate=10)
        assert sharpe_no_rfr > sharpe_high_rfr

    def test_result_is_annualised(self, mixed_values):
        """
        Assert that the result changes when periods_per_year is altered
        (e.g. 252 vs. 12), confirming the annualisation factor
        sqrt(periods_per_year) is applied correctly.
        """
        sharpe_252 = calculate_sharpe_ratio(mixed_values, 0, 252)
        sharpe_280 = calculate_sharpe_ratio(mixed_values, 0, 280)
        # Higher periods_per_year → larger sqrt() multiplier → larger absolute Sharpe
        assert abs(sharpe_280) > abs(sharpe_252)