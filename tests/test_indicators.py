"""
tests/test_indicators.py
------------------------
Unit tests for indicators.py (calculate_sma).
"""

import pytest
from src.indicators import calculate_sma


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def prices_10() -> list[float]:
    """Ten evenly spaced prices for easy hand-calculation."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


@pytest.fixture
def constant_prices() -> list[float]:
    """All prices identical — SMA must equal that constant."""
    return [42.0] * 10


# ── calculate_sma ──────────────────────────────────────────────────────────────

class TestCalculateSma:

    def test_output_length_matches_input(self, prices_10):
        """
        Assert that the returned list has the same length as the input,
        so callers can zip prices and SMAs index-by-index.
        """
        result = calculate_sma(prices_10, window=3)
        assert len(result) == len(prices_10)

    def test_leading_nones_equal_window_minus_one(self, prices_10):
        """
        Assert that the first (window - 1) entries are None, because
        there is insufficient data to compute a full window average.
        """
        window = 4
        result = calculate_sma(prices_10, window=window)
        assert result[: window - 1] == [None] * (window - 1)

    def test_no_nones_when_window_equals_one(self, prices_10):
        """
        Assert that window=1 produces no None values, since every single
        price is its own 1-period average.
        """
        result = calculate_sma(prices_10, window=1)
        assert None not in result

    def test_first_valid_value_is_correct(self, prices_10):
        """
        Assert that the first non-None SMA value equals the arithmetic mean
        of the first `window` prices.
        prices_10[:3] = [1, 2, 3] → mean = 2.0
        """
        result = calculate_sma(prices_10, window=3)
        assert result[2] == pytest.approx(2.0)

    def test_last_value_is_correct(self, prices_10):
        """
        Assert that the last SMA value equals the mean of the last `window`
        prices: [8, 9, 10] → mean = 9.0.
        """
        result = calculate_sma(prices_10, window=3)
        assert result[-1] == pytest.approx(9.0)

    def test_constant_prices_give_constant_sma(self, constant_prices):
        """
        Assert that when all prices are the same value, every non-None
        SMA entry equals that value (mean of identical numbers = number).
        """
        window = 5
        result = calculate_sma(constant_prices, window=window)
        for val in result[window - 1 :]:
            assert val == pytest.approx(42.0)

    def test_window_of_full_length_returns_one_value(self, prices_10):
        """
        Assert that when window == len(prices), exactly one non-None value
        is produced (the mean of all prices), and the rest are None.
        """
        n = len(prices_10)
        result = calculate_sma(prices_10, window=n)
        assert result[:-1] == [None] * (n - 1)
        assert result[-1] == pytest.approx(sum(prices_10) / n)

    def test_raises_when_window_is_zero(self, prices_10):
        """
        Assert that a ValueError is raised for window < 1, as documented
        in the function's Raises section.
        """
        with pytest.raises(ValueError):
            calculate_sma(prices_10, window=0)

    def test_raises_when_window_exceeds_length(self, prices_10):
        """
        Assert that a ValueError is raised when window > len(prices),
        since there are not enough data points to fill even one window.
        """
        with pytest.raises(ValueError):
            calculate_sma(prices_10, window=len(prices_10) + 1)
