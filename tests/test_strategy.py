"""
tests/# Helpers

def make_records(closes: list[float]) -> list[dict]:t_strategy.py
----------------------
Unit tests for strategy.py (SMAStrategy.__init__, generate_signals).
"""

import pytest
from src.strategy import SMAStrategy


# Helpers


def _make_records(prices: list[float]) -> list[dict]:
    """Convert a plain list of prices into the OHLCV-dict format expected by generate_signals."""
    return [{"Date": f"2024-01-{i+1:02d}", "Close": p} for i, p in enumerate(prices)]


# Tests for SMAStrategy.__init__


class TestSMAStrategyInit:

    def test_raises_when_short_equals_long(self):
        """
        Assert that a ValueError is raised when short_window == long_window,
        since the crossover logic requires short < long.
        """
        with pytest.raises(ValueError):
            SMAStrategy(short_window=20, long_window=20)

    def test_raises_when_short_greater_than_long(self):
        """
        Assert that a ValueError is raised when short_window > long_window.
        """
        with pytest.raises(ValueError):
            SMAStrategy(short_window=50, long_window=20)

    def test_valid_windows_are_stored(self):
        """
        Assert that valid short_window and long_window values are stored
        as instance attributes after successful construction.
        """
        s = SMAStrategy(short_window=10, long_window=30)
        assert s.short_window == 10
        assert s.long_window == 30


# Tests for SMAStrategy.generate_signals


class TestGenerateSignals:

    def test_output_length_matches_input(self):
        """
        Assert that generate_signals returns one dict per record,
        preserving the same length as the input list.
        """
        records = _make_records([float(i) for i in range(1, 61)])
        s = SMAStrategy(short_window=5, long_window=10)
        signals = s.generate_signals(records)
        assert len(signals) == len(records)

    def test_each_signal_has_required_keys(self):
        """
        Assert that every dict in the result contains 'Date', 'Close',
        and 'Signal' keys.
        """
        records = _make_records([float(i) for i in range(1, 61)])
        s = SMAStrategy(short_window=5, long_window=10)
        for sig in s.generate_signals(records):
            assert "Date" in sig
            assert "Close" in sig
            assert "Signal" in sig

    def test_signal_values_are_valid(self):
        """
        Assert that every 'Signal' value is one of the three allowed
        strings: 'BUY', 'SELL', or 'HOLD'.
        """
        records = _make_records([float(i) for i in range(1, 61)])
        s = SMAStrategy(short_window=5, long_window=10)
        valid = {"BUY", "SELL", "HOLD"}
        for sig in s.generate_signals(records):
            assert sig["Signal"] in valid

    def test_raises_when_too_few_records(self):
        """
        Assert that a ValueError is raised when the number of records is
        less than long_window, since the slow SMA cannot be computed.
        """
        records = _make_records([1.0] * 9)
        s = SMAStrategy(short_window=5, long_window=10)
        with pytest.raises(ValueError):
            s.generate_signals(records)

    def test_golden_cross_produces_buy(self):
        """
        Assert that a golden cross (fast SMA crossing above slow SMA)
        generates at least one BUY signal in a controlled price series
        that starts low and then rises sharply.
        """
        # Phase 1: slow declining ramp so short SMA < long SMA
        # Phase 2: sudden sustained spike so short SMA crosses above long SMA
        declining = [100.0 - i * 0.5 for i in range(50)]  # 100 -> 75.5
        spike = [5_000.0] * 10
        records = _make_records(declining + spike)
        s = SMAStrategy(short_window=5, long_window=20)
        signals = s.generate_signals(records)
        assert any(sig["Signal"] == "BUY" for sig in signals)

    def test_death_cross_produces_sell(self):
        """
        Assert that a death cross (fast SMA crossing below slow SMA)
        generates at least one SELL signal in a price series that starts
        high and then drops sharply.
        """
        # Phase 1: slow rising ramp so short SMA > long SMA
        # Phase 2: sudden sustained crash so short SMA crosses below long SMA
        rising = [100.0 + i * 0.5 for i in range(50)]  # 100 -> 124.5
        crash = [1.0] * 10
        records = _make_records(rising + crash)
        s = SMAStrategy(short_window=5, long_window=20)
        signals = s.generate_signals(records)
        assert any(sig["Signal"] == "SELL" for sig in signals)

    def test_flat_prices_produce_only_hold(self):
        """
        Assert that a perfectly flat price series (no crossover possible)
        generates only HOLD signals, since the fast and slow SMAs are
        always equal and no crossover event occurs.
        """
        records = _make_records([50.0] * 60)
        s = SMAStrategy(short_window=5, long_window=20)
        signals = s.generate_signals(records)
        assert all(sig["Signal"] == "HOLD" for sig in signals)

    def test_dates_and_closes_are_preserved(self):
        """
        Assert that the 'Date' and 'Close' values in the output exactly
        match the corresponding input records, confirming the strategy
        does not modify or reorder the source data.
        """
        records = _make_records([float(i) for i in range(1, 61)])
        s = SMAStrategy(short_window=5, long_window=10)
        signals = s.generate_signals(records)
        for original, result in zip(records, signals):
            assert result["Date"] == original["Date"]
            assert result["Close"] == original["Close"]
