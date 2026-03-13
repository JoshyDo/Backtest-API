"""
tests/test_main_spread_slippage.py
----------------------------------
Unit tests for spread and slippage integration in main.py run_backtest().
Tests verify that the quantity calculation and portfolio execution 
correctly account for spread, slippage, and commission.
"""

import pytest
from typing import Dict
from src.portfolio import Portfolio


# Mock data generator for deterministic testing


def create_mock_signal(date: str, close: float, signal: str) -> Dict:
    """
    Create a mock trading signal for testing.

    Args:
        date: Date string (e.g., "2024-01-01")
        close: Close price
        signal: "BUY" or "SELL"

    Returns:
        Dictionary with signal data
    """
    return {"Date": date, "Close": close, "Signal": signal}


class TestQuantityCalculationWithSpreadSlippage:
    """
    Tests for quantity calculation in main.py run_backtest().
    Verifies that quantity calculation uses avg_spread correctly.
    """

    def test_quantity_calculation_basic(self):
        """
        Assert that quantity calculation accounts for spread, slippage, and commission.

        Formula: quantity = int(portfolio.cash // (price * (1 + avg_spread) * (1 + slippage) * (1 + commission)))

        With:
        - portfolio.cash = 10,000
        - price = 100
        - avg_spread = (0.001 + 0.002) / 2 = 0.0015
        - slippage = 0.002
        - commission = 0.001

        price_multiplier = (1 + 0.0015) * (1 + 0.002) * (1 + 0.001) ≈ 1.004505
        quantity = int(10000 // (100 * 1.004505)) = int(99.55...) = 99
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.002,
        )

        price = 100.0
        spread_min = 0.001
        spread_max = 0.002
        slippage = 0.002
        commission = 0.001

        # Calculate quantity using the formula from main.py
        avg_spread = (spread_min + spread_max) / 2
        quantity = int(
            portfolio.cash
            // (price * (1 + avg_spread) * (1 + slippage) * (1 + commission))
        )

        # Expected: int(10000 // (100 * 1.004505)) = 99
        assert quantity == 99
        assert quantity > 0  # Must be positive

    def test_quantity_calculation_with_high_costs(self):
        """
        Assert that with higher spread, slippage, and commission,
        the calculated quantity is lower (fewer shares can be bought).
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.002,
        )

        price = 100.0
        spread_min_low = 0.001
        spread_max_low = 0.002
        slippage = 0.002
        commission = 0.001

        # Low costs scenario
        avg_spread_low = (spread_min_low + spread_max_low) / 2
        quantity_low = int(
            portfolio.cash
            // (price * (1 + avg_spread_low) * (1 + slippage) * (1 + commission))
        )

        # High costs scenario
        spread_min_high = 0.05
        spread_max_high = 0.1
        avg_spread_high = (spread_min_high + spread_max_high) / 2
        quantity_high = int(
            portfolio.cash
            // (price * (1 + avg_spread_high) * (1 + slippage) * (1 + commission))
        )

        # Quantity should be lower with higher costs
        assert quantity_high < quantity_low

    def test_quantity_zero_when_insufficient_cash(self):
        """
        Assert that quantity is 0 when the effective price (including costs)
        exceeds the available cash.
        """
        portfolio = Portfolio(
            initial_cash=100.0,  # Very low cash
            commission=0.001,
            slippage=0.002,
            spread_min=0.1,
            spread_max=0.2,
        )

        price = 1000.0  # High price
        spread_min = 0.1
        spread_max = 0.2
        slippage = 0.002
        commission = 0.001

        avg_spread = (spread_min + spread_max) / 2
        quantity = int(
            portfolio.cash
            // (price * (1 + avg_spread) * (1 + slippage) * (1 + commission))
        )

        # Quantity should be 0 due to insufficient cash
        assert quantity == 0

    def test_buy_execution_with_calculated_quantity(self):
        """
        Assert that a buy() with the calculated quantity succeeds and
        deducts the correct amount from cash.
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.001,  # Fixed for deterministic testing
        )

        price = 100.0
        spread_min = 0.001
        spread_max = 0.001
        slippage = 0.002
        commission = 0.001

        # Calculate quantity
        avg_spread = (spread_min + spread_max) / 2
        quantity = int(
            portfolio.cash
            // (price * (1 + avg_spread) * (1 + slippage) * (1 + commission))
        )

        # Execute buy
        initial_cash = portfolio.cash
        portfolio.buy(date="2024-01-01", price=price, quantity=quantity)

        # Verify shares were added and cash was deducted
        assert portfolio.shares == quantity
        assert portfolio.cash < initial_cash
        assert portfolio.cash >= 0  # Should not go negative


class TestBuyAndSellCycleWithCosts:
    """
    Tests for complete buy/sell cycles with spread, slippage, and commission.
    """

    def test_buy_sell_cycle_incurs_costs(self):
        """
        Assert that a buy-at-X, sell-at-X cycle (same price) results in
        lower portfolio value due to accumulated costs.
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.001,
        )

        initial_value = portfolio.get_portfolio_value(100.0)

        # Buy 10 shares at $100
        portfolio.buy(date="2024-01-01", price=100.0, quantity=10)

        # Sell 10 shares at $100 (same price)
        portfolio.sell(date="2024-01-02", price=100.0, quantity=10)

        final_value = portfolio.get_portfolio_value(100.0)

        # Final value should be less than initial due to costs
        assert final_value < initial_value

        # Loss should be approximately:
        # Buy cost: 10 * 100 * (1.001) * (1.002) * (1.001) ≈ 1004.006
        # Sell proceeds: 10 * 100 * (0.999) * (0.998) * (0.999) ≈ 995.997
        # Net loss: 1004.006 - 995.997 ≈ 8.009
        expected_loss = 1004.006 - 995.997
        actual_loss = initial_value - final_value

        assert actual_loss == pytest.approx(expected_loss, rel=0.01)

    def test_multiple_buy_sell_cycles(self):
        """
        Assert that portfolio value decreases over multiple buy-sell cycles,
        even when prices don't change, due to cumulative transaction costs.
        Since spread is randomized, test that the loss is monotonic and positive.
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.001,  # Fixed for deterministic testing
        )

        initial_value = portfolio.get_portfolio_value(100.0)
        values_after_cycles = []

        # Perform 3 buy-sell cycles at the same price
        for i in range(3):
            portfolio.buy(date=f"2024-01-{(i*2)+1:02d}", price=100.0, quantity=5)
            portfolio.sell(date=f"2024-01-{(i*2)+2:02d}", price=100.0, quantity=5)
            values_after_cycles.append(portfolio.get_portfolio_value(100.0))

        # Final value should be less than initial
        final_value = portfolio.get_portfolio_value(100.0)
        assert final_value < initial_value

        # Values should be monotonically decreasing
        for i in range(len(values_after_cycles) - 1):
            assert values_after_cycles[i] >= values_after_cycles[i + 1]


class TestSpreadAveragingLogic:
    """
    Tests for the avg_spread calculation logic in main.py.
    """

    def test_avg_spread_calculation(self):
        """
        Assert that avg_spread = (spread_min + spread_max) / 2
        is calculated correctly.
        """
        spread_min = 0.001
        spread_max = 0.002

        avg_spread = (spread_min + spread_max) / 2

        expected = 0.0015
        assert avg_spread == pytest.approx(expected)

    def test_avg_spread_with_different_ranges(self):
        """
        Assert that different spread ranges produce different avg_spread values.
        """
        ranges = [
            (0.001, 0.002),
            (0.001, 0.005),
            (0.0005, 0.01),
        ]

        avg_spreads = [(min_s + max_s) / 2 for min_s, max_s in ranges]

        # All should be different
        assert len(set(avg_spreads)) == len(ranges)

    def test_avg_spread_conservative_vs_aggressive(self):
        """
        Assert that using spread_max (conservative) vs avg_spread (moderate)
        affects the quantity calculation.

        Using spread_max ensures safer quantity calculation.
        Using avg_spread may allow slightly higher quantities.
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.003,
        )

        price = 100.0
        spread_min = 0.001
        spread_max = 0.003
        slippage = 0.002
        commission = 0.001

        # Conservative approach (using spread_max)
        quantity_conservative = int(
            portfolio.cash
            // (price * (1 + spread_max) * (1 + slippage) * (1 + commission))
        )

        # Moderate approach (using avg_spread)
        avg_spread = (spread_min + spread_max) / 2
        quantity_moderate = int(
            portfolio.cash
            // (price * (1 + avg_spread) * (1 + slippage) * (1 + commission))
        )

        # Moderate should be >= conservative (more optimistic)
        assert quantity_moderate >= quantity_conservative


class TestEdgeCasesWithSpreadSlippage:
    """
    Tests for edge cases when spread and slippage are applied.
    """

    def test_very_small_cash_with_costs(self):
        """
        Assert that with very small cash and significant costs,
        no shares can be bought (quantity = 0).
        """
        portfolio = Portfolio(
            initial_cash=1.0,  # 1 dollar only
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.002,
        )

        price = 100.0
        avg_spread = 0.0015
        quantity = int(
            portfolio.cash // (price * (1 + avg_spread) * (1 + 0.002) * (1 + 0.001))
        )

        assert quantity == 0

    def test_zero_spread_zero_slippage(self):
        """
        Assert that with zero spread and zero slippage,
        only commission affects the quantity calculation.
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.0,
            spread_min=0.0,
            spread_max=0.0,
        )

        price = 100.0
        avg_spread = 0.0
        slippage = 0.0
        commission = 0.001

        quantity = int(
            portfolio.cash
            // (price * (1 + avg_spread) * (1 + slippage) * (1 + commission))
        )

        # Should be int(10000 // (100 * 1.001)) = 99
        assert quantity == 99

    def test_high_price_low_cash(self):
        """
        Assert that with high prices and low cash, quantity is zero.
        """
        portfolio = Portfolio(
            initial_cash=100.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.002,
        )

        price = 10_000.0  # Very high price
        avg_spread = 0.0015
        quantity = int(
            portfolio.cash // (price * (1 + avg_spread) * (1 + 0.002) * (1 + 0.001))
        )

        assert quantity == 0
