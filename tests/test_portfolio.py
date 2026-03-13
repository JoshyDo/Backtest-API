"""
tests/test_portfolio.py
-----------------------
Unit tests for portfolio.py (Portfolio.buy, Portfolio.sell,
Portfolio.get_portfolio_value).
"""

import pytest
from src.portfolio import Portfolio


# Fixtures


@pytest.fixture
def default_portfolio() -> Portfolio:
    """
    Returns a fresh Portfolio with default arguments:
    initial_cash=10_000.0, commission=0.001.
    Used as the standard starting point for most tests.
    """
    return Portfolio()


@pytest.fixture
def zero_commission_portfolio() -> Portfolio:
    """
    Returns a fresh Portfolio with commission=0.0.
    Used to test buy/sell maths without commission noise,
    making expected values easier to calculate by hand.
    """
    return Portfolio(commission=0.0)


@pytest.fixture
def low_cash_portfolio() -> Portfolio:
    """
    Returns a Portfolio with a very small cash balance (e.g. $1.00).
    Used to trigger the 'insufficient cash' RuntimeError in buy().
    """
    return Portfolio(initial_cash=1.00)


# Tests for Portfolio.buy


class TestBuy:

    def test_cash_decreases_after_buy(self, default_portfolio):
        """
        Assert that portfolio.cash is reduced after a valid buy order.
        After buying, cash must be strictly less than initial_cash.
        """
        before = default_portfolio.cash
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        after = default_portfolio.cash
        assert before > after

    def test_shares_increase_after_buy(self, default_portfolio):
        """
        Assert that portfolio.shares equals the quantity passed to buy()
        after a single buy order on a portfolio that starts with 0 shares.
        """
        quantity = 10
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=quantity)
        assert default_portfolio.shares == quantity

    def test_cost_includes_commission(self, default_portfolio):
        """
        Assert that the cash deducted includes spread, slippage, and commission.
        With randomized spread, we test the range of costs.
        """
        price, quantity = 100.0, 10
        before = default_portfolio.cash
        default_portfolio.buy(date="2024-01-01", price=price, quantity=quantity)

        # The actual cost depends on the random spread, but should be within range:
        # min: quantity * price * (1 + spread_min) * (1 + slippage) * (1 + commission)
        # max: quantity * price * (1 + spread_max) * (1 + slippage) * (1 + commission)
        min_cost = (
            quantity
            * price
            * (1 + default_portfolio.spread_min)
            * (1 + default_portfolio.slippage)
            * (1 + default_portfolio.commission)
        )
        max_cost = (
            quantity
            * price
            * (1 + default_portfolio.spread_max)
            * (1 + default_portfolio.slippage)
            * (1 + default_portfolio.commission)
        )

        actual_cost = before - default_portfolio.cash
        assert min_cost <= actual_cost <= max_cost

    def test_buy_records_transaction(self, default_portfolio):
        """
        Assert that portfolio.transactions contains exactly one entry after
        a single buy(), and that its 'Type' key equals 'BUY'.
        """
        default_portfolio.buy(date="2024-01-01", price=100, quantity=10)
        assert len(default_portfolio.transactions) == 1
        assert default_portfolio.transactions[0]["Type"] == "BUY"

    def test_buy_raises_on_insufficient_cash(self, low_cash_portfolio):
        """
        Assert that a RuntimeError is raised when the total cost of the order
        exceeds the available cash balance.
        Use pytest.raises(RuntimeError) as the context manager.
        """
        with pytest.raises(RuntimeError):
            low_cash_portfolio.buy(date="2024-01-01", price=100000, quantity=10)

    def test_buy_raises_on_non_positive_quantity(self, default_portfolio):
        """
        Assert that a ValueError is raised when quantity <= 0 is passed
        to buy(), before any cash or share balance is modified.
        Use pytest.raises(ValueError) as the context manager.
        """
        with pytest.raises(ValueError):
            default_portfolio.buy(date="2024-01-01", price=10, quantity=-100)


# Tests for Portfolio.sell


class TestSell:

    def test_cash_increases_after_sell(self, default_portfolio):
        """Cash should be higher after selling some shares."""
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        before = default_portfolio.cash
        default_portfolio.sell(date="2024-01-02", price=100.0, quantity=10)
        assert default_portfolio.cash > before

    def test_shares_decrease_after_sell(self, default_portfolio):
        """Shares decrease by exactly the sold quantity."""
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        default_portfolio.sell(date="2024-01-02", price=100.0, quantity=4)
        assert default_portfolio.shares == 6

    def test_proceeds_exclude_commission(self, zero_commission_portfolio):
        """With commission=0 but default spread/slippage, there is a small loss."""
        price, quantity = 100.0, 10
        zero_commission_portfolio.buy(date="2024-01-01", price=price, quantity=quantity)
        zero_commission_portfolio.sell(
            date="2024-01-02", price=price, quantity=quantity
        )
        assert zero_commission_portfolio.cash < 10000.0

    def test_sell_records_transaction(self, default_portfolio):
        """Second transaction should be a SELL record."""
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        default_portfolio.sell(date="2024-01-02", price=100.0, quantity=10)
        assert len(default_portfolio.transactions) == 2
        assert default_portfolio.transactions[1]["Type"] == "SELL"

    def test_sell_raises_on_insufficient_shares(self, default_portfolio):
        """Selling more than held shares should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            default_portfolio.sell(date="2024-01-01", price=100.0, quantity=1)

    def test_sell_raises_on_non_positive_quantity(self, default_portfolio):
        """Non-positive quantity should raise ValueError."""
        with pytest.raises(ValueError):
            default_portfolio.sell(date="2024-01-01", price=100.0, quantity=-5)


def test_sell_with_spread_and_slippage_loss(zero_commission_portfolio):
    """End-to-end check that spread/slippage cause loss even with zero commission."""
    price, quantity = 100.0, 10
    zero_commission_portfolio.buy(date="2024-01-01", price=price, quantity=quantity)
    zero_commission_portfolio.sell(
        date="2024-01-02", price=price, quantity=quantity
    )
    assert zero_commission_portfolio.cash < 10000.0


# Tests for Portfolio.get_portfolio_value


class TestGetPortfolioValue:

    def test_equals_cash_when_no_shares(self, default_portfolio):
        """
        Assert that get_portfolio_value(price) returns exactly portfolio.cash
        when portfolio.shares == 0, regardless of the price argument.
        """
        assert default_portfolio.get_portfolio_value(999.0) == default_portfolio.cash

    def test_includes_share_value(self, default_portfolio):
        """
        Assert that get_portfolio_value(price) equals
        portfolio.cash + portfolio.shares * price after a buy order.
        Pick a specific price and quantity to make the expected value
        easy to compute by hand.
        """
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        expected = default_portfolio.cash + default_portfolio.shares * 150.0
        assert default_portfolio.get_portfolio_value(150.0) == pytest.approx(expected)

    def test_value_changes_with_price(self, default_portfolio):
        """
        Assert that get_portfolio_value() returns different results for
        two different price arguments when portfolio.shares > 0.
        A higher price must produce a higher portfolio value.
        """
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        assert default_portfolio.get_portfolio_value(
            200.0
        ) > default_portfolio.get_portfolio_value(100.0)


# Tests for Spread and Slippage


class TestSpreadAndSlippage:

    @pytest.fixture
    def portfolio_with_spread_slippage(self) -> Portfolio:
        """
        Returns a Portfolio with fixed spread_min/spread_max and slippage.
        Used to test deterministic spread/slippage behavior.
        spread_min=0.001, spread_max=0.001 → fixed spread of 0.001 (0.1%)
        slippage=0.002 (0.2%)
        commission=0.001 (0.1%)
        """
        return Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.001,  # Fixed spread to make testing deterministic
        )

    def test_buy_cost_includes_spread_slippage_commission(
        self, portfolio_with_spread_slippage
    ):
        """
        Assert that buy cost equals:
        quantity * price * (1 + spread) * (1 + slippage) * (1 + commission)

        With spread=0.001, slippage=0.002, commission=0.001:
        price_multiplier = (1.001) * (1.002) * (1.001) ≈ 1.004006
        quantity=10, price=100.0
        expected_cost = 10 * 100.0 * 1.004006 ≈ 1004.006
        """
        price, quantity = 100.0, 10
        before = portfolio_with_spread_slippage.cash

        portfolio_with_spread_slippage.buy(
            date="2024-01-01", price=price, quantity=quantity
        )

        # Expected cost: quantity * price * (1 + spread) * (1 + slippage) * (1 + commission)
        # = 10 * 100 * (1 + 0.001) * (1 + 0.002) * (1 + 0.001)
        # = 1000 * 1.001 * 1.002 * 1.001
        expected_cost = quantity * price * (1.001) * (1.002) * (1.001)

        assert portfolio_with_spread_slippage.cash == pytest.approx(
            before - expected_cost, rel=1e-5
        )

    def test_sell_proceeds_includes_spread_slippage_commission(
        self, portfolio_with_spread_slippage
    ):
        """
        Assert that sell proceeds equals:
        quantity * price * (1 - spread) * (1 - slippage) * (1 - commission)

        With spread=0.001, slippage=0.002, commission=0.001:
        price_multiplier = (1 - 0.001) * (1 - 0.002) * (1 - 0.001) ≈ 0.995997
        Buy 10 shares at $100, then sell all at $100.
        Net proceeds ≈ 10 * 100 * 0.995997 ≈ 995.997
        """
        price, quantity = 100.0, 10

        # First, buy shares
        portfolio_with_spread_slippage.buy(
            date="2024-01-01", price=price, quantity=quantity
        )
        cash_after_buy = portfolio_with_spread_slippage.cash

        # Then, sell all shares
        portfolio_with_spread_slippage.sell(
            date="2024-01-02", price=price, quantity=quantity
        )

        # Expected proceeds: quantity * price * (1 - spread) * (1 - slippage) * (1 - commission)
        # = 10 * 100 * (1 - 0.001) * (1 - 0.002) * (1 - 0.001)
        # = 1000 * 0.999 * 0.998 * 0.999
        expected_proceeds = quantity * price * (0.999) * (0.998) * (0.999)

        assert portfolio_with_spread_slippage.cash == pytest.approx(
            cash_after_buy + expected_proceeds, rel=1e-5
        )

    def test_buy_with_spread_raises_insufficient_cash(self):
        """
        Assert that when spread increases the effective price significantly,
        an otherwise affordable order becomes unaffordable.
        With high spread (0.1), a buy that would fit without spread may fail.
        """
        portfolio = Portfolio(
            initial_cash=1_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.1,
            spread_max=0.1,
        )

        # Try to buy 10 shares at $100 with high spread (0.1)
        # price_multiplier ≈ (1.1) * (1.002) * (1.001) ≈ 1.1033
        # cost ≈ 10 * 100 * 1.1033 ≈ 1103.3
        # available cash = 1000.0 → should fail
        with pytest.raises(RuntimeError):
            portfolio.buy(date="2024-01-01", price=100.0, quantity=10)

    def test_buy_with_variable_spread(self):
        """
        Assert that spread varies between spread_min and spread_max
        across multiple buy() calls. Since spread is random, we test
        that costs differ across multiple buys (with high probability).
        """
        portfolio = Portfolio(
            initial_cash=100_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.01,  # Much larger range
        )

        costs = []
        for i in range(10):
            before = portfolio.cash
            portfolio.buy(date=f"2024-01-0{i+1}", price=100.0, quantity=1)
            cost = before - portfolio.cash
            costs.append(cost)

        # With 10 buys and high spread variance, at least some costs should differ
        assert len(set([round(c, 4) for c in costs])) > 1

    def test_sell_with_variable_spread(self):
        """
        Assert that spread varies between spread_min and spread_max
        across multiple sell() calls. Proceeds should differ due to
        random spread variation.
        """
        portfolio = Portfolio(
            initial_cash=100_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.001,
            spread_max=0.01,
        )

        # Buy 100 shares first
        portfolio.buy(date="2024-01-01", price=100.0, quantity=100)

        proceeds = []
        for i in range(10):
            before = portfolio.cash
            portfolio.sell(date=f"2024-01-0{i+1}", price=100.0, quantity=1)
            proceed = portfolio.cash - before
            proceeds.append(proceed)

        # With 10 sells and high spread variance, at least some proceeds should differ
        assert len(set([round(p, 4) for p in proceeds])) > 1

    def test_buy_then_sell_includes_all_costs(self):
        """
        Assert that buying and selling at the same price results in
        lower portfolio value due to accumulated spread, slippage, and
        commission costs across both transactions.
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.002,
            spread_max=0.002,
        )

        initial_value = portfolio.get_portfolio_value(100.0)

        # Buy at $100
        portfolio.buy(date="2024-01-01", price=100.0, quantity=10)

        # Sell at $100 (same price)
        portfolio.sell(date="2024-01-02", price=100.0, quantity=10)

        final_value = portfolio.get_portfolio_value(100.0)

        # Final value should be less than initial due to all transaction costs
        assert final_value < initial_value

    def test_spread_slippage_commission_order_matters(self):
        """
        Assert that the order of applying costs (spread, slippage, commission)
        is multiplicative and matches the formula:
        price * (1 ± spread) * (1 ± slippage) * (1 ± commission)

        This test verifies the difference between:
        - Additive: 1 + 0.002 + 0.002 + 0.001 = 1.005
        - Multiplicative: 1.002 * 1.002 * 1.001 ≈ 1.005006
        """
        portfolio = Portfolio(
            initial_cash=10_000.0,
            commission=0.001,
            slippage=0.002,
            spread_min=0.002,
            spread_max=0.002,
        )

        price, quantity = 100.0, 10
        before = portfolio.cash

        portfolio.buy(date="2024-01-01", price=price, quantity=quantity)

        # Multiplicative formula
        multiplicative_cost = quantity * price * (1.002) * (1.002) * (1.001)

        # Verify actual cost matches multiplicative approach
        actual_cost = before - portfolio.cash
        assert actual_cost == pytest.approx(multiplicative_cost, rel=1e-5)
