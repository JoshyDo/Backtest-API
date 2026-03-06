"""
tests/test_portfolio.py
-----------------------
Unit tests for portfolio.py (Portfolio.buy, Portfolio.sell,
Portfolio.get_portfolio_value).
"""

import pytest
from portfolio import Portfolio


# ── Fixtures ───────────────────────────────────────────────────────────────────

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


# ── Portfolio.buy ──────────────────────────────────────────────────────────────

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
        Assert that the cash deducted equals quantity * price * (1 + commission),
        confirming commission is factored into the total cost correctly.
        """
        price, quantity = 100.0, 10
        expected_cost = quantity * price * (1 + default_portfolio.commission)
        before = default_portfolio.cash
        default_portfolio.buy(date="2024-01-01", price=price, quantity=quantity)
        assert default_portfolio.cash == pytest.approx(before - expected_cost)

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


# ── Portfolio.sell ─────────────────────────────────────────────────────────────

class TestSell:

    def test_cash_increases_after_sell(self, default_portfolio):
        """
        Assert that portfolio.cash is greater after a sell() than before it.
        Set up shares first with a buy(), then sell() and compare cash.
        """
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        before = default_portfolio.cash
        default_portfolio.sell(date="2024-01-02", price=100.0, quantity=10)
        assert default_portfolio.cash > before

    def test_shares_decrease_after_sell(self, default_portfolio):
        """
        Assert that portfolio.shares is reduced by exactly the quantity
        passed to sell().
        Set up shares first with a buy(), then sell a subset and check.
        """
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        default_portfolio.sell(date="2024-01-02", price=100.0, quantity=4)
        assert default_portfolio.shares == 6

    def test_proceeds_exclude_commission(self, zero_commission_portfolio):
        """
        Assert that with commission=0.0, cash after selling equals
        quantity * price (net proceeds = quantity * price * (1 - 0)).
        First buy shares, then sell them, then check the cash balance.
        """
        price, quantity = 100.0, 10
        zero_commission_portfolio.buy(date="2024-01-01", price=price, quantity=quantity)
        zero_commission_portfolio.sell(date="2024-01-02", price=price, quantity=quantity)
        # With no commission, selling at the same price restores cash exactly
        assert zero_commission_portfolio.cash == pytest.approx(10_000.0)

    def test_sell_records_transaction(self, default_portfolio):
        """
        Assert that after one buy() and one sell(), portfolio.transactions
        has exactly two entries, and the second entry's 'Type' equals 'SELL'.
        """
        default_portfolio.buy(date="2024-01-01", price=100.0, quantity=10)
        default_portfolio.sell(date="2024-01-02", price=100.0, quantity=10)
        assert len(default_portfolio.transactions) == 2
        assert default_portfolio.transactions[1]["Type"] == "SELL"

    def test_sell_raises_on_insufficient_shares(self, default_portfolio):
        """
        Assert that a RuntimeError is raised when the quantity to sell
        exceeds portfolio.shares (including selling from a 0-share portfolio).
        Use pytest.raises(RuntimeError) as the context manager.
        """
        with pytest.raises(RuntimeError):
            default_portfolio.sell(date="2024-01-01", price=100.0, quantity=1)

    def test_sell_raises_on_non_positive_quantity(self, default_portfolio):
        """
        Assert that a ValueError is raised when quantity <= 0 is passed
        to sell(), before any cash or share balance is modified.
        Use pytest.raises(ValueError) as the context manager.
        """
        with pytest.raises(ValueError):
            default_portfolio.sell(date="2024-01-01", price=100.0, quantity=-5)


# ── Portfolio.get_portfolio_value ──────────────────────────────────────────────

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
        assert default_portfolio.get_portfolio_value(200.0) > default_portfolio.get_portfolio_value(100.0)
