import random
"""
Manages the virtual portfolio (cash, positions, transaction costs).
"""


class Portfolio:
    """
    Simulates a virtual portfolio for backtesting.

    Tracks cash balance, share holdings, and all executed transactions.
    Every buy/sell is charged a percentage commission.
    """

    def __init__(self, initial_cash: float = 10_000.0, commission: float = 0.001, 
                 slippage: float = 0.002, spread_min: float = 0.001, 
                 spread_max: float = 0.003) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.shares: float = 0.0
        self.commission = commission
        self.slippage = slippage
        self.spread_min = spread_min
        self.spread_max = spread_max
        self.transactions: list[dict] = []

    def buy(self, date: str, price: float, quantity: float) -> None:
        """
        Executes a buy order: deducts cost from cash, adds shares.

        Raises:
            ValueError:  If quantity <= 0.
            RuntimeError: If insufficient cash.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")

        spread = random.uniform(self.spread_min, self.spread_max)
        price_with_all_costs = price * (1 + spread) * (1 + self.slippage) * (1 + self.commission)
        total_cost = quantity * price_with_all_costs

        if self.cash < total_cost:
            raise RuntimeError(
                f"Insufficient capital: ${self.cash:.2f} available, but ${total_cost:.2f} required."
            )

        self.cash -= total_cost
        self.shares += quantity
        self.transactions.append({
            "Date":     date,
            "Type":     "BUY",
            "Price":    price,
            "Quantity": quantity,
            "Cost":     total_cost
        })

    def sell(self, date: str, price: float, quantity: float) -> None:
        """
        Executes a sell order: adds proceeds to cash, removes shares.

        Raises:
            ValueError:  If quantity <= 0.
            RuntimeError: If insufficient shares.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")

        if self.shares < quantity:
            raise RuntimeError(
                f"Insufficient shares: {self.shares} held, but {quantity} requested."
            )
        spread = random.uniform(self.spread_min, self.spread_max)
        price_with_all_costs = price * (1 - spread) * (1 - self.slippage) * (1 - self.commission)
        net_proceeds = quantity * price_with_all_costs

        self.cash += net_proceeds
        self.shares -= quantity
        self.transactions.append({
            "Date":     date,
            "Type":     "SELL",
            "Price":    price,
            "Quantity": quantity,
            "Proceeds": net_proceeds
        })

    def get_portfolio_value(self, current_price: float) -> float:
        """Returns total portfolio value: cash + shares * current_price."""
        return self.cash + self.shares * current_price