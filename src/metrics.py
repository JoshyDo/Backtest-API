"""
Calculation of performance metrics for backtest evaluation.
"""

import math
import statistics


def calculate_max_drawdown(portfolio_values: list[float]) -> float:
    """
    Calculates the Maximum Drawdown (MDD) of a portfolio value time series.

    The Maximum Drawdown measures the largest percentage loss from a previous
    peak to a subsequent trough.

    Args:
        portfolio_values: Time-ordered list of daily portfolio values
                          (oldest first). All values must be > 0.

    Returns:
        The Maximum Drawdown as a negative decimal (e.g. -0.34 means -34%).
        Returns 0.0 if the list is empty or no drawdown occurred.

    Raises:
        ValueError: If any value is <= 0.
    """
    if not portfolio_values:
        return 0.0
    if any(v <= 0 for v in portfolio_values):
        raise ValueError("All portfolio values must be > 0.")

    peak = portfolio_values[0]
    max_dd = 0.0
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (value - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def calculate_sharpe_ratio(
    portfolio_values: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculates the annualised Sharpe Ratio based on daily returns.

    Formula (annualised):
        Daily return   r(t)  = (V(t) - V(t-1)) / V(t-1)
        Excess return  er(t) = r(t) - (risk_free_rate / periods_per_year)
        Sharpe Ratio         = mean(er) / std(er) * sqrt(periods_per_year)

    Args:
        portfolio_values: Time-ordered daily portfolio values. Needs >= 2 entries.
        risk_free_rate:   Annual risk-free rate as decimal (e.g. 0.02 for 2%).
        periods_per_year: Trading days per year. Default: 252.

    Returns:
        The annualised Sharpe Ratio. Returns 0.0 if std of returns is 0.

    Raises:
        ValueError: If fewer than 2 values are provided.
    """
    if len(portfolio_values) < 2:
        raise ValueError("At least 2 portfolio values are required.")

    daily_rate = risk_free_rate / periods_per_year
    excess_returns = [
        (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1] - daily_rate
        for i in range(1, len(portfolio_values))
    ]

    avg = sum(excess_returns) / len(excess_returns)
    std = statistics.stdev(excess_returns)

    if std == 0:
        return 0.0

    return (avg / std) * math.sqrt(periods_per_year)
