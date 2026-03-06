"""
Technical indicators for the backtesting engine.
"""


def calculate_sma(prices: list[float], window: int) -> list[float | None]:
    """
    Calculates the Simple Moving Average (SMA) for a given price series.

    Uses a rolling sum for O(n) performance instead of re-summing each window.

    Args:
        prices: Ordered list of closing prices (oldest first).
        window: Number of periods for the moving average. Must be >= 1.

    Returns:
        A list of the same length as `prices`. The first `window - 1` entries
        are None (insufficient data); the rest contain the SMA value.

    Raises:
        ValueError: If `window` < 1 or `window` > len(prices).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > len(prices):
        raise ValueError(f"window ({window}) exceeds number of prices ({len(prices)})")

    sma: list[float | None] = [None] * (window - 1)
    rolling_sum = sum(prices[:window])
    sma.append(rolling_sum / window)

    for i in range(window, len(prices)):
        rolling_sum += prices[i] - prices[i - window]
        sma.append(rolling_sum / window)

    return sma
