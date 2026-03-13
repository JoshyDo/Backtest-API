"""
Trading signal generation using SMA crossover logic.
"""

from .indicators import calculate_sma


class SMAStrategy:
    """
    SMA crossover strategy that generates buy/sell signals.

    - BUY  (Golden Cross): fast SMA crosses above slow SMA.
    - SELL (Death Cross):  fast SMA crosses below slow SMA.
    - HOLD: no crossover event.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be less than long_window ({long_window})"
            )
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, records: list[dict]) -> list[dict]:
        """
        Assigns a trading signal to each day based on SMA crossover detection.

        Args:
            records: OHLCV dicts from load_csv_data(). Each must have "Date" and "Close".

        Returns:
            List of dicts with keys: Date, Close, Signal ("BUY"/"SELL"/"HOLD").

        Raises:
            ValueError: If records has fewer entries than long_window.
        """
        if len(records) < self.long_window:
            raise ValueError(
                f"Need at least {self.long_window} records, got {len(records)}"
            )

        prices = [record["Close"] for record in records]
        short_sma = calculate_sma(prices, self.short_window)
        long_sma = calculate_sma(prices, self.long_window)

        result = []
        for i, day in enumerate(records):
            s_curr = short_sma[i]
            l_curr = long_sma[i]
            s_prev = short_sma[i - 1] if i > 0 else None
            l_prev = long_sma[i - 1] if i > 0 else None

            if s_curr is None or l_curr is None or s_prev is None or l_prev is None:
                signal = "HOLD"
            elif s_curr > l_curr and s_prev < l_prev:
                signal = "BUY"
            elif s_curr < l_curr and s_prev > l_prev:
                signal = "SELL"
            else:
                signal = "HOLD"

            result.append(
                {"Date": day["Date"], "Close": day["Close"], "Signal": signal}
            )

        return result
