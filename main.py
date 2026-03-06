"""
main.py
-------
Entry point of the backtesting engine.
Orchestrates the full pipeline from data download to performance evaluation.
"""

import logging

from data_loader import download_historical_data, load_csv_data
from metrics import calculate_max_drawdown, calculate_sharpe_ratio
from portfolio import Portfolio
from strategy import SMAStrategy

# ── Configuration ──────────────────────────────────────────────────────────────
TICKER = "AAPL"
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"
CSV_PATH = f"data/{TICKER}.csv"

INITIAL_CASH = 10_000.0  # Starting capital in USD
COMMISSION = 0.001  # Transaction fee: 0.1 %
SHORT_WINDOW = 20  # Short-term SMA period (days)
LONG_WINDOW = 50  # Long-term SMA period (days)
# ───────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    # 1. Download & load data
    log.info("Downloading %s data (%s to %s)...", TICKER, START_DATE, END_DATE)
    download_historical_data(TICKER, START_DATE, END_DATE, CSV_PATH)

    records = load_csv_data(CSV_PATH)
    log.info("%d records loaded.", len(records))

    # 2. Generate trading signals
    strategy = SMAStrategy(short_window=SHORT_WINDOW, long_window=LONG_WINDOW)
    signals = strategy.generate_signals(records)
    log.info(
        "%d signals generated (SMA %d/%d).",
        len(signals), SHORT_WINDOW, LONG_WINDOW,
    )

    # 3. Run backtest
    portfolio = Portfolio(initial_cash=INITIAL_CASH, commission=COMMISSION)
    portfolio_values: list[float] = []

    for signal in signals:
        if signal["Signal"] == "BUY":
            quantity = int(portfolio.cash // (signal["Close"] * (1 + portfolio.commission)))
            if quantity > 0:
                log.info(
                    "BUY  %s | Price: %.2f | Shares: %d",
                    signal["Date"], signal["Close"], quantity,
                )
                portfolio.buy(signal["Date"], signal["Close"], quantity)

        elif signal["Signal"] == "SELL" and portfolio.shares > 0:
            log.info(
                "SELL %s | Price: %.2f | Shares: %d",
                signal["Date"], signal["Close"], int(portfolio.shares),
            )
            portfolio.sell(signal["Date"], signal["Close"], portfolio.shares)

        portfolio_values.append(portfolio.get_portfolio_value(signal["Close"]))

    # 4. Performance metrics
    if not portfolio_values:
        log.warning("No portfolio values — nothing to report.")
        return

    mdd = calculate_max_drawdown(portfolio_values)
    sharpe = calculate_sharpe_ratio(portfolio_values)

    # Buy-and-Hold comparison
    first_price = records[0]["Close"]
    last_price = records[-1]["Close"]
    buy_hold_end = (INITIAL_CASH / first_price) * last_price

    strategy_return = (portfolio_values[-1] - INITIAL_CASH) / INITIAL_CASH
    buyhold_return = (buy_hold_end - INITIAL_CASH) / INITIAL_CASH

    print("\n── Backtest Results ───────────────────────────")
    print(f"  Initial Capital    : $ {INITIAL_CASH:>10,.2f}")
    print(f"  Final Value (SMA)  : $ {portfolio_values[-1]:>10,.2f}")
    print(f"  Final Value (B&H)  : $ {buy_hold_end:>10,.2f}")
    print(f"  Return  (SMA)      :   {strategy_return:>10.2%}")
    print(f"  Return  (B&H)      :   {buyhold_return:>10.2%}")
    print(f"  Maximum Drawdown   :   {mdd:>10.2%}")
    print(f"  Sharpe Ratio       :   {sharpe:>10.4f}")
    print("───────────────────────────────────────────────")


if __name__ == "__main__":
    main()
