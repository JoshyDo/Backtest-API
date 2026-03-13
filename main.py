"""
CLI entry point that orchestrates the full backtesting pipeline:
  1. Download/load historical data via yfinance
  2. Generate SMA crossover trading signals
  3. Execute virtual portfolio with commission tracking
  4. Calculate performance metrics and benchmarks
  5. Optionally optimize parameters via grid-search

Usage:
    python main.py                          # Run single backtest
    python main.py                          # Run grid-search optimization (see GRID_SEARCH_ENABLED)

Configuration:
    Edit the constants below to customize backtest parameters.
"""

import logging
from typing import Optional

from src.data_loader import download_historical_data, load_csv_data
from src.metrics import calculate_max_drawdown, calculate_sharpe_ratio
from src.portfolio import Portfolio
from src.strategy import SMAStrategy
from src.optimizer import run_grid_search
from src.cpp_optimizer import is_cpp_available
from src.walk_forward import WalkForwardAnalyzer, print_wfa_summary

# ============================================================================
# CONFIGURATION: Modify these settings to customize the backtest
# ============================================================================

# Data & Backtest Parameters
TICKER = "DAX"  # Stock ticker symbol (e.g., "AAPL", "^GSPC", "BAS.DE")
START_DATE = "2000-01-01"  # Backtest start date (YYYY-MM-DD, inclusive)
END_DATE = "2027-01-01"  # Backtest end date (YYYY-MM-DD, exclusive)
CSV_PATH = f"data/{TICKER}.csv"  # Path to cache downloaded data

INITIAL_CASH = 10_000.0  # Starting capital (USD)
COMMISSION = 0.001  # Transaction fee as decimal (0.1% = 0.001)
SLIPPAGE = 0.002
SPREAD_MIN = 0.001
SPREAD_MAX = 0.002
SHORT_WINDOW = 20  # Fast SMA period (days)
LONG_WINDOW = 50  # Slow SMA period (days)

# Grid Search Optimization Parameters
# Set GRID_SEARCH_ENABLED = True to find optimal SMA parameters
# Otherwise, runs single backtest with SHORT_WINDOW & LONG_WINDOW above
GRID_SEARCH_ENABLED = True

# Parameter ranges for grid search (MUCH more conservative)
GRID_SEARCH_FAST_MIN = 15  # ← Höher
GRID_SEARCH_FAST_MAX = 35  # ← Viel niedriger
GRID_SEARCH_SLOW_MIN = 40  # ← Höher
GRID_SEARCH_SLOW_MAX = 100  # ← Viel niedriger

# Das sind nur 20 × 60 = 1200 Kombinationen
# Viel bessere Generalisierung!

# Walk-Forward Analysis Parameters
# Set WALK_FORWARD_ENABLED = True to use WFA for out-of-sample validation
# Otherwise, runs standard Grid Search on full historical data (in-sample only)
WALK_FORWARD_ENABLED = True

# Walk-Forward window configuration (trading days)
WFA_IS_WINDOW_DAYS = 504  # ← Erhöhe auf 2 Jahre
WFA_OOS_WINDOW_DAYS = 252  # ← Erhöhe auf 1 Jahr
WFA_STEP_SIZE_DAYS = 252  # ← Verschiebe um 1 Jahr
WFA_WARMUP_DAYS = 406  # ← Erhöhe

# ============================================================================
# END CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def run_backtest(
    data: Optional[list[dict]] = None,
    ticker: str = TICKER,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    csv_path: str = CSV_PATH,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION,
    slippage: float = SLIPPAGE,
    spread_min=SPREAD_MIN,
    spread_max=SPREAD_MAX,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW,
    print_results: bool = True,
) -> list[float]:
    """
    Executes a complete backtesting pipeline and returns portfolio value history.

    This function:
      1. Downloads/loads historical OHLCV data
      2. Generates SMA crossover trading signals
      3. Simulates portfolio execution with commission tracking
      4. Calculates performance metrics (Sharpe, max drawdown, etc.)
      5. Prints formatted results (optional)

    Args:
        data: Pre-loaded OHLCV records (if None, downloads from ticker/date range)
        ticker: Stock ticker symbol (e.g., "AAPL", "^GSPC")
        start_date: Backtest start date "YYYY-MM-DD"
        end_date: Backtest end date "YYYY-MM-DD"
        csv_path: Path to save/load CSV data
        initial_cash: Starting capital in USD
        commission: Transaction fee as decimal (0.001 = 0.1%)
        short_window: Fast SMA period in days
        long_window: Slow SMA period in days
        print_results: If True, prints formatted performance summary

    Returns:
        list[float]: Portfolio values at each trading day (for metric calculations)

    Raises:
        ValueError: If data validation fails or parameters are invalid
        FileNotFoundError: If CSV file cannot be loaded and ticker data unavailable
    """
    # 1. Download & load data (only if not provided)
    if data is None:
        log.info("Downloading %s data (%s to %s)...", ticker, start_date, end_date)
        download_historical_data(ticker, start_date, end_date, csv_path)
        records = load_csv_data(csv_path)
        log.info("%d records loaded.", len(records))
    else:
        records = data

    # 2. Generate trading signals
    strategy = SMAStrategy(short_window=short_window, long_window=long_window)
    signals = strategy.generate_signals(records)
    log.debug(
        "%d signals generated (SMA %d/%d).",
        len(signals),
        short_window,
        long_window,
    )

    # 3. Run backtest
    portfolio = Portfolio(
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        spread_min=spread_min,
        spread_max=spread_max,
    )
    portfolio_values: list[float] = []

    for signal in signals:
        if signal["Signal"] == "BUY":
            # Multiplikativ: (1 + spread) * (1 + slippage) * (1 + commission)
            quantity = int(
                portfolio.cash
                // (
                    signal["Close"]
                    * (1 + spread_max)
                    * (1 + slippage)
                    * (1 + commission)
                )
            )
            if quantity > 0:
                portfolio.buy(signal["Date"], signal["Close"], quantity)

        elif signal["Signal"] == "SELL" and portfolio.shares > 0:
            log.debug(
                "SELL %s | Price: %.2f | Shares: %d",
                signal["Date"],
                signal["Close"],
                int(portfolio.shares),
            )
            portfolio.sell(signal["Date"], signal["Close"], portfolio.shares)

        portfolio_values.append(portfolio.get_portfolio_value(signal["Close"]))

    # 4. Performance metrics
    if not portfolio_values:
        log.warning("No portfolio values - nothing to report.")
        return portfolio_values

    mdd = calculate_max_drawdown(portfolio_values)
    sharpe = calculate_sharpe_ratio(portfolio_values)

    # Only print if requested (default True for CLI, False for optimization)
    if print_results:
        # Buy-and-Hold comparison
        first_price = records[0]["Close"]
        last_price = records[-1]["Close"]
        buy_hold_end = (initial_cash / first_price) * last_price

        strategy_return = (portfolio_values[-1] - initial_cash) / initial_cash
        buyhold_return = (buy_hold_end - initial_cash) / initial_cash

        # Calculate annualized return (assuming ~252 trading days per year, 5 years of data)
        num_years = len(records) / 252
        annualized_return = (
            (portfolio_values[-1] / initial_cash) ** (1 / num_years)
        ) - 1

        print("\nBacktest Results")
        print("=" * 50)
        print(f"  SMA Parameters     : ({short_window:>2d}, {long_window:>2d})")
        print(f"  Initial Capital    : $ {initial_cash:>10,.2f}")
        print(f"  Final Value (SMA)  : $ {portfolio_values[-1]:>10,.2f}")
        print(f"  Final Value (B&H)  : $ {buy_hold_end:>10,.2f}")
        print(f"  Return  (SMA)      :   {strategy_return:>10.2%}")
        print(f"  Return  (B&H)      :   {buyhold_return:>10.2%}")
        print(f"  Return per Year    :   {annualized_return:>10.2%}")
        print(f"  Maximum Drawdown   :   {mdd:>10.2%}")
        print(f"  Sharpe Ratio       :   {sharpe:>10.4f}")
        print("=" * 50)

    return portfolio_values


def wfa_grid_search_wrapper(is_data: list[dict]) -> dict:
    """
    Wrapper für Grid Search in Walk-Forward Analysis.
    Nutzt die globalen Grid-Search-Parameter.
    """
    results = run_grid_search(
        data=is_data,
        fast_range=range(GRID_SEARCH_FAST_MIN, GRID_SEARCH_FAST_MAX),
        slow_range=range(GRID_SEARCH_SLOW_MIN, GRID_SEARCH_SLOW_MAX),
        backtest_func=run_backtest,
    )

    # Extrahiere beste Parameter (beste Sharpe Ratio)
    best_result = results["best_sharpe"]

    return {
        "best_short": best_result["short_window"],
        "best_long": best_result["long_window"],
        "best_sharpe": best_result["sharpe_ratio"],
    }


def wfa_backtest_wrapper(
    oos_data: list[dict], short_window: int, long_window: int
) -> list[float]:
    """
    Wrapper für Backtest in Walk-Forward Analysis.
    Gibt Portfolio-Werte zurück für OOS-Validierung.
    """
    portfolio_values = run_backtest(
        data=oos_data,
        short_window=short_window,
        long_window=long_window,
        print_results=False,
    )

    return portfolio_values


if __name__ == "__main__":
    # Single backtest with fixed parameters
    if not GRID_SEARCH_ENABLED:
        run_backtest()

    # Grid-search optimization to find best SMA parameters
    else:
        # Suppress detailed logging during optimization
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("src.data_loader").setLevel(logging.WARNING)
        logging.getLogger("src.strategy").setLevel(logging.WARNING)
        logging.getLogger("src.portfolio").setLevel(logging.WARNING)

        # Load data once
        download_historical_data(TICKER, START_DATE, END_DATE, CSV_PATH)
        data = load_csv_data(CSV_PATH)

        # Calculate grid size
        fast_count = GRID_SEARCH_FAST_MAX - GRID_SEARCH_FAST_MIN
        slow_count = GRID_SEARCH_SLOW_MAX - GRID_SEARCH_SLOW_MIN
        total_combinations = fast_count * slow_count

        engine = "C++ Multithreaded" if is_cpp_available() else "Python"
        print(
            f"\n[Grid Search] {engine} | {len(data)} records | SMA {GRID_SEARCH_FAST_MIN}-{GRID_SEARCH_FAST_MAX-1} x {GRID_SEARCH_SLOW_MIN}-{GRID_SEARCH_SLOW_MAX-1} | {total_combinations} combos"
        )

        best_params = run_grid_search(
            data=data,
            fast_range=range(GRID_SEARCH_FAST_MIN, GRID_SEARCH_FAST_MAX),
            slow_range=range(GRID_SEARCH_SLOW_MIN, GRID_SEARCH_SLOW_MAX),
            backtest_func=run_backtest,
        )

        # Extract best parameters
        best_sharpe = best_params["best_sharpe"]
        best_returns = best_params["best_returns"]

        # Run final backtest with best Sharpe parameters
        print(f"\n{'='*70}")
        print(
            f"BEST SHARPE: SMA ({best_sharpe['short_window']}, {best_sharpe['long_window']})"
        )
        print(f"{'='*70}")
        run_backtest(
            data=data,
            short_window=best_sharpe["short_window"],
            long_window=best_sharpe["long_window"],
            print_results=True,
        )

        # Run final backtest with best Returns parameters (if different)
        if (
            best_returns["short_window"] != best_sharpe["short_window"]
            or best_returns["long_window"] != best_sharpe["long_window"]
        ):
            print(f"\n{'='*70}")
            print(
                f"BEST RETURNS: SMA ({best_returns['short_window']}, {best_returns['long_window']})"
            )
            print(f"{'='*70}")
            run_backtest(
                data=data,
                short_window=best_returns["short_window"],
                long_window=best_returns["long_window"],
                print_results=True,
            )

    # Walk-Forward Analysis
    if WALK_FORWARD_ENABLED:
        print("\n" + "=" * 100)
        print("Starting Walk-Forward Analysis...")
        print("=" * 100)

        analyzer = WalkForwardAnalyzer(
            data=data,
            is_window_days=WFA_IS_WINDOW_DAYS,
            oos_window_days=WFA_OOS_WINDOW_DAYS,
            step_size_days=WFA_STEP_SIZE_DAYS,
            warmup_days=WFA_WARMUP_DAYS,
        )

        wfa_results = analyzer.run(
            grid_search_func=wfa_grid_search_wrapper,  # Wrapper for standard grid search
            backtest_func=wfa_backtest_wrapper,  # Wrapper for backtest
            initial_capital=INITIAL_CASH,
            # Inner Cross-Validation (Layer 2) parameters
            fast_range=range(GRID_SEARCH_FAST_MIN, GRID_SEARCH_FAST_MAX),
            slow_range=range(GRID_SEARCH_SLOW_MIN, GRID_SEARCH_SLOW_MAX),
            use_inner_cv=True,  # Enable inner CV with robustness checks
        )

        print_wfa_summary(wfa_results, initial_capital=INITIAL_CASH)

# TODO: - Dokumentieren in ReadMe
# TODO: - Löschen von .md docs und eine gute md generieren (txt)
