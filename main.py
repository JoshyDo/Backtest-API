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

# ============================================================================
# CONFIGURATION: Modify these settings to customize the backtest
# ============================================================================

# Data & Backtest Parameters
TICKER = "DAX"                      # Stock ticker symbol (e.g., "AAPL", "^GSPC", "BAS.DE")
START_DATE = "2000-01-01"           # Backtest start date (YYYY-MM-DD, inclusive)
END_DATE = "2027-01-01"             # Backtest end date (YYYY-MM-DD, exclusive)
CSV_PATH = f"data/{TICKER}.csv"     # Path to cache downloaded data

INITIAL_CASH = 10_000.0             # Starting capital (USD)
COMMISSION = 0.001                  # Transaction fee as decimal (0.1% = 0.001)
SLIPPAGE = 0.002
SPREAD_MIN = 0.001
SPREAD_MAX = 0.002
SHORT_WINDOW = 20                   # Fast SMA period (days)
LONG_WINDOW = 50                    # Slow SMA period (days)

# Grid Search Optimization Parameters
# Set GRID_SEARCH_ENABLED = True to find optimal SMA parameters
# Otherwise, runs single backtest with SHORT_WINDOW & LONG_WINDOW above
GRID_SEARCH_ENABLED = True

# Parameter ranges for grid search (exclusive upper bounds)
GRID_SEARCH_FAST_MIN = 5            # Short SMA: minimum days
GRID_SEARCH_FAST_MAX = 100          # Short SMA: maximum days (exclusive)
GRID_SEARCH_SLOW_MIN = 15           # Long SMA: minimum days
GRID_SEARCH_SLOW_MAX = 500          # Long SMA: maximum days (exclusive)

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
    spread_min = SPREAD_MIN,
    spread_max = SPREAD_MAX,
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
        len(signals), short_window, long_window,
    )

    # 3. Run backtest
    portfolio = Portfolio(initial_cash=initial_cash, commission=commission, slippage=slippage, 
                          spread_min=spread_min, spread_max=spread_max)
    portfolio_values: list[float] = []

    for signal in signals:
        if signal["Signal"] == "BUY":
            # Multiplikativ: (1 + spread) * (1 + slippage) * (1 + commission)
            quantity = int(portfolio.cash // (signal["Close"] * (1 + spread_max) * (1 + slippage) * (1 + commission)))
            if quantity > 0:
                portfolio.buy(signal["Date"], signal["Close"], quantity)

        elif signal["Signal"] == "SELL" and portfolio.shares > 0:
            log.debug(
                "SELL %s | Price: %.2f | Shares: %d",
                signal["Date"], signal["Close"], int(portfolio.shares),
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
        num_years = (len(records) / 252)
        annualized_return = ((portfolio_values[-1] / initial_cash) ** (1 / num_years)) - 1

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


if __name__ == "__main__":
    # Single backtest with fixed parameters
    if not GRID_SEARCH_ENABLED:
        run_backtest()
    
    # Grid-search optimization to find best SMA parameters
    else:
        print("\n" + "="*70)
        print("GRID-SEARCH OPTIMIZATION - Finding optimal SMA parameters")
        print("="*70 + "\n")
        
        # Suppress detailed logging during optimization
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("src.data_loader").setLevel(logging.WARNING)
        logging.getLogger("src.strategy").setLevel(logging.WARNING)
        logging.getLogger("src.portfolio").setLevel(logging.WARNING)
        
        # Load data once
        log.warning("Loading data for optimization...")
        download_historical_data(TICKER, START_DATE, END_DATE, CSV_PATH)
        data = load_csv_data(CSV_PATH)
        log.warning(f"{len(data)} records loaded")
        
        # Calculate grid size
        fast_count = GRID_SEARCH_FAST_MAX - GRID_SEARCH_FAST_MIN
        slow_count = GRID_SEARCH_SLOW_MAX - GRID_SEARCH_SLOW_MIN
        total_combinations = fast_count * slow_count
        
        # === EXHAUSTIVE: Full grid search ===
        print("\n" + "="*70)
        print("EXHAUSTIVE GRID-SEARCH OPTIMIZATION")
        print("="*70)
        if is_cpp_available():
            print("Using C++ Multithreaded Optimizer (10-50x faster)")
        else:
            print("Using Pure Python Optimizer (C++ not available)")
        print(f"\nTesting all parameter combinations")
        print(f"  Range: SMA {GRID_SEARCH_FAST_MIN}-{GRID_SEARCH_FAST_MAX-1} x {GRID_SEARCH_SLOW_MIN}-{GRID_SEARCH_SLOW_MAX-1}")
        print(f"  Total combinations: {fast_count} x {slow_count} = {total_combinations}\n")
        
        best_params = run_grid_search(
            data=data,
            fast_range=range(GRID_SEARCH_FAST_MIN, GRID_SEARCH_FAST_MAX),
            slow_range=range(GRID_SEARCH_SLOW_MIN, GRID_SEARCH_SLOW_MAX),
            backtest_func=run_backtest,
        )
        
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        
        # Display both best Sharpe and best Returns
        best_sharpe = best_params['best_sharpe']
        best_returns = best_params['best_returns']
        
        print(f"\n{'BEST SHARPE RATIO:':.<40}")
        print(f"  Short Window (SMA): {best_sharpe['short_window']:>2d} days")
        print(f"  Long Window (SMA):  {best_sharpe['long_window']:>3d} days")
        print(f"  Sharpe Ratio:       {best_sharpe['sharpe_ratio']:>8.4f}")
        print(f"  Final Value:        ${best_sharpe['final_value']:>10,.2f}")
        print(f"  Max Drawdown:       {best_sharpe['max_drawdown']:>8.2%}")
        
        print(f"\n{'BEST RETURNS:':.<40}")
        print(f"  Short Window (SMA): {best_returns['short_window']:>2d} days")
        print(f"  Long Window (SMA):  {best_returns['long_window']:>3d} days")
        print(f"  Sharpe Ratio:       {best_returns['sharpe_ratio']:>8.4f}")
        print(f"  Final Value:        ${best_returns['final_value']:>10,.2f}")
        print(f"  Max Drawdown:       {best_returns['max_drawdown']:>8.2%}")
        print("="*70 + "\n")
        
        # Run final backtest with best Sharpe parameters
        print("Running final backtest with BEST SHARPE parameters...\n")
        run_backtest(
            data=data,
            short_window=best_sharpe['short_window'],
            long_window=best_sharpe['long_window'],
            print_results=True,
        )
        
        # Run final backtest with best Returns parameters (if different)
        if (best_returns['short_window'] != best_sharpe['short_window'] or 
            best_returns['long_window'] != best_sharpe['long_window']):
            print("\n\nRunning final backtest with BEST RETURNS parameters...\n")
            run_backtest(
                data=data,
                short_window=best_returns['short_window'],
                long_window=best_returns['long_window'],
                print_results=True,
            )