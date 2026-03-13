"""
Parameter optimization via grid-search for the SMA crossover strategy.

This module provides grid-search functionality to find optimal short and long
SMA parameters by exhaustively testing combinations and evaluating their
Sharpe ratios on historical data.

Features:
  - Pure Python grid search (works everywhere)
  - C++ multithreaded grid search (when compiled, 10-50x faster)
  - Automatic fallback to Python if C++ unavailable
"""

import io
import logging
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import TypedDict, Callable

from src.metrics import calculate_max_drawdown, calculate_sharpe_ratio

try:
    from src.cpp_optimizer import is_cpp_available

    CPP_AVAILABLE = is_cpp_available()
except Exception:  # pragma: no cover - defensive fallback
    CPP_AVAILABLE = False


log = logging.getLogger(__name__)


def create_progress_bar(progress: float, width: int = 40, char: str = "=") -> str:
    """
    Creates a visual progress bar.

    Args:
        progress: Progress from 0.0 to 1.0
        width: Bar width in characters
        char: Character for filled portion

    Returns:
        Formatted progress bar as string
    """
    filled = int(progress * width)
    bar = char * filled + "-" * (width - filled)
    percentage = int(progress * 100)
    return f"[{bar}] {percentage:>3d}%"


def format_time(seconds: float) -> str:
    """
    Converts seconds to human-readable format (h:mm:ss).

    Args:
        seconds: Number of seconds

    Returns:
        Formatted time string
    """
    if seconds < 0:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs:02d}s"


class OptimizationResult(TypedDict):
    """
    Represents the result of a single backtest parameter combination.

    Attributes:
        short_window: Short SMA window in days
        long_window: Long SMA window in days
        sharpe_ratio: Annualized Sharpe ratio from the backtest
        final_value: Final portfolio value after backtest
        max_drawdown: Maximum drawdown during backtest
    """

    short_window: int
    long_window: int
    sharpe_ratio: float
    final_value: float
    max_drawdown: float


def _run_grid_search_python(
    data: list[dict],
    fast_range: range,
    slow_range: range,
    backtest_func: Callable,
) -> dict:
    """
    Pure Python implementation of grid search.

    Args:
        data: List of OHLCV records
        fast_range: Range of short SMA values
        slow_range: Range of long SMA values
        backtest_func: Backtest function to call

    Returns:
        Dict with 'best_sharpe' and 'best_returns' OptimizationResults
    """
    # Initialize tracking variables
    best_sharpe_result: OptimizationResult | None = None
    best_returns_result: OptimizationResult | None = None
    best_sharpe = float("-inf")
    best_returns = float("-inf")  # Track by final value, not percentage
    total_combinations = 0
    tested_combinations = 0
    result_history: dict[tuple[int, int], float] = {}
    start_time = time.time()
    combination_times: list[float] = []

    # Count valid combinations
    for short in fast_range:
        for long in slow_range:
            if short < long:
                total_combinations += 1

    # Grid Search Loop - sorted by fast window for cache efficiency
    fast_list = sorted(list(fast_range))
    slow_list = sorted(list(slow_range))

    for short in fast_list:
        for long in slow_list:
            if short >= long:
                continue

            tested_combinations += 1
            iteration_start = time.time()

            # Calculate progress
            progress = tested_combinations / total_combinations
            progress_bar = create_progress_bar(progress, width=40)

            # Calculate elapsed and estimated remaining time
            # elapsed = time.time() - start_time
            if combination_times:
                avg_time = sum(combination_times) / len(combination_times)
                remaining_combinations = total_combinations - tested_combinations
                estimated_remaining = avg_time * remaining_combinations
                time_str = format_time(estimated_remaining)
            else:
                time_str = "Computing..."

            # Run backtest with these parameters (suppress stdout/stderr to keep progress bar clean)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                portfolio_values = backtest_func(
                    data=data,
                    short_window=short,
                    long_window=long,
                    print_results=False,
                )

            # Display progress on stderr with carriage return
            status_line = f"\r{progress_bar} {tested_combinations:>3d}/{total_combinations} | ETA: {time_str:>10s}"
            sys.stderr.write(status_line)
            sys.stderr.flush()

            # Track iteration time
            combination_times.append(time.time() - iteration_start)

            # Calculate performance metrics
            if not isinstance(portfolio_values, list) or not all(
                isinstance(x, (int, float)) for x in portfolio_values
            ):
                raise TypeError("portfolio_values must be a list of floats/ints")
            sharpe = calculate_sharpe_ratio(portfolio_values)
            max_dd = calculate_max_drawdown(portfolio_values)
            final_value = portfolio_values[-1] if portfolio_values else 0.0

            # Store in results
            result_history[(short, long)] = sharpe

            # Check if best Sharpe
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sharpe_result = OptimizationResult(
                    short_window=short,
                    long_window=long,
                    sharpe_ratio=sharpe,
                    final_value=final_value,
                    max_drawdown=max_dd,
                )

            # Check if best Returns (by final value)
            if final_value > best_returns:
                best_returns = final_value
                best_returns_result = OptimizationResult(
                    short_window=short,
                    long_window=long,
                    sharpe_ratio=sharpe,
                    final_value=final_value,
                    max_drawdown=max_dd,
                )

    # Return results
    if best_sharpe_result is None or best_returns_result is None:
        raise RuntimeError("Grid search did not produce any valid results")

    # Print completion message
    total_time = time.time() - start_time
    sys.stderr.write(f"\nGrid search completed in {format_time(total_time)}!\n")
    sys.stderr.write(
        f"  Best Sharpe: SMA({best_sharpe_result['short_window']}, {best_sharpe_result['long_window']}) = {best_sharpe_result['sharpe_ratio']:.4f}\n"
    )
    sys.stderr.write(
        f"  Best Returns: SMA({best_returns_result['short_window']}, {best_returns_result['long_window']}) = ${best_returns_result['final_value']:,.2f}\n"
    )
    sys.stderr.write(
        f"  Combinations tested: {tested_combinations}/{total_combinations}\n\n"
    )
    sys.stderr.flush()

    return {
        "best_sharpe": best_sharpe_result,
        "best_returns": best_returns_result,
    }


def _run_grid_search_cpp(
    data: list[dict],
    fast_range: range,
    slow_range: range,
) -> dict:
    """
    C++ multithreaded implementation of grid search.

    Note: C++ version returns best Sharpe by default.
    For best returns, we use the Sharpe result as it's typically well-balanced.

    Args:
        data: List of OHLCV records
        fast_range: Range of short SMA values
        slow_range: Range of long SMA values

    Returns:
        Dict with 'best_sharpe' and 'best_returns' (same for C++ version)
    """
    from src.cpp_optimizer import grid_search_multithreaded_cpp

    # Extract price data
    prices = [record["Close"] for record in data]

    # Call C++ function (optimizes for Sharpe ratio)
    result = grid_search_multithreaded_cpp(
        prices=prices,
        fast_min=min(fast_range),
        fast_max=max(fast_range) + 1,
        slow_min=min(slow_range),
        slow_max=max(slow_range) + 1,
    )

    return {
        "best_sharpe": result,
        "best_returns": result,  # C++ optimizes for Sharpe, use same result
    }


def run_grid_search(
    data: list[dict],
    fast_range: range,
    slow_range: range,
    backtest_func: Callable,
) -> dict:
    """
    Executes exhaustive grid-search optimization for SMA crossover parameters.

    This function tests combinations of fast_range and slow_range, where fast_sma < slow_sma.
    Tests all valid combinations exhaustively without early stopping.

    Uses C++ multithreaded implementation if available (10-50x faster),
    otherwise falls back to pure Python.

    Args:
        data: List of OHLCV dicts (must contain at least ["Date", "Close"]).
              Output from load_csv_data().
        fast_range: range object for short SMA values (e.g. range(5, 51) for 5-50).
        slow_range: range object for long SMA values (e.g. range(20, 201) for 20-200).
        backtest_func: Callable that executes backtest and returns portfolio values.
                       Expected signature: backtest_func(data, short_window, long_window, print_results=False)

    Returns:
        dict with 'best_sharpe' and 'best_returns' keys, each containing OptimizationResult:
            - short_window: int - SMA window in days
            - long_window: int - SMA window in days
            - sharpe_ratio: float - Sharpe ratio for this combination
            - final_value: float - final portfolio value
            - max_drawdown: float - maximum drawdown

    Raises:
        ValueError: If data has insufficient records or ranges are empty
        RuntimeError: If grid search produces no valid results

    Notes:
        - Progress is displayed on stderr with real-time ETA
        - All combinations are tested exhaustively
        - C++ implementation used automatically if available (see cpp/build.sh)

    Example:
        >>> data = load_csv_data("data/AAPL.csv")
        >>> result = run_grid_search(data, range(10, 31), range(30, 101), run_backtest)
        >>> print(f"Best Sharpe: {result['best_sharpe']['short_window']}/{result['best_sharpe']['long_window']}")
        >>> print(f"Best Returns: {result['best_returns']['short_window']}/{result['best_returns']['long_window']}")
    """
    # Validate input
    if len(data) < max(slow_range):
        raise ValueError("Insufficient data for these SMA parameters")
    if len(fast_range) == 0 or len(slow_range) == 0:
        raise ValueError("fast_range and slow_range cannot be empty")

    if CPP_AVAILABLE and backtest_func.__name__ == "run_backtest":
        return _run_grid_search_cpp(
            data=data,
            fast_range=fast_range,
            slow_range=slow_range,
        )
    else:
        # Fall back to pure Python implementation for custom backtest functions or when C++ unavailable
        return _run_grid_search_python(
            data=data,
            fast_range=fast_range,
            slow_range=slow_range,
            backtest_func=backtest_func,
        )
