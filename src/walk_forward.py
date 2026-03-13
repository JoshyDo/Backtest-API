"""
src/walk_forward.py
-------------------
Walk-Forward Analysis (WFA) architecture for out-of-sample validation.

Prevents overfitting by iteratively training on historical data (In-Sample)
and testing on unseen future data (Out-Of-Sample).

Key concepts:
- In-Sample (IS): Training window where Grid Search finds optimal parameters
- Out-Of-Sample (OOS): Test window where optimal parameters are validated
- Rolling Window: Iteratively shifts forward through time
- Data Leakage Prevention: Strict temporal boundaries between IS and OOS
"""

from typing import Callable, List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import numpy as np

from src.metrics import calculate_sharpe_ratio

log = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """
    Represents a single Walk-Forward iteration window.

    Attributes:
        iteration_num: Which iteration this is (0-based)
        is_start_idx: Start index of In-Sample (training) data
        is_end_idx: End index of In-Sample (training) data (exclusive)
        oos_start_idx: Start index of Out-Of-Sample (test) data
        oos_end_idx: End index of Out-Of-Sample (test) data (exclusive)
        is_data: Actual In-Sample records (for Grid Search)
        oos_data: Actual Out-Of-Sample records (for backtest validation)
    """

    iteration_num: int
    is_start_idx: int
    is_end_idx: int
    oos_start_idx: int
    oos_end_idx: int
    is_data: List[Dict]
    oos_data: List[Dict]

    @property
    def is_length(self) -> int:
        """Length of In-Sample window in days."""
        return self.is_end_idx - self.is_start_idx

    @property
    def oos_length(self) -> int:
        """Length of Out-Of-Sample window in days."""
        return self.oos_end_idx - self.oos_start_idx

    def __repr__(self) -> str:
        return (
            f"WFWindow(iter={self.iteration_num}, "
            f"IS=[{self.is_start_idx}:{self.is_end_idx}], "
            f"OOS=[{self.oos_start_idx}:{self.oos_end_idx}])"
        )


@dataclass
class WalkForwardResult:
    """
    Results from a single Walk-Forward iteration.

    Attributes:
        iteration_num: Which iteration this result belongs to
        best_short_window: Optimal SMA short period found in IS
        best_long_window: Optimal SMA long period found in IS
        best_sharpe_is: Sharpe ratio achieved on IS data
        best_sharpe_oos: Sharpe ratio achieved on OOS data (validation)
        oos_portfolio_values: Portfolio value history during OOS test
        oos_final_value: Final portfolio value at end of OOS period
        oos_returns: Daily returns during OOS period (for aggregation)
    """

    iteration_num: int
    best_short_window: int
    best_long_window: int
    best_sharpe_is: float
    best_sharpe_oos: float
    oos_portfolio_values: List[float]
    oos_final_value: float
    oos_returns: List[float]


@dataclass
class InnerCVResult:
    """
    Result from inner cross-validation of a single parameter combination.

    Attributes:
        short_window: Short SMA period
        long_window: Long SMA period
        train_sharpe: Sharpe ratio on training split
        validate_sharpe: Sharpe ratio on validation split
        adjusted_sharpe: Penalty-adjusted Sharpe (not division-based)
        generalization_gap: Absolute difference between train and validate Sharpe
        is_robust: Whether this parameter combination meets quality thresholds
    """

    short_window: int
    long_window: int
    train_sharpe: float
    validate_sharpe: float
    adjusted_sharpe: float
    generalization_gap: float
    is_robust: bool


def calculate_adjusted_sharpe(train_sharpe: float, validate_sharpe: float) -> float:
    """
    Calculate adjusted Sharpe ratio using penalty-based approach (not division).

    CRITICAL BUG FIX #1:
    Replace division-based efficiency = validate_sharpe / train_sharpe
    Reason: Division breaks with negative Sharpe ratios and creates false positives

    Formula:
        adjusted_sharpe = validate_sharpe - 0.5 * |train_sharpe - validate_sharpe|

    This penalizes overfitting (large gap between train and validate) while rewarding
    validation performance. Does NOT break on negative Sharpes.

    Args:
        train_sharpe: Sharpe ratio on training data
        validate_sharpe: Sharpe ratio on validation data

    Returns:
        Adjusted Sharpe (float): Lower bound = validate_sharpe - 0.5 * max_possible_gap
    """
    generalization_gap = abs(train_sharpe - validate_sharpe)
    adjusted = validate_sharpe - 0.5 * generalization_gap
    return adjusted


def split_is_data(
    is_data: List[Dict], train_ratio: float = 0.7
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split In-Sample window into train and validation sets (70/30).

    Args:
        is_data: Full In-Sample data
        train_ratio: Ratio for training (default 70%)

    Returns:
        Tuple of (train_data, validate_data) preserving temporal order
    """
    split_idx = int(len(is_data) * train_ratio)
    train_data = is_data[:split_idx]
    validate_data = is_data[split_idx:]

    if len(train_data) < 100 or len(validate_data) < 50:
        log.warning(
            f"Small CV split: train={len(train_data)}, validate={len(validate_data)}. "
            f"Recommend IS window >= 500 days"
        )

    return train_data, validate_data


def run_inner_cross_validation(
    is_data: List[Dict],
    fast_range: range,
    slow_range: range,
    backtest_func: Callable,
    warmup_data: List[Dict],
) -> Tuple[List[InnerCVResult], int, int]:
    """
    Run inner cross-validation on Grid Search parameter combinations.

    CRITICAL BUG FIX #2 & #3:
    - Ranking by percentile (top quartile), not absolute thresholds
    - Parameter stability using relative percentage distance (not euclidean)

    This prevents crashes during bear markets when NO parameters exceed threshold,
    and prevents parameter drift from small numbers (10→15) being treated same as
    large numbers (200→205).

    Args:
        is_data: In-Sample training data
        fast_range: Range of fast SMA values
        slow_range: Range of slow SMA values
        backtest_func: Function to run single backtest
        warmup_data: Warmup data for indicator calculation

    Returns:
        Tuple of (robust_candidates, best_short, best_long):
        - robust_candidates: List of InnerCVResult for parameters that pass CV
        - best_short: Short window of best parameter
        - best_long: Long window of best parameter

    Process:
        1. Split IS data: 70% train, 30% validate
        2. Test ALL combinations on BOTH splits
        3. Calculate adjusted Sharpe with penalty for overfitting
        4. Select robust candidates: top quartile (75th percentile) performers
        5. Return best among robust candidates
    """
    # Split IS data for inner CV
    train_data, validate_data = split_is_data(is_data)

    # Test all combinations on train and validate
    all_cv_results = []

    for short_window in fast_range:
        for long_window in slow_range:
            if short_window >= long_window:
                continue

            # Test on training split
            train_values = backtest_func(
                warmup_data + train_data, short_window, long_window
            )
            train_sharpe = calculate_sharpe_ratio(train_values)

            # Test on validation split
            validate_values = backtest_func(
                warmup_data + validate_data, short_window, long_window
            )
            validate_sharpe = calculate_sharpe_ratio(validate_values)

            # Calculate adjusted Sharpe (penalty-based, not division-based)
            adjusted_sharpe = calculate_adjusted_sharpe(train_sharpe, validate_sharpe)
            generalization_gap = abs(train_sharpe - validate_sharpe)

            cv_result = InnerCVResult(
                short_window=short_window,
                long_window=long_window,
                train_sharpe=train_sharpe,
                validate_sharpe=validate_sharpe,
                adjusted_sharpe=adjusted_sharpe,
                generalization_gap=generalization_gap,
                is_robust=False,  # Will be set based on percentile
            )
            all_cv_results.append(cv_result)

    if not all_cv_results:
        raise ValueError("No valid parameter combinations from inner CV")

    # Sort by adjusted Sharpe (best first)
    all_cv_results.sort(key=lambda x: x.adjusted_sharpe, reverse=True)

    # CRITICAL BUG FIX #2: Use percentile ranking, not absolute thresholds
    # Mark top 25% as robust (75th percentile)
    percentile_threshold = int(len(all_cv_results) * 0.25)
    if percentile_threshold == 0:
        percentile_threshold = 1

    robust_candidates = all_cv_results[:percentile_threshold]

    for result in robust_candidates:
        result.is_robust = True

    # Best parameter is top of robust list
    best_result = robust_candidates[0]

    log.info(
        f"Inner CV: {len(all_cv_results)} combos tested, {len(robust_candidates)} robust. "
        f"Best: {best_result.short_window}/{best_result.long_window} "
        f"(train={best_result.train_sharpe:.4f}, val={best_result.validate_sharpe:.4f}, "
        f"adj={best_result.adjusted_sharpe:.4f})"
    )

    return robust_candidates, best_result.short_window, best_result.long_window


def calculate_relative_parameter_distance(
    params1: Tuple[int, int], params2: Tuple[int, int]
) -> float:
    """
    Calculate relative (percentage-based) distance between parameter pairs.

    CRITICAL BUG FIX #3:
    Replace euclidean distance with relative percentage change.
    Reason: Fast-SMA change 10→15 (50%) should NOT be treated same as
            Slow-SMA 200→205 (2.5%)

    Formula:
        distance = sqrt((Δshort/short)² + (Δlong/long)²)

    Args:
        params1: (short_window_1, long_window_1)
        params2: (short_window_2, long_window_2)

    Returns:
        Relative distance (float, 0 = identical, >1 = significant change)
    """
    short1, long1 = params1
    short2, long2 = params2

    # Handle division by zero
    if short1 == 0 or long1 == 0:
        return float("inf")

    relative_short = (short2 - short1) / short1
    relative_long = (long2 - long1) / long1

    distance = np.sqrt(relative_short**2 + relative_long**2)
    return distance


class WalkForwardAnalyzer:
    """
    Orchestrates Walk-Forward Analysis by:
    1. Creating rolling windows of IS/OOS data
    2. Running Grid Search on each IS window
    3. Validating on each OOS window
    4. Aggregating results into single out-of-sample equity curve
    """

    def __init__(
        self,
        data: List[dict],
        is_window_days: int = 504,
        oos_window_days: int = 252,
        step_size_days: int = 252,
        warmup_days: int = 50,
    ):
        # Validierung
        if warmup_days < 100:  # Faustregel für SMA
            log.warning(
                f"warmup_days ({warmup_days}) ist klein. "
                f"Empfohlen: >= 100 (besser: >= long_window)"
            )

        total_window = warmup_days + is_window_days + oos_window_days
        if total_window > len(data):
            raise ValueError(
                f"Nicht genug Daten: brauche {total_window} Tage "
                f"(warmup {warmup_days} + is {is_window_days} + oos {oos_window_days}), "
                f"habe aber nur {len(data)}"
            )

        self.data = data
        self.is_window_days = is_window_days
        self.oos_window_days = oos_window_days
        self.step_size_days = step_size_days
        self.warmup_days = warmup_days
        self.total_records = len(data)
        self.total_window_length = warmup_days + is_window_days + oos_window_days

        # Validation
        if self.total_window_length > self.total_records:
            raise ValueError(
                f"Total window length ({self.total_window_length}) exceeds "
                f"available data ({self.total_records})"
            )

        log.info(
            "Walk-Forward configured: IS=%d days, OOS=%d days, Step=%d days, Warmup=%d days",
            is_window_days,
            oos_window_days,
            step_size_days,
            warmup_days,
        )

    def generate_windows(self) -> List[WalkForwardWindow]:
        """
        Generate all Walk-Forward windows from data.

        Returns:
            List of WalkForwardWindow objects defining IS/OOS boundaries

        Architecture note:
            - Each window has a WARMUP phase (before IS) for indicator calculation
            - IS phase: Where Grid Search optimizes parameters
            - OOS phase: Where optimized parameters are tested (validation)
            - Windows do NOT overlap to prevent data leakage
            - Step size determines how much each iteration advances
        """
        windows = []
        iteration = 0
        start_idx = 0

        while (start_idx + self.total_window_length) <= self.total_records:
            warmup_end = start_idx + self.warmup_days

            is_start = warmup_end
            is_end = is_start + self.is_window_days

            oos_start = is_end
            oos_end = oos_start + self.oos_window_days

            is_data = self.data[is_start:is_end]
            oos_data = self.data[oos_start:oos_end]

            window = WalkForwardWindow(
                iteration_num=iteration,
                is_start_idx=is_start,
                is_end_idx=is_end,
                oos_start_idx=oos_start,
                oos_end_idx=oos_end,
                is_data=is_data,
                oos_data=oos_data,
            )
            windows.append(window)

            start_idx += self.step_size_days
            iteration += 1

        return windows

    def run(
        self,
        grid_search_func: Callable,
        backtest_func: Callable,
        initial_capital: float = 10_000.0,
        fast_range: Optional[range] = None,
        slow_range: Optional[range] = None,
        use_inner_cv: bool = True,
    ) -> Dict:
        """
        Execute complete Walk-Forward Analysis with optional nested cross-validation.

        Args:
            grid_search_func: Function that runs Grid Search on IS data
                Signature: grid_search_func(data, ...) -> {best_short, best_long, best_sharpe}
            backtest_func: Function that runs single backtest
                Signature: backtest_func(data, short_window, long_window, ...) -> list[float] (portfolio values)
            initial_capital: Starting capital for backtests
            fast_range: Range for fast SMA (needed for inner CV)
            slow_range: Range for slow SMA (needed for inner CV)
            use_inner_cv: Whether to use inner cross-validation (Layer 2)

        Returns:
            Dictionary with:
            - 'iterations': List of WalkForwardResult objects (enhanced with CV metrics)
            - 'aggregated_oos_equity': Combined equity curve from all OOS periods
            - 'final_sharpe_oos': Sharpe ratio calculated on aggregated OOS equity
            - 'final_value': Final portfolio value after all OOS periods
            - 'num_windows': Number of iterations
            - 'inner_cv_enabled': Whether inner CV was used

        Architecture (Layer 2 - Inner Cross-Validation):
            - Loop over each window
            - For each iteration:
              1. Split IS window: 70% Train / 30% Validate
              2. If use_inner_cv: Run inner CV on ALL Grid Search combos
                 - Test each combo on Train and Validate separately
                 - Calculate adjusted Sharpe (penalty-based)
                 - Select robust parameters (top 25% percentile)
                 - Apply parameter stability constraint (relative distance from prev iteration)
              3. Otherwise: Use standard grid_search_func
              4. Run backtest on OOS with selected parameters
              5. Store result with CV metrics
            - Aggregate all OOS equity
            - Calculate final Sharpe on aggregated OOS data
        """
        windows = self.generate_windows()

        if not windows:
            raise ValueError("No valid Walk-Forward windows generated from data")

        log.info("Generated %d Walk-Forward windows", len(windows))
        log.info(f"Inner CV enabled: {use_inner_cv}")

        results = []
        last_best_params: Optional[Tuple[int, int]] = (
            None  # For parameter stability tracking
        )

        for window in windows:
            log.info(
                f"Iteration {window.iteration_num}, IS range: {window.is_start_idx}-{window.is_end_idx}, OOS range: {window.oos_start_idx}-{window.oos_end_idx}"
            )

            # Get warmup data for this window
            warmup_start = window.is_start_idx - self.warmup_days
            warmup_data = self.data[warmup_start : window.is_start_idx]

            # Step 1: Get best parameters (with inner CV if enabled)
            if use_inner_cv and fast_range is not None and slow_range is not None:
                # Layer 2: Inner Cross-Validation with robustness checks
                log.info(f"Running inner CV for iteration {window.iteration_num}...")
                robust_candidates, best_short, best_long = run_inner_cross_validation(
                    is_data=window.is_data,
                    fast_range=fast_range,
                    slow_range=slow_range,
                    backtest_func=backtest_func,
                    warmup_data=warmup_data,
                )

                # Apply parameter stability constraint (Layer 3)
                if last_best_params is not None and isinstance(last_best_params, tuple):
                    current_params = (best_short, best_long)
                    relative_distance = calculate_relative_parameter_distance(
                        last_best_params, current_params
                    )

                    # If distance is small, prefer current parameters
                    # If large, check if there's a closer robust candidate
                    if relative_distance > 0.25:  # 25% relative change threshold
                        log.warning(
                            f"Large parameter drift: {last_best_params} → {current_params} "
                            f"(relative distance: {relative_distance:.4f}). "
                            f"Searching for more stable alternative..."
                        )

                        # Find closest parameter in robust candidates
                        prev_params = last_best_params
                        closest_candidate = min(
                            robust_candidates,
                            key=lambda x: calculate_relative_parameter_distance(
                                prev_params, (x.short_window, x.long_window)
                            ),
                        )

                        closest_distance = calculate_relative_parameter_distance(
                            prev_params,
                            (
                                closest_candidate.short_window,
                                closest_candidate.long_window,
                            ),
                        )

                        # If a significantly closer candidate exists, use it
                        if closest_distance < relative_distance * 0.8:
                            log.info(
                                f"Selecting more stable parameters: "
                                f"{closest_candidate.short_window}/{closest_candidate.long_window} "
                                f"(distance: {closest_distance:.4f})"
                            )
                            best_short = closest_candidate.short_window
                            best_long = closest_candidate.long_window

                best_sharpe_is = None  # Will be calculated on full IS data

            else:
                # Fallback: Standard grid search (Layer 1 only)
                log.info(
                    f"Running standard grid search for iteration {window.iteration_num}..."
                )
                best_params = grid_search_func(window.is_data)
                best_short = best_params["best_short"]
                best_long = best_params["best_long"]
                best_sharpe_is = best_params["best_sharpe"]

            # Update parameter tracking
            last_best_params = (best_short, best_long)

            # Step 2: Calculate IS Sharpe if not already done
            if best_sharpe_is is None:
                warmup_and_is_data = self.data[warmup_start : window.is_end_idx]
                is_values = backtest_func(warmup_and_is_data, best_short, best_long)
                is_only_values = is_values[-self.is_window_days :]
                best_sharpe_is = calculate_sharpe_ratio(is_only_values)

            # Step 3: Test on OOS
            warmup_and_oos_start = window.oos_start_idx - self.warmup_days
            warmup_and_oos_data = self.data[warmup_and_oos_start : window.oos_end_idx]

            oos_portfolio_values = backtest_func(
                warmup_and_oos_data, best_short, best_long
            )
            oos_only_values = oos_portfolio_values[-self.oos_window_days :]
            oos_returns = calculate_daily_returns(oos_only_values)

            result = WalkForwardResult(
                iteration_num=window.iteration_num,
                best_short_window=best_short,
                best_long_window=best_long,
                best_sharpe_is=best_sharpe_is,
                best_sharpe_oos=calculate_sharpe_ratio(oos_only_values),
                oos_portfolio_values=oos_portfolio_values,
                oos_final_value=oos_only_values[-1],
                oos_returns=oos_returns,
            )

            results.append(result)

        aggregated_equity, aggregated_returns = aggregate_oos_equity(
            initial_capital, results
        )

        final_sharpe_oos = calculate_sharpe_ratio(aggregated_equity)

        return {
            "iterations": results,
            "aggregated_oos_equity": aggregated_equity,
            "final_sharpe_oos": final_sharpe_oos,
            "final_value": aggregated_equity[-1],
            "num_windows": len(windows),
            "inner_cv_enabled": use_inner_cv,
            "oos_window_days": self.oos_window_days,
            "data": self.data,
        }


def calculate_daily_returns(portfolio_values: List[float]) -> List[float]:
    """
    Convert portfolio value series to daily returns.

    Args:
        portfolio_values: List of portfolio values over time

    Returns:
        List of daily returns (length = len(portfolio_values) - 1)

    Formula:
        daily_return[i] = (portfolio_values[i+1] - portfolio_values[i]) / portfolio_values[i]
    """
    if len(portfolio_values) < 2:
        raise ValueError("Portfolio_values must have at least 2 values")

    daily_returns = []
    for i in range(len(portfolio_values) - 1):
        current_value = portfolio_values[i]
        next_value = portfolio_values[i + 1]

        if current_value <= 0:
            raise ValueError(
                f"Portfolio value at index {i} is {current_value}, must be > 0"
            )

        daily_return = (next_value - current_value) / current_value
        daily_returns.append(daily_return)

    return daily_returns


def aggregate_oos_equity(
    initial_capital: float,
    iteration_results: List[WalkForwardResult],
) -> Tuple[List[float], List[float]]:
    """
    Aggregate Out-Of-Sample results into single equity curve.

    Args:
        initial_capital: Starting capital
        iteration_results: List of WalkForwardResult from each WFA iteration

    Returns:
        Tuple of (aggregated_equity_curve, aggregated_daily_returns)

    Logic:
        1. Start equity = initial_capital
        2. For each iteration result:
           - Extract OOS daily returns
           - For each return:
             - New equity = last_equity * (1 + return)
             - Append to aggregated list
        3. Ensures capital carries forward between iterations
    """
    if not iteration_results:
        raise ValueError("iteration_results cannot be empty")

    aggregated_equity = [initial_capital]

    for result in iteration_results:
        if result.oos_returns is None or len(result.oos_returns) == 0:
            log.warning(
                f"Iteration {result.iteration_num} has no OOS returns, skipping"
            )
            continue

        oos_returns = result.oos_returns

        for daily_return in oos_returns:
            if daily_return < -1.0:  # Sanity check: return can't be worse than -100%
                log.warning(
                    f"Iteration {result.iteration_num}: Suspicious return {daily_return:.4f}, capping at -0.99"
                )
                daily_return = -0.99

            new_equity = aggregated_equity[-1] * (1 + daily_return)
            aggregated_equity.append(new_equity)

    if len(aggregated_equity) < 2:
        raise ValueError("Aggregated equity curve has insufficient data points")

    aggregated_returns = calculate_daily_returns(aggregated_equity)

    log.info(
        f"Aggregated {len(iteration_results)} iterations into {len(aggregated_equity)} equity points"
    )

    return (aggregated_equity, aggregated_returns)


def print_wfa_summary(
    results: Dict,
    initial_capital: float = 10_000.0,
) -> None:
    """
    Print formatted summary of Walk-Forward Analysis results.

    Enhanced to show inner CV metrics and robustness analysis.

    Args:
        results: Dict with 'iterations' (list of WalkForwardResult) and aggregated metrics
        initial_capital: Starting capital for return calculation
    """
    if not results.get("iterations"):
        print("No Walk-Forward results to display")
        return

    iterations = results["iterations"]
    final_value = results.get("final_value", 0)
    final_sharpe_oos = results.get("final_sharpe_oos", 0)
    inner_cv_enabled = results.get("inner_cv_enabled", False)
    oos_window_days = results.get("oos_window_days", 252)
    data = results.get("data", [])

    # Header
    print("\n" + "=" * 130)
    print("WALK-FORWARD ANALYSIS SUMMARY".center(130))
    print(
        f"Inner CV (Layer 2 Robustness): {'ENABLED ✓' if inner_cv_enabled else 'DISABLED'}".center(
            130
        )
    )
    print("=" * 130)

    # Column headers - adjust based on whether inner CV is enabled
    if inner_cv_enabled:
        print(
            f"{'Iter':<6} {'Short':<8} {'Long':<8} {'IS Sharpe':<12} {'OOS Sharpe':<12} {'Drift':<10} {'Status':<12}"
        )
    else:
        print(
            f"{'Iter':<6} {'Short':<8} {'Long':<8} {'IS Sharpe':<12} {'OOS Sharpe':<12} {'Drift':<10}"
        )
    print("-" * 130)

    # Iteration details
    total_drift = 0
    extreme_drift_count = 0

    for result in iterations:
        drift = result.best_sharpe_is - result.best_sharpe_oos
        total_drift += drift

        # Determine status badge
        if drift > 2.0:
            status = "⚠ SEVERE"
            extreme_drift_count += 1
        elif drift > 1.0:
            status = "⚡ HIGH"
        elif drift < 0:
            status = "✓ GOOD"
        else:
            status = "NORMAL"

        if inner_cv_enabled:
            print(
                f"{result.iteration_num:<6} "
                f"{result.best_short_window:<8} "
                f"{result.best_long_window:<8} "
                f"{result.best_sharpe_is:<12.4f} "
                f"{result.best_sharpe_oos:<12.4f} "
                f"{drift:<10.4f} "
                f"{status:<12}"
            )
        else:
            print(
                f"{result.iteration_num:<6} "
                f"{result.best_short_window:<8} "
                f"{result.best_long_window:<8} "
                f"{result.best_sharpe_is:<12.4f} "
                f"{result.best_sharpe_oos:<12.4f} "
                f"{drift:<10.4f}"
            )

    # Footer
    print("-" * 130)

    total_return = (
        (final_value - initial_capital) / initial_capital if initial_capital > 0 else 0
    )
    avg_drift = total_drift / len(iterations) if iterations else 0
    
    # Calculate annualized return (assuming ~252 trading days per year)
    num_years = len(iterations) * (oos_window_days / 252) if iterations else 1
    annualized_return = ((final_value / initial_capital) ** (1 / max(num_years, 1))) - 1

    # Calculate Buy & Hold return (from first to last price in data)
    buyhold_return = 0.0
    if data and len(data) >= 2:
        first_price = data[0].get("Close", 0)
        last_price = data[-1].get("Close", 0)
        if first_price > 0:
            buyhold_return = (last_price - first_price) / first_price
            buyhold_annualized = ((last_price / first_price) ** (1 / num_years)) - 1
        else:
            buyhold_annualized = 0.0
    else:
        buyhold_annualized = 0.0

    print(f"{'Final Portfolio Value:':<40} ${final_value:,.2f}")
    print(f"{'Total Return (WFA):':<40} {total_return * 100:,.2f}%")
    print(f"{'Total Return (Buy & Hold):':<40} {buyhold_return * 100:,.2f}%")
    print(f"{'Return per Year (WFA):':<40} {annualized_return * 100:,.2f}%")
    print(f"{'Return per Year (Buy & Hold):':<40} {buyhold_annualized * 100:,.2f}%")
    print(f"{'Final OOS Sharpe Ratio:':<40} {final_sharpe_oos:.4f}")
    print(f"{'Average IS-OOS Drift:':<40} {avg_drift:.4f}")
    print(f"{'Number of Iterations:':<40} {len(iterations)}")

    # Drift analysis and warnings
    if extreme_drift_count > 0:
        print(
            f"{'⚠️  SEVERE Drift Events (>2.0):':<40} {extreme_drift_count}/{len(iterations)}"
        )

    # Inner CV assessment
    if inner_cv_enabled:
        if avg_drift < 0.5:
            print(f"{'Inner CV Robustness:':<40} ✓ EXCELLENT (drift < 0.5)")
        elif avg_drift < 1.0:
            print(f"{'Inner CV Robustness:':<40} ✓ GOOD (drift < 1.0)")
        elif avg_drift < 1.5:
            print(f"{'Inner CV Robustness:':<40} ⚡ MODERATE (drift < 1.5)")
        else:
            print(
                f"{'Inner CV Robustness:':<40} ⚠ WEAK (drift >= 1.5) - Consider expanding IS window"
            )

    print("=" * 130 + "\n")
