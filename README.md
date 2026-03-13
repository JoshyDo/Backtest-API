# Backtesting Engine

A production-quality Python backtesting framework for evaluating an SMA crossover trading strategy on historical market data with advanced walk-forward analysis and inner cross-validation.

## Key Features

- **SMA Crossover Strategy** – Configurable short/long moving averages with golden/death cross signals
- **Grid-Search Optimization** – Pure Python + optional multithreaded C++ backend for parameter tuning
- **Walk-Forward Analysis (WFA)** – Rolling in-sample/out-of-sample windows with data leakage prevention
- **Inner Cross-Validation (Layer 2)** – Robust parameter selection using:
  - Penalty-based adjusted Sharpe (handles negative returns)
  - Percentile-based robust candidate filtering
  - Relative parameter distance stability constraints
- **Realistic Portfolio Simulation** – Commission, bid-ask spread, and slippage
- **Performance Metrics** – Sharpe ratio, max drawdown, returns, drift analysis
- **Data Management** – yfinance download with automatic CSV caching
- **High Test Coverage** – 164 unit tests, 89% code coverage, 100% core modules

## Quick Start

### Installation

```bash
git clone https://github.com/JoshyDo/Backtest-API.git
cd Backtest-API

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### First Run

```bash
python main.py
```

By default, this will:
1. **Download data** – Fetches OHLCV for DAX (or configured ticker) and caches to CSV
2. **Run grid search** – Searches conservative SMA ranges (15–35 short, 40–100 long)
3. **Execute WFA** – Rolls through ~2-year in-sample, ~1-year out-of-sample windows
4. **Inner CV enabled** – Each IS window is split 70/30 for robust parameter selection
5. **Print results** – Displays summary stats, iteration details, and drift analysis

## Configuration

Edit the configuration block at the top of `main.py` to customize behavior:

```python
# ============ DATA & BACKTEST ============
TICKER = "DAX"
START_DATE = "2000-01-01"
END_DATE = "2027-01-01"
INITIAL_CASH = 10_000.0
COMMISSION = 0.001      # 0.1% per trade
SLIPPAGE = 0.002        # 0.2% market impact
SPREAD_MIN = 0.001      # 0.1% bid-ask min
SPREAD_MAX = 0.002      # 0.2% bid-ask max

# ============ BASE SMA PARAMETERS ============
SHORT_WINDOW = 20
LONG_WINDOW = 50

# ============ GRID SEARCH ============
GRID_SEARCH_ENABLED = True
GRID_SEARCH_FAST_MIN = 15    # short SMA min
GRID_SEARCH_FAST_MAX = 35    # short SMA max
GRID_SEARCH_SLOW_MIN = 40    # long SMA min
GRID_SEARCH_SLOW_MAX = 100   # long SMA max

# ============ WALK-FORWARD ANALYSIS ============
WALK_FORWARD_ENABLED = True
WFA_IS_WINDOW_DAYS = 504    # ~2 trading years
WFA_OOS_WINDOW_DAYS = 252   # ~1 trading year
WFA_STEP_SIZE_DAYS = 252    # roll forward 1 year
WFA_WARMUP_DAYS = 406       # indicator warmup
```

### Execution Modes

| Config | Behavior |
|--------|----------|
| `GRID_SEARCH_ENABLED=False` | Single backtest with `SHORT_WINDOW`/`LONG_WINDOW` |
| `GRID_SEARCH_ENABLED=True`, `WALK_FORWARD_ENABLED=False` | Grid search → find best params on full period |
| Both `True` | Grid search + WFA with inner CV on each window |

## Walk-Forward Analysis & Inner Cross-Validation

The WFA system (`src/walk_forward.py`) prevents overfitting through:

### Architecture

**Layer 1 (Standard WFA)**
- Rolling in-sample (IS) window for parameter optimization
- Out-of-sample (OOS) window for validation
- Warmup period to prevent indicator leakage
- Data flows forward in time only

**Layer 2 (Inner CV)**
- Split each IS into 70% train / 30% validate
- Run grid search on train split
- Evaluate candidates on validate split
- Select robust parameters (top 25% percentile)
- Apply stability constraint (relative distance from prior window)

**Layer 3 (Parameter Stability)**
- Track parameter changes between windows
- If relative distance exceeds threshold (25%), search for closer alternative
- Prefer stability to single-period performance

### Usage Example

```python
from src.walk_forward import WalkForwardAnalyzer, print_wfa_summary

analyzer = WalkForwardAnalyzer(
    data=data,
    is_window_days=504,      # 2-year in-sample
    oos_window_days=252,     # 1-year out-of-sample
    step_size_days=252,      # roll forward 1 year
    warmup_days=406,         # warmup buffer
)

results = analyzer.run(
    grid_search_func=wfa_grid_search_wrapper,
    backtest_func=wfa_backtest_wrapper,
    initial_capital=10_000.0,
    fast_range=range(15, 35),
    slow_range=range(40, 100),
    use_inner_cv=True,  # Enable Layer 2
)

print_wfa_summary(results, initial_capital=10_000.0)
```

### Key Innovations

1. **Penalty-Based Adjusted Sharpe**  
   Handles negative Sharpe ratios correctly:
   ```
   adjusted = validate_sharpe - 0.5 * |train_sharpe - validate_sharpe|
   ```

2. **Percentile-Based Selection**  
   Works in all market regimes (bull, bear, flat):
   ```
   robust_count = max(1, int(len(candidates) * 0.25))
   ```

3. **Relative Distance Metric**  
   Penalizes small-number changes more than large:
   ```
   distance = sqrt(((short_new - short_old) / short_old)² + ...)
   ```

For deep dives, see:
- `LAYER2_INNER_CV_GUIDE.md` – Detailed math and design rationale
- `BUG_FIXES_VISUAL_COMPARISON.md` – Comparison with naive approaches

## Quick Reference

### Common Commands

```bash
# Run default backtest
python main.py

# Run tests
pytest

# Run tests with coverage
pytest --cov=src

# Build C++ optimizer
bash cpp/build.sh

# Download fresh data
python -c "from src.data_loader import download_historical_data; download_historical_data('DAX', 'data')"
```

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `Portfolio` | `portfolio.py` | Tracks cash, shares, transactions |
| `SMAStrategy` | `strategy.py` | Generates buy/sell/hold signals |
| `WalkForwardAnalyzer` | `walk_forward.py` | Manages rolling windows + inner CV |
| `WalkForwardResult` | `walk_forward.py` | Stores per-window metrics |

### Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `run_grid_search` | `optimizer.py` | Parameter optimization (Python/C++) |
| `run_backtest` | `strategy.py` | Executes strategy on data |
| `calculate_sharpe_ratio` | `metrics.py` | Risk-adjusted return metric |
| `calculate_adjusted_sharpe` | `walk_forward.py` | Penalty-based generalization metric |
| `print_wfa_summary` | `walk_forward.py` | Formatted results table |

---

**Questions or improvements?** Open an issue or submit a pull request.

## Testing & Coverage

Run the full test suite:

```bash
pytest
```

With coverage report:

```bash
pytest --cov=src --cov-report=term-missing
```

**Current Coverage:**

```
src/__init__.py         100%  ✓
src/data_loader.py      100%  ✓
src/indicators.py       100%  ✓
src/metrics.py          100%  ✓
src/optimizer.py        100%  ✓
src/portfolio.py        100%  ✓
src/strategy.py         100%  ✓
src/walk_forward.py      82%  (mostly log branches)
src/cpp_optimizer.py     76%  (library binding fallbacks)
─────────────────────────────
TOTAL                    89%
```

**Test Suite:**  
164 tests across 9 modules, all passing. Test categories:

- `test_indicators.py` – SMA calculation edge cases
- `test_metrics.py` – Sharpe, drawdown on various market conditions
- `test_portfolio.py` – Buy/sell mechanics, spread/slippage/commission
- `test_strategy.py` – Golden/death cross logic, signal generation
- `test_optimizer.py` – Grid search, progress bar, time formatting
- `test_cpp_optimizer.py` – C++ backend availability and correctness
- `test_data_loader.py` – CSV load/download, cache behavior
- `test_inner_cv.py` – Inner CV metrics, adjusted Sharpe, parameter distance
- `test_walk_forward.py` – WFA window generation, aggregation, stability

## C++ Optimizer

The optional multithreaded C++ backend can accelerate grid search by 5–10×.

### Build

```bash
bash cpp/build.sh
```

On macOS with Apple Silicon, ensure you're using an x86-compatible Python or adjust compiler flags accordingly.

### How It Works

`src/cpp_optimizer.py` automatically detects the compiled library (`.dylib` on macOS, `.so` on Linux) and:

- **If found + valid:** Routes large grid searches to C++ (multithreaded SMA evaluation)
- **If missing/invalid:** Falls back silently to pure Python implementation

The C++ path is used only for the standard `run_backtest` function; custom backtest implementations use Python.

### Fallback Behavior

If compilation fails or the library isn't available, the system continues with pure Python—no manual intervention needed.

## File Structure

```
src/
  __init__.py              # Package initialization
  data_loader.py           # yfinance download + CSV cache
  indicators.py            # SMA and related indicators (100% covered)
  metrics.py               # Sharpe, max drawdown (100% covered)
  optimizer.py             # Grid search + C++ dispatch (100% covered)
  portfolio.py             # Portfolio & trade simulation (100% covered)
  strategy.py              # SMA crossover signals (100% covered)
  walk_forward.py          # WFA + inner CV engine (82% covered)
  cpp_optimizer.py         # C++ backend wrapper (76% covered)

tests/
  test_*.py               # 164 unit tests, all passing
  conftest.py             # Shared fixtures

cpp/
  backtest_optimizer.cpp  # Multithreaded grid search
  build.sh                # Compilation script

main.py                   # CLI entrypoint with configuration
requirements.txt          # Python dependencies
```

## Disclaimer

This codebase is for **research and educational purposes only**. It is not a substitute for professional financial advice.

- Historical backtests do not guarantee future performance
- Always test strategies on independent out-of-sample data
- Never trade with real capital without understanding all risks
- Use proper position sizing and risk management

## License

See `LICENSE` for details.
