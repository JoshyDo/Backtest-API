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

All configuration is in `main.py`. Edit the top-level variables to customize:

```python
# ═══════════════════════════════════════════════════════════════════════
# DATA & BACKTEST
# ═══════════════════════════════════════════════════════════════════════
TICKER = "DAX"                  # Ticker symbol
START_DATE = "2000-01-01"       # Backtest start date
END_DATE = "2027-01-01"         # Backtest end date
INITIAL_CASH = 10_000.0         # Starting capital

# Transaction costs
COMMISSION = 0.001              # 0.1% per trade
SLIPPAGE = 0.002                # 0.2% market impact
SPREAD_MIN = 0.001              # 0.1% bid-ask minimum
SPREAD_MAX = 0.002              # 0.2% bid-ask maximum

# ═══════════════════════════════════════════════════════════════════════
# SMA PARAMETERS (for single backtest mode)
# ═══════════════════════════════════════════════════════════════════════
SHORT_WINDOW = 20               # Fast-moving average
LONG_WINDOW = 50                # Slow-moving average

# ═══════════════════════════════════════════════════════════════════════
# GRID SEARCH (parameter optimization)
# ═══════════════════════════════════════════════════════════════════════
GRID_SEARCH_ENABLED = True      # Enable grid search?
GRID_SEARCH_FAST_MIN = 15       # Short SMA: min
GRID_SEARCH_FAST_MAX = 35       # Short SMA: max
GRID_SEARCH_SLOW_MIN = 40       # Long SMA: min
GRID_SEARCH_SLOW_MAX = 100      # Long SMA: max

# ═══════════════════════════════════════════════════════════════════════
# WALK-FORWARD ANALYSIS (rolling windows + inner CV)
# ═══════════════════════════════════════════════════════════════════════
WALK_FORWARD_ENABLED = True     # Enable WFA?
WFA_IS_WINDOW_DAYS = 504        # In-sample period (~2 trading years)
WFA_OOS_WINDOW_DAYS = 252       # Out-of-sample period (~1 trading year)
WFA_STEP_SIZE_DAYS = 252        # Roll forward by (1 trading year)
WFA_WARMUP_DAYS = 406           # Warmup buffer (prevent indicator leakage)
```

### Execution Modes

| Setting | Behavior |
|---------|----------|
| `GRID_SEARCH_ENABLED=False` | Single backtest with fixed `SHORT_WINDOW` / `LONG_WINDOW` |
| `GRID_SEARCH_ENABLED=True` only | Grid search on full period to find best parameters |
| Both enabled | Grid search + WFA with inner CV on rolling windows |

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
| `WalkForwardWindow` | `walk_forward.py` | Single WFA iteration window (IS/OOS split) |
| `WalkForwardResult` | `walk_forward.py` | Stores per-window metrics (Sharpe, params, equity) |
| `InnerCVResult` | `walk_forward.py` | Inner CV result for parameter combination |

### Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `run_grid_search` | `optimizer.py` | Parameter optimization (Python/C++) |
| `run_backtest` | `strategy.py` | Executes strategy on data |
| `calculate_sma` | `indicators.py` | Simple moving average calculation |
| `calculate_sharpe_ratio` | `metrics.py` | Risk-adjusted return metric |
| `calculate_max_drawdown` | `metrics.py` | Maximum peak-to-trough decline |
| `calculate_adjusted_sharpe` | `walk_forward.py` | Penalty-based generalization metric |
| `calculate_relative_parameter_distance` | `walk_forward.py` | Relative parameter stability metric |
| `run_inner_cross_validation` | `walk_forward.py` | Layer 2 robust parameter selection |
| `split_is_data` | `walk_forward.py` | Split in-sample into train/validate |
| `aggregate_oos_equity` | `walk_forward.py` | Combine OOS results across windows |
| `print_wfa_summary` | `walk_forward.py` | Formatted WFA results table |

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

**Current Coverage (164 tests passing):**

```
src/__init__.py         100%  ✓
src/data_loader.py      100%  ✓
src/indicators.py       100%  ✓
src/metrics.py          100%  ✓
src/optimizer.py        100%  ✓
src/portfolio.py        100%  ✓
src/strategy.py         100%  ✓
src/walk_forward.py      82%  (log branches, extreme drift paths)
src/cpp_optimizer.py     76%  (library binding fallbacks)
─────────────────────────────
TOTAL                    89%  (548 statements, 58 missed)
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

**Note:** This project was developed with assistance from AI tools for architecture generation and testing, as this is a learning project. All functionality has been thoroughly reviewed and tested.

## License

See `LICENSE` for details.
