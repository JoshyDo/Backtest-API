# Backtesting Engine

A Python backtesting framework that evaluates an SMA crossover trading strategy against historical stock market data. Includes grid-search parameter optimization, comprehensive performance metrics, and buy-and-hold benchmarking.

## Features

Core Features:
- SMA crossover strategy with configurable window periods
- Grid-search optimization to find best-performing parameters
- Performance metrics: Sharpe ratio, max drawdown, annualized returns
- Virtual portfolio with realistic commission tracking
- Buy-and-hold benchmark comparison
- Automatic data download via yfinance with CSV caching

Production-Ready:
- 100+ comprehensive unit tests with **93% code coverage**
- Type hints throughout for code clarity
- Clean architecture with modular design
- Efficient algorithms (O(n) SMA calculation)
- Detailed docstrings and error handling
- C++ multithreaded optimizer for 10-50x performance boost

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JoshyDo/backtest-api.git
cd backtest-api

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Backtest

Simply execute the main script:

```bash
python main.py
```

This will run a backtest with the configured parameters and display results in the terminal.

### Configuration

Edit the configuration at the top of `main.py`:

```python
TICKER = "DAX"              # Stock ticker symbol
START_DATE = "2021-01-01"   # Backtest period start
END_DATE = "2027-01-01"     # Backtest period end
INITIAL_CASH = 10_000.0     # Starting capital (USD)
COMMISSION = 0.001          # Transaction fee (0.1%)
SHORT_WINDOW = 20           # Fast SMA period (days)
LONG_WINDOW = 50            # Slow SMA period (days)
GRID_SEARCH_ENABLED = True  # Enable parameter optimization
```

### Grid Search Optimization

To find optimal SMA parameters, set `GRID_SEARCH_ENABLED = True`:

```python
GRID_SEARCH_FAST_MIN = 5      # Short SMA range: 5-99 days
GRID_SEARCH_FAST_MAX = 100
GRID_SEARCH_SLOW_MIN = 15     # Long SMA range: 15-499 days
GRID_SEARCH_SLOW_MAX = 500
```

The system will test all valid combinations exhaustively and display the best parameters found.

**Performance**: With C++ optimizer enabled, tests 46,075 parameter combinations in seconds (10-50x faster than pure Python).

**Dual Results**: Grid search now returns both:
- **Best Sharpe Ratio**: Risk-adjusted returns optimization
- **Best Returns**: Absolute return optimization

This allows you to choose between conservative (lower volatility) or aggressive (maximum returns) strategies.


## Example Output

```
Backtest Results
  SMA Parameters     : (12, 47)
  Initial Capital    : $  10,000.00
  Final Value (SMA)  : $  11,234.56
  Final Value (B&H)  : $  13,456.78
  Return  (SMA)      :      12.35%
  Return  (B&H)      :      34.57%
  Return per Year    :       2.35%
  Maximum Drawdown   :     -18.32%
  Sharpe Ratio       :       0.6234
```

## Architecture

```
src/
├── __init__.py        - Package exports
├── data_loader.py     - Download & cache historical OHLCV data
├── indicators.py      - SMA calculation (optimized O(n) algorithm)
├── metrics.py         - Sharpe ratio & max drawdown calculation
├── portfolio.py       - Virtual portfolio with commission tracking
├── strategy.py        - SMA crossover signal generation
└── optimizer.py       - Grid-search parameter optimization

tests/
├── test_data_loader.py    - Data loading tests
├── test_indicators.py      - SMA calculation tests
├── test_metrics.py         - Performance metrics tests
├── test_optimizer.py       - Grid-search optimization tests
├── test_portfolio.py       - Portfolio simulation tests
└── test_strategy.py        - Signal generation tests

main.py               - CLI entry point & orchestration
requirements.txt      - Package dependencies
conftest.py           - Pytest configuration
```

## SMA Crossover Strategy

The strategy generates signals based on two Simple Moving Averages:

- BUY Signal (Golden Cross): Fast SMA crosses above slow SMA
- SELL Signal (Death Cross): Fast SMA crosses below slow SMA
- HOLD: No crossover detected

Example with SMA(20, 50):

```
Day 1-19:    HOLD (insufficient data for SMA(50))
Day 20:      Calculate both SMAs
Day 50+:     Generate BUY/SELL signals based on crossovers
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/ --cov-report=term-missing

# Run specific test file
pytest tests/test_strategy.py -v

# Run a single test
pytest tests/test_metrics.py::TestCalculateSharpeRatio -v
```

**Test Coverage**: 100+ tests achieve **93% code coverage** across all modules.

Test files include:
- `test_data_loader.py` - Data loading tests (100% coverage)
- `test_indicators.py` - SMA calculation tests (100% coverage)
- `test_metrics.py` - Performance metrics tests (100% coverage)
- `test_optimizer.py` - Grid-search optimization tests (94% coverage)
- `test_portfolio.py` - Portfolio simulation tests (100% coverage)
- `test_strategy.py` - Signal generation tests (100% coverage)
- `test_cpp_optimizer.py` - C++ optimizer integration tests (77% coverage)

Module coverage breakdown:
```
src/__init__.py       - 100%
src/cpp_optimizer.py  -  77%  (ctypes binding layer, some paths hard to test)
src/data_loader.py    - 100%
src/indicators.py     - 100%
src/metrics.py        - 100%
src/optimizer.py      -  94%  (C++ detection paths excluded from Python tests)
src/portfolio.py      - 100%
src/strategy.py       - 100%
```

## Performance Metrics

### Sharpe Ratio
Measures risk-adjusted returns. Formula:

```
Sharpe = (avg_daily_return - risk_free_rate) / std_dev_return * sqrt(252)
```

Higher is better. Values > 1.0 indicate good risk-adjusted returns.

### Maximum Drawdown
Largest peak-to-trough decline:

```
MDD = (current_low - previous_peak) / previous_peak
```

Represents worst-case loss. More negative values indicate larger drawdowns.

### Annualized Return
Total return scaled to yearly basis:

```
Annual Return = (final_value / initial_value)^(1/years) - 1
```

## Optimization Algorithm

The grid-search optimizer exhaustively tests all combinations of parameters to find the best Sharpe ratio:

1. **Validation**: Ensures data has sufficient records for both SMA periods
2. **Combination Enumeration**: Tests all (short, long) pairs where short < long
3. **Backtest Execution**: Runs full backtests for each parameter combination
4. **Metric Calculation**: Computes Sharpe ratio, max drawdown, final value
5. **Result Selection**: Returns parameters with highest Sharpe ratio AND highest returns
6. **Progress Tracking**: Shows real-time progress with ETA (Python) or runs in parallel (C++)

A typical 75×185 grid (13,875 parameter sets) tests all combinations exhaustively.

### C++ Multithreaded Optimizer

For production use, the system automatically uses a compiled C++ optimizer when available:

**Building the C++ Optimizer**:
```bash
bash cpp/build.sh
```

**Benefits**:
- 10-50x faster than pure Python
- Automatic multithreading across all CPU cores
- Seamless fallback to Python if C++ not available
- Same results as Python implementation

**Technical Details**:
- Compiled to native shared library (`.dylib` on macOS, `.so` on Linux)
- Called via ctypes from Python without external dependencies
- Thread-safe result aggregation with mutex protection
- Optimized cache locality for SMA calculations

## Dependencies

| Package   | Version | Purpose                    |
|-----------|---------|----------------------------|
| yfinance  | ≥0.2.0  | Historical market data API |

Python 3.9+ required.

## Contributing

Contributions welcome! Areas for future improvement:

- Additional technical indicators (EMA, RSI, MACD)
- Visualization of optimization results (heatmap in `plot_heatmap()`)
- Multi-asset portfolio support
- Walk-forward analysis for robustness testing
- Performance visualization (matplotlib/plotly)
- Execution simulation with slippage/spread
- Paper trading interface
- Risk management features (stop-loss, position sizing)
- Machine learning parameter optimization

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Disclaimer

This backtesting engine is for educational and research purposes only. Past performance does not guarantee future results. Live trading involves substantial risk. Always test strategies thoroughly before deploying real capital.
