# Backtesting Engine

A Python backtesting engine that evaluates an SMA crossover trading strategy against historical stock data.

## How It Works

```
yfinance API  -->  download_historical_data()  -->  AAPL.csv
                                                       |
                   load_csv_data()              <------+
                        |
                   SMAStrategy.generate_signals()  -->  BUY / SELL / HOLD
                        |
                   Portfolio.buy() / sell()        -->  simulated trades
                        |
                   calculate_max_drawdown()
                   calculate_sharpe_ratio()         -->  performance report
```

## Project Structure

```
src/
├── __init__.py       - Package exports
├── data_loader.py    - Download & parse historical OHLCV data (yfinance + csv)
├── indicators.py     - Technical indicators (SMA with rolling sum)
├── metrics.py        - Max drawdown & Sharpe ratio calculation
├── portfolio.py      - Virtual portfolio with commission tracking
└── strategy.py       - SMA crossover signal generation

tests/
├── test_data_loader.py
├── test_indicators.py
├── test_metrics.py
├── test_portfolio.py
└── test_strategy.py   - 47 comprehensive unit tests

data/                - Historical stock data (CSV)
main.py              - Orchestration and results output
conftest.py          - pytest configuration
requirements.txt     - Dependencies
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Edit the constants at the top of `main.py`:

| Parameter      | Default       | Description                     |
|----------------|---------------|---------------------------------|
| `TICKER`       | `"AAPL"`      | Stock ticker symbol             |
| `START_DATE`   | `"2021-01-01"`| Backtest start date             |
| `END_DATE`     | `"2026-01-01"`| Backtest end date               |
| `INITIAL_CASH` | `10,000`      | Starting capital (USD)          |
| `COMMISSION`   | `0.001`       | Transaction fee (0.1%)          |
| `SHORT_WINDOW` | `20`          | Fast SMA period (days)          |
| `LONG_WINDOW`  | `50`          | Slow SMA period (days)          |

## Example Output

```
── Backtest Results ───────────────────────────
  Initial Capital    : $  10,000.00
  Final Value (SMA)  : $  11,234.56
  Final Value (B&H)  : $  13,456.78
  Return  (SMA)      :      12.35%
  Return  (B&H)      :      34.57%
  Maximum Drawdown   :     -15.23%
  Sharpe Ratio       :      0.8421
───────────────────────────────────────────────
```

## License

MIT
