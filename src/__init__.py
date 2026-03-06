"""
Backtesting Engine Package

A Python backtesting engine that evaluates an SMA crossover trading strategy
against historical stock data.
"""

__version__ = "1.0.0"

from .data_loader import download_historical_data, load_csv_data
from .indicators import calculate_sma
from .metrics import calculate_max_drawdown, calculate_sharpe_ratio
from .portfolio import Portfolio
from .strategy import SMAStrategy

__all__ = [
    "download_historical_data",
    "load_csv_data",
    "calculate_sma",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "Portfolio",
    "SMAStrategy",
]
