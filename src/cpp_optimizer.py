"""
Python wrapper for C++ multithreaded grid search optimizer.

This module provides a ctypes interface to the compiled C++ library,
enabling fast parallel backtest optimization from Python.
"""

import ctypes
import os
import sys
import platform
from pathlib import Path
from typing import TypedDict


class OptimizationResult(TypedDict):
    """Result structure matching C++ BacktestResult"""
    short_window: int
    long_window: int
    sharpe_ratio: float
    final_value: float
    max_drawdown: float


class _OptimizationResultC(ctypes.Structure):
    """ctypes structure for C++ OptimizationResult"""
    _fields_ = [
        ("short_window", ctypes.c_int),
        ("long_window", ctypes.c_int),
        ("sharpe_ratio", ctypes.c_double),
        ("final_value", ctypes.c_double),
        ("max_drawdown", ctypes.c_double),
    ]


def _get_library_path() -> str:
    """
    Find the compiled C++ library.
    
    Returns:
        Path to the compiled library (.dylib on macOS, .so on Linux)
    
    Raises:
        FileNotFoundError: If library not found
        RuntimeError: If OS not supported
    """
    script_dir = Path(__file__).parent.parent / "cpp"
    
    if platform.system() == "Darwin":
        lib_path = script_dir / "backtest_optimizer.dylib"
    elif platform.system() == "Linux":
        lib_path = script_dir / "backtest_optimizer.so"
    else:
        raise RuntimeError(f"Unsupported OS: {platform.system()}")
    
    if not lib_path.exists():
        raise FileNotFoundError(
            f"C++ library not found at {lib_path}\n"
            f"Please build it first: bash cpp/build.sh"
        )
    
    return str(lib_path)


def _load_library() -> ctypes.CDLL:
    """
    Load the C++ library using ctypes.
    
    Returns:
        Loaded ctypes library
    
    Raises:
        FileNotFoundError: If library not found
        OSError: If library cannot be loaded
    """
    lib_path = _get_library_path()
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise OSError(f"Failed to load C++ library: {e}")
    
    return lib


# Load library once at module import
try:
    _lib = _load_library()
except Exception as e:
    print(f"Warning: Could not load C++ optimizer: {e}")
    print("Grid search will fall back to pure Python implementation.")
    _lib = None


def grid_search_multithreaded_cpp(
    prices: list[float],
    fast_min: int,
    fast_max: int,
    slow_min: int,
    slow_max: int,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    num_threads: int = -1,
) -> OptimizationResult:
    """
    Run grid search optimization using C++ with multithreading.
    
    Args:
        prices: List of closing prices
        fast_min: Minimum short SMA window (inclusive)
        fast_max: Maximum short SMA window (exclusive)
        slow_min: Minimum long SMA window (inclusive)
        slow_max: Maximum long SMA window (exclusive)
        initial_cash: Starting capital in USD
        commission: Transaction commission as decimal (0.001 = 0.1%)
        num_threads: Number of worker threads (-1 = auto-detect)
    
    Returns:
        OptimizationResult with best parameters and metrics
    
    Raises:
        RuntimeError: If C++ library not available
        ValueError: If parameters invalid
        OSError: If library call fails
    
    Example:
        >>> prices = [100.0, 101.5, 102.3, ...]
        >>> result = grid_search_multithreaded_cpp(
        ...     prices, fast_min=5, fast_max=50,
        ...     slow_min=20, slow_max=100
        ... )
        >>> print(f"Best: SMA({result['short_window']}, {result['long_window']})")
        >>> print(f"Sharpe: {result['sharpe_ratio']:.4f}")
    """
    if _lib is None:
        raise RuntimeError(
            "C++ library not loaded. Please build it: bash cpp/build.sh"
        )
    
    # Validate inputs
    if not prices or len(prices) < 2:
        raise ValueError("Need at least 2 price points")
    
    if fast_min < 1 or fast_max <= fast_min:
        raise ValueError("Invalid fast window range")
    
    if slow_min < 1 or slow_max <= slow_min:
        raise ValueError("Invalid slow window range")
    
    # Convert Python list to C array
    prices_c = (ctypes.c_double * len(prices))(*prices)
    
    # Set up function signature
    _lib.grid_search_multithreaded.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # prices
        ctypes.c_int,                      # num_prices
        ctypes.c_int,                      # fast_min
        ctypes.c_int,                      # fast_max
        ctypes.c_int,                      # slow_min
        ctypes.c_int,                      # slow_max
        ctypes.c_double,                   # initial_cash
        ctypes.c_double,                   # commission
        ctypes.c_int,                      # num_threads
        ctypes.POINTER(_OptimizationResultC),  # result
    ]
    _lib.grid_search_multithreaded.restype = ctypes.c_int
    
    # Allocate result structure
    result_c = _OptimizationResultC()
    
    # Call C++ function
    error_code = _lib.grid_search_multithreaded(
        prices_c,
        len(prices),
        fast_min,
        fast_max,
        slow_min,
        slow_max,
        initial_cash,
        commission,
        num_threads,
        ctypes.byref(result_c),
    )
    
    if error_code != 0:
        error_messages = {
            1: "Invalid input parameters (null pointer or empty data)",
            2: "Invalid parameter ranges (fast/slow windows)",
            3: "No valid parameter combinations found",
        }
        raise OSError(f"Grid search failed: {error_messages.get(error_code, f'Unknown error {error_code}')}")
    
    # Convert result back to Python dict
    return OptimizationResult(
        short_window=result_c.short_window,
        long_window=result_c.long_window,
        sharpe_ratio=result_c.sharpe_ratio,
        final_value=result_c.final_value,
        max_drawdown=result_c.max_drawdown,
    )


def is_cpp_available() -> bool:
    """
    Check if C++ library is available and loaded.
    
    Returns:
        True if C++ optimizer is ready to use, False otherwise
    """
    return _lib is not None
