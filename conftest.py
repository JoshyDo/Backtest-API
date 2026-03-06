# conftest.py
# -----------
# Adds the project root to sys.path so that pytest can import root-level
# modules (data_loader, indicators, strategy, portfolio, metrics) from
# within the tests/ subdirectory.

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
