"""
Responsible for downloading and reading historical price data.
"""

import csv
import os

import yfinance as yf


def download_historical_data(ticker: str, start: str, end: str, output_path: str) -> str:
    """
    Downloads historical daily prices via yfinance and saves them as CSV.

    Args:
        ticker:      Stock ticker symbol (e.g. "AAPL").
        start:       Start date "YYYY-MM-DD" (inclusive).
        end:         End date "YYYY-MM-DD" (exclusive).
        output_path: File path for the CSV output.

    Returns:
        Absolute path to the saved CSV file.

    Raises:
        ValueError: If no data is found for the given ticker/period.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        raise ValueError(f"No data found for '{ticker}' in {start} to {end}.")

    if hasattr(raw.columns, "levels"):
        raw.columns = raw.columns.get_level_values(0)

    raw.index.name = "Date"
    raw.to_csv(output_path)

    return os.path.abspath(output_path)


def load_csv_data(filepath: str) -> list[dict]:
    """
    Reads a CSV file with historical price data into a list of dicts.

    Numeric columns (Open, High, Low, Close, Volume) are converted to float.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of OHLCV dictionaries with Date as string and numeric values as float.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    records: list[dict] = []
    numeric_columns = {"Open", "High", "Low", "Close", "Volume"}

    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for col in numeric_columns:
                if col in row and row[col] not in ("", "null", "None"):
                    row[col] = float(row[col])
            records.append(dict(row))

    return records
