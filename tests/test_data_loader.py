"""
tests/test_data_loader.py
-------------------------
Unit tests for data_loader.py (download_historical_data, load_csv_data).
"""

import os
import tempfile
import pytest
from src.data_loader import download_historical_data, load_csv_data


class TestLoadCsvData:
    """Tests for load_csv_data function."""

    def test_load_csv_data_returns_list(self):
        """Assert that load_csv_data returns a list of dictionaries."""
        # Use existing test data
        records = load_csv_data("data/AAPL.csv")
        assert isinstance(records, list)
        assert len(records) > 0
        assert isinstance(records[0], dict)

    def test_csv_data_has_required_columns(self):
        """Assert that loaded records contain required OHLCV columns."""
        records = load_csv_data("data/AAPL.csv")
        required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
        for record in records:
            assert required_columns.issubset(set(record.keys()))

    def test_csv_numeric_columns_are_floats(self):
        """Assert that numeric columns (Open, High, Low, Close, Volume) are converted to floats."""
        records = load_csv_data("data/AAPL.csv")
        for record in records:
            assert isinstance(record["Open"], float)
            assert isinstance(record["High"], float)
            assert isinstance(record["Low"], float)
            assert isinstance(record["Close"], float)
            assert isinstance(record["Volume"], float)

    def test_csv_date_is_string(self):
        """Assert that Date column remains a string."""
        records = load_csv_data("data/AAPL.csv")
        for record in records:
            assert isinstance(record["Date"], str)

    def test_load_csv_raises_on_missing_file(self):
        """Assert that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_csv_data("data/nonexistent_file.csv")

    def test_load_csv_preserves_row_order(self):
        """Assert that records are loaded in the same order as in the CSV file."""
        records = load_csv_data("data/AAPL.csv")
        # Verify that dates are in chronological order (oldest first)
        dates = [record["Date"] for record in records]
        assert dates == sorted(dates)

    def test_csv_data_handles_empty_values(self):
        """Assert that empty or null string values are handled gracefully."""
        records = load_csv_data("data/AAPL.csv")
        # Verify that no None values appear in numeric fields
        for record in records:
            if "Close" in record:
                assert record["Close"] is not None and record["Close"] != ""


class TestDownloadHistoricalData:
    """Tests for download_historical_data function."""

    def test_download_creates_csv_file(self):
        """Assert that download_historical_data creates a CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_download.csv")
            try:
                result_path = download_historical_data(
                    ticker="AAPL",
                    start="2025-01-01",
                    end="2025-01-31",
                    output_path=output_path,
                )
                assert os.path.exists(result_path)
                assert result_path.endswith(".csv")
            except ValueError:
                # Skip if no data available for the date range
                pytest.skip("No data available for the specified ticker/date range")

    def test_download_returns_absolute_path(self):
        """Assert that download_historical_data returns an absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_download.csv")
            try:
                result_path = download_historical_data(
                    ticker="AAPL",
                    start="2025-01-01",
                    end="2025-01-31",
                    output_path=output_path,
                )
                assert os.path.isabs(result_path)
            except ValueError:
                pytest.skip("No data available for the specified ticker/date range")

    def test_download_csv_has_ohlcv_columns(self):
        """Assert that downloaded CSV contains OHLCV columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_download.csv")
            try:
                result_path = download_historical_data(
                    ticker="AAPL",
                    start="2025-01-01",
                    end="2025-01-31",
                    output_path=output_path,
                )
                records = load_csv_data(result_path)
                assert len(records) > 0
                required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
                assert required_columns.issubset(set(records[0].keys()))
            except ValueError:
                pytest.skip("No data available for the specified ticker/date range")

    def test_download_raises_on_invalid_ticker(self):
        """Assert that ValueError is raised for invalid ticker symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_invalid.csv")
            with pytest.raises(ValueError):
                download_historical_data(
                    ticker="INVALID_TICKER_XYZ123",
                    start="2025-01-01",
                    end="2025-01-31",
                    output_path=output_path,
                )

    def test_download_creates_directory_if_not_exists(self):
        """Assert that download_historical_data creates output directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "nested", "dir", "test_download.csv")
            try:
                result_path = download_historical_data(
                    ticker="AAPL",
                    start="2025-01-01",
                    end="2025-01-31",
                    output_path=nested_path,
                )
                assert os.path.exists(result_path)
                assert os.path.dirname(result_path).endswith("dir")
            except ValueError:
                pytest.skip("No data available for the specified ticker/date range")
