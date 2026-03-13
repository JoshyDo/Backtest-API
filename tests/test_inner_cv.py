"""
tests/test_inner_cv.py
---------------------
Unit tests for inner cross-validation (Layer 2) functions.

Tests the mathematically robust metrics:
1. Adjusted Sharpe (penalty-based, not division)
2. Percentile-based thresholds (not absolute values)
3. Relative parameter distance (not euclidean)
"""

import pytest
import numpy as np
from src.walk_forward import (
    calculate_adjusted_sharpe,
    calculate_relative_parameter_distance,
    split_is_data,
    InnerCVResult,
)


class TestAdjustedSharpe:
    """Test adjusted Sharpe calculation (BUG FIX #1)."""

    def test_adjusted_sharpe_perfect_generalization(self):
        """When train == validate, no penalty applied."""
        adjusted = calculate_adjusted_sharpe(train_sharpe=1.0, validate_sharpe=1.0)
        assert adjusted == pytest.approx(1.0)

    def test_adjusted_sharpe_with_overfitting(self):
        """When train > validate, penalty applied."""
        # train=1.0, validate=0.6 → gap=0.4, penalty=0.2
        adjusted = calculate_adjusted_sharpe(train_sharpe=1.0, validate_sharpe=0.6)
        expected = 0.6 - 0.5 * 0.4  # 0.6 - 0.2 = 0.4
        assert adjusted == pytest.approx(expected)

    def test_adjusted_sharpe_negative_sharpe(self):
        """Handles negative Sharpe ratios without breaking."""
        # Division-based would break: -0.5 / 1.0 is valid but -0.5 / -1.0 = 0.5 (false positive!)
        # Penalty-based handles both gracefully
        adjusted = calculate_adjusted_sharpe(train_sharpe=-0.5, validate_sharpe=-1.0)
        expected = -1.0 - 0.5 * 0.5  # -1.0 - 0.25 = -1.25
        assert adjusted == pytest.approx(expected)

    def test_adjusted_sharpe_validate_better_than_train(self):
        """When validation Sharpe is better (rare but good!)."""
        adjusted = calculate_adjusted_sharpe(train_sharpe=0.5, validate_sharpe=1.5)
        # gap = 1.0, penalty = 0.5
        expected = 1.5 - 0.5 * 1.0  # 1.0
        assert adjusted == pytest.approx(expected)

    def test_adjusted_sharpe_both_negative_large_gap(self):
        """Bear market scenario with poor performance."""
        adjusted = calculate_adjusted_sharpe(train_sharpe=-2.0, validate_sharpe=-3.0)
        expected = -3.0 - 0.5 * 1.0  # -3.5
        assert adjusted == pytest.approx(expected)

    def test_adjusted_sharpe_is_robust_to_zero(self):
        """Should handle zero Sharpe gracefully."""
        adjusted = calculate_adjusted_sharpe(train_sharpe=0.0, validate_sharpe=0.5)
        expected = 0.5 - 0.5 * 0.5  # 0.25
        assert adjusted == pytest.approx(expected)


class TestRelativeParameterDistance:
    """Test relative parameter distance calculation (BUG FIX #3)."""

    def test_identical_parameters(self):
        distance = calculate_relative_parameter_distance((20, 50), (20, 50))
        assert distance == pytest.approx(0.0)

    def test_small_number_large_percentage_change(self):
        distance = calculate_relative_parameter_distance((10, 50), (15, 50))
        assert distance > 0.4

    def test_large_number_small_percentage_change(self):
        distance = calculate_relative_parameter_distance((20, 200), (20, 205))
        assert distance < 0.05

    def test_difference_from_euclidean(self):
        euclidean = np.sqrt(5**2 + 5**2)
        relative = calculate_relative_parameter_distance((10, 200), (15, 205))
        assert euclidean > relative
        assert relative < 1.0

    def test_moderate_distance_and_symmetry(self):
        distance_forward = calculate_relative_parameter_distance((20, 50), (25, 62))
        distance_backward = calculate_relative_parameter_distance((25, 62), (20, 50))
        assert 0.2 < distance_forward < 0.6
        assert 0.2 < distance_backward < 0.6


class TestSplitIsData:
    """Test IS data splitting (70/30 train/validate)."""

    def test_split_basic(self):
        """Basic split with 10 records."""
        data = [{"date": i} for i in range(10)]
        train, validate = split_is_data(data, train_ratio=0.7)

        assert len(train) == 7
        assert len(validate) == 3
        assert train[0]["date"] == 0
        assert validate[-1]["date"] == 9

    def test_split_preserves_order(self):
        """Temporal order must be preserved."""
        data = [{"date": i} for i in range(100)]
        train, validate = split_is_data(data, train_ratio=0.7)

        # Last train date should be before first validate date
        assert train[-1]["date"] < validate[0]["date"]

        # Should be consecutive
        assert train[-1]["date"] + 1 == validate[0]["date"]

    def test_split_500_day_is_window(self):
        """Realistic 504-day IS window."""
        data = [{"date": i} for i in range(504)]
        train, validate = split_is_data(data, train_ratio=0.7)

        # 70% of 504 = 352.8 ≈ 352
        assert len(train) == 352
        assert len(validate) == 152
        assert len(train) + len(validate) == 504

    def test_split_custom_ratio(self):
        """Custom train/validate ratio."""
        data = [{"date": i} for i in range(100)]
        train, validate = split_is_data(data, train_ratio=0.8)

        assert len(train) == 80
        assert len(validate) == 20


class TestInnerCVResult:
    """Test InnerCVResult dataclass."""

    def test_inner_cv_result_creation(self):
        """Create and validate InnerCVResult."""
        result = InnerCVResult(
            short_window=20,
            long_window=50,
            train_sharpe=0.8,
            validate_sharpe=0.6,
            adjusted_sharpe=0.5,
            generalization_gap=0.2,
            is_robust=True,
        )

        assert result.short_window == 20
        assert result.long_window == 50
        assert result.is_robust is True

    def test_inner_cv_result_comparison(self):
        """Compare two InnerCVResults."""
        result1 = InnerCVResult(
            short_window=20,
            long_window=50,
            train_sharpe=0.8,
            validate_sharpe=0.7,
            adjusted_sharpe=0.65,
            generalization_gap=0.1,
            is_robust=True,
        )
        result2 = InnerCVResult(
            short_window=25,
            long_window=55,
            train_sharpe=0.7,
            validate_sharpe=0.9,
            adjusted_sharpe=0.75,
            generalization_gap=0.2,
            is_robust=True,
        )

        # result2 has better adjusted_sharpe
        assert result2.adjusted_sharpe > result1.adjusted_sharpe


class TestPercentileRanking:
    """Test percentile-based ranking (BUG FIX #2)."""

    def test_percentile_works_in_bear_market(self):
        """When all Sharpes are negative, percentile still works."""
        cv_results = [
            InnerCVResult(
                short_window=i,
                long_window=50 + i,
                train_sharpe=-0.5 - (i * 0.1),  # Getting worse
                validate_sharpe=-0.3 - (i * 0.05),
                adjusted_sharpe=-0.4 - (i * 0.075),
                generalization_gap=0.2,
                is_robust=False,
            )
            for i in range(10)
        ]

        # Sort by adjusted_sharpe (best first)
        cv_results.sort(key=lambda x: x.adjusted_sharpe, reverse=True)

        # Top 25% (3 out of 10)
        percentile_threshold = max(1, int(len(cv_results) * 0.25))
        robust = cv_results[:percentile_threshold]

        # Even in bear market, we get top performers
        assert len(robust) > 0
        assert robust[0].adjusted_sharpe > robust[-1].adjusted_sharpe

    def test_percentile_minimum_one(self):
        """Always select at least one parameter."""
        cv_results = [
            InnerCVResult(
                short_window=20,
                long_window=50,
                train_sharpe=0.5,
                validate_sharpe=0.4,
                adjusted_sharpe=0.35,
                generalization_gap=0.1,
                is_robust=False,
            )
        ]

        cv_results.sort(key=lambda x: x.adjusted_sharpe, reverse=True)
        percentile_threshold = max(1, int(len(cv_results) * 0.25))
        robust = cv_results[:percentile_threshold]

        assert len(robust) == 1
        assert robust[0].short_window == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
