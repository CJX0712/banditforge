"""Tests for evaluation metrics."""

from eval.metrics import relative_reduction, significance_mean_diff


def test_significance_true():
    # |10 - 12| = 2 > 0.5*(1+1) = 1
    assert significance_mean_diff(10.0, 1.0, 12.0, 1.0) is True


def test_significance_false():
    # |10 - 10.5| = 0.5 < 0.5*(1+1) = 1
    assert significance_mean_diff(10.0, 1.0, 10.5, 1.0) is False


def test_relative_reduction():
    assert abs(relative_reduction(100.0, 50.0) - 0.5) < 1e-12
    assert abs(relative_reduction(100.0, 110.0) + 0.1) < 1e-12


def test_relative_reduction_zero_base():
    assert relative_reduction(0.0, 5.0) == 0.0
