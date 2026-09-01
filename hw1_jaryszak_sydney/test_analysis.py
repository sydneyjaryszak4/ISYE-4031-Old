from __future__ import annotations

import numpy as np
import pytest

from analysis import (
    part1c_descriptive_stats,
    part1d_mean_ci,
    part1e_prediction_interval,
    part2_forecasts,
    part3d_verify,
    train,
)


def test_held_out_changes_do_not_affect_training_summaries_or_forecasts() -> None:
    """Held-out values must not enter training summaries or baseline construction."""
    altered_test = [0, 0, 0, 0, 0, 0]
    placeholder_test = [1, 2, 3, 4, 5, 6]

    baseline_stats = part1c_descriptive_stats(train)
    baseline_forecasts = part2_forecasts(train, placeholder_test)

    altered_forecasts = part2_forecasts(train, altered_test)

    assert part1c_descriptive_stats(train) == baseline_stats
    assert np.array_equal(
        altered_forecasts["mean_baseline"], baseline_forecasts["mean_baseline"]
    )
    assert np.array_equal(
        altered_forecasts["last_value_baseline"],
        baseline_forecasts["last_value_baseline"],
    )


def test_sample_standard_deviation_uses_n_minus_one() -> None:
    """Sample variance and standard deviation must use n - 1 in the denominator."""
    x = np.asarray(train, dtype=float)
    n = len(x)
    summaries = part1c_descriptive_stats(train)

    expected_variance = float(np.sum((x - x.mean()) ** 2) / (n - 1))
    expected_std = float(np.sqrt(expected_variance))

    assert summaries["variance_denominator"] == n - 1
    assert summaries["variance"] == pytest.approx(expected_variance)
    assert summaries["std_dev"] == pytest.approx(expected_std)
    assert summaries["std_dev"] == pytest.approx(x.std(ddof=1))


def test_prediction_interval_is_wider_than_mean_confidence_interval() -> None:
    """A prediction interval for one new observation must be wider than a mean CI."""
    mean_ci = part1d_mean_ci(train)
    prediction_interval = part1e_prediction_interval(train)

    mean_width = mean_ci["interval"][1] - mean_ci["interval"][0]
    prediction_width = (
        prediction_interval["interval"][1] - prediction_interval["interval"][0]
    )

    assert prediction_width > mean_width


def test_repeated_runs_produce_identical_results() -> None:
    """Core numerical outputs must be reproducible across repeated runs."""
    first_descriptive = part1c_descriptive_stats(train)
    first_mean_ci = part1d_mean_ci(train)
    first_prediction = part1e_prediction_interval(train)
    first_forecasts = part2_forecasts(train, [1, 2, 3, 4, 5, 6])
    first_part3 = part3d_verify()

    second_descriptive = part1c_descriptive_stats(train)
    second_mean_ci = part1d_mean_ci(train)
    second_prediction = part1e_prediction_interval(train)
    second_forecasts = part2_forecasts(train, [1, 2, 3, 4, 5, 6])
    second_part3 = part3d_verify()

    assert first_descriptive == second_descriptive
    assert first_mean_ci == second_mean_ci
    assert first_prediction == second_prediction
    assert first_part3 == second_part3
    assert np.array_equal(
        first_forecasts["mean_baseline"], second_forecasts["mean_baseline"]
    )
    assert np.array_equal(
        first_forecasts["last_value_baseline"],
        second_forecasts["last_value_baseline"],
    )
