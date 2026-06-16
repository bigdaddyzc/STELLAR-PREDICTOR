"""Calibration guard for the graded reliability score.

The reliability score (``ReliabilityFilter.reliability_score``) claims to
express how much a prediction should be trusted. This test verifies that
claim against leave-one-out ground truth: predictions the model rates as
more reliable should, on average, land closer to the true hidden planet.
Without this, the score could drift into a decorative number uncorrelated
with real accuracy.
"""

from __future__ import annotations

from stellar_predictor.data.known_systems import (
    STELLAR_INFO,
    ground_truth_planets,
    system_keys,
)
from stellar_predictor.patterns.predictor import GapPredictor
from stellar_predictor.patterns.reliability import ReliabilityFilter

_EARTH_MASSES_PER_SOLAR = 332946.0


def _collect_score_error_pairs():
    """Run LOO over every system, pairing each matched gap's reliability
    score with its true relative position error."""
    flt = ReliabilityFilter()
    pairs = []  # (reliability_score, pos_error)
    for system_name in system_keys():
        planets = ground_truth_planets(system_name)
        stellar_mass = STELLAR_INFO.get(system_name, {}).get("mass", 1.0)
        for k in range(1, len(planets) - 1):
            hidden_name, true_a, _, _ = planets[k]
            inner_name = planets[k - 1][0]
            reduced = [
                (n, a, m / _EARTH_MASSES_PER_SOLAR, e)
                for j, (n, a, m, e) in enumerate(planets) if j != k
            ]
            result = GapPredictor(stellar_mass=stellar_mass).predict(reduced)
            best, best_err = None, float("inf")
            for g in result.predicted_gaps:
                if g.inner_planet != inner_name:
                    continue
                err = abs(g.predicted_a - true_a) / true_a
                if err < best_err:
                    best_err, best = err, g
            if best is not None:
                verdict = flt.evaluate(best)
                pairs.append((verdict.score, best_err))
    return pairs


def test_reliability_score_anticorrelates_with_error():
    """Higher reliability score should mean lower position error.

    Split matched predictions at the median score; the high-reliability
    half must have a smaller mean position error than the low half.
    """
    pairs = _collect_score_error_pairs()
    assert len(pairs) >= 6, "not enough matched gaps to assess calibration"

    pairs.sort(key=lambda p: p[0])  # by score ascending
    mid = len(pairs) // 2
    low_half = pairs[:mid]
    high_half = pairs[mid:]

    mean_err_low = sum(e for _, e in low_half) / len(low_half)
    mean_err_high = sum(e for _, e in high_half) / len(high_half)

    assert mean_err_high <= mean_err_low + 1e-9, (
        f"reliability score is miscalibrated: high-score gaps mean err "
        f"{mean_err_high:.3f} > low-score gaps mean err {mean_err_low:.3f}"
    )


def test_all_scores_in_unit_range():
    """Every reliability score stays within [0, 1]."""
    for score, _ in _collect_score_error_pairs():
        assert 0.0 <= score <= 1.0
