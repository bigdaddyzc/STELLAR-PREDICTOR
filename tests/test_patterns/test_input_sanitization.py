"""Edge cases: degenerate or malformed planet inputs must not crash prediction."""

from __future__ import annotations

import math

import pytest

from stellar_predictor.data.models import ExoplanetSystem
from stellar_predictor.patterns.predictor import GapPredictor
from stellar_predictor.patterns.stability import MAX_ECCENTRICITY, StabilityAnalyzer

extract = StabilityAnalyzer.extract_planet_data_full

BASE = [("c", 1.0, 1e-6), ("d", 2.0, 1e-6), ("e", 4.0, 1e-6)]


@pytest.mark.parametrize("bad_a", [-1.0, 0.0, float("nan"), float("inf"), None, "x"])
def test_tuple_input_drops_unplaceable_axes(bad_a):
    rows = extract([("b", bad_a, 1e-6), *BASE])
    assert [r[0] for r in rows] == ["c", "d", "e"]


def test_tuple_input_with_bad_axis_predicts_finite_gaps():
    """Regression: a negative axis used to raise ZeroDivisionError, NaN gave NaN gaps."""
    for bad in (-1.0, float("nan")):
        result = GapPredictor().predict([("b", bad, 1e-6), *BASE])
        assert result.num_known_planets == 3
        assert result.predicted_gaps
        assert all(math.isfinite(g.predicted_a) and g.predicted_a > 0
                   for g in result.predicted_gaps)


def test_eccentricity_is_clamped():
    rows = extract([("b", 1.0, 1e-6, 1.5), ("c", 2.0, 1e-6, -0.2), ("d", 4.0, 1e-6, float("nan"))])
    assert [r[3] for r in rows] == [MAX_ECCENTRICITY, 0.0, 0.0]


def test_bad_mass_becomes_zero():
    rows = extract([("b", 1.0, float("nan")), ("c", 2.0, -5.0)])
    assert [r[2] for r in rows] == [0.0, 0.0]


def test_exoplanet_input_sanitized():
    system = ExoplanetSystem(name="t", stellar_mass=1.0, planets=[
        {"name": "b", "a": 0.1, "mass": float("nan"), "eccentricity": 2.0},
        {"name": "c", "a": -0.2, "mass": 1.0},
        {"name": "d", "a": 0.3, "mass": 3.0, "e": 0.1},
    ])
    rows = extract(system)
    assert [r[0] for r in rows] == ["b", "d"]
    assert rows[0][2] == pytest.approx(1e-6)  # unknown mass placeholder
    assert rows[0][3] == MAX_ECCENTRICITY
    assert rows[1][3] == pytest.approx(0.1)  # legacy "e" key still honoured


@pytest.mark.parametrize("planets", [[], [("b", 1.0, 1e-6)]])
def test_fewer_than_two_planets_warns_without_gaps(planets):
    result = GapPredictor().predict(planets)
    assert result.predicted_gaps == []
    assert result.tb_fit is None
    assert any("at least 2 planets" in w for w in result.warnings)


def test_two_planets_skip_tb_fit_but_score_gap():
    result = GapPredictor().predict([("b", 1.0, 1e-6), ("c", 4.0, 1e-6)])
    assert result.tb_fit is None  # below min_known_planets
    assert len(result.predicted_gaps) >= 1
    gap = result.predicted_gaps[0]
    assert 1.0 < gap.predicted_a < 4.0
    assert 0.0 <= gap.combined_score <= 1.0


def test_unsorted_input_is_sorted():
    rows = extract([("e", 4.0, 1e-6), ("c", 1.0, 1e-6), ("d", 2.0, 1e-6)])
    assert [r[1] for r in rows] == [1.0, 2.0, 4.0]


def test_scores_stay_in_unit_interval_for_tightly_packed_system():
    result = GapPredictor().predict([("b", 1.0, 1e-3), ("c", 1.01, 1e-3), ("d", 1.02, 1e-3)])
    for g in result.predicted_gaps:
        assert 0.0 <= g.combined_score <= 1.0
        assert math.isfinite(g.predicted_a)
