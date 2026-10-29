"""Scoring v2: TB gating, resonant-chain down-weighting, period units."""

from __future__ import annotations

import math

import pytest

from stellar_predictor.data.known_systems import STELLAR_INFO, build_system
from stellar_predictor.patterns.predictor import GapPredictor
from stellar_predictor.patterns.reliability import filter_gaps
from stellar_predictor.prediction.pipeline import PredictionPipeline


@pytest.fixture(scope="module")
def solar():
    return PredictionPipeline().analyze(build_system("solar_system"))


@pytest.fixture(scope="module")
def trappist():
    return PredictionPipeline().analyze(build_system("trappist1"))


def _pair(result, inner, outer):
    return max((g for g in result.predicted_gaps
                if (g.inner_planet, g.outer_planet) == (inner, outer)),
               key=lambda g: g.combined_score)


def test_tb_gate_caps_stability_only_signal():
    assert GapPredictor._tb_gate(0.0) == pytest.approx(0.15)
    assert GapPredictor._tb_gate(1.0) == pytest.approx(1.0)
    assert GapPredictor._tb_gate(0.5) > GapPredictor._tb_gate(0.1)


def test_tb_step_score_prefers_whole_missing_slots():
    beta = 1.534
    mars_jupiter = GapPredictor._tb_step_score(1.5237, 5.2034, beta)   # ~2.9 steps
    saturn_uranus = GapPredictor._tb_step_score(9.5371, 19.1913, beta)  # ~1.6 steps
    mercury_venus = GapPredictor._tb_step_score(0.3871, 0.7233, beta)   # ~1.5 steps
    earth_mars = GapPredictor._tb_step_score(1.0, 1.5237, beta)          # ~1 step
    assert mars_jupiter > 0.7
    assert mars_jupiter > 2 * saturn_uranus
    assert mars_jupiter > 4 * mercury_venus
    assert earth_mars == 0.0


def test_solar_top_gap_separated_from_false_positives(solar):
    top = solar.predicted_gaps[0]
    assert (top.inner_planet, top.outer_planet) == ("Mars", "Jupiter")
    for inner, outer in (("Mercury", "Venus"), ("Saturn", "Uranus")):
        fp = _pair(solar, inner, outer)
        assert top.combined_score - fp.combined_score >= 0.30, (inner, outer, fp.combined_score)


def test_zero_tb_gaps_cannot_score_high(solar):
    for g in solar.predicted_gaps:
        if g.titius_bode_score == 0.0:
            assert g.combined_score < 0.30, (g.inner_planet, g.outer_planet, g.combined_score)


def test_resonant_chain_interior_gaps_downweighted(trappist):
    interior = [g for g in trappist.predicted_gaps if "outer edge" not in g.outer_planet]
    assert interior
    assert max(g.combined_score for g in interior) < 0.20
    reliable, _ = filter_gaps(trappist.predicted_gaps)
    assert all("outer edge" in g.outer_planet for g in reliable)
    assert any("resonant chain" in w for w in trappist.warnings)


def test_chain_lock_separates_chain_from_tb_system():
    trappist = [0.01154, 0.01580, 0.02227, 0.02925, 0.03849, 0.04683, 0.06189]
    solar = [0.3871, 0.7233, 1.0, 1.5237, 5.2034, 9.5371, 19.1913, 30.069]
    assert GapPredictor._resonant_chain_lock(trappist) == pytest.approx(1.0)
    assert GapPredictor._resonant_chain_lock(solar) < 0.5
    assert GapPredictor.chain_penalty_factor(0.29) == pytest.approx(1.0)
    assert GapPredictor.chain_penalty_factor(1.0) < 0.5


def test_extrapolation_factor_decreases_per_step():
    f = [GapPredictor.extrapolation_factor(k) for k in (1, 2, 3)]
    assert f[0] < 1.0 and f[0] > f[1] > f[2]


def test_period_uses_stellar_mass(trappist):
    star = STELLAR_INFO["trappist1"]["mass"]
    for g in trappist.predicted_gaps:
        assert g.predicted_period == pytest.approx(
            math.sqrt(g.predicted_a ** 3 / star), rel=1e-2)  # predicted_a is rounded
