"""Retrodiction *ranking* regression tests (scoring v2).

``test_retrodiction.py`` only checks that a hidden planet's position is
recovered somewhere in the candidate list. These tests check that the
scoring puts the true location at the *top*, which is what a user reads.

Measured at v0.7 (scoring v2), see the PR description for the before/after:

* pre-Neptune (Mercury..Uranus): best beyond-Uranus candidate 30.97 AU
  (Neptune 30.07 AU, 3.0 % error), ranked #1 of the outer-edge candidates.
  Overall it ranks #3, behind only the Mars-Jupiter gap and its sub-slot,
  which mark the real asteroid belt. (v0.6: 3rd of the outer candidates,
  behind 73 AU and 48 AU.)
* pre-Uranus (Mercury..Saturn): Uranus sits ~1.6 TB steps beyond Saturn
  (19.19 / 9.54 = 2.01 vs beta ~1.52), so no single-step log-linear slot can
  land on it. The +1 slot (13.7 AU) ranks first; the +2 slot (21.2 AU, 10.7 %
  error) ranks second and its uncertainty band contains 19.19 AU. We assert
  top-2 with that documented tolerance rather than pretending #1 is
  physically reachable.
* TRAPPIST-1 minus e: the d-f gap spans ~2 TB steps, and its candidate
  (0.0279 AU vs 0.0293 AU, 4.6 % error) is ranked #1 overall.
"""

from __future__ import annotations

import pytest

from stellar_predictor.data.known_systems import STELLAR_INFO, ground_truth_planets
from stellar_predictor.patterns.predictor import GapPredictor

_EARTH_PER_SUN = 332946.0


def _tuples(system: str, drop: tuple[str, ...] = ()) -> list[tuple]:
    return [(n, a, m / _EARTH_PER_SUN, e)
            for n, a, m, e in ground_truth_planets(system) if n not in drop]


def _true_a(system: str, name: str) -> float:
    return next(a for n, a, _, _ in ground_truth_planets(system) if n == name)


def _outer_edge(gaps, inner: str):
    return [g for g in gaps if g.inner_planet == inner and "outer edge" in g.outer_planet]


@pytest.fixture(scope="module")
def pre_neptune():
    return GapPredictor(stellar_mass=1.0).predict(_tuples("solar_system", ("Neptune",)))


@pytest.fixture(scope="module")
def pre_uranus():
    return GapPredictor(stellar_mass=1.0).predict(
        _tuples("solar_system", ("Uranus", "Neptune")))


def test_pre_neptune_true_location_ranked_first_beyond_uranus(pre_neptune):
    neptune = _true_a("solar_system", "Neptune")
    edges = _outer_edge(pre_neptune.predicted_gaps, "Uranus")
    assert len(edges) >= 2
    top = edges[0]
    err = abs(top.predicted_a - neptune) / neptune
    assert err < 0.05, f"top beyond-Uranus candidate {top.predicted_a:.2f} AU, err {err:.1%}"


def test_pre_neptune_only_asteroid_belt_outranks_neptune(pre_neptune):
    neptune = _true_a("solar_system", "Neptune")
    gaps = pre_neptune.predicted_gaps
    rank = next(i for i, g in enumerate(gaps) if abs(g.predicted_a - neptune) / neptune < 0.05)
    above = gaps[:rank]
    assert all((g.inner_planet, g.outer_planet) == ("Mars", "Jupiter") for g in above), (
        [f"{g.inner_planet}-{g.outer_planet}" for g in above])


def test_extrapolation_penalty_orders_outer_edge_by_step(pre_neptune):
    edges = _outer_edge(pre_neptune.predicted_gaps, "Uranus")
    steps = [int(g.outer_planet.split("+")[1].rstrip(")")) for g in edges]
    assert steps == sorted(steps), f"outer-edge candidates out of step order: {steps}"
    scores = [g.combined_score for g in edges]
    assert all(a > b for a, b in zip(scores, scores[1:])), scores


def test_pre_uranus_within_top2_documented_tolerance(pre_uranus):
    uranus = _true_a("solar_system", "Uranus")
    edges = _outer_edge(pre_uranus.predicted_gaps, "Saturn")
    top2 = edges[:2]
    best = min(top2, key=lambda g: abs(g.predicted_a - uranus))
    err = abs(best.predicted_a - uranus) / uranus
    assert err < 0.15, f"closest top-2 candidate {best.predicted_a:.2f} AU, err {err:.1%}"
    assert best.predicted_a_lower <= uranus <= best.predicted_a_upper


def test_trappist1_minus_e_recovered_first():
    star = STELLAR_INFO["trappist1"]["mass"]
    true_a = _true_a("trappist1", "TRAPPIST-1 e")
    res = GapPredictor(stellar_mass=star).predict(_tuples("trappist1", ("TRAPPIST-1 e",)))
    top = res.predicted_gaps[0]
    assert (top.inner_planet, top.outer_planet) == ("TRAPPIST-1 d", "TRAPPIST-1 f")
    assert abs(top.predicted_a - true_a) / true_a < 0.10
