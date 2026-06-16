"""Regression tests for eccentricity key reading in extract_planet_data_full.

P0 bug: stability.py was reading p.get("e") but ExoplanetSystem planets
store eccentricity under "eccentricity". This suite ensures the fix holds.
"""

from __future__ import annotations

from stellar_predictor.data.known_systems import build_exoplanet
from stellar_predictor.data.models import ExoplanetSystem
from stellar_predictor.patterns.stability import StabilityAnalyzer


def test_hd219134_eccentricities_nonzero():
    """HD 219134 has planets with significant eccentricity; none should read as 0."""
    system = build_exoplanet("hd219134")
    data = StabilityAnalyzer.extract_planet_data_full(system)
    eccs = {name: ecc for name, _, _, ecc in data}

    # HD 219134 e: e=0.34 — the most severe case
    assert abs(eccs["HD 219134 e"] - 0.34) < 0.01, (
        f"HD 219134 e eccentricity read as {eccs['HD 219134 e']}, expected ~0.34"
    )
    # HD 219134 d: e=0.138
    assert abs(eccs["HD 219134 d"] - 0.138) < 0.01
    # HD 219134 c: e=0.062
    assert abs(eccs["HD 219134 c"] - 0.062) < 0.01


def test_eccentricity_key_canonical():
    """ExoplanetSystem with 'eccentricity' key must propagate correctly."""
    system = ExoplanetSystem(name="test", stellar_mass=1.0)
    system.planets.append({"name": "A", "a": 1.0, "mass": 1.0, "eccentricity": 0.25})
    system.planets.append({"name": "B", "a": 2.0, "mass": 1.0, "eccentricity": 0.10})

    data = StabilityAnalyzer.extract_planet_data_full(system)
    eccs = {name: ecc for name, _, _, ecc in data}

    assert abs(eccs["A"] - 0.25) < 1e-9
    assert abs(eccs["B"] - 0.10) < 1e-9


def test_eccentricity_key_legacy_e():
    """Legacy planet dicts using 'e' key must still work."""
    system = ExoplanetSystem(name="legacy", stellar_mass=1.0)
    system.planets.append({"name": "X", "a": 1.0, "mass": 1.0, "e": 0.15})
    system.planets.append({"name": "Y", "a": 2.0, "mass": 1.0, "e": 0.05})

    data = StabilityAnalyzer.extract_planet_data_full(system)
    eccs = {name: ecc for name, _, _, ecc in data}

    assert abs(eccs["X"] - 0.15) < 1e-9
    assert abs(eccs["Y"] - 0.05) < 1e-9


def test_eccentricity_missing_defaults_to_zero():
    """Planet dicts with no eccentricity key at all must default to 0.0."""
    system = ExoplanetSystem(name="no_ecc", stellar_mass=1.0)
    system.planets.append({"name": "P", "a": 1.0, "mass": 1.0})
    system.planets.append({"name": "Q", "a": 2.0, "mass": 1.0})

    data = StabilityAnalyzer.extract_planet_data_full(system)
    eccs = {name: ecc for name, _, _, ecc in data}

    assert eccs["P"] == 0.0
    assert eccs["Q"] == 0.0


def test_tuple_input_with_eccentricity():
    """List-of-tuples input (name, a, mass, e) must propagate eccentricity."""
    tuples = [
        ("Inner", 1.0, 3e-6, 0.05),
        ("Middle", 2.0, 1e-5, 0.10),
        ("Outer", 4.0, 5e-4, 0.20),
    ]
    data = StabilityAnalyzer.extract_planet_data_full(tuples)
    eccs = {name: ecc for name, _, _, ecc in data}

    assert abs(eccs["Inner"] - 0.05) < 1e-9
    assert abs(eccs["Middle"] - 0.10) < 1e-9
    assert abs(eccs["Outer"] - 0.20) < 1e-9
