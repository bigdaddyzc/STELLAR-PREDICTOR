"""Canonical definitions of the built-in planetary systems.

Single source of truth for the 5 supported systems, shared by the web task
layer, the accuracy-evaluation scripts, and the validation test suite. Keeping
the data here avoids the drift that previously existed between three separate
copies (web/tasks.py, scripts/eval_accuracy.py, and ad-hoc test fixtures).

Data provenance:
- Solar System: J2000 mean orbital elements (JPL).
- Exoplanet systems: NASA Exoplanet Archive (masses in Earth masses).
"""

from __future__ import annotations

import numpy as np

from stellar_predictor.data.models import (
    CelestialBody,
    ExoplanetSystem,
    OrbitalElements,
    StellarSystem,
)

# Solar System: (name, mass_solar, a_au, e, inc, Omega, omega, M) — J2000.
SOLAR_SYSTEM_PLANETS = [
    ("Mercury", 1.66012e-7, 0.3871, 0.2056, 0.1222, 0.8436, 0.5088, 4.4026),
    ("Venus", 2.44783e-6, 0.7233, 0.0068, 0.0592, 1.3383, 0.9577, 3.1761),
    ("Earth", 3.00273e-6, 1.0000, 0.0167, 0.0000, -0.1965, 1.7968, 6.2400),
    ("Mars", 3.22715e-7, 1.5237, 0.0934, 0.0323, 0.8653, -1.1951, 0.3381),
    ("Jupiter", 9.54786e-4, 5.2034, 0.0484, 0.0227, 1.7534, 0.2389, 0.3411),
    ("Saturn", 2.85837e-4, 9.5371, 0.0542, 0.0434, 1.9847, 1.6130, 5.5647),
    ("Uranus", 4.36624e-5, 19.1913, 0.0472, 0.0135, 1.2956, 1.6929, 2.4844),
    ("Neptune", 5.15138e-5, 30.0690, 0.0086, 0.0309, 2.2999, -1.4869, 4.4715),
]

# Stellar properties (mass M_sun, radius R_sun, effective temperature K).
STELLAR_INFO = {
    "solar_system": {"mass": 1.0, "radius": 1.0, "teff": 5778.0},
    "trappist1": {"mass": 0.089, "radius": 0.119, "teff": 2566.0},
    "kepler11": {"mass": 0.96, "radius": 1.06, "teff": 5663.0},
    "kepler33": {"mass": 1.26, "radius": 1.58, "teff": 5904.0},
    "hd219134": {"mass": 0.81, "radius": 0.778, "teff": 4699.0},
}

# Exoplanet systems: key -> (stellar_mass_solar, [(name, mass_earth, a_au, ecc), ...]).
EXOPLANET_DATA: dict[str, tuple] = {
    "trappist1": (0.089, [
        ("TRAPPIST-1 b", 1.374, 0.01154, 0.006),
        ("TRAPPIST-1 c", 1.308, 0.01580, 0.006),
        ("TRAPPIST-1 d", 0.388, 0.02227, 0.008),
        ("TRAPPIST-1 e", 0.692, 0.02925, 0.005),
        ("TRAPPIST-1 f", 1.039, 0.03849, 0.010),
        ("TRAPPIST-1 g", 1.321, 0.04683, 0.002),
        ("TRAPPIST-1 h", 0.326, 0.06189, 0.086),
    ]),
    "kepler11": (0.96, [
        ("Kepler-11 b", 1.9, 0.091, 0.045),
        ("Kepler-11 c", 2.9, 0.107, 0.026),
        ("Kepler-11 d", 7.3, 0.155, 0.004),
        ("Kepler-11 e", 8.0, 0.195, 0.012),
        ("Kepler-11 f", 2.0, 0.250, 0.013),
        ("Kepler-11 g", 0.95, 0.466, 0.15),
    ]),
    "kepler33": (1.26, [
        ("Kepler-33 b", 0.16, 0.0677, 0.0),
        ("Kepler-33 c", 0.29, 0.1189, 0.0),
        ("Kepler-33 d", 0.48, 0.1662, 0.0),
        ("Kepler-33 e", 0.36, 0.2138, 0.0),
        ("Kepler-33 f", 0.40, 0.2535, 0.0),
    ]),
    "hd219134": (0.81, [
        ("HD 219134 b", 4.74, 0.0388, 0.0),
        ("HD 219134 c", 4.36, 0.0653, 0.062),
        ("HD 219134 d", 16.17, 0.237, 0.138),
        ("HD 219134 e", 70.90, 2.563, 0.34),
    ]),
}

# Planet dict keys (kept in one place so producers/consumers never drift).
PLANET_KEY_A = "a"
PLANET_KEY_MASS = "mass"
PLANET_KEY_ECC = "eccentricity"
PLANET_KEY_NAME = "name"


def build_solar_system() -> StellarSystem:
    """Construct the full 8-planet Solar System as a StellarSystem."""
    system = StellarSystem(name="Solar System", source="J2000 orbital elements")
    system.add_body(CelestialBody("Sun", 1.0, np.zeros(3), np.zeros(3)))
    for name, mass, a, e, i, Om, om, M in SOLAR_SYSTEM_PLANETS:
        system.add_body(CelestialBody(
            name=name, mass=mass,
            orbital_elements=OrbitalElements(a, e, i, Om, om, M),
        ))
    return system


def build_exoplanet(key: str) -> ExoplanetSystem:
    """Construct an ExoplanetSystem from EXOPLANET_DATA by key."""
    star_mass, planets = EXOPLANET_DATA[key]
    system = ExoplanetSystem(name=key, stellar_mass=star_mass)
    for pname, mass, a, ecc in planets:
        system.planets.append({
            PLANET_KEY_NAME: pname,
            PLANET_KEY_A: a,
            PLANET_KEY_MASS: mass,   # Earth masses (converted to solar downstream)
            PLANET_KEY_ECC: ecc,
        })
    return system


def build_system(system_name: str):
    """Build any supported system by its canonical key, or None if unknown."""
    if system_name == "solar_system":
        return build_solar_system()
    if system_name in EXOPLANET_DATA:
        return build_exoplanet(system_name)
    return None


def system_keys() -> list[str]:
    """All supported system keys."""
    return ["solar_system", *EXOPLANET_DATA.keys()]


def ground_truth_planets(system_name: str) -> list[tuple[str, float, float, float]]:
    """Return sorted (name, a_au, mass_earth, ecc) ground-truth for a system.

    Used by the validation harness to score leave-one-out recall. Masses are
    in Earth masses for both Solar System and exoplanet systems.
    """
    if system_name == "solar_system":
        rows = [
            (name, a, mass * 332946.0, e)
            for name, mass, a, e, *_ in SOLAR_SYSTEM_PLANETS
        ]
    elif system_name in EXOPLANET_DATA:
        _, planets = EXOPLANET_DATA[system_name]
        rows = [(name, a, mass, ecc) for name, mass, a, ecc in planets]
    else:
        return []
    return sorted(rows, key=lambda r: r[1])
