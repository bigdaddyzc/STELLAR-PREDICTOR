"""Tests for GapPredictor."""

import numpy as np
import pytest

from stellar_predictor.data.models import CelestialBody, ExoplanetSystem, OrbitalElements, StellarSystem
from stellar_predictor.patterns.predictor import GapPredictor


def _build_solar_system() -> StellarSystem:
    planets = [
        ("Mercury", 1.66012e-7, 0.3871, 0.2056, 0.1222, 0.8436, 0.5088, 4.4026),
        ("Venus", 2.44783e-6, 0.7233, 0.0068, 0.0592, 1.3383, 0.9577, 3.1761),
        ("Earth", 3.00273e-6, 1.0000, 0.0167, 0.0, -0.1965, 1.7968, 6.2400),
        ("Mars", 3.22715e-7, 1.5237, 0.0934, 0.0323, 0.8653, -1.1951, 0.3381),
        ("Jupiter", 9.54786e-4, 5.2034, 0.0484, 0.0227, 1.7534, 0.2389, 0.3411),
        ("Saturn", 2.85837e-4, 9.5371, 0.0542, 0.0434, 1.9847, 1.6130, 5.5647),
        ("Uranus", 4.36624e-5, 19.1913, 0.0472, 0.0135, 1.2956, 1.6929, 2.4844),
        ("Neptune", 5.15138e-5, 30.0690, 0.0086, 0.0309, 2.2999, -1.4869, 4.4715),
    ]
    system = StellarSystem(name="Solar System")
    system.add_body(CelestialBody("Sun", 1.0, np.zeros(3), np.zeros(3)))
    for name, mass, a, e, i, Om, om, M in planets:
        system.add_body(CelestialBody(
            name=name, mass=mass,
            orbital_elements=OrbitalElements(a, e, i, Om, om, M),
        ))
    return system


class TestGapPredictor:
    def test_solar_system_prediction(self):
        """Should identify the Mars-Jupiter gap (Asteroid Belt region)."""
        predictor = GapPredictor(stellar_mass=1.0, min_known_planets=3)
        result = predictor.predict(_build_solar_system())

        assert result.system_name == "Solar System"
        assert result.num_known_planets == 8
        assert result.tb_fit is not None
        assert result.tb_fit.r_squared > 0.80
        assert len(result.predicted_gaps) > 0

        # Mars-Jupiter gap should appear
        mars_jupiter = [g for g in result.predicted_gaps
                        if g.inner_a > 1.0 and g.inner_a < 2.0
                        and g.outer_a > 4.0 and g.outer_a < 6.0]
        assert len(mars_jupiter) > 0

    def test_exoplanet_system(self):
        """Should work on ExoplanetSystem data."""
        system = ExoplanetSystem(
            name="Kepler-11",
            host_name="Kepler-11",
            stellar_mass=0.95,
            planets=[
                {"name": "b", "a": 0.091, "mass": 1.9, "period": 10.3},
                {"name": "c", "a": 0.107, "mass": 2.9, "period": 13.0},
                {"name": "d", "a": 0.155, "mass": 7.3, "period": 22.7},
                {"name": "e", "a": 0.195, "mass": 8.0, "period": 32.0},
                {"name": "f", "a": 0.250, "mass": 2.0, "period": 46.7},
                {"name": "g", "a": 0.466, "mass": 0.95, "period": 118.4},
            ],
        )
        predictor = GapPredictor(stellar_mass=0.95, min_known_planets=3)
        result = predictor.predict(system)

        assert result.num_known_planets == 6
        assert result.tb_fit is not None
        assert len(result.predicted_gaps) > 0

        # The gap between f (0.25 AU) and g (0.466 AU) should be ranked
        f_g_gaps = [g for g in result.predicted_gaps
                    if g.inner_planet == "f" and g.outer_planet == "g"]
        assert len(f_g_gaps) > 0

    def test_two_planet_system(self):
        """Should still run stability analysis with < 3 planets."""
        predictor = GapPredictor(stellar_mass=1.0, min_known_planets=3)
        system = StellarSystem(name="Two Planet")
        system.add_body(CelestialBody("Sun", 1.0, np.zeros(3), np.zeros(3)))
        system.add_body(CelestialBody(
            "a", 1e-5, orbital_elements=OrbitalElements(1.0, 0.0, 0, 0, 0, 0),
        ))
        system.add_body(CelestialBody(
            "b", 1e-5, orbital_elements=OrbitalElements(4.0, 0.0, 0, 0, 0, 0),
        ))
        result = predictor.predict(system)
        assert result.num_known_planets == 2
        assert result.tb_fit is None  # Too few planets
        assert len(result.stability_regions) > 0

    def test_execution_time(self):
        """Pattern analysis should be very fast (< 1 second)."""
        predictor = GapPredictor(stellar_mass=1.0)
        result = predictor.predict(_build_solar_system())
        assert result.execution_time_s < 1.0
