"""Integration test: Solar System pattern analysis."""

import numpy as np
import pytest

from stellar_predictor.data.models import CelestialBody, OrbitalElements, StellarSystem
from stellar_predictor.prediction.pipeline import PredictionPipeline


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


class TestSolarSystemAnalysis:
    def test_pattern_analysis_succeeds(self):
        """Full pattern analysis on Solar System produces valid output."""
        pipeline = PredictionPipeline(enable_verification=False)
        system = _build_solar_system()
        result = pipeline.analyze(system)

        assert result.system_name == "Solar System"
        assert result.num_known_planets == 8
        assert result.tb_fit is not None
        assert result.tb_fit.r_squared > 0.80
        assert len(result.predicted_gaps) > 0

        top_gap = result.predicted_gaps[0]
        assert top_gap.combined_score > 0.0
        assert top_gap.predicted_a > 0
        assert top_gap.predicted_period > 0

    def test_mars_jupiter_gap_detected(self):
        """The Mars-Jupiter gap should be ranked as a prediction.

        This corresponds to the Asteroid Belt region (~2.1–3.3 AU).
        """
        pipeline = PredictionPipeline()
        result = pipeline.analyze(_build_solar_system())

        # Find the gap between Mars (1.52 AU) and Jupiter (5.20 AU)
        mars_jupiter_gaps = [
            g for g in result.predicted_gaps
            if g.inner_a > 1.0 and g.inner_a < 2.0
            and g.outer_a > 4.0 and g.outer_a < 6.0
        ]
        assert len(mars_jupiter_gaps) > 0

        gap = mars_jupiter_gaps[0]
        # Predicted semi-major axis should be within the gap
        assert 1.5 < gap.predicted_a < 5.5
        # The asteroid belt's Ceres is at ~2.77 AU — should be close to prediction
        assert 2.0 < gap.predicted_a < 4.5

    def test_all_planets_have_tb_assignment(self):
        """All 8 planets should get a TB index."""
        pipeline = PredictionPipeline()
        result = pipeline.analyze(_build_solar_system())

        assert result.tb_fit is not None
        # All planets should appear in the index map
        planet_names = [b.name for b in _build_solar_system().planets]
        for name in planet_names:
            assert name in result.tb_fit.index_map, f"{name} missing from TB fit"
