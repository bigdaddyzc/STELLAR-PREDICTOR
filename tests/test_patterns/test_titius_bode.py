"""Tests for Titius-Bode fitting."""

from stellar_predictor.patterns.titius_bode import TitiusBodeFit


class TestTitiusBodeFit:
    def test_perfect_log_linear(self):
        """A system with perfectly log-linear spacing should have R^2 ~ 1.0."""
        axes = [0.1, 0.2, 0.4, 0.8, 1.6]
        names = ["p1", "p2", "p3", "p4", "p5"]
        result = TitiusBodeFit.best_fit(axes, names)
        assert result.r_squared > 0.99

    def test_solar_system_like(self):
        """Solar System spacing (approximate log-linear)."""
        axes = [0.39, 0.72, 1.0, 1.52, 5.2, 9.54, 19.2, 30.1]
        names = ["M", "V", "E", "Ma", "J", "S", "U", "N"]
        result = TitiusBodeFit.best_fit(axes, names)
        assert result.r_squared > 0.85

    def test_gap_detection(self):
        """Removing a planet should create a detectable large gap."""
        import numpy as np

        from stellar_predictor.data.models import CelestialBody, OrbitalElements, StellarSystem
        from stellar_predictor.patterns.predictor import GapPredictor

        # Build a system with 4 planets: 0.1, 0.2, 0.8, 1.6
        # The gap between 0.2 and 0.8 is large — a "missing" planet at ~0.4
        system = StellarSystem(name="test")
        system.add_body(CelestialBody("Sun", 1.0, np.zeros(3), np.zeros(3)))
        for name, a_val in zip(["p1", "p2", "p4", "p5"], [0.1, 0.2, 0.8, 1.6]):
            system.add_body(CelestialBody(
                name, 1e-6,
                orbital_elements=OrbitalElements(a_val, 0, 0, 0, 0, 0),
            ))

        predictor = GapPredictor(stellar_mass=1.0, min_known_planets=3)
        result = predictor.predict(system)
        # The p2→p4 gap should be identified
        gaps = [g for g in result.predicted_gaps
                if g.inner_planet == "p2" and g.outer_planet == "p4"]
        assert len(gaps) > 0
        # Predicted axis should be around 0.4 AU (midpoint or TB prediction)
        assert 0.3 < gaps[0].predicted_a < 0.6

    def test_two_planet_system(self):
        """With only 2 planets, TB fit should still work."""
        axes = [1.0, 4.0]
        names = ["p1", "p2"]
        result = TitiusBodeFit.best_fit(axes, names)
        assert result.r_squared > 0.99

    def test_three_planet_tight(self):
        """Three tightly packed planets should fit well."""
        axes = [0.05, 0.1, 0.2]
        names = ["a", "b", "c"]
        result = TitiusBodeFit.best_fit(axes, names)
        assert result.r_squared > 0.95
        assert 1.5 < result.beta < 2.5

    def test_classical_fit(self):
        """Classical Titius-Bode: a = 0.4 + 0.3 * 2^n for inner solar system."""
        axes = [0.39, 0.72, 1.0, 1.52, 5.2, 9.54]
        names = ["M", "V", "E", "Ma", "J", "S"]
        result = TitiusBodeFit.fit_classical(axes, names)
        assert result.r_squared > 0.80
