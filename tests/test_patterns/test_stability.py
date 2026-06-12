"""Tests for Hill-radius stability analysis."""

from stellar_predictor.patterns.stability import StabilityAnalyzer


class TestStabilityAnalyzer:
    def test_hill_radius_calculation(self):
        """Earth's Hill radius should be ~0.01 AU."""
        analyzer = StabilityAnalyzer(stellar_mass=1.0)
        rh = analyzer.hill_radius(1.0, 3.0e-6)
        expected = 1.0 * (3.0e-6 / 3.0) ** (1.0 / 3.0)
        assert abs(rh - expected) < 1e-8

    def test_stable_gap_detection(self):
        """Two widely separated planets should have a stable gap."""
        analyzer = StabilityAnalyzer(stellar_mass=1.0, critical_sep=10.0)
        planets = [("inner", 1.0, 3e-6), ("outer", 10.0, 3e-6)]
        gaps = analyzer.find_stability_gaps(planets)
        assert len(gaps) > 0
        assert any(g.gap_ratio >= 1.0 for g in gaps)

    def test_no_gap_tight_packing(self):
        """Tightly packed planets should show low stability scores."""
        analyzer = StabilityAnalyzer(stellar_mass=0.08, critical_sep=10.0)
        planets = [
            ("b", 0.011, 1e-6),
            ("c", 0.012, 1e-6),
            ("d", 0.013, 1e-6),
        ]
        gaps = analyzer.find_stability_gaps(planets)
        # With such tight spacing, no gap should be "wide open"
        assert all(g.gap_ratio < 3.0 for g in gaps)

    def test_solar_system_stability(self):
        """Solar System inner planets have some stable regions."""
        analyzer = StabilityAnalyzer(stellar_mass=1.0, critical_sep=10.0)
        planets = [
            ("Venus", 0.72, 2.45e-6),
            ("Earth", 1.00, 3.00e-6),
            ("Mars", 1.52, 3.23e-7),
            ("Jupiter", 5.20, 9.55e-4),
        ]
        gaps = analyzer.find_stability_gaps(planets)
        assert len(gaps) == 3  # One between each pair
        # Mars-Jupiter should have the widest gap
        mars_jupiter = gaps[2]
        assert mars_jupiter.width_au > 1.0
