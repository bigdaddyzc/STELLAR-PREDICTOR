"""Unit tests for stellar_predictor.physics.properties.

Validates analytical formulas against reference values for known bodies:
Earth, Neptune, Jupiter. Tolerances are intentionally generous (10–20%)
because the M-R relations are empirical fits, not exact physics.
"""

from __future__ import annotations

import pytest

from stellar_predictor.physics.properties import (
    classify_planet,
    density_gcm3,
    equilibrium_temperature,
    hill_sphere_au,
    mass_radius_relation,
    planet_from_mass,
    surface_gravity_ms2,
)

# ---- mass_radius_relation ----

class TestMassRadiusRelation:
    def test_earth(self):
        # Earth: 1 M_earth -> ~1 R_earth (Zeng+2016 rocky regime)
        r = mass_radius_relation(1.0)
        assert 0.9 < r < 1.1, f"Earth radius {r:.3f} outside [0.9, 1.1]"

    def test_rocky_scaling(self):
        # Twice the mass should give a larger radius (monotone in rocky regime)
        assert mass_radius_relation(2.0) > mass_radius_relation(1.0)

    def test_neptune_class(self):
        # Neptune: ~17 M_earth -> roughly 3.5–4.5 R_earth
        r = mass_radius_relation(17.15)
        assert 3.0 < r < 5.5, f"Neptune-class radius {r:.3f} outside [3.0, 5.5]"

    def test_jupiter_class(self):
        # Jupiter: ~318 M_earth -> roughly 10–12 R_earth (degenerate regime)
        r = mass_radius_relation(317.8)
        assert 9.0 < r < 13.5, f"Jupiter-class radius {r:.3f} outside [9.0, 13.5]"

    def test_degenerate_radius_decreases_with_mass(self):
        # In the degenerate gas-giant regime (>130 M_earth), radius should
        # decrease (or stay nearly flat) as mass increases
        r_lo = mass_radius_relation(200.0)
        r_hi = mass_radius_relation(1000.0)
        assert r_lo >= r_hi * 0.90, "Degenerate regime: radius should not grow quickly"

    def test_zero_mass_returns_minimum(self):
        assert mass_radius_relation(0.0) == pytest.approx(0.1)

    def test_continuity_at_regime_boundaries(self):
        # No discontinuity across the 2 M_earth and 130 M_earth breakpoints
        r_below_2 = mass_radius_relation(1.99)
        r_above_2 = mass_radius_relation(2.01)
        assert abs(r_below_2 - r_above_2) < 0.05, "Discontinuity at 2 M_earth boundary"

        r_below_130 = mass_radius_relation(129.0)
        r_above_130 = mass_radius_relation(131.0)
        assert abs(r_below_130 - r_above_130) < 0.20, "Discontinuity at 130 M_earth boundary"


# ---- equilibrium_temperature ----

class TestEquilibriumTemperature:
    def test_earth_sun_system(self):
        # Earth at 1 AU around the Sun (T*=5778 K, R*=1 R_sun, albedo=0.3) -> ~255 K
        t = equilibrium_temperature(1.0, 5778.0, stellar_radius_rsun=1.0, albedo=0.3)
        assert 230.0 < t < 280.0, f"Earth T_eq {t:.1f} K outside [230, 280]"

    def test_closer_planet_hotter(self):
        t_close = equilibrium_temperature(0.5, 5778.0)
        t_far = equilibrium_temperature(2.0, 5778.0)
        assert t_close > t_far

    def test_higher_albedo_cooler(self):
        t_low = equilibrium_temperature(1.0, 5778.0, albedo=0.1)
        t_high = equilibrium_temperature(1.0, 5778.0, albedo=0.9)
        assert t_low > t_high

    def test_hotter_star_hotter_planet(self):
        t_cool = equilibrium_temperature(1.0, 4000.0)
        t_hot = equilibrium_temperature(1.0, 6000.0)
        assert t_hot > t_cool

    def test_scaling_with_distance(self):
        # T_eq ~ a^(-1/2); doubling distance reduces T by ~sqrt(2)
        t1 = equilibrium_temperature(1.0, 5778.0)
        t4 = equilibrium_temperature(4.0, 5778.0)
        ratio = t1 / t4
        assert 1.8 < ratio < 2.2, f"T scaling ratio {ratio:.2f} deviates from sqrt(4)=2"


# ---- surface_gravity_ms2 ----

class TestSurfaceGravity:
    def test_earth(self):
        r_earth = mass_radius_relation(1.0)
        g = surface_gravity_ms2(1.0, r_earth)
        assert 8.5 < g < 11.0, f"Earth surface gravity {g:.2f} m/s² outside [8.5, 11.0]"

    def test_more_massive_rocky_planet_higher_g(self):
        r1 = mass_radius_relation(1.0)
        r2 = mass_radius_relation(5.0)
        g1 = surface_gravity_ms2(1.0, r1)
        g2 = surface_gravity_ms2(5.0, r2)
        assert g2 > g1


# ---- density_gcm3 ----

class TestDensity:
    def test_earth_density(self):
        # By definition: density_gcm3(1, 1) == 5.51
        d = density_gcm3(1.0, 1.0)
        assert d == pytest.approx(5.51)

    def test_gas_giant_less_dense_than_rocky(self):
        r_rocky = mass_radius_relation(1.0)
        r_giant = mass_radius_relation(317.8)
        d_rocky = density_gcm3(1.0, r_rocky)
        d_giant = density_gcm3(317.8, r_giant)
        assert d_giant < d_rocky


# ---- classify_planet ----

class TestClassifyPlanet:
    @pytest.mark.parametrize("mass,expected_en", [
        (1.0, "Rocky"),
        (5.0, "Super-Earth"),
        (20.0, "Neptune-like"),
        (80.0, "Sub-Saturn"),
        (300.0, "Gas Giant"),
        (700.0, "Super-Jupiter"),
    ])
    def test_classifications(self, mass, expected_en):
        _, en = classify_planet(mass)
        assert en == expected_en


# ---- hill_sphere_au ----

class TestHillSphere:
    def test_earth_hill_sphere(self):
        # Earth Hill sphere ~0.01 AU
        h = hill_sphere_au(1.0, 1.0, 1.0)
        assert 0.005 < h < 0.02

    def test_more_massive_planet_larger_hill_sphere(self):
        h_small = hill_sphere_au(1.0, 1.0, 1.0)
        h_large = hill_sphere_au(1.0, 317.8, 1.0)
        assert h_large > h_small


# ---- planet_from_mass (smoke test) ----

class TestPlanetFromMass:
    def test_returns_expected_keys(self):
        result = planet_from_mass(1.0, 10.0, 1.0, 1.0, 5778.0, 1.0)
        expected_keys = {
            "semi_major_axis", "orbital_period", "mass", "radius",
            "density", "eq_temperature", "surface_gravity",
            "hill_sphere", "planet_type", "system_age",
        }
        assert expected_keys.issubset(result.keys())

    def test_all_values_are_strings(self):
        result = planet_from_mass(5.0, 50.0, 2.0, 1.0)
        for key, entry in result.items():
            assert isinstance(entry["value"], str), f"{key} value is not a string"

    def test_single_mass_no_range(self):
        # When low == high, value should not contain "–"
        result = planet_from_mass(10.0, 10.0, 1.5, 1.0)
        assert "–" not in result["mass"]["value"]
