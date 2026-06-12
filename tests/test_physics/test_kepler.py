"""Tests for Kepler equation solver and orbital element conversions."""

import numpy as np
import pytest

from stellar_predictor.physics.kepler import (
    cartesian_to_orbital_elements,
    eccentric_to_true_anomaly,
    kepler_solve,
    orbital_elements_to_cartesian,
)


class TestKeplerSolve:
    def test_circular_orbit(self):
        """For e=0, E should equal M."""
        for M in np.linspace(0, 2 * np.pi, 20, endpoint=False):
            E = kepler_solve(M, 0.0)
            assert abs(E - M) < 1e-10

    def test_known_solution(self):
        """Verify against known M = E - e*sin(E)."""
        E_true = 1.5
        e = 0.3
        M = E_true - e * np.sin(E_true)
        E_solved = kepler_solve(M, e)
        assert abs(E_solved - E_true) < 1e-10

    def test_high_eccentricity(self):
        """Should converge for high eccentricity."""
        e = 0.95
        for M in np.linspace(0.01, 2 * np.pi - 0.01, 10):
            E = kepler_solve(M, e)
            # Verify: M = E - e*sin(E)
            M_check = E - e * np.sin(E)
            assert abs(M_check - (M % (2 * np.pi))) < 1e-10

    def test_zero_mean_anomaly(self):
        """M=0 should give E=0 for any eccentricity."""
        for e in [0, 0.1, 0.5, 0.9]:
            E = kepler_solve(0.0, e)
            assert abs(E) < 1e-10


class TestOrbitalConversions:
    def test_circular_orbit_roundtrip(self):
        """Circular orbit should roundtrip through conversions."""
        a, e, i = 1.0, 0.0, 0.0
        Omega, omega, nu = 0.0, 0.0, np.pi / 4

        pos, vel = orbital_elements_to_cartesian(a, e, i, Omega, omega, nu)
        elements = cartesian_to_orbital_elements(pos, vel)

        assert abs(elements["a"] - a) < 1e-8
        assert abs(elements["e"] - e) < 1e-8

    def test_earth_like_orbit(self):
        """Earth-like orbit should have correct radius at perihelion."""
        a, e = 1.0, 0.0167
        nu = 0.0  # At perihelion

        pos, vel = orbital_elements_to_cartesian(a, e, 0.0, 0.0, 0.0, nu)
        r = np.linalg.norm(pos)
        expected_r = a * (1 - e)  # Perihelion distance

        assert abs(r - expected_r) < 1e-10

    def test_inclined_orbit(self):
        """Inclined orbit should have z-component."""
        a, e, i = 5.0, 0.1, np.radians(30)
        nu = np.pi / 2

        pos, vel = orbital_elements_to_cartesian(a, e, i, 0.0, 0.0, nu)
        assert abs(pos[2]) > 0.1  # Should have z-component

    def test_energy_conservation(self):
        """Velocity should be consistent with vis-viva equation."""
        a, e = 2.0, 0.3
        mu = 4 * np.pi**2

        for nu in np.linspace(0, 2 * np.pi, 10):
            pos, vel = orbital_elements_to_cartesian(a, e, 0.0, 0.0, 0.0, nu, mu=mu)
            r = np.linalg.norm(pos)
            v = np.linalg.norm(vel)
            # vis-viva: v^2 = mu * (2/r - 1/a)
            v_expected = np.sqrt(mu * (2 / r - 1 / a))
            assert abs(v - v_expected) < 1e-8
