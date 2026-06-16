"""Tests for N-body simulator."""

import numpy as np

from stellar_predictor.data.models import CelestialBody, OrbitalElements, StellarSystem
from stellar_predictor.physics.nbody import NBodySimulator


def make_two_body_system() -> StellarSystem:
    """Create a simple Sun + Earth-like system."""
    system = StellarSystem(name="Test")
    system.add_body(CelestialBody(
        name="Star",
        mass=1.0,
        position=np.zeros(3),
        velocity=np.zeros(3),
    ))
    system.add_body(CelestialBody(
        name="Planet",
        mass=3e-6,
        orbital_elements=OrbitalElements(
            semi_major_axis=1.0,
            eccentricity=0.0,
            inclination=0.0,
            longitude_ascending=0.0,
            argument_perihelion=0.0,
            mean_anomaly=0.0,
        ),
    ))
    return system


class TestNBodySimulator:
    def test_circular_orbit_period(self):
        """Circular orbit at 1 AU should have period ~1 year."""
        system = make_two_body_system()
        sim = NBodySimulator(system)
        result = sim.simulate(t_end=1.0, n_steps=100)

        # After 1 year, planet should return near starting position
        pos_start = result.positions["Planet"][0]
        pos_end = result.positions["Planet"][-1]
        distance = np.linalg.norm(pos_end - pos_start)

        assert distance < 0.01  # Should be very close

    def test_energy_conservation(self):
        """Energy should be conserved over integration."""
        system = make_two_body_system()
        sim = NBodySimulator(system)

        e0 = sim.get_energy()
        sim.simulate(t_end=10.0, n_steps=100)
        e1 = sim.get_energy()

        relative_error = abs((e1 - e0) / e0)
        assert relative_error < 1e-10  # IAS15 is very accurate

    def test_add_test_body(self):
        """Should be able to add a test body after initialization."""
        system = make_two_body_system()
        sim = NBodySimulator(system)

        sim.add_test_body(mass=1e-4, semi_major_axis=5.0, name="Jupiter-like")
        result = sim.simulate(t_end=1.0, n_steps=50)

        assert "Jupiter-like" in result.positions
        assert result.positions["Jupiter-like"].shape == (50, 3)

    def test_simulation_result_shape(self):
        """Simulation results should have correct shapes."""
        system = make_two_body_system()
        sim = NBodySimulator(system)
        result = sim.simulate(t_end=5.0, n_steps=200)

        assert result.times.shape == (200,)
        assert result.positions["Star"].shape == (200, 3)
        assert result.positions["Planet"].shape == (200, 3)
        assert result.velocities["Planet"].shape == (200, 3)


class TestNBodyMultiPlanet:
    def test_jupiter_saturn_interaction(self):
        """Jupiter and Saturn should perturb each other measurably."""
        system = StellarSystem(name="Test")
        system.add_body(CelestialBody(name="Sun", mass=1.0,
                                       position=np.zeros(3), velocity=np.zeros(3)))
        system.add_body(CelestialBody(
            name="Jupiter",
            mass=9.55e-4,
            orbital_elements=OrbitalElements(5.2, 0.048, 0.023, 1.75, 0.24, 0.34),
        ))
        system.add_body(CelestialBody(
            name="Saturn",
            mass=2.86e-4,
            orbital_elements=OrbitalElements(9.54, 0.054, 0.043, 1.98, 1.61, 5.56),
        ))

        sim = NBodySimulator(system)
        result = sim.simulate(t_end=30.0, n_steps=300)  # ~2.5 Jupiter orbits

        # Saturn's orbit should deviate from pure Keplerian
        # Check that distance from Sun varies (not perfectly periodic)
        saturn_r = np.linalg.norm(result.positions["Saturn"], axis=1)
        # Standard deviation of distance should show perturbation effects
        assert np.std(saturn_r) > 0.01
