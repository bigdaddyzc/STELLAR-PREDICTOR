"""N-body gravitational simulation wrapping REBOUND."""

from __future__ import annotations

from typing import Optional

import numpy as np
import rebound

from stellar_predictor.data.models import CelestialBody, SimulationResult, StellarSystem


class NBodySimulator:
    """N-body simulator using REBOUND's IAS15 integrator."""

    def __init__(
        self,
        system: StellarSystem,
        integrator: str = "ias15",
        units: tuple[str, str, str] = ("AU", "yr", "Msun"),
    ):
        self.system = system
        self.sim = rebound.Simulation()
        self.sim.integrator = integrator
        self.sim.units = units

        for body in system.bodies:
            self._add_body(body)

        self.sim.move_to_com()
        self._body_names = [b.name for b in system.bodies]

    def _add_body(self, body: CelestialBody) -> None:
        if body.orbital_elements is not None:
            oe = body.orbital_elements
            self.sim.add(
                m=body.mass,
                a=oe.semi_major_axis,
                e=oe.eccentricity,
                inc=oe.inclination,
                Omega=oe.longitude_ascending,
                omega=oe.argument_perihelion,
                M=oe.mean_anomaly,
            )
        elif body.position is not None and body.velocity is not None:
            # velocity: convert AU/day to AU/yr
            v_yr = body.velocity * 365.25
            self.sim.add(
                m=body.mass,
                x=body.position[0],
                y=body.position[1],
                z=body.position[2],
                vx=v_yr[0],
                vy=v_yr[1],
                vz=v_yr[2],
            )
        else:
            self.sim.add(m=body.mass)

    def simulate(
        self,
        t_end: float,
        n_steps: int = 1000,
        t_start: float = 0.0,
    ) -> SimulationResult:
        """Run simulation and record positions/velocities.

        Args:
            t_end: End time in years
            n_steps: Number of output steps
            t_start: Start time in years

        Returns:
            SimulationResult with time series of positions and velocities
        """
        times = np.linspace(t_start, t_end, n_steps)
        n_bodies = len(self._body_names)

        positions = {name: np.zeros((n_steps, 3)) for name in self._body_names}
        velocities = {name: np.zeros((n_steps, 3)) for name in self._body_names}

        for i, t in enumerate(times):
            self.sim.integrate(t)
            for j, name in enumerate(self._body_names):
                p = self.sim.particles[j]
                positions[name][i] = [p.x, p.y, p.z]
                velocities[name][i] = [p.vx, p.vy, p.vz]

        # Convert times to days for consistency
        return SimulationResult(
            times=times * 365.25,
            positions=positions,
            velocities=velocities,
        )

    def add_test_body(
        self,
        mass: float,
        semi_major_axis: float,
        eccentricity: float = 0.0,
        inclination: float = 0.0,
        longitude_ascending: float = 0.0,
        argument_perihelion: float = 0.0,
        mean_anomaly: float = 0.0,
        name: str = "test_body",
    ) -> None:
        """Add a test body to the simulation."""
        self.sim.add(
            m=mass,
            a=semi_major_axis,
            e=eccentricity,
            inc=inclination,
            Omega=longitude_ascending,
            omega=argument_perihelion,
            M=mean_anomaly,
        )
        self._body_names.append(name)
        self.sim.move_to_com()

    def get_energy(self) -> float:
        """Get total energy of the system (for conservation checks)."""
        return self.sim.energy()

    def clone(self) -> "NBodySimulator":
        """Create a copy of this simulator at its current state."""
        new = object.__new__(NBodySimulator)
        new.sim = self.sim.copy()
        new.system = self.system
        new._body_names = list(self._body_names)
        return new
