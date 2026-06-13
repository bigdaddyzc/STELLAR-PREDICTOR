"""Core data models for celestial bodies and stellar systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.time import Time


@dataclass
class OrbitalElements:
    """Keplerian orbital elements."""

    semi_major_axis: float  # AU
    eccentricity: float
    inclination: float  # radians
    longitude_ascending: float  # radians (Ω)
    argument_perihelion: float  # radians (ω)
    mean_anomaly: float  # radians (M at epoch)
    epoch: Optional[Time] = None

    @property
    def period(self) -> Optional[float]:
        """Orbital period in years (requires parent mass context)."""
        return None


@dataclass
class CelestialBody:
    """A celestial body with physical and orbital properties."""

    name: str
    mass: float  # solar masses
    position: Optional[np.ndarray] = None  # 3D cartesian, AU
    velocity: Optional[np.ndarray] = None  # 3D cartesian, AU/day
    radius: Optional[float] = None  # solar radii
    orbital_elements: Optional[OrbitalElements] = None
    metadata: dict = field(default_factory=dict)

    @property
    def mass_kg(self) -> float:
        return self.mass * 1.989e30

    @property
    def mass_earth(self) -> float:
        return self.mass * 332946.0

    @property
    def mass_jupiter(self) -> float:
        return self.mass * 1047.35


@dataclass
class StellarSystem:
    """A stellar system containing stars and orbiting bodies."""

    name: str
    bodies: list[CelestialBody] = field(default_factory=list)
    epoch: Optional[Time] = None
    source: str = ""

    @property
    def stars(self) -> list[CelestialBody]:
        return [b for b in self.bodies if b.mass > 0.08]

    @property
    def planets(self) -> list[CelestialBody]:
        return [b for b in self.bodies if b.mass <= 0.08]

    @property
    def total_mass(self) -> float:
        return sum(b.mass for b in self.bodies)

    def add_body(self, body: CelestialBody) -> None:
        self.bodies.append(body)

    def remove_body(self, name: str) -> Optional[CelestialBody]:
        """Remove a body by name and return it."""
        for i, b in enumerate(self.bodies):
            if b.name.lower() == name.lower():
                return self.bodies.pop(i)
        return None

    def get_body(self, name: str) -> Optional[CelestialBody]:
        for b in self.bodies:
            if b.name.lower() == name.lower():
                return b
        return None


@dataclass
class ExoplanetSystem:
    """Lightweight representation of an exoplanetary system.

    Holds stellar properties and a list of known planets for pattern analysis.
    Each planet is a dict with keys: name, a, mass, period, e, i, radius_earth.
    """
    name: str
    host_name: str = ""
    stellar_mass: float = 1.0
    stellar_radius: float = 1.0
    stellar_teff: float = 5778.0
    planets: list[dict] = field(default_factory=list)

    @property
    def num_planets(self) -> int:
        return len(self.planets)

    def sorted_axes(self) -> list[float]:
        return sorted(p["a"] for p in self.planets if p.get("a") is not None)

    def sorted_planets(self) -> list[dict]:
        return sorted(
            [p for p in self.planets if p.get("a") is not None],
            key=lambda p: p["a"],
        )


@dataclass
class GapResult:
    """A predicted orbital gap where an unknown body could exist."""
    inner_a: float
    outer_a: float
    predicted_a: float
    predicted_period: float
    titius_bode_score: float = 0.0
    stability_score: float = 0.0
    combined_score: float = 0.0
    estimated_mass_range: tuple[float, float] = (0.1, 10.0)
    method: str = ""
    inner_planet: str = ""
    outer_planet: str = ""
    predicted_a_lower: float = 0.0
    predicted_a_upper: float = 0.0
    predicted_eccentricity: float = 0.0
    resonance_score: float = 0.0


@dataclass
class SimulationResult:
    """Results from an N-body simulation."""

    times: np.ndarray  # array of times (days)
    positions: dict[str, np.ndarray]  # name -> (N, 3) positions in AU
    velocities: dict[str, np.ndarray]  # name -> (N, 3) velocities in AU/day


@dataclass
class Residuals:
    """Residual data between observed and modeled positions."""

    times: np.ndarray  # days
    values: np.ndarray  # residual magnitudes
    body_name: str
    components: Optional[np.ndarray] = None  # (N, 3) x/y/z residuals
