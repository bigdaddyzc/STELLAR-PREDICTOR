"""Candidate body representation with uncertainty estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateBody:
    """A predicted unknown celestial body with parameter uncertainties.

    Each parameter is stored as (median, lower_bound, upper_bound) representing
    the 50th, 16th, and 84th percentiles (1-sigma equivalent).
    """

    mass: tuple[float, float, float]  # solar masses
    semi_major_axis: tuple[float, float, float]  # AU
    eccentricity: tuple[float, float, float]
    inclination: tuple[float, float, float]  # radians
    period: tuple[float, float, float]  # years
    confidence: float  # 0-1 detection confidence
    method: str  # which detection method produced this

    longitude_ascending: tuple[float, float, float] | None = None
    argument_perihelion: tuple[float, float, float] | None = None

    @property
    def mass_earth(self) -> float:
        """Best estimate mass in Earth masses."""
        return self.mass[0] * 332946.0

    @property
    def mass_jupiter(self) -> float:
        """Best estimate mass in Jupiter masses."""
        return self.mass[0] * 1047.35

    def summary(self) -> str:
        """Human-readable summary of the candidate."""
        lines = [
            f"Candidate Body (confidence: {self.confidence:.1%})",
            f"  Method: {self.method}",
            f"  Mass: {self.mass_earth:.2f} M_Earth "
            f"({self.mass[0]:.2e} M_Sun, range [{self.mass[1]:.2e}, {self.mass[2]:.2e}])",
            f"  Semi-major axis: {self.semi_major_axis[0]:.2f} AU "
            f"(range [{self.semi_major_axis[1]:.2f}, {self.semi_major_axis[2]:.2f}])",
            f"  Eccentricity: {self.eccentricity[0]:.3f} "
            f"(range [{self.eccentricity[1]:.3f}, {self.eccentricity[2]:.3f}])",
            f"  Period: {self.period[0]:.2f} yr "
            f"(range [{self.period[1]:.2f}, {self.period[2]:.2f}])",
            f"  Inclination: {np.degrees(self.inclination[0]):.1f}° "
            f"(range [{np.degrees(self.inclination[1]):.1f}°, {np.degrees(self.inclination[2]):.1f}°])",
        ]
        return "\n".join(lines)
