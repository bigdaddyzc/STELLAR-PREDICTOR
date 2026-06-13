"""Hill-radius orbital stability analysis for detecting planet-hosting gaps.

Computes mutual Hill radii between adjacent planets and identifies regions
where additional planets could be dynamically stable.

v0.3: eccentricity-aware separation — for eccentric neighbors the usable
gap shrinks to the span between the inner planet's apoapsis and the outer
planet's periapsis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class HillRadius:
    """Hill radius for a planet."""
    planet_name: str
    a_au: float
    m_planet: float       # solar masses
    m_star: float          # solar masses
    r_hill_au: float


@dataclass
class StabilityRegion:
    """A region of orbital space that could host an additional planet."""
    inner_boundary_au: float
    outer_boundary_au: float
    width_au: float
    min_mass_estimate: float   # M_Earth
    max_mass_estimate: float   # M_Earth
    gap_ratio: float           # separation / critical_sep (>1 means stable room)
    neighbor_inner: str
    neighbor_outer: str


class StabilityAnalyzer:
    """Analyze orbital stability and detect regions where planets can exist."""

    def __init__(self, stellar_mass: float = 1.0, critical_sep: float = 10.0):
        """Args:
            stellar_mass: Host star mass in solar masses.
            critical_sep: Minimum separation in mutual Hill radii for stability.
        """
        self.stellar_mass = stellar_mass
        self.critical_sep = critical_sep

    def hill_radius(self, a_au: float, m_planet: float) -> float:
        """Compute Hill radius in AU: R_H = a * (m / (3*M))^(1/3)."""
        return a_au * (m_planet / (3.0 * self.stellar_mass)) ** (1.0 / 3.0)

    def hill_radii(self, planets: list[tuple[str, float, float]]
                   ) -> list[HillRadius]:
        """Calculate Hill radii for sorted planets.

        Args:
            planets: list of (name, a_au, mass_solar)
        """
        return [
            HillRadius(name, a, m, self.stellar_mass,
                       r_hill_au=self.hill_radius(a, m))
            for name, a, m in planets
        ]

    def mutual_hill_separation(self, inner: HillRadius,
                               outer: HillRadius) -> float:
        """Separation between two planets in mutual Hill radii."""
        mu = (inner.m_planet + outer.m_planet) / (3.0 * self.stellar_mass)
        r_h_mutual = (inner.a_au + outer.a_au) / 2.0 * mu ** (1.0 / 3.0)
        if r_h_mutual < 1e-12:
            return float("inf")
        return (outer.a_au - inner.a_au) / r_h_mutual

    def find_stability_gaps(self, planets: list[tuple[str, float, float]],
                            eccentricities: Optional[Sequence[float]] = None
                            ) -> list[StabilityRegion]:
        """Find gaps where an additional planet could be stable.

        For each adjacent pair, computes the separation in mutual Hill radii.
        If separation > critical_sep, the gap is flagged as stability-compatible.

        When ``eccentricities`` are provided (aligned with ``planets``), the
        usable span is measured from the inner planet's apoapsis
        a_in * (1 + e_in) to the outer planet's periapsis a_out * (1 - e_out):
        eccentric neighbors sweep a wider radial band, leaving less room
        for a stable intermediate orbit.
        """
        if len(planets) < 2:
            return []

        hills = self.hill_radii(planets)
        eccs = list(eccentricities) if eccentricities is not None else []
        if len(eccs) < len(planets):
            eccs = eccs + [0.0] * (len(planets) - len(eccs))

        regions = []

        for i in range(len(hills) - 1):
            inner, outer = hills[i], hills[i + 1]
            e_in = max(0.0, float(eccs[i] or 0.0))
            e_out = max(0.0, float(eccs[i + 1] or 0.0))

            # Eccentricity-aware usable span: inner apoapsis -> outer periapsis
            eff_inner = inner.a_au * (1.0 + e_in)
            eff_outer = outer.a_au * (1.0 - e_out)
            span = eff_outer - eff_inner

            mu = (inner.m_planet + outer.m_planet) / (3.0 * self.stellar_mass)
            a_mid = (inner.a_au + outer.a_au) / 2.0
            r_h_mutual = a_mid * mu ** (1.0 / 3.0)
            r_crit = self.critical_sep * r_h_mutual
            gap_width = span - r_crit

            if gap_width > 0:
                inner_boundary = eff_inner + r_h_mutual * (self.critical_sep / 2.0)
                outer_boundary = eff_outer - r_h_mutual * (self.critical_sep / 2.0)

                gap_ratio = span / max(r_crit, 1e-12)

                max_mass = self._max_stable_mass(a_mid, r_h_mutual)
                # v0.4: Tighter mass priors — min from Hill packing, max gated by gap width
                mass_if_fills_30pct = (gap_width / a_mid) ** 3 * 3.0 * self.stellar_mass * 332946.0 * 0.027
                min_mass_earth = max(0.1, min(max_mass * 0.01, mass_if_fills_30pct))
                max_mass_earth = min(max_mass * 3.0, 10000.0)
                regions.append(StabilityRegion(
                    inner_boundary_au=float(inner_boundary),
                    outer_boundary_au=float(outer_boundary),
                    width_au=float(gap_width),
                    min_mass_estimate=min_mass_earth,
                    max_mass_estimate=max_mass_earth,
                    gap_ratio=float(gap_ratio),
                    neighbor_inner=inner.planet_name,
                    neighbor_outer=outer.planet_name,
                ))
            else:
                # Gap exists but too tight for stability
                gap_ratio = max(span, 0.0) / max(r_crit, 1e-12)
                regions.append(StabilityRegion(
                    inner_boundary_au=float(inner.a_au),
                    outer_boundary_au=float(outer.a_au),
                    width_au=0.0,
                    min_mass_estimate=0.0,
                    max_mass_estimate=0.0,
                    gap_ratio=float(gap_ratio),
                    neighbor_inner=inner.planet_name,
                    neighbor_outer=outer.planet_name,
                ))

        return regions

    def _max_stable_mass(self, a_au: float, r_h_mutual: float) -> float:
        """Maximum planet mass (M_Earth) that fits in the stability gap."""
        max_mass_solar = 3.0 * self.stellar_mass * (r_h_mutual / a_au) ** 3
        return float(max_mass_solar * 332946.0)

    def allowed_mass_range(self, region: StabilityRegion
                           ) -> tuple[float, float]:
        """Mass range for a planet in this region."""
        return (region.min_mass_estimate, min(region.max_mass_estimate, 10000.0))

    @staticmethod
    def extract_planet_data_full(system) -> list[tuple[str, float, float, float]]:
        """Extract sorted (name, a, mass, eccentricity) from any system representation."""
        from stellar_predictor.data.models import StellarSystem, ExoplanetSystem

        if isinstance(system, StellarSystem):
            data = []
            for body in system.planets:
                a = None
                ecc = 0.0
                if body.orbital_elements:
                    a = body.orbital_elements.semi_major_axis
                    ecc = body.orbital_elements.eccentricity or 0.0
                elif body.position is not None:
                    a = float(np.sqrt(np.sum(body.position**2)))
                if a is not None and a > 0:
                    data.append((body.name, a, body.mass, float(ecc)))
            return sorted(data, key=lambda x: x[1])

        if isinstance(system, ExoplanetSystem):
            data = []
            for p in system.sorted_planets():
                a = p.get("a")
                mass = p.get("mass", 0.0)
                name = p.get("name", "?")
                ecc = p.get("e", 0.0) or 0.0
                if a is not None and a > 0:
                    mass_solar = mass / 332946.0 if mass > 0 else 1e-6
                    data.append((name, float(a), float(mass_solar), float(ecc)))
            return data

        # Assume list of (name, a, mass_solar[, e]) tuples
        result = []
        for item in sorted(system, key=lambda x: x[1]):
            if len(item) >= 4:
                result.append((item[0], item[1], item[2], float(item[3])))
            else:
                result.append((item[0], item[1], item[2], 0.0))
        return result

    @staticmethod
    def extract_planet_data(system) -> list[tuple[str, float, float]]:
        """Extract sorted (name, a, mass) from any system representation."""
        return [(n, a, m) for n, a, m, _ in
                StabilityAnalyzer.extract_planet_data_full(system)]
