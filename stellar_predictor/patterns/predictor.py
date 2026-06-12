"""GapPredictor: combines Titius-Bode and stability analysis to predict unknown planets.

This is the primary prediction engine. It operates purely analytically on
known planet parameters — no N-body simulation required.

v0.3 improvements:
- Mean-motion resonance scoring as a third signal alongside TB + stability
- Eccentricity-aware Hill stability gaps
- Confidence intervals for the predicted semi-major axes derived from
  TB-fit residual scatter
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from stellar_predictor.data.models import ExoplanetSystem, GapResult, StellarSystem
from stellar_predictor.patterns.stability import (
    StabilityAnalyzer,
    StabilityRegion,
)
from stellar_predictor.patterns.titius_bode import TitiusBodeFit, TBResult

try:
    from config.settings import (
        RESONANCES,
        RESONANCE_SCORE_WEIGHT,
        RESONANCE_TOLERANCE,
    )
except ImportError:  # standalone fallback
    RESONANCES = [(2, 1), (3, 2), (5, 3), (4, 3)]
    RESONANCE_SCORE_WEIGHT = 0.1
    RESONANCE_TOLERANCE = 0.05


@dataclass
class PredictionResult:
    """Complete prediction for a stellar system."""
    system_name: str
    num_known_planets: int
    tb_fit: Optional[TBResult] = None
    stability_regions: list[StabilityRegion] = field(default_factory=list)
    predicted_gaps: list[GapResult] = field(default_factory=list)
    execution_time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


class GapPredictor:
    """Predict unknown planets by combining orbital spacing patterns and stability."""

    def __init__(self, stellar_mass: float = 1.0,
                 min_known_planets: int = 3,
                 tb_weight: float = 0.5,
                 stability_weight: float = 0.5,
                 resonance_weight: float = RESONANCE_SCORE_WEIGHT):
        self.stellar_mass = stellar_mass
        self.min_known_planets = min_known_planets
        self.tb_weight = tb_weight
        self.stability_weight = stability_weight
        self.resonance_weight = resonance_weight
        self._tb_fitter = TitiusBodeFit()
        self._stability = StabilityAnalyzer(stellar_mass=stellar_mass)

    def predict(self, system) -> PredictionResult:
        """Run full prediction on a stellar system.

        Accepts StellarSystem, ExoplanetSystem, or list of (name, a, mass_solar).
        """
        t0 = time.perf_counter()
        warnings = []
        system_name = self._get_system_name(system)
        planet_full = StabilityAnalyzer.extract_planet_data_full(system)
        planet_data = [(n, a, m) for n, a, m, _ in planet_full]
        eccs = [e for _, _, _, e in planet_full]

        if len(planet_data) < 2:
            warnings.append("Need at least 2 planets for gap analysis")
            return PredictionResult(
                system_name=system_name,
                num_known_planets=len(planet_data),
                warnings=warnings,
            )

        axes = [a for _, a, _ in planet_data]
        names = [n for n, _, _ in planet_data]
        masses_sun = [m for _, _, m in planet_data]

        # 1. Titius-Bode fit (with mass weighting when available)
        tb_result = None
        if len(planet_data) >= self.min_known_planets:
            if any(m > 1e-10 for m in masses_sun):
                tb_result = TitiusBodeFit.best_fit(axes, names, masses=masses_sun)
            else:
                tb_result = TitiusBodeFit.best_fit(axes, names)
            if tb_result.r_squared < 0.5:
                warnings.append(
                    f"Titius-Bode fit quality low (R^2={tb_result.r_squared:.2f}). "
                    "Gap predictions may be unreliable."
                )
            if tb_result.loocv_rmse > 0.25:
                warnings.append(
                    f"TB fit LOOCV RMSE high ({tb_result.loocv_rmse:.2f} in log-a). "
                    "Fit may not generalize."
                )

        # 2. Stability analysis (eccentricity-aware)
        stability_regions = self._stability.find_stability_gaps(
            planet_data, eccentricities=eccs)

        # 3. Cross-reference TB gaps with stability regions
        predicted_gaps = self._cross_reference(
            axes, names, tb_result, stability_regions, planet_data, eccs
        )

        # 4. Normalize scores at system level
        if predicted_gaps:
            max_score = max(g.combined_score for g in predicted_gaps)
            if max_score > 1e-9:
                factor = 1.0 / np.sqrt(max_score)
                for g in predicted_gaps:
                    g.combined_score = round(min(1.0, g.combined_score * factor), 3)

        # Sort by combined score descending
        predicted_gaps.sort(key=lambda g: g.combined_score, reverse=True)

        elapsed = time.perf_counter() - t0
        return PredictionResult(
            system_name=system_name,
            num_known_planets=len(planet_data),
            tb_fit=tb_result,
            stability_regions=stability_regions,
            predicted_gaps=predicted_gaps,
            execution_time_s=elapsed,
            warnings=warnings,
        )

    def predict_multi_system(self, systems: list
                             ) -> dict[str, PredictionResult]:
        """Run prediction on multiple systems."""
        return {self._get_system_name(s): self.predict(s) for s in systems}

    def _resonance_score(self, predicted_a: float, inner_a: float,
                         outer_a: float) -> float:
        """Score [0, 1] for proximity of the predicted orbit to low-order
        mean-motion resonances with its neighbors.

        Bodies in (or near) low-order commensurabilities with neighbors
        (e.g. 3:2, 2:1) are dynamically protected — a predicted orbit near
        such a resonance is more plausible than one in between.
        """
        best = 0.0
        if inner_a <= 0 or outer_a <= inner_a or predicted_a <= 0:
            return best
        for p, q in RESONANCES:
            ratio = (p / q) ** (2.0 / 3.0)  # Kepler: a ~ P^(2/3)
            # Resonant orbit exterior to the inner neighbor,
            # and interior to the outer neighbor
            for a_res in (inner_a * ratio, outer_a / ratio):
                if inner_a < a_res < outer_a:
                    rel = abs(predicted_a - a_res) / a_res
                    if rel < RESONANCE_TOLERANCE:
                        best = max(best, 1.0 - rel / RESONANCE_TOLERANCE)
        return best

    def _cross_reference(self, axes: list[float], names: list[str],
                         tb_result: Optional[TBResult],
                         stability_regions: list[StabilityRegion],
                         planet_data: list[tuple[str, float, float]],
                         eccs: Optional[list[float]] = None
                         ) -> list[GapResult]:
        """Cross-reference TB gaps with stability regions to produce scored gaps.

        Two-pass approach:
        1. Compute per-gap TB, stability and resonance scores
        2. Cross-validate non-adjacent gaps for TB consistency boost
        """
        n = len(axes)
        gaps: list[GapResult] = []
        if eccs is None or len(eccs) < n:
            eccs = [0.0] * n

        # TB residual scatter (log-space) for confidence intervals
        tb_sigma = 0.0
        if tb_result is not None and tb_result.residuals:
            tb_sigma = float(np.std(tb_result.residuals))

        # ---- Pass 1: per-gap scoring ----
        for i in range(n - 1):
            inner_a = axes[i]
            outer_a = axes[i + 1]
            inner_name = names[i]
            outer_name = names[i + 1]

            tb_score = 0.0
            predicted_a = (inner_a * outer_a) ** 0.5  # geometric mean default

            if tb_result is not None and tb_result.r_squared >= 0.5:
                expected_ratio = tb_result.beta
                actual_ratio = outer_a / inner_a if inner_a > 0 else float("inf")
                ratio_excess = actual_ratio / max(expected_ratio, 1.01)

                if ratio_excess > 1.15:
                    # Scale: excess=1.15 → 0.3, excess=3.0 → 0.9
                    tb_score = min(0.9, 0.3 + 0.4 * (ratio_excess - 1.0))
                    predicted_a = (inner_a * outer_a) ** 0.5

                # Check TB gap match for index-level prediction
                tb_gaps = self._tb_fitter.score_gaps(tb_result, axes, names)
                for tg in tb_gaps:
                    if (tg.inner_planet == inner_name and
                            tg.outer_planet == outer_name):
                        predicted_a = tg.predicted_a
                        tb_score = max(tb_score, 0.7)
                        if inner_a < predicted_a < outer_a:
                            tb_score = max(tb_score, 0.9)
                        break
            else:
                tb_score = 0.3

            # Stability score — bias predicted_a toward stable region
            stability_score = 0.0
            mass_range = (0.1, 10.0)
            predicted_a_lower = max(inner_a * 1.05, inner_a + 0.001)
            predicted_a_upper = min(outer_a * 0.95, outer_a - 0.001)

            if i < len(stability_regions):
                sr = stability_regions[i]
                if sr.gap_ratio >= 1.0:
                    stability_score = min(1.0, sr.gap_ratio / 5.0)
                    mass_range = self._stability.allowed_mass_range(sr)
                    # Bias predicted position toward the stable zone inner 40%
                    if sr.width_au > 0:
                        preferred = sr.inner_boundary_au + 0.4 * sr.width_au
                        if inner_a < preferred < outer_a:
                            predicted_a = preferred
                    predicted_a_lower = max(predicted_a_lower, sr.inner_boundary_au)
                    predicted_a_upper = min(predicted_a_upper, sr.outer_boundary_au)

            # Resonance signal
            resonance = self._resonance_score(predicted_a, inner_a, outer_a)

            # Confidence interval from TB residual scatter (log-space sigma)
            if tb_sigma > 0:
                ci_lower = predicted_a * float(np.exp(-tb_sigma))
                ci_upper = predicted_a * float(np.exp(tb_sigma))
                new_lower = max(predicted_a_lower, ci_lower)
                new_upper = min(predicted_a_upper, ci_upper)
                if new_lower < new_upper:
                    predicted_a_lower, predicted_a_upper = new_lower, new_upper

            w_total = self.tb_weight + self.stability_weight + self.resonance_weight
            combined = (self.tb_weight * tb_score +
                        self.stability_weight * stability_score +
                        self.resonance_weight * resonance) / max(w_total, 1e-9)

            period = predicted_a ** 1.5
            method = "titius_bode+stability" if tb_result else "stability_only"
            if resonance > 0.5:
                method += "+resonance"

            gaps.append(GapResult(
                inner_a=inner_a, outer_a=outer_a,
                predicted_a=round(predicted_a, 4),
                predicted_period=round(period, 2),
                titius_bode_score=round(tb_score, 3),
                stability_score=round(stability_score, 3),
                combined_score=round(combined, 3),
                estimated_mass_range=mass_range,
                method=method,
                inner_planet=inner_name,
                outer_planet=outer_name,
                predicted_a_lower=round(predicted_a_lower, 4),
                predicted_a_upper=round(predicted_a_upper, 4),
                predicted_eccentricity=round((eccs[i] + eccs[i + 1]) / 2.0, 4),
                resonance_score=round(resonance, 3),
            ))

        # ---- Pass 2: cross-gap consistency ----
        if tb_result is not None and tb_result.r_squared >= 0.5:
            beta = tb_result.beta
            for i in range(len(gaps)):
                for j in range(i + 2, min(i + 4, len(gaps))):
                    # Check if gaps i and j are TB-consistent across multiple spacings
                    inner_a = gaps[i].inner_a
                    outer_a = gaps[j].outer_a
                    if inner_a > 0:
                        steps = j - i + 1
                        expected_step_ratio = beta ** steps
                        actual_span = outer_a / inner_a
                        if abs(actual_span - expected_step_ratio) / max(expected_step_ratio, 1.01) < 0.20:
                            # Consistent! Boost both gaps
                            boost = 0.10
                            gaps[i].combined_score = round(
                                min(1.0, gaps[i].combined_score + boost), 3)
                            gaps[j].combined_score = round(
                                min(1.0, gaps[j].combined_score + boost), 3)

        return gaps

    @staticmethod
    def _get_system_name(system) -> str:
        if isinstance(system, (StellarSystem, ExoplanetSystem)):
            return system.name
        if isinstance(system, str):
            return system
        return "unknown"
