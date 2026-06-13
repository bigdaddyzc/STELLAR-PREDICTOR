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

import math
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
        OUTER_EDGE_STEPS,
        OUTER_EDGE_MIN_R2,
        MULTI_PLANET_MIN_STEPS,
        TB_BASE_WEIGHT,
        STABILITY_BASE_WEIGHT,
        RESONANCE_BASE_WEIGHT,
    )
except ImportError:  # standalone fallback
    RESONANCES = [(2, 1), (3, 2), (4, 3), (5, 3), (5, 4), (3, 1), (5, 2), (7, 3)]
    RESONANCE_SCORE_WEIGHT = 0.15
    RESONANCE_TOLERANCE = 0.05
    OUTER_EDGE_STEPS = 3
    OUTER_EDGE_MIN_R2 = 0.7
    MULTI_PLANET_MIN_STEPS = 3
    TB_BASE_WEIGHT = 0.50
    STABILITY_BASE_WEIGHT = 0.40
    RESONANCE_BASE_WEIGHT = 0.10


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

        # 4. Normalize scores at system level (logistic squash preserving rank order)
        if predicted_gaps:
            for g in predicted_gaps:
                # Logistic: x/(x+0.15) maps 0.15->0.50, 0.30->0.67, 0.50->0.77, 0.90->0.86
                g.combined_score = round(min(1.0, g.combined_score / (g.combined_score + 0.15)), 3)

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
        """Score [0, 1] for proximity of predicted orbit to low-order MMRs.

        v0.4: Gaussian proximity weighting with order penalty.
        Lower-order resonances (e.g. 2:1) protect more strongly than
        higher-order ones (e.g. 5:4).
        """
        best = 0.0
        if inner_a <= 0 or outer_a <= inner_a or predicted_a <= 0:
            return best
        for p, q in RESONANCES:
            ratio = (p / q) ** (2.0 / 3.0)  # Kepler: a ~ P^(2/3)
            order = abs(p - q)
            # Penalize higher-order resonances (weaker dynamical protection)
            order_factor = max(0.3, 1.0 - 0.2 * (order - 1))

            for a_res in (inner_a * ratio, outer_a / ratio):
                if inner_a < a_res < outer_a:
                    rel = abs(predicted_a - a_res) / a_res
                    if rel < RESONANCE_TOLERANCE:
                        # Gaussian proximity: 1.0 at exact, ~0.5 at half-tolerance
                        proximity = math.exp(-3.0 * (rel / RESONANCE_TOLERANCE) ** 2)
                        score = proximity * order_factor
                        best = max(best, score)
        return best

    def _stability_score_sigmoid(self, gap_ratio: float) -> float:
        """Sigmoid stability score: gradual saturation.

        gap_ratio=2 -> 0.31, gap_ratio=5 -> 0.68, gap_ratio=10 -> 0.89
        """
        if gap_ratio < 1.0:
            return 0.0
        return max(0.0, 2.0 / (1.0 + math.exp(-gap_ratio / 2.5)) - 1.0)

    def _compute_dynamic_weights(self, tb_result: Optional[TBResult]
                                  ) -> tuple[float, float, float]:
        """Compute dynamic scoring weights based on TB fit quality."""
        if tb_result is not None and tb_result.r_squared >= 0.5:
            tb_conf = max(0.3, min(0.99, tb_result.r_squared))
            tb_w = TB_BASE_WEIGHT * tb_conf
            stab_w = STABILITY_BASE_WEIGHT * (1.0 + 0.25 * (1.0 - tb_conf))
            res_w = RESONANCE_BASE_WEIGHT * tb_conf
        else:
            tb_w, stab_w, res_w = 0.0, 0.85, 0.15
        return tb_w, stab_w, res_w

    def _cross_reference(self, axes: list[float], names: list[str],
                         tb_result: Optional[TBResult],
                         stability_regions: list[StabilityRegion],
                         planet_data: list[tuple[str, float, float]],
                         eccs: Optional[list[float]] = None
                         ) -> list[GapResult]:
        """Cross-reference TB gaps with stability regions to produce scored gaps.

        v0.4 improvements:
        - Sigmoid stability scoring (gradual saturation)
        - Dynamic weights based on TB fit quality
        - Gaussian resonance scoring with order penalty
        - Amplitude-modulated cross-gap consistency boost
        - Uncertainty propagation to mass, period, and combined score
        - Multi-planet-per-wide-gap detection
        - Resonance-aware predicted_a bias
        """
        n = len(axes)
        gaps: list[GapResult] = []
        if eccs is None or len(eccs) < n:
            eccs = [0.0] * n

        # TB residual scatter (log-space) for confidence intervals
        tb_sigma = 0.0
        if tb_result is not None and tb_result.residuals:
            tb_sigma = float(np.std(tb_result.residuals))

        # Dynamic weights based on fit quality
        tb_w, stab_w, res_w = self._compute_dynamic_weights(tb_result)

        masses_sun = [m for _, _, m in planet_data]

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

            # Stability score — sigmoid-based
            stability_score = 0.0
            mass_range = (0.1, 10.0)
            predicted_a_lower = max(inner_a * 1.05, inner_a + 0.001)
            predicted_a_upper = min(outer_a * 0.95, outer_a - 0.001)

            if i < len(stability_regions):
                sr = stability_regions[i]
                if sr.gap_ratio >= 1.0:
                    stability_score = self._stability_score_sigmoid(sr.gap_ratio)
                    mass_range = self._stability.allowed_mass_range(sr)

                    # Resonance-aware predicted_a bias (v0.4)
                    if sr.width_au > 0:
                        center = (sr.inner_boundary_au + sr.outer_boundary_au) / 2.0
                        # Find resonance positions in the gap and pull toward
                        # the one closest to the current predicted_a
                        nearest_res = None
                        nearest_dist = float("inf")
                        for p, q in RESONANCES:
                            ratio = (p / q) ** (2.0 / 3.0)
                            for a_res in (inner_a * ratio, outer_a / ratio):
                                if sr.inner_boundary_au < a_res < sr.outer_boundary_au:
                                    dist = abs(a_res - predicted_a)
                                    if dist < nearest_dist:
                                        nearest_dist = dist
                                        nearest_res = a_res
                        if nearest_res is not None:
                            # Pull gently (0.3) toward nearest resonance
                            preferred = predicted_a + (nearest_res - predicted_a) * 0.3
                        else:
                            preferred = center
                        preferred = max(sr.inner_boundary_au + 0.1 * sr.width_au,
                                        min(sr.outer_boundary_au - 0.1 * sr.width_au,
                                            preferred))
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

            # ---- v0.4: Uncertainty propagation ----
            if tb_sigma > 0:
                # Mass uncertainty (mass ~ a^3 from Hill sphere)
                mass_mid = (mass_range[0] + mass_range[1]) / 2.0
                mass_sigma = 3.0 * tb_sigma
                ml = max(mass_range[0], mass_mid * np.exp(-mass_sigma))
                mu = min(mass_range[1], mass_mid * np.exp(mass_sigma))
                mass_range = (round(ml, 3), round(mu, 3))

            # ---- v0.4: Multi-planet-in-wide-gap detection ----
            sub_gaps_for_this_pair: list[GapResult] = []
            if tb_result is not None and tb_result.r_squared >= 0.5:
                tb_steps = round(math.log(outer_a / inner_a) / math.log(max(tb_result.beta, 1.01)))
                if tb_steps >= MULTI_PLANET_MIN_STEPS:
                    for sub_idx in range(1, tb_steps):
                        sub_a = inner_a * (tb_result.beta ** sub_idx)
                        if sub_a <= inner_a * 1.02 or sub_a >= outer_a * 0.98:
                            continue
                        if i < len(stability_regions):
                            sr = stability_regions[i]
                            if sr.inner_boundary_au < sub_a < sr.outer_boundary_au:
                                sub_tb = 0.6
                                sub_stab = self._stability_score_sigmoid(
                                    (outer_a - inner_a) / max(sr.gap_ratio, 0.01))
                                sub_res = self._resonance_score(sub_a, inner_a, outer_a)
                                sub_w = tb_w + stab_w + res_w
                                sub_combined = (tb_w * sub_tb + stab_w * sub_stab
                                                + res_w * sub_res) / max(sub_w, 1e-9)
                                sub_gaps_for_this_pair.append(GapResult(
                                    inner_a=inner_a, outer_a=outer_a,
                                    predicted_a=round(sub_a, 4),
                                    predicted_period=round(sub_a ** 1.5, 2),
                                    titius_bode_score=round(sub_tb, 3),
                                    stability_score=round(sub_stab, 3),
                                    combined_score=round(sub_combined, 3),
                                    estimated_mass_range=mass_range,
                                    method="sub_gap_tb_fit",
                                    inner_planet=inner_name,
                                    outer_planet=outer_name,
                                    predicted_a_lower=round(sub_a * 0.85, 4),
                                    predicted_a_upper=round(sub_a * 1.15, 4),
                                    predicted_eccentricity=round((eccs[i] + eccs[i + 1]) / 2.0, 4),
                                    resonance_score=round(sub_res, 3),
                                ))

            # ---- Combined score with dynamic weights ----
            w_total = tb_w + stab_w + res_w
            combined = (tb_w * tb_score + stab_w * stability_score
                        + res_w * resonance) / max(w_total, 1e-9)

            # Uncertainty penalty on combined score
            if tb_sigma > 0:
                uncertainty_penalty = 1.0 / (1.0 + 2.0 * tb_sigma)
                combined *= uncertainty_penalty

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

            # Add sub-gaps after the primary gap
            gaps.extend(sub_gaps_for_this_pair)

        # ---- Pass 1b: Outer-edge prediction (beyond last known planet) ----
        if tb_result is not None and tb_result.r_squared >= OUTER_EDGE_MIN_R2:
            outer_a = axes[-1]
            outer_name = names[-1]
            last_idx = tb_result.index_map.get(names[-1],
                                               tb_result.start_index + n - 1)
            for edge_step in range(1, OUTER_EDGE_STEPS + 1):
                pred_index = last_idx + edge_step
                predicted_a = tb_result.alpha * (tb_result.beta ** pred_index)
                if predicted_a > 500:  # sanity cap — beyond ~500 AU is noise
                    break

                # TB score: moderate, based on fit quality
                edge_tb_score = min(0.7, 0.3 + 0.4 * (tb_result.r_squared - 0.5))

                # Stability score: Hill radius separation from last known planet
                last_mass = masses_sun[-1] if masses_sun else 1e-6
                r_h_last = outer_a * (last_mass / (3.0 * self.stellar_mass)) ** (1.0 / 3.0)
                sep = (predicted_a - outer_a) / max(
                    (outer_a + predicted_a) / 2.0 * ((last_mass + 1e-8) / (3.0 * self.stellar_mass)) ** (1.0 / 3.0),
                    1e-12)
                edge_stab = self._stability_score_sigmoid(
                    sep / max(self._stability.critical_sep, 1.0) * 3.0)

                # Mass prior: wider for outer systems
                edge_mass_range = (0.5, min(500.0, 15.0 * (predicted_a / outer_a)))

                e_w = tb_w + stab_w + res_w
                e_combined = (tb_w * edge_tb_score + stab_w * edge_stab) / max(e_w, 1e-9)

                gaps.append(GapResult(
                    inner_a=outer_a,
                    outer_a=predicted_a * 1.3,
                    predicted_a=round(predicted_a, 4),
                    predicted_period=round(predicted_a ** 1.5, 2),
                    titius_bode_score=round(edge_tb_score, 3),
                    stability_score=round(edge_stab, 3),
                    combined_score=round(e_combined, 3),
                    estimated_mass_range=edge_mass_range,
                    method="tb_extrapolation+stability_edge",
                    inner_planet=outer_name,
                    outer_planet=f"(outer edge +{edge_step})",
                    predicted_a_lower=round(max(outer_a * 1.05, predicted_a * 0.8), 4),
                    predicted_a_upper=round(predicted_a * 1.2, 4),
                    predicted_eccentricity=round(eccs[-1], 4) if eccs else 0.0,
                    resonance_score=0.0,
                ))

        # ---- Pass 2: cross-gap consistency (amplitude-modulated) ----
        if tb_result is not None and tb_result.r_squared >= 0.5:
            beta = tb_result.beta
            for i in range(len(gaps)):
                for j in range(i + 2, min(i + 4, len(gaps))):
                    inner_a = gaps[i].inner_a
                    outer_a = gaps[j].outer_a
                    if inner_a > 0:
                        steps = j - i + 1
                        expected_step_ratio = beta ** steps
                        actual_span = outer_a / inner_a
                        # Relative error in TB consistency
                        rel_err = abs(actual_span - expected_step_ratio) / max(expected_step_ratio, 1.01)

                        if rel_err < 0.20:
                            # Amplitude-modulated boost: better match = higher boost
                            boost = 0.15 * max(0.0, 1.0 - rel_err * 5.0)
                            # Modulate by gap quality
                            avg_quality = (gaps[i].combined_score + gaps[j].combined_score) / 2.0
                            quality_factor = 0.5 + 0.5 * avg_quality
                            boost *= quality_factor
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
