"""GapPredictor: combines Titius-Bode and stability analysis to predict unknown planets.

This is the primary prediction engine. It operates purely analytically on
known planet parameters — no N-body simulation required.

v0.3 improvements:
- Mean-motion resonance scoring as a third signal alongside TB + stability
- Eccentricity-aware Hill stability gaps
- Confidence intervals for the predicted semi-major axes derived from
  TB-fit residual scatter

v0.7 (scoring v2):
- TB evidence is a continuous "missing slots" measure (how many TB steps the
  gap spans, and how close that is to a whole number of slots).
- The combined score is *gated* by TB evidence, so a wide-but-unremarkable
  gap (high stability, TB ~ 0) can no longer score high on stability alone.
- Resonant chains are down-weighted globally, and a pair that is itself
  locked near a first-order MMR is penalised further (a body in between
  would break the lock).
- Outer-edge extrapolations are penalised per TB step beyond the outermost
  known planet.
- Orbital periods use the stellar mass: P = sqrt(a^3 / M_star) (years).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from stellar_predictor.data.models import ExoplanetSystem, GapResult, StellarSystem
from stellar_predictor.patterns.stability import (
    StabilityAnalyzer,
    StabilityRegion,
)
from stellar_predictor.patterns.titius_bode import TBResult, TitiusBodeFit

try:
    from config.settings import (
        CHAIN_LOCK_THRESHOLD,
        CHAIN_PENALTY,
        CROSS_GAP_MAX_BOOST,
        CROSS_GAP_REL_ERR_TOLERANCE,
        EDGE_STABILITY_CAP,
        EXTRAPOLATION_BASE,
        EXTRAPOLATION_DECAY,
        MULTI_PLANET_MIN_STEPS,
        OUTER_EDGE_MIN_R2,
        OUTER_EDGE_STEPS,
        PAIR_RESONANCE_PENALTY,
        RESONANCE_BASE_WEIGHT,
        RESONANCE_PULL_FRACTION,
        RESONANCE_SCORE_WEIGHT,
        RESONANCE_TOLERANCE,
        RESONANCES,
        STABILITY_BASE_WEIGHT,
        STABILITY_SIGMOID_SCALE,
        SYSTEM_LOGISTIC_OFFSET,
        TB_BASE_WEIGHT,
        TB_GATE_FLOOR,
        TB_RATIO_EXCESS_THRESHOLD,
        TB_STEP_INTEGRALITY_WIDTH,
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
    TB_RATIO_EXCESS_THRESHOLD = 1.15
    STABILITY_SIGMOID_SCALE = 2.5
    RESONANCE_PULL_FRACTION = 0.30
    CROSS_GAP_MAX_BOOST = 0.15
    CROSS_GAP_REL_ERR_TOLERANCE = 0.20
    SYSTEM_LOGISTIC_OFFSET = 0.15
    TB_GATE_FLOOR = 0.15
    TB_STEP_INTEGRALITY_WIDTH = 0.30
    CHAIN_PENALTY = 0.60
    CHAIN_LOCK_THRESHOLD = 0.50
    PAIR_RESONANCE_PENALTY = 0.70
    EXTRAPOLATION_BASE = 0.60
    EXTRAPOLATION_DECAY = 0.60
    EDGE_STABILITY_CAP = 0.70


@dataclass
class PredictionResult:
    """Complete prediction for a stellar system."""
    system_name: str
    num_known_planets: int
    tb_fit: TBResult | None = None
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

        chain_lock = self._resonant_chain_lock(axes)
        if chain_lock > CHAIN_LOCK_THRESHOLD:
            warnings.append(
                f"Compact resonant chain detected ({chain_lock:.0%} of adjacent "
                "pairs locked near a low-order MMR); gap scores are down-weighted "
                "because a hidden body would break the chain."
            )

        # 4. Normalize scores at system level (logistic squash preserving rank order)
        if predicted_gaps:
            for g in predicted_gaps:
                # Logistic: x/(x+0.15) maps 0.15->0.50, 0.30->0.67, 0.50->0.77, 0.90->0.86
                g.combined_score = round(min(1.0, g.combined_score / (g.combined_score + SYSTEM_LOGISTIC_OFFSET)), 3)

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

    def _period_yr(self, a_au: float) -> float:
        """Kepler's third law in AU / yr / M_sun: P = sqrt(a^3 / M_star)."""
        m = self.stellar_mass if self.stellar_mass and self.stellar_mass > 0 else 1.0
        return float(math.sqrt(max(a_au, 0.0) ** 3 / m))

    @staticmethod
    def _tb_step_score(inner_a: float, outer_a: float, beta: float) -> float:
        """TB evidence [0, 0.9] that whole TB slots are missing between two planets.

        ``steps = ln(a_out/a_in) / ln(beta)`` is the number of TB steps the
        pair spans. One step means "no slot missing". Evidence rises from
        1.3 to 2.0 steps and is modulated by integrality: 2.86 steps
        (Mars-Jupiter) looks like whole missing slots, 1.5 steps
        (Mercury-Venus) is just an irregular spacing.
        """
        if inner_a <= 0 or outer_a <= inner_a or beta <= 1.01:
            return 0.0
        steps = math.log(outer_a / inner_a) / math.log(beta)
        missing = min(1.0, max(0.0, (steps - 1.3) / 0.7))
        if missing <= 0.0:
            return 0.0
        k = max(2, round(steps))
        integrality = math.exp(-((steps - k) / TB_STEP_INTEGRALITY_WIDTH) ** 2)
        return 0.9 * missing * (0.5 + 0.5 * integrality)

    @staticmethod
    def _tb_gate(tb_score: float) -> float:
        """Multiplicative gate: stability/resonance only count with TB support."""
        tb = min(1.0, max(0.0, tb_score))
        return TB_GATE_FLOOR + (1.0 - TB_GATE_FLOOR) * tb

    @staticmethod
    def _pair_resonance_factor(inner_a: float, outer_a: float) -> float:
        """Penalty [1-PAIR_RESONANCE_PENALTY, 1] for a pair locked near a first-order MMR."""
        if inner_a <= 0 or outer_a <= inner_a:
            return 1.0
        period_ratio = (outer_a / inner_a) ** 1.5
        best = 0.0
        for p, q in RESONANCES:
            # Tight first-order locks only (3:2, 4:3, 5:4). A 2:1 pair can
            # host a body in between (2:1 = 4:3 x 3:2), so it is not penalised.
            if p - q != 1 or q < 2:
                continue
            rel = abs(period_ratio - p / q) / (p / q)
            tol = 2.0 * RESONANCE_TOLERANCE
            if rel < tol:
                best = max(best, math.exp(-3.0 * (rel / tol) ** 2))
        return 1.0 - PAIR_RESONANCE_PENALTY * best

    @staticmethod
    def extrapolation_factor(step: int) -> float:
        """Penalty for a candidate ``step`` TB slots beyond the outermost planet."""
        return EXTRAPOLATION_BASE * EXTRAPOLATION_DECAY ** max(0, step - 1)

    def _stability_score_sigmoid(self, gap_ratio: float) -> float:
        """Sigmoid stability score: gradual saturation.

        gap_ratio=2 -> 0.31, gap_ratio=5 -> 0.68, gap_ratio=10 -> 0.89
        """
        if gap_ratio < 1.0:
            return 0.0
        return max(0.0, 2.0 / (1.0 + math.exp(-gap_ratio / STABILITY_SIGMOID_SCALE)) - 1.0)

    def _detect_resonance_chain(self, axes: list[float]) -> float:
        """Fraction [0, 1] of adjacent pairs sitting near a low-order MMR.

        Resonant chains (e.g. TRAPPIST-1) have most adjacent period ratios
        clustered on low-order commensurabilities; classic Titius-Bode
        systems (Solar System, Kepler-33) do not. This gates how strongly
        resonance influences the predicted position and score: meaningful
        in chains, noise otherwise.
        """
        if len(axes) < 3:
            return 0.0
        near = 0
        total = 0
        for i in range(len(axes) - 1):
            if axes[i] <= 0:
                continue
            period_ratio = (axes[i + 1] / axes[i]) ** 1.5
            total += 1
            for p, q in RESONANCES:
                if q < p:
                    rel = abs(period_ratio - p / q) / (p / q)
                    if rel < 2.0 * RESONANCE_TOLERANCE:
                        near += 1
                        break
        return near / total if total else 0.0

    @staticmethod
    def _resonant_chain_lock(axes: list[float]) -> float:
        """Fraction [0, 1] of adjacent pairs that are *compact and locked*.

        Stricter than :meth:`_detect_resonance_chain` (which uses a doubled
        tolerance and all eight MMRs, so the Solar System scores 0.86): a
        pair counts only if its period ratio is <= ~2.1 and within
        ``RESONANCE_TOLERANCE`` of an MMR with p/q <= 2. TRAPPIST-1 -> 1.0,
        Solar System -> 0.29.
        """
        if len(axes) < 3:
            return 0.0
        locked = 0
        total = 0
        for i in range(len(axes) - 1):
            if axes[i] <= 0:
                continue
            total += 1
            pr = (axes[i + 1] / axes[i]) ** 1.5
            if pr > 2.0 * (1.0 + RESONANCE_TOLERANCE):
                continue
            for p, q in RESONANCES:
                if p / q <= 2.0 and abs(pr - p / q) / (p / q) < RESONANCE_TOLERANCE:
                    locked += 1
                    break
        return locked / total if total else 0.0

    @staticmethod
    def chain_penalty_factor(chain_lock: float) -> float:
        """Global score factor; only systems that are mostly locked are penalised."""
        excess = max(0.0, chain_lock - CHAIN_LOCK_THRESHOLD) / (1.0 - CHAIN_LOCK_THRESHOLD)
        return 1.0 - CHAIN_PENALTY * min(1.0, excess)

    def _tb_confidence(self, tb_result: TBResult | None) -> float:
        """Confidence [0, 1] that the TB position estimate is trustworthy.

        Combines in-sample R^2 with LOOCV generalisation: a fit that
        generalises poorly (high LOOCV RMSE in log-a) is down-weighted so
        stability/resonance corrections take over.
        """
        if tb_result is None or tb_result.r_squared < 0.5:
            return 0.0
        conf = max(0.0, min(1.0, tb_result.r_squared))
        if tb_result.loocv_rmse > 0:
            conf /= (1.0 + 2.0 * tb_result.loocv_rmse)
        return conf

    def _compute_dynamic_weights(self, tb_result: TBResult | None,
                                 chain_strength: float = 0.0
                                 ) -> tuple[float, float, float]:
        """Compute dynamic scoring weights based on TB fit quality.

        Resonance weight scales with ``chain_strength``: halved for
        non-resonant systems, up to ~2x for strong resonant chains.
        """
        chain_factor = 0.5 + 1.5 * chain_strength
        if tb_result is not None and tb_result.r_squared >= 0.5:
            tb_conf = max(0.3, min(0.99, tb_result.r_squared))
            tb_w = TB_BASE_WEIGHT * tb_conf
            stab_w = STABILITY_BASE_WEIGHT * (1.0 + 0.25 * (1.0 - tb_conf))
            res_w = RESONANCE_BASE_WEIGHT * tb_conf * chain_factor
        else:
            tb_w, stab_w, res_w = 0.0, 0.85, 0.15 * chain_factor
        return tb_w, stab_w, res_w

    def _cross_reference(self, axes: list[float], names: list[str],
                         tb_result: TBResult | None,
                         stability_regions: list[StabilityRegion],
                         planet_data: list[tuple[str, float, float]],
                         eccs: list[float] | None = None
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

        # Resonance-chain strength gates how much resonance influences
        # position and score (meaningful in chains, noise in TB systems).
        chain_strength = self._detect_resonance_chain(axes)

        # Dynamic weights based on fit quality + chain strength
        tb_w, stab_w, res_w = self._compute_dynamic_weights(
            tb_result, chain_strength)

        # Confidence in the TB position estimate (anchors the prediction)
        tb_conf = self._tb_confidence(tb_result)

        masses_sun = [m for _, _, m in planet_data]

        # ---- Pass 1: per-gap scoring ----
        for i in range(n - 1):
            inner_a = axes[i]
            outer_a = axes[i + 1]
            inner_name = names[i]
            outer_name = names[i + 1]

            tb_score = 0.0
            # Eccentricity-aware geometric mean: anchor between inner
            # apoapsis and outer periapsis. Reduces to sqrt(a_in*a_out)
            # for circular orbits, which is the exact position of a hidden
            # body in an equal-ratio (Titius-Bode) chain.
            e_in = max(0.0, eccs[i])
            e_out = max(0.0, eccs[i + 1])
            eff_inner = inner_a * (1.0 + e_in)
            eff_outer = outer_a * (1.0 - e_out)
            if eff_outer > eff_inner > 0:
                geom_anchor = (eff_inner * eff_outer) ** 0.5
            else:
                geom_anchor = (inner_a * outer_a) ** 0.5
            predicted_a = geom_anchor
            tb_anchor = None  # TB index-level prediction when available

            if tb_result is not None and tb_result.r_squared >= 0.5:
                # v0.7: continuous "missing slots" evidence (replaces the
                # ratio-excess ramp + flat 0.9 for any index skip, which
                # scored Saturn-Uranus as high as Mars-Jupiter).
                tb_score = self._tb_step_score(inner_a, outer_a, tb_result.beta)

                # TB index-level prediction anchors the *position* only
                tb_gaps = self._tb_fitter.score_gaps(tb_result, axes, names)
                for tg in tb_gaps:
                    if (tg.inner_planet == inner_name and
                            tg.outer_planet == outer_name):
                        tb_anchor = tg.predicted_a
                        break
            else:
                tb_score = 0.3

            # ---- Position anchor: blend the TB index prediction with the
            # eccentricity-aware geometric mean by TB confidence. A trusted
            # TB fit dominates; otherwise the geometric mean carries the
            # estimate. (Previously the stability-band arithmetic center
            # unconditionally overwrote this, biasing predictions outward.)
            if tb_anchor is not None and inner_a < tb_anchor < outer_a:
                predicted_a = tb_conf * tb_anchor + (1.0 - tb_conf) * geom_anchor
            else:
                predicted_a = geom_anchor

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

                    # Gentle corrections to the anchor, scaled by (1 - tb_conf)
                    # so a trusted TB position is never overwritten. The
                    # resonance pull is further gated by resonance-chain
                    # membership (noise in non-resonant TB systems).
                    if sr.width_au > 0:
                        correction_room = 1.0 - tb_conf

                        # (a) Resonance pull — only meaningful in chains
                        if chain_strength > 0.0 and correction_room > 0.0:
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
                                pull = RESONANCE_PULL_FRACTION * chain_strength * correction_room
                                predicted_a += (nearest_res - predicted_a) * pull

                        # (b) Clamp into the stable band; if the anchor
                        # escaped it entirely, fall back to the geometric
                        # (not arithmetic) center of the band.
                        lo_band = sr.inner_boundary_au + 0.1 * sr.width_au
                        hi_band = sr.outer_boundary_au - 0.1 * sr.width_au
                        if 0 < lo_band < hi_band and not (lo_band <= predicted_a <= hi_band):
                            geom_center = (sr.inner_boundary_au
                                           * sr.outer_boundary_au) ** 0.5
                            predicted_a = min(hi_band, max(lo_band, geom_center))

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
                                # Sub-slots inherit (and never exceed) the
                                # parent gap's TB evidence.
                                sub_tb = min(0.6, 0.75 * tb_score)
                                sub_stab = self._stability_score_sigmoid(
                                    (outer_a - inner_a) / max(sr.gap_ratio, 0.01))
                                sub_res = self._resonance_score(sub_a, inner_a, outer_a)
                                sub_w = tb_w + stab_w + res_w
                                sub_combined = (tb_w * sub_tb + stab_w * sub_stab
                                                + res_w * sub_res) / max(sub_w, 1e-9)
                                sub_combined *= (self._tb_gate(sub_tb)
                                                 * self._pair_resonance_factor(
                                                     inner_a, outer_a))
                                sub_gaps_for_this_pair.append(GapResult(
                                    inner_a=inner_a, outer_a=outer_a,
                                    predicted_a=round(sub_a, 4),
                                    predicted_period=round(self._period_yr(sub_a), 6),
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

            # v0.7: TB gating + per-pair resonance lock penalty
            combined *= self._tb_gate(tb_score)
            combined *= self._pair_resonance_factor(inner_a, outer_a)

            period = self._period_yr(predicted_a)
            method = "titius_bode+stability" if tb_result else "stability_only"
            if resonance > 0.5:
                method += "+resonance"

            gaps.append(GapResult(
                inner_a=inner_a, outer_a=outer_a,
                predicted_a=round(predicted_a, 4),
                predicted_period=round(period, 6),
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
        edge_steps = self._predict_outer_edge(
            gaps, axes, names, tb_result, masses_sun, eccs,
            tb_w, stab_w, res_w,
        )

        # ---- Pass 2: cross-gap consistency (amplitude-modulated) ----
        self._apply_cross_gap_consistency(gaps, tb_result)

        # ---- Pass 3 (v0.7): extrapolation + resonant-chain penalties ----
        # Applied last so no boost can re-order them away.
        for g, step in edge_steps:
            g.combined_score *= self.extrapolation_factor(step)
        chain_factor = self.chain_penalty_factor(self._resonant_chain_lock(axes))
        for g in gaps:
            g.combined_score = round(g.combined_score * chain_factor, 3)

        return gaps

    def _predict_outer_edge(self, gaps: list[GapResult], axes: list[float],
                            names: list[str], tb_result: TBResult | None,
                            masses_sun: list[float], eccs: list[float],
                            tb_w: float, stab_w: float, res_w: float
                            ) -> list[tuple[GapResult, int]]:
        """Append TB-extrapolated predictions beyond the outermost planet.

        Mutates ``gaps`` in place and returns ``(gap, step)`` pairs so the
        caller can apply the per-step extrapolation penalty. Only runs when
        the TB fit is good enough (``OUTER_EDGE_MIN_R2``); steps out to
        ``OUTER_EDGE_STEPS`` slots, stopping at a 500 AU sanity cap.
        """
        added: list[tuple[GapResult, int]] = []
        if tb_result is None or tb_result.r_squared < OUTER_EDGE_MIN_R2:
            return added
        n = len(axes)
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
            sep = (predicted_a - outer_a) / max(
                (outer_a + predicted_a) / 2.0 * ((last_mass + 1e-8) / (3.0 * self.stellar_mass)) ** (1.0 / 3.0),
                1e-12)
            edge_stab = self._stability_score_sigmoid(
                sep / max(self._stability.critical_sep, 1.0) * 3.0)
            # v0.7: open space beyond the outermost planet is not positive
            # evidence, so the edge stability term saturates instead of
            # growing with distance (which used to rank +3 above +1).
            edge_stab = min(edge_stab, EDGE_STABILITY_CAP)

            # Mass prior: wider for outer systems
            edge_mass_range = (0.5, min(500.0, 15.0 * (predicted_a / outer_a)))

            e_w = tb_w + stab_w + res_w
            e_combined = (tb_w * edge_tb_score + stab_w * edge_stab) / max(e_w, 1e-9)
            e_combined *= self._tb_gate(edge_tb_score)

            edge_gap = GapResult(
                inner_a=outer_a,
                outer_a=predicted_a * 1.3,
                predicted_a=round(predicted_a, 4),
                predicted_period=round(self._period_yr(predicted_a), 6),
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
            )
            gaps.append(edge_gap)
            added.append((edge_gap, edge_step))
        return added

    def _apply_cross_gap_consistency(self, gaps: list[GapResult],
                                     tb_result: TBResult | None) -> None:
        """Boost gaps whose multi-step spans are TB-consistent.

        Mutates ``gaps[*].combined_score`` in place. A span of ``steps`` gaps
        that matches ``beta**steps`` (within 20%) earns an amplitude-modulated
        boost scaled by the two endpoints' quality.
        """
        if tb_result is None or tb_result.r_squared < 0.5:
            return
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

                    if rel_err < CROSS_GAP_REL_ERR_TOLERANCE:
                        # Amplitude-modulated boost: better match = higher boost
                        boost = CROSS_GAP_MAX_BOOST * max(0.0, 1.0 - rel_err * 5.0)
                        # Modulate by gap quality
                        avg_quality = (gaps[i].combined_score + gaps[j].combined_score) / 2.0
                        quality_factor = 0.5 + 0.5 * avg_quality
                        boost *= quality_factor
                        # v0.7: multiplicative, so a gated (TB~0) gap
                        # cannot be lifted back up by a neighbour's boost.
                        gaps[i].combined_score = min(
                            1.0, gaps[i].combined_score * (1.0 + boost))
                        gaps[j].combined_score = min(
                            1.0, gaps[j].combined_score * (1.0 + boost))

    @staticmethod
    def _get_system_name(system) -> str:
        if isinstance(system, StellarSystem | ExoplanetSystem):
            return system.name
        if isinstance(system, str):
            return system
        return "unknown"
