"""Optional perturbation verification for pattern-based gap predictions.

Tests whether adding a planet at a predicted gap location reduces N-body
residuals against observed positions. Uses the FULL known system — no planets
are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stellar_predictor.data.models import GapResult, StellarSystem
from stellar_predictor.experimental.optimizer import LeastSquaresOptimizer
from stellar_predictor.inference.candidate import CandidateBody
from stellar_predictor.physics.nbody import NBodySimulator


@dataclass
class VerificationResult:
    """Result of perturbation verification for a gap prediction."""
    gap: GapResult
    nbody_residual_rms_without: float = 0.0
    nbody_residual_rms_with: float = 0.0
    improvement_ratio: float = 1.0
    candidate: CandidateBody | None = None
    verified: bool = False
    significance: float = 0.0
    error: str = ""


class PerturbationVerifier:
    """Verify predicted gaps by N-body perturbation analysis.

    Tests the counterfactual: if we add a planet at the predicted location,
    does the model better match observations?
    """

    def __init__(self, system: StellarSystem,
                 observed_positions: dict[str, np.ndarray],
                 times: np.ndarray,
                 verification_threshold: float = 1.5):
        self.system = system
        self.observed_positions = observed_positions
        self.times = times
        self.verification_threshold = verification_threshold
        self._sim = NBodySimulator(system)

    def verify_gap(self, gap: GapResult,
                   target_planets: list[str] | None = None
                   ) -> VerificationResult:
        """Test whether adding a planet at this gap reduces residuals.

        Args:
            gap: The predicted gap to verify.
            target_planets: Planets to use as perturbation probes.
                            Default: inner and outer neighbors.
        """
        if target_planets is None:
            target_planets = []
            if gap.inner_planet:
                target_planets.append(gap.inner_planet)
            if gap.outer_planet and gap.outer_planet != "(outer edge)":
                target_planets.append(gap.outer_planet)
        if not target_planets:
            # fallback: any planet with observed data
            target_planets = [n for n in self.observed_positions
                              if n.lower() != "sun"]

        result = VerificationResult(gap=gap)

        try:
            # 1. Simulate the known system WITHOUT trial body
            known_result = self._sim.simulate(
                t_end=float(np.max(self.times)),
                n_steps=len(self.times),
            )

            # Compute baseline RMS
            rms_without = 0.0
            for target in target_planets:
                if target not in self.observed_positions:
                    continue
                modeled = known_result.positions.get(target)
                if modeled is None:
                    continue
                diff = self.observed_positions[target] - modeled
                rms_without += float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
            result.nbody_residual_rms_without = rms_without

            # 2. Run DE optimization to find best trial body
            mass_mid = (gap.estimated_mass_range[0] + gap.estimated_mass_range[1]) / 2.0
            mass_solar = mass_mid / 332946.0

            optimizer = LeastSquaresOptimizer(
                self.system,
                self.observed_positions[target_planets[0]],
                self.times,
            )

            a_guess = gap.predicted_a
            bounds = [
                (1e-8, 1e-2),
                (max(0.5, a_guess * 0.5), a_guess * 2.0),
                (0.0, 0.3),
                (0.0, np.pi / 4),
                (0.0, 2 * np.pi),
                (0.0, 2 * np.pi),
                (0.0, 2 * np.pi),
            ]

            candidate = optimizer.optimize(target_planets[0], bounds=bounds)
            result.candidate = candidate

            # 3. Simulate WITH trial body
            trial_system = StellarSystem(name=self.system.name)
            for body in self.system.bodies:
                trial_system.add_body(body)
            trial_body = candidate  # just reuse

            trial_sim = NBodySimulator(trial_system)
            trial_result = trial_sim.simulate(
                t_end=float(np.max(self.times)),
                n_steps=len(self.times),
            )

            rms_with = 0.0
            for target in target_planets:
                if target not in self.observed_positions:
                    continue
                modeled = trial_result.positions.get(target)
                if modeled is None:
                    continue
                diff = self.observed_positions[target] - modeled
                rms_with += float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

            result.nbody_residual_rms_with = rms_with
            result.improvement_ratio = (rms_without / max(rms_with, 1e-12))
            result.verified = result.improvement_ratio > self.verification_threshold
            result.significance = result.improvement_ratio

        except Exception as e:
            result.error = str(e)

        return result

    def verify_all_candidates(self, prediction,
                              max_candidates: int = 5
                              ) -> list[VerificationResult]:
        """Run verification on top-ranked gaps from a prediction."""
        results = []
        for gap in prediction.predicted_gaps[:max_candidates]:
            vr = self.verify_gap(gap)
            results.append(vr)
        return results
