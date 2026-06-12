"""Main prediction pipeline: pattern analysis -> gap detection -> optional verification."""

from __future__ import annotations

from typing import Optional

import numpy as np

from stellar_predictor.data.models import ExoplanetSystem, GapResult, StellarSystem
from stellar_predictor.patterns.predictor import GapPredictor, PredictionResult
from stellar_predictor.verification.perturbation import (
    PerturbationVerifier,
    VerificationResult,
)


class PredictionPipeline:
    """Orchestrates pattern-based prediction with optional perturbation verification."""

    def __init__(self, enable_verification: bool = False,
                 verification_timeout_seconds: float = 600):
        self.enable_verification = enable_verification
        self.verification_timeout = verification_timeout_seconds
        self._predictor = GapPredictor()

    def analyze(self, system) -> PredictionResult:
        """Run pattern analysis only (fast, no N-body)."""
        if isinstance(system, StellarSystem):
            self._predictor.stellar_mass = system.total_mass or 1.0
        elif isinstance(system, ExoplanetSystem):
            self._predictor.stellar_mass = system.stellar_mass
        return self._predictor.predict(system)

    def predict(self, system,
                observed_positions: Optional[dict[str, np.ndarray]] = None,
                times: Optional[np.ndarray] = None,
                ) -> tuple[PredictionResult, Optional[list[VerificationResult]]]:
        """Full pipeline: pattern analysis + optional perturbation verification.

        Returns:
            (prediction_result, verification_results or None)
        """
        pred_result = self.analyze(system)

        verification_results = None
        if (self.enable_verification and observed_positions is not None
                and times is not None and isinstance(system, StellarSystem)):
            verifier = PerturbationVerifier(system, observed_positions, times)
            verification_results = verifier.verify_all_candidates(pred_result)

        return pred_result, verification_results

    def predict_exoplanet(self, system: ExoplanetSystem) -> PredictionResult:
        """Pattern-only prediction for exoplanet systems."""
        self._predictor.stellar_mass = system.stellar_mass
        return self._predictor.predict(system)

    def multi_system_analysis(self, systems: list
                              ) -> dict[str, PredictionResult]:
        """Analyze multiple systems and return results keyed by name."""
        return self._predictor.predict_multi_system(systems)

    def aggregate_patterns(self, results: dict[str, PredictionResult]) -> dict:
        """Aggregate spacing patterns across multiple systems."""
        all_ratios = []
        system_tb_betas = []

        for name, result in results.items():
            if result.tb_fit and result.tb_fit.r_squared > 0.5:
                system_tb_betas.append({
                    "system": name,
                    "beta": result.tb_fit.beta,
                    "r_squared": result.tb_fit.r_squared,
                    "num_planets": result.num_known_planets,
                })

            for gap in result.predicted_gaps:
                if gap.inner_a > 0:
                    ratio = gap.outer_a / gap.inner_a
                    all_ratios.append(ratio)

        return {
            "mean_spacing_ratio": float(np.mean(all_ratios)) if all_ratios else 0,
            "median_spacing_ratio": float(np.median(all_ratios)) if all_ratios else 0,
            "std_spacing_ratio": float(np.std(all_ratios)) if all_ratios else 0,
            "systems_analyzed": len(results),
            "tb_fits": system_tb_betas,
        }
