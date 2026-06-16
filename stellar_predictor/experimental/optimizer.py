"""Least-squares optimizer for initial parameter estimation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize

from config.settings import DE_MAX_ITER, DE_SEED, DE_TOL
from stellar_predictor.data.models import StellarSystem
from stellar_predictor.inference.candidate import CandidateBody
from stellar_predictor.physics.nbody import NBodySimulator


class LeastSquaresOptimizer:
    """Estimate hidden body parameters by minimizing residuals."""

    def __init__(self, system: StellarSystem, observed_positions: np.ndarray, times: np.ndarray):
        """
        Args:
            system: Known system (without the hidden body)
            observed_positions: (N, 3) observed positions of the perturbed body
            times: (N,) times in days
        """
        self.system = system
        self.observed_positions = observed_positions
        self.times = times
        self.times_years = times / 365.25

    def cost_function(self, params: np.ndarray, target_body_name: str) -> float:
        """Compute chi-squared between model and observations.

        Args:
            params: [mass, a, e, inc, Omega, omega, M0]
            target_body_name: Name of the body whose positions we're comparing
        """
        mass, a, e, inc, Omega, omega, M0 = params

        # Reject unphysical parameters
        if mass <= 0 or a <= 0 or e < 0 or e >= 1:
            return 1e20

        # Create simulation with trial body
        sim = NBodySimulator(self.system)
        sim.add_test_body(
            mass=mass,
            semi_major_axis=a,
            eccentricity=e,
            inclination=inc,
            longitude_ascending=Omega,
            argument_perihelion=omega,
            mean_anomaly=M0,
            name="trial_body",
        )

        # Run simulation
        t_end = self.times_years[-1] - self.times_years[0]
        if t_end <= 0:
            return 1e20

        result = sim.simulate(t_end=t_end, n_steps=len(self.times))

        # Compare target body positions
        if target_body_name not in result.positions:
            return 1e20

        modeled = result.positions[target_body_name]
        chi2 = np.sum((self.observed_positions - modeled) ** 2)
        return chi2

    def optimize(
        self,
        target_body_name: str,
        bounds: list[tuple[float, float]] | None = None,
        method: str = "differential_evolution",
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> CandidateBody:
        """Find optimal parameters for the hidden body.

        Args:
            target_body_name: Body whose trajectory is perturbed
            bounds: Parameter bounds [(mass_lo, mass_hi), (a_lo, a_hi), ...]
            method: Optimization method
            progress_callback: Optional callback(status, progress_fraction)

        Returns:
            CandidateBody with best-fit parameters
        """
        if bounds is None:
            bounds = [
                (1e-8, 1e-2),    # mass (solar masses)
                (1.0, 100.0),    # semi-major axis (AU)
                (0.0, 0.5),      # eccentricity
                (0.0, np.pi),    # inclination
                (0.0, 2 * np.pi),  # Omega
                (0.0, 2 * np.pi),  # omega
                (0.0, 2 * np.pi),  # M0
            ]

        cost_fn = lambda params: self.cost_function(params, target_body_name)

        if method == "differential_evolution":
            de_callback = None
            if progress_callback:
                self._de_iter = 0
                def de_callback(xk, convergence):
                    self._de_iter += 1
                    progress_callback("optimizing", min(self._de_iter / DE_MAX_ITER, 0.99))

            result = differential_evolution(
                cost_fn,
                bounds=bounds,
                seed=DE_SEED,
                maxiter=DE_MAX_ITER,
                tol=DE_TOL,
                polish=True,
                popsize=20,
                callback=de_callback,
            )
        else:
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
            result = minimize(cost_fn, x0, method="Nelder-Mead", options={"maxiter": 10000})

        best = result.x
        # Estimate uncertainties from Hessian (crude approximation)
        uncertainties = self._estimate_uncertainties(best, bounds, cost_fn)

        # Convert to period using Kepler's third law (P^2 = a^3 for solar mass)
        a = best[1]
        period = a**1.5  # years, approximate for M_star ~ 1 M_sun

        return CandidateBody(
            mass=(best[0], best[0] * 0.5, best[0] * 2.0),
            semi_major_axis=(best[1], best[1] * uncertainties[1], best[1] / uncertainties[1]),
            eccentricity=(best[2], max(0, best[2] - 0.1), min(0.99, best[2] + 0.1)),
            inclination=(best[3], best[3] - 0.2, best[3] + 0.2),
            period=(period, period * 0.7, period * 1.5),
            confidence=self._estimate_confidence(result.fun),
            method="least_squares_orbital_residual",
            longitude_ascending=(best[4], best[4] - 0.3, best[4] + 0.3),
            argument_perihelion=(best[5], best[5] - 0.3, best[5] + 0.3),
        )

    def _estimate_uncertainties(
        self,
        best: np.ndarray,
        bounds: list[tuple[float, float]],
        cost_fn: Callable,
    ) -> np.ndarray:
        """Crude uncertainty estimation via parameter perturbation."""
        n_params = len(best)
        uncertainties = np.ones(n_params)

        chi2_min = cost_fn(best)
        if chi2_min <= 0:
            return uncertainties

        for i in range(n_params):
            delta = (bounds[i][1] - bounds[i][0]) * 0.01
            perturbed = best.copy()
            perturbed[i] += delta
            chi2_perturbed = cost_fn(perturbed)

            if chi2_perturbed > chi2_min:
                # Scale uncertainty by how sensitive chi2 is to this parameter
                ratio = chi2_perturbed / chi2_min
                uncertainties[i] = 1.0 + 1.0 / max(ratio - 1, 0.01)

        return uncertainties

    def _estimate_confidence(self, chi2_min: float) -> float:
        """Rough confidence estimate based on chi-squared improvement."""
        n_points = len(self.times)
        reduced_chi2 = chi2_min / max(n_points - 7, 1)

        # Map reduced chi2 to confidence (heuristic)
        if reduced_chi2 < 1e-6:
            return 0.95
        elif reduced_chi2 < 1e-4:
            return 0.8
        elif reduced_chi2 < 1e-2:
            return 0.5
        else:
            return 0.2
