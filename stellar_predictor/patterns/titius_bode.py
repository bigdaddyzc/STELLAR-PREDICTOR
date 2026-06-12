"""Generalized Titius-Bode law fitting for planetary system spacing analysis.

Fits the form a_n = alpha * beta^n (log-linear) to known planet semi-major
axes, then identifies missing indices as predicted gaps where unknown planets
could exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class TBGap:
    """A gap in the Titius-Bode index sequence."""
    index: int
    inner_planet: str
    outer_planet: str
    predicted_a: float
    is_edge: bool = False


@dataclass
class TBResult:
    """Result of Titius-Bode fitting."""
    alpha: float           # a_n = alpha * beta^n (log-linear form)
    beta: float
    r_squared: float
    predicted_axes: list[float]
    residuals: list[float]
    index_map: dict[str, int]
    outliers: list[str] = field(default_factory=list)
    gaps: list[TBGap] = field(default_factory=list)
    start_index: int = 0
    # Classical form: a = a0 + b * c^n
    a0: float = 0.0
    b: float = 0.0
    c: float = 0.0


def fit_titius_bode(axes: list[float], names: list[str]) -> TBResult:
    """Convenience function: best-fit Titius-Bode law to a sorted list of axes."""
    return TitiusBodeFit.best_fit(axes, names)


class TitiusBodeFit:
    """Fit generalized Titius-Bode law to planetary system spacing."""

    @staticmethod
    def fit_log_linear(axes: list[float], names: list[str]) -> TBResult:
        """Fit ln(a) = ln(alpha) + n * ln(beta) via linear regression.

        Tries multiple index anchorings to find the best fit. For N planets,
        the first planet's index can be 0, 1, 2, ... up to max_start_index.
        """
        if len(axes) < 2:
            return TBResult(
                alpha=axes[0] if axes else 0, beta=1.0, r_squared=0.0,
                predicted_axes=[], residuals=[], index_map={},
            )

        n_planets = len(axes)
        log_a = np.log(axes)
        max_start = min(5, n_planets)

        best_r2 = -np.inf
        best_result = None

        for start in range(max_start):
            indices = np.arange(start, start + n_planets, dtype=float)
            # Linear regression: log_a = ln(alpha) + indices * ln(beta)
            A = np.column_stack([np.ones(n_planets), indices])
            coeffs, residuals_vec, rank, singular = np.linalg.lstsq(
                A, log_a, rcond=None
            )
            ln_alpha, ln_beta = coeffs[0], coeffs[1]
            predicted_log = A @ coeffs
            ss_res = np.sum((log_a - predicted_log) ** 2)
            ss_tot = np.sum((log_a - np.mean(log_a)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

            if r2 > best_r2 + 1e-9 or (abs(r2 - best_r2) < 1e-9 and best_result is not None and start < best_result.start_index):
                best_r2 = r2
                alpha = np.exp(ln_alpha)
                beta = np.exp(ln_beta)
                residuals = (log_a - predicted_log).tolist()
                index_map = {names[i]: start + i for i in range(n_planets)}

                predicted_axes = [alpha * beta**j for j in range(start, start + n_planets + 3)]

                best_result = TBResult(
                    alpha=alpha, beta=beta, r_squared=r2,
                    predicted_axes=predicted_axes,
                    residuals=residuals,
                    index_map=index_map,
                    start_index=start,
                )

        return best_result

    @staticmethod
    def fit_classical(axes: list[float], names: list[str]) -> TBResult:
        """Fit a = a0 + b * c^n with nonlinear least squares."""
        if len(axes) < 3:
            return TitiusBodeFit.fit_log_linear(axes, names)

        n_planets = len(axes)

        def classical(n, a0, b, c):
            return a0 + b * c**n

        best_r2 = -np.inf
        best_result = None

        for start in range(min(5, n_planets)):
            indices = np.arange(start, start + n_planets, dtype=float)
            try:
                popt, _ = curve_fit(
                    classical, indices, axes,
                    p0=[0.4, 0.3, 1.7],
                    bounds=([0, 0, 0.5], [5, 10, 3.0]),
                    maxfev=5000,
                )
                predicted = classical(indices, *popt)
                ss_res = np.sum((axes - predicted) ** 2)
                ss_tot = np.sum((axes - np.mean(axes)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

                if r2 > best_r2 + 1e-9 or (abs(r2 - best_r2) < 1e-9 and best_result is not None and start < best_result.start_index):
                    best_r2 = r2
                    residuals = (np.log(axes) - np.log(predicted)).tolist()
                    index_map = {names[i]: start + i for i in range(n_planets)}
                    pred_axes = [classical(j, *popt) for j in range(start, start + n_planets + 3)]
                    best_result = TBResult(
                        alpha=popt[0], beta=popt[2], r_squared=r2,
                        predicted_axes=pred_axes,
                        residuals=residuals,
                        index_map=index_map,
                        start_index=start,
                        a0=popt[0], b=popt[1], c=popt[2],
                    )
            except Exception:
                continue

        if best_result is None:
            return TitiusBodeFit.fit_log_linear(axes, names)
        return best_result

    @staticmethod
    def fit_weighted(axes: list[float], names: list[str],
                     masses: list[float]) -> TBResult:
        """Fit ln(a) = ln(alpha) + n * ln(beta) with mass-based weights.

        Weight w_i = (mass_i / max_mass)^0.25 — quartic root dampens
        the dominance of gas giants while still giving them more say
        than low-mass inner planets.
        """
        if len(axes) < 2:
            return TBResult(
                alpha=axes[0] if axes else 0, beta=1.0, r_squared=0.0,
                predicted_axes=[], residuals=[], index_map={},
            )

        n_planets = len(axes)
        log_a = np.log(axes)
        max_mass = max(masses)
        weights = np.array([(m / max_mass) ** 0.25 for m in masses])
        W = np.diag(weights)
        max_start = min(5, n_planets)

        best_r2 = -np.inf
        best_result = None

        for start in range(max_start):
            indices = np.arange(start, start + n_planets, dtype=float)
            A = np.column_stack([np.ones(n_planets), indices])
            # Weighted least squares: (W A) coeffs = W log_a
            WA = W @ A
            Wlog_a = W @ log_a
            coeffs, residuals_vec, rank, singular = np.linalg.lstsq(
                WA, Wlog_a, rcond=None
            )
            ln_alpha, ln_beta = coeffs[0], coeffs[1]
            predicted_log = A @ coeffs
            # Weighted R^2
            ss_res = np.sum(weights * (log_a - predicted_log) ** 2)
            mean_log = np.sum(weights * log_a) / np.sum(weights)
            ss_tot = np.sum(weights * (log_a - mean_log) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

            if r2 > best_r2 + 1e-9 or (abs(r2 - best_r2) < 1e-9
                    and best_result is not None and start < best_result.start_index):
                best_r2 = r2
                alpha = np.exp(ln_alpha)
                beta = np.exp(ln_beta)
                residuals = (log_a - predicted_log).tolist()
                index_map = {names[i]: start + i for i in range(n_planets)}
                predicted_axes = [alpha * beta**j for j in range(start, start + n_planets + 3)]
                best_result = TBResult(
                    alpha=alpha, beta=beta, r_squared=r2,
                    predicted_axes=predicted_axes,
                    residuals=residuals,
                    index_map=index_map,
                    start_index=start,
                )

        return best_result

    @staticmethod
    def best_fit(axes: list[float], names: list[str],
                 masses: list[float] | None = None) -> TBResult:
        """Try both fits, return log-linear as primary with classical as supplementary.

        If masses are provided, also tries mass-weighted fit and picks the
        one with highest R^2.
        """
        log_result = TitiusBodeFit.fit_log_linear(axes, names)

        # Also try weighted if masses are available
        if masses is not None and len(masses) == len(axes):
            weighted = TitiusBodeFit.fit_weighted(axes, names, masses)
            if weighted.r_squared > log_result.r_squared + 0.02:
                log_result = weighted

        if len(axes) < 3:
            return log_result
        try:
            classical = TitiusBodeFit.fit_classical(axes, names)
            if classical.r_squared > log_result.r_squared:
                log_result.a0 = classical.a0
                log_result.b = classical.b
                log_result.c = classical.c
        except Exception:
            pass
        return log_result

    def score_gaps(self, result: TBResult, axes: list[float],
                   names: list[str]) -> list[TBGap]:
        """Identify missing indices that represent significant gaps.

        A gap is significant if the predicted axis for that index falls
        between existing planet axes where there is a real separation.
        """
        if result.r_squared < 0.5:
            return []

        n_planets = len(axes)
        max_index = result.start_index + n_planets - 1
        occupied_indices = set(result.index_map.values())
        gaps = []

        for n in range(result.start_index, max_index + 1):
            if n not in occupied_indices:
                pred_a = result.alpha * result.beta**n
                # Find the planets on either side
                inner_name = ""
                outer_name = ""
                inner_a = 0
                outer_a = float("inf")
                for name, idx in result.index_map.items():
                    a_val = axes[names.index(name)]
                    if idx < n and a_val > inner_a:
                        inner_a = a_val
                        inner_name = name
                    if idx > n and a_val < outer_a:
                        outer_a = a_val
                        outer_name = name

                if inner_a > 0 and outer_a < float("inf"):
                    gap = TBGap(
                        index=n, inner_planet=inner_name,
                        outer_planet=outer_name, predicted_a=pred_a,
                    )
                    gaps.append(gap)

        # Also check beyond the last planet
        last_n = max_index
        for n in range(last_n + 1, last_n + 4):
            pred_a = result.alpha * result.beta**n
            gaps.append(TBGap(
                index=n, inner_planet=names[-1],
                outer_planet="(outer edge)", predicted_a=pred_a,
                is_edge=True,
            ))

        return gaps

    def gap_significance(self, result: TBResult, gap: TBGap,
                         axes: list[float], names: list[str]) -> float:
        """Return 0-1 score for TB-based gap significance."""
        if result.r_squared < 0.5:
            return 0.0
        residuals_arr = np.abs(result.residuals)
        if len(residuals_arr) == 0:
            return 0.0
        sigma = np.std(residuals_arr) if len(residuals_arr) > 1 else 0.1
        mean_res = np.mean(residuals_arr)
        if sigma < 1e-6:
            return 0.5
        z_score = (mean_res / sigma) if sigma > 0 else 0
        return min(1.0, max(0.0, 0.5 + 0.3 * z_score))
