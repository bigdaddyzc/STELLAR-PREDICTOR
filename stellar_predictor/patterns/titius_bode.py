"""Generalized Titius-Bode law fitting for planetary system spacing analysis.

Fits the form a_n = alpha * beta^n (log-linear) to known planet semi-major
axes, then identifies missing indices as predicted gaps where unknown planets
could exist.

v0.3 improvements:
- Skip-aware fitting: index assignments may leave empty slots (missing
  planets). The best assignment is selected with BIC to avoid overfitting,
  and skipped indices become direct gap predictions.
- Leave-one-out cross-validation (LOOCV) for fit robustness diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import curve_fit

try:
    from config.settings import TB_MAX_SKIPS, TB_SKIP_MIN_R2_GAIN
except ImportError:  # standalone fallback
    TB_MAX_SKIPS = 3
    TB_SKIP_MIN_R2_GAIN = 0.01


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
    # v0.3: skip-aware fitting diagnostics
    n_skips: int = 0
    bic: float = 0.0
    loocv_rmse: float = 0.0


def fit_titius_bode(axes: list[float], names: list[str]) -> TBResult:
    """Convenience function: best-fit Titius-Bode law to a sorted list of axes."""
    return TitiusBodeFit.best_fit(axes, names)


def _linear_fit(indices: np.ndarray, log_a: np.ndarray):
    """OLS fit of log_a = c0 + c1 * indices. Returns (coeffs, predicted, ss_res, r2)."""
    A = np.column_stack([np.ones(len(indices)), indices])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_a, rcond=None)
    predicted = A @ coeffs
    ss_res = float(np.sum((log_a - predicted) ** 2))
    ss_tot = float(np.sum((log_a - np.mean(log_a)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    return coeffs, predicted, ss_res, r2


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
            coeffs, predicted_log, ss_res, r2 = _linear_fit(indices, log_a)

            if r2 > best_r2 + 1e-9 or (abs(r2 - best_r2) < 1e-9 and best_result is not None and start < best_result.start_index):
                best_r2 = r2
                alpha = np.exp(coeffs[0])
                beta = np.exp(coeffs[1])
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
    def fit_with_skips(axes: list[float], names: list[str],
                       max_skips: int | None = None,
                       max_start: int = 3) -> TBResult:
        """Fit ln(a) = ln(alpha) + n * ln(beta) allowing skipped indices.

        Real systems can have undiscovered planets: their slot in the
        Titius-Bode sequence is empty (e.g. the Asteroid Belt between Mars
        and Jupiter). This method enumerates monotonic index assignments
        where each inter-planet step is 1 or 2 (a step of 2 leaves one
        empty slot), limited to ``max_skips`` total skips, and selects the
        assignment with the lowest BIC (each skip counts as an extra model
        parameter, preventing overfitting).

        Skipped indices are returned as TBGap entries with their predicted
        semi-major axes.
        """
        n_planets = len(axes)
        if n_planets < 3:
            return TitiusBodeFit.fit_log_linear(axes, names)
        if max_skips is None:
            max_skips = TB_MAX_SKIPS

        log_a = np.log(axes)
        best = None  # (sort_key, start, k, indices, coeffs, predicted, r2, bic)

        for start in range(max_start):
            for k in range(max_skips + 1):
                for slots in combinations_with_replacement(range(n_planets - 1), k):
                    increments = np.ones(n_planets - 1, dtype=int)
                    for s in slots:
                        increments[s] += 1
                    indices = np.concatenate(
                        ([start], start + np.cumsum(increments))
                    ).astype(float)

                    coeffs, predicted, ss_res, r2 = _linear_fit(indices, log_a)
                    bic = (n_planets * np.log(max(ss_res / n_planets, 1e-12))
                           + (2 + k) * np.log(n_planets))
                    sort_key = (bic, -r2, k, start)
                    if best is None or sort_key < best[0]:
                        best = (sort_key, start, k, indices, coeffs,
                                predicted, r2, bic)

        _, start, k, indices, coeffs, predicted, r2, bic = best
        alpha = float(np.exp(coeffs[0]))
        beta = float(np.exp(coeffs[1]))
        int_indices = [int(i) for i in indices]
        index_map = {names[i]: int_indices[i] for i in range(n_planets)}
        occupied = set(int_indices)
        last_idx = int_indices[-1]

        predicted_axes = [alpha * beta**j for j in range(start, last_idx + 4)]
        residuals = (log_a - predicted).tolist()

        # Skipped indices are direct gap predictions
        gaps: list[TBGap] = []
        for idx in range(start, last_idx + 1):
            if idx not in occupied:
                inner_candidates = [(i2, nm) for nm, i2 in index_map.items() if i2 < idx]
                outer_candidates = [(i2, nm) for nm, i2 in index_map.items() if i2 > idx]
                if inner_candidates and outer_candidates:
                    gaps.append(TBGap(
                        index=idx,
                        inner_planet=max(inner_candidates)[1],
                        outer_planet=min(outer_candidates)[1],
                        predicted_a=alpha * beta**idx,
                    ))

        loocv = TitiusBodeFit._loocv_rmse(np.asarray(indices, dtype=float), log_a)

        return TBResult(
            alpha=alpha, beta=beta, r_squared=float(r2),
            predicted_axes=predicted_axes,
            residuals=residuals,
            index_map=index_map,
            gaps=gaps,
            start_index=start,
            n_skips=k,
            bic=float(bic),
            loocv_rmse=loocv,
        )

    @staticmethod
    def _loocv_rmse(indices: np.ndarray, log_a: np.ndarray) -> float:
        """Leave-one-out cross-validation RMSE (log-space) for a fixed
        index assignment. Lower values indicate a more robust fit."""
        n = len(log_a)
        if n < 4:
            return 0.0
        errors = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            coeffs, _, _, _ = _linear_fit(indices[mask], log_a[mask])
            pred_i = coeffs[0] + coeffs[1] * indices[i]
            errors.append((log_a[i] - pred_i) ** 2)
        return float(np.sqrt(np.mean(errors)))

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
        """Try all fits and return the best.

        Order of preference:
        1. Log-linear (optionally mass-weighted when clearly better)
        2. Skip-aware fit — preferred when allowing missing-planet slots
           clearly improves the fit (R^2 gain above TB_SKIP_MIN_R2_GAIN)
        3. Classical form parameters attached as supplementary info

        LOOCV RMSE is computed for the chosen fit.
        """
        log_result = TitiusBodeFit.fit_log_linear(axes, names)

        # Also try weighted if masses are available
        if masses is not None and len(masses) == len(axes):
            weighted = TitiusBodeFit.fit_weighted(axes, names, masses)
            if weighted.r_squared > log_result.r_squared + 0.02:
                log_result = weighted

        # v0.3: skip-aware fit — captures missing-planet slots
        if len(axes) >= 4:
            try:
                skip_result = TitiusBodeFit.fit_with_skips(axes, names)
                if (skip_result.n_skips > 0
                        and skip_result.r_squared
                        > log_result.r_squared + TB_SKIP_MIN_R2_GAIN):
                    log_result = skip_result
            except Exception:
                pass

        # LOOCV diagnostic for the chosen fit
        if not log_result.loocv_rmse and log_result.index_map:
            try:
                indices = np.array(
                    [log_result.index_map[nm] for nm in names], dtype=float)
                log_result.loocv_rmse = TitiusBodeFit._loocv_rmse(
                    indices, np.log(np.asarray(axes, dtype=float)))
            except Exception:
                pass

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

        occupied_indices = set(result.index_map.values())
        if not occupied_indices:
            return []
        max_index = max(occupied_indices)
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
