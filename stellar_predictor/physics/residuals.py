"""Residual analysis: compare observed vs modeled trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lombscargle

from stellar_predictor.data.models import Residuals


class ResidualAnalyzer:
    """Compute and analyze residuals between observed and modeled positions."""

    def compute(
        self,
        observed_positions: np.ndarray,
        modeled_positions: np.ndarray,
        times: np.ndarray,
        body_name: str = "",
    ) -> Residuals:
        """Compute position residuals.

        Args:
            observed_positions: (N, 3) array in AU
            modeled_positions: (N, 3) array in AU
            times: (N,) time array in days

        Returns:
            Residuals object with magnitude and components
        """
        components = observed_positions - modeled_positions
        magnitudes = np.linalg.norm(components, axis=1)

        return Residuals(
            times=times,
            values=magnitudes,
            body_name=body_name,
            components=components,
        )

    def periodogram(
        self,
        residuals: Residuals,
        freq_range: tuple[float, float] | None = None,
        n_frequencies: int = 10000,
    ) -> PeriodogramResult:
        """Compute Lomb-Scargle periodogram of residuals.

        Args:
            residuals: Residual data
            freq_range: (min_freq, max_freq) in cycles/day, or auto
            n_frequencies: Number of frequency points

        Returns:
            PeriodogramResult with frequencies, power, and peak info
        """
        times = residuals.times

        # Use signed vector component with highest variance to avoid
        # frequency doubling from rectified (always-positive) magnitudes
        if residuals.components is not None:
            variances = np.var(residuals.components, axis=0)
            best_axis = int(np.argmax(variances))
            values = residuals.components[:, best_axis]
        else:
            values = residuals.values

        # Remove mean
        values = values - np.mean(values)

        # Auto frequency range based on data span and sampling
        dt = np.median(np.diff(times))
        T = times[-1] - times[0]

        if freq_range is None:
            f_min = 1.0 / T
            f_max = 0.5 / dt
        else:
            f_min, f_max = freq_range

        # Angular frequencies for scipy's lombscargle
        freqs = np.linspace(f_min, f_max, n_frequencies)
        angular_freqs = 2 * np.pi * freqs

        power = lombscargle(times, values, angular_freqs, normalize=True)

        # Find peak
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]
        peak_period = 1.0 / peak_freq  # days

        return PeriodogramResult(
            frequencies=freqs,
            power=power,
            peak_frequency=peak_freq,
            peak_period=peak_period,
            peak_power=power[peak_idx],
        )

    def estimate_signal_amplitude(self, residuals: Residuals) -> float:
        """Estimate the amplitude of the dominant signal in residuals (AU)."""
        if residuals.components is not None:
            variances = np.var(residuals.components, axis=0)
            best_axis = int(np.argmax(variances))
            return np.std(residuals.components[:, best_axis]) * np.sqrt(2)
        return np.std(residuals.values) * np.sqrt(2)


@dataclass
class PeriodogramResult:
    """Result of a Lomb-Scargle periodogram analysis."""

    frequencies: np.ndarray  # cycles/day
    power: np.ndarray  # normalized power
    peak_frequency: float  # cycles/day
    peak_period: float  # days
    peak_power: float

    @property
    def peak_period_years(self) -> float:
        return self.peak_period / 365.25
