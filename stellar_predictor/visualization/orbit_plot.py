"""Orbit and residual visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from stellar_predictor.data.models import Residuals, SimulationResult, StellarSystem
from stellar_predictor.inference.candidate import CandidateBody


def plot_system_orbits(
    system: StellarSystem,
    simulation: SimulationResult | None = None,
    candidates: list[CandidateBody] | None = None,
    projection: str = "2d",
    figsize: tuple[float, float] = (10, 10),
    title: str | None = None,
) -> plt.Figure:
    """Plot orbital paths of a stellar system.

    Args:
        system: The stellar system
        simulation: Optional simulation result with trajectories
        candidates: Optional candidate bodies to overlay
        projection: "2d" or "3d"
        figsize: Figure size
        title: Plot title
    """
    fig = plt.figure(figsize=figsize)

    if projection == "3d":
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    colors = plt.cm.tab10(np.linspace(0, 1, len(system.bodies)))

    if simulation is not None:
        for i, body in enumerate(system.bodies):
            pos = simulation.positions.get(body.name)
            if pos is None:
                continue

            if projection == "3d":
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                        color=colors[i], label=body.name, alpha=0.7)
                ax.scatter(*pos[-1], color=colors[i], s=50, zorder=5)
            else:
                ax.plot(pos[:, 0], pos[:, 1],
                        color=colors[i], label=body.name, alpha=0.7)
                ax.scatter(pos[-1, 0], pos[-1, 1],
                           color=colors[i], s=50, zorder=5)
    else:
        # Plot from orbital elements if no simulation
        for i, body in enumerate(system.bodies):
            if body.orbital_elements is not None:
                oe = body.orbital_elements
                theta = np.linspace(0, 2 * np.pi, 200)
                r = oe.semi_major_axis * (1 - oe.eccentricity**2) / (
                    1 + oe.eccentricity * np.cos(theta)
                )
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                if projection == "3d":
                    z = np.zeros_like(x)
                    ax.plot(x, y, z, color=colors[i], label=body.name, alpha=0.7)
                else:
                    ax.plot(x, y, color=colors[i], label=body.name, alpha=0.7)

    # Plot candidates
    if candidates:
        for j, cand in enumerate(candidates):
            a = cand.semi_major_axis[0]
            e = cand.eccentricity[0]
            theta = np.linspace(0, 2 * np.pi, 200)
            r = a * (1 - e**2) / (1 + e * np.cos(theta))
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            if projection == "3d":
                ax.plot(x, y, np.zeros_like(x), "--r", linewidth=2,
                        label=f"Candidate {j+1}", alpha=0.8)
            else:
                ax.plot(x, y, "--r", linewidth=2,
                        label=f"Candidate {j+1}", alpha=0.8)

    # Sun at origin
    if projection == "3d":
        ax.scatter(0, 0, 0, color="gold", s=200, marker="*", zorder=10)
    else:
        ax.scatter(0, 0, color="gold", s=200, marker="*", zorder=10)

    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    if projection == "3d":
        ax.set_zlabel("z (AU)")

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or f"{system.name} - Orbital Plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_residuals(
    residuals: Residuals,
    periodogram=None,
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """Plot residual time series and optionally periodogram.

    Args:
        residuals: Residual data
        periodogram: Optional PeriodogramResult
        figsize: Figure size
    """
    n_plots = 2 if periodogram else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Residual time series
    times_years = residuals.times / 365.25
    axes[0].plot(times_years, residuals.values * 1000, "b-", alpha=0.7)  # Convert to mAU
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Residual (mAU)")
    axes[0].set_title(f"Position Residuals - {residuals.body_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="k", linestyle="--", alpha=0.3)

    # Periodogram
    if periodogram:
        periods_years = 1.0 / (periodogram.frequencies * 365.25)
        axes[1].plot(periods_years, periodogram.power, "r-", alpha=0.7)
        axes[1].axvline(periodogram.peak_period_years, color="k", linestyle="--",
                        label=f"Peak: {periodogram.peak_period_years:.1f} yr")
        axes[1].set_xlabel("Period (years)")
        axes[1].set_ylabel("Power")
        axes[1].set_title("Lomb-Scargle Periodogram")
        axes[1].set_xscale("log")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
