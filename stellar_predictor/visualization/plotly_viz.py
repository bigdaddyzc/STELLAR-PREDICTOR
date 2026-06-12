"""Interactive Plotly-based visualizations for the web interface."""

from __future__ import annotations

from typing import Optional

import numpy as np

from stellar_predictor.data.models import GapResult, Residuals, SimulationResult, StellarSystem
from stellar_predictor.inference.candidate import CandidateBody
from stellar_predictor.physics.residuals import PeriodogramResult

def _alpha_hex(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)' for semi-transparent overlays."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _lighten_hex(hex_color: str, amount: float) -> str:
    """Lighten a hex color by mixing with white. Returns '#rrggbb'."""
    h = hex_color.lstrip("#")
    r = int(int(h[0:2], 16) + (255 - int(h[0:2], 16)) * amount)
    g = int(int(h[2:4], 16) + (255 - int(h[2:4], 16)) * amount)
    b = int(int(h[4:6], 16) + (255 - int(h[4:6], 16)) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


PLANET_COLORS = {
    "Mercury": "#8c8c8c",
    "Venus": "#e6c35c",
    "Earth": "#4a90d9",
    "Mars": "#c1440e",
    "Jupiter": "#c88b3a",
    "Saturn": "#e8d080",
    "Uranus": "#7ecbc4",
    "Neptune": "#4b70dd",
    "Sun": "#ffd700",
}


def orbit_plot_3d(
    simulation: SimulationResult,
    system: StellarSystem,
    candidates: Optional[list[CandidateBody]] = None,
    actual_body_sim: Optional[dict] = None,
    predicted_gaps: Optional[list[GapResult]] = None,
) -> dict:
    """Generate Plotly JSON for 3D orbit visualization.

    Args:
        simulation: Simulation result with position time series
        system: The stellar system (for body metadata)
        candidates: Predicted candidate bodies
        actual_body_sim: Optional dict with 'name' and 'positions' (N,3) for ground truth
        predicted_gaps: Optional list of predicted orbital gaps to show as dashed rings

    Returns:
        Plotly figure dict ready for Plotly.newPlot()
    """
    traces = []

    for body in system.bodies:
        pos = simulation.positions.get(body.name)
        if pos is None:
            continue

        if body.name == "Sun":
            traces.append({
                "type": "scatter3d",
                "mode": "markers",
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
                "name": "Sun",
                "marker": {"size": 12, "color": "#ffd700", "symbol": "diamond"},
            })
            continue

        color = PLANET_COLORS.get(body.name, "#888888")
        traces.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
            "z": pos[:, 2].tolist(),
            "name": body.name,
            "line": {"color": color, "width": 3},
            "opacity": 0.8,
        })
        traces.append({
            "type": "scatter3d",
            "mode": "markers",
            "x": [float(pos[-1, 0])],
            "y": [float(pos[-1, 1])],
            "z": [float(pos[-1, 2])],
            "name": f"{body.name} (current)",
            "marker": {"size": 6, "color": color},
            "showlegend": False,
        })

    if candidates:
        for i, cand in enumerate(candidates):
            a = cand.semi_major_axis[0]
            e = cand.eccentricity[0]
            inc = cand.inclination[0]
            omega = cand.argument_perihelion[0] if cand.argument_perihelion else 0
            Omega = cand.longitude_ascending[0] if cand.longitude_ascending else 0

            theta = np.linspace(0, 2 * np.pi, 300)
            r = a * (1 - e**2) / (1 + e * np.cos(theta))

            x_orb = r * np.cos(theta)
            y_orb = r * np.sin(theta)
            z_orb = np.zeros_like(x_orb)

            cos_O, sin_O = np.cos(Omega), np.sin(Omega)
            cos_i, sin_i = np.cos(inc), np.sin(inc)
            cos_w, sin_w = np.cos(omega), np.sin(omega)

            x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb + \
                (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb
            y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb + \
                (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb
            z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

            traces.append({
                "type": "scatter3d",
                "mode": "lines",
                "x": x.tolist(),
                "y": y.tolist(),
                "z": z.tolist(),
                "name": f"Predicted Body {i+1}",
                "line": {"color": "#ff4444", "width": 4, "dash": "dash"},
            })

    if predicted_gaps:
        gap_colors = ["#ffaa00", "#ff8800", "#ff6600", "#ff4400", "#ff2200"]
        for i, gap in enumerate(predicted_gaps):
            theta = np.linspace(0, 2 * np.pi, 200)
            r = gap.predicted_a
            x_ring = r * np.cos(theta)
            y_ring = r * np.sin(theta)
            z_ring = np.zeros_like(x_ring)
            color = gap_colors[min(i, len(gap_colors) - 1)]
            traces.append({
                "type": "scatter3d",
                "mode": "lines",
                "x": x_ring.tolist(),
                "y": y_ring.tolist(),
                "z": z_ring.tolist(),
                "name": f"Gap: {gap.predicted_a:.1f} AU (score={gap.combined_score:.2f})",
                "line": {"color": color, "width": 3, "dash": "dash"},
                "opacity": 0.7,
            })

    if actual_body_sim:
        pos = actual_body_sim["positions"]
        traces.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
            "z": pos[:, 2].tolist(),
            "name": f"{actual_body_sim['name']} (actual)",
            "line": {"color": "#00cc66", "width": 2},
            "opacity": 0.6,
        })

    layout = {
        "scene": {
            "xaxis": {"title": "x (AU)"},
            "yaxis": {"title": "y (AU)"},
            "zaxis": {"title": "z (AU)"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 0.8, "y": 0.8, "z": 0.6}},
        },
        "title": "Solar System - Orbit Visualization",
        "showlegend": True,
        "legend": {"x": 0.02, "y": 0.98},
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
    }

    return {"data": traces, "layout": layout}


def residual_plot(
    residuals: Residuals,
    periodogram: Optional[PeriodogramResult] = None,
) -> dict:
    """Generate Plotly JSON for residual and periodogram subplots.

    Args:
        residuals: Residual data with components
        periodogram: Optional periodogram result

    Returns:
        Plotly figure dict
    """
    times_years = (residuals.times / 365.25).tolist()

    traces = []

    if residuals.components is not None:
        labels = ["x", "y", "z"]
        colors = ["#4a90d9", "#e6553a", "#50c878"]
        for i, (label, color) in enumerate(zip(labels, colors)):
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": times_years,
                "y": (residuals.components[:, i] * 1000).tolist(),
                "name": f"Residual {label}",
                "line": {"color": color, "width": 1.5},
                "opacity": 0.7,
                "xaxis": "x",
                "yaxis": "y",
            })

    traces.append({
        "type": "scatter",
        "mode": "lines",
        "x": times_years,
        "y": (residuals.values * 1000).tolist(),
        "name": "Magnitude",
        "line": {"color": "#333333", "width": 2},
        "xaxis": "x",
        "yaxis": "y",
    })

    if periodogram:
        periods_years = (1.0 / (periodogram.frequencies * 365.25)).tolist()
        power = periodogram.power.tolist()

        traces.append({
            "type": "scatter",
            "mode": "lines",
            "x": periods_years,
            "y": power,
            "name": "Periodogram",
            "line": {"color": "#c1440e", "width": 2},
            "xaxis": "x2",
            "yaxis": "y2",
        })

        traces.append({
            "type": "scatter",
            "mode": "lines",
            "x": [periodogram.peak_period_years, periodogram.peak_period_years],
            "y": [0, float(periodogram.peak_power)],
            "name": f"Peak: {periodogram.peak_period_years:.1f} yr",
            "line": {"color": "#000000", "width": 2, "dash": "dash"},
            "xaxis": "x2",
            "yaxis": "y2",
        })

    layout = {
        "grid": {"rows": 2, "columns": 1, "pattern": "independent"},
        "xaxis": {"title": "Time (years)", "domain": [0, 1], "anchor": "y"},
        "yaxis": {"title": "Residual (mAU)", "domain": [0.55, 1.0], "anchor": "x"},
        "xaxis2": {"title": "Period (years)", "type": "log", "domain": [0, 1], "anchor": "y2"},
        "yaxis2": {"title": "Power", "domain": [0, 0.42], "anchor": "x2"},
        "title": f"Residual Analysis - {residuals.body_name}",
        "showlegend": True,
        "height": 600,
        "margin": {"t": 50, "b": 50},
    }

    return {"data": traces, "layout": layout}


def comparison_data(
    candidate: CandidateBody,
    actual_params: Optional[dict] = None,
) -> dict:
    """Generate comparison table data.

    Args:
        candidate: Predicted candidate body
        actual_params: Optional dict with keys: mass_solar, semi_major_axis, eccentricity, period, inclination

    Returns:
        Dict with 'rows' list of parameter comparisons
    """
    rows = [
        {
            "parameter": "Mass",
            "unit": "M_Earth",
            "predicted": round(candidate.mass_earth, 2),
            "predicted_range": [round(candidate.mass[1] * 332946, 2), round(candidate.mass[2] * 332946, 2)],
        },
        {
            "parameter": "Semi-major axis",
            "unit": "AU",
            "predicted": round(candidate.semi_major_axis[0], 2),
            "predicted_range": [round(candidate.semi_major_axis[1], 2), round(candidate.semi_major_axis[2], 2)],
        },
        {
            "parameter": "Eccentricity",
            "unit": "",
            "predicted": round(candidate.eccentricity[0], 4),
            "predicted_range": [round(candidate.eccentricity[1], 4), round(candidate.eccentricity[2], 4)],
        },
        {
            "parameter": "Period",
            "unit": "years",
            "predicted": round(candidate.period[0], 1),
            "predicted_range": [round(candidate.period[1], 1), round(candidate.period[2], 1)],
        },
        {
            "parameter": "Inclination",
            "unit": "deg",
            "predicted": round(np.degrees(candidate.inclination[0]), 2),
            "predicted_range": [round(np.degrees(candidate.inclination[1]), 2), round(np.degrees(candidate.inclination[2]), 2)],
        },
    ]

    if actual_params:
        actual_map = {
            "Mass": actual_params.get("mass_earth"),
            "Semi-major axis": actual_params.get("semi_major_axis"),
            "Eccentricity": actual_params.get("eccentricity"),
            "Period": actual_params.get("period"),
            "Inclination": actual_params.get("inclination_deg"),
        }
        for row in rows:
            actual = actual_map.get(row["parameter"])
            if actual is not None:
                row["actual"] = actual
                if actual != 0:
                    row["error_pct"] = round(abs(row["predicted"] - actual) / actual * 100, 1)

    return {
        "confidence": candidate.confidence,
        "method": candidate.method,
        "rows": rows,
    }


def titius_bode_plot(tb_result, system_name: str = "",
                     gaps: Optional[list[GapResult]] = None) -> dict:
    """Generate Plotly JSON for Titius-Bode fit visualization.

    Shows log(a) vs index n with regression line, observed planets as points,
    and predicted gaps as hollow markers.
    """
    index_map = tb_result.index_map
    names = list(index_map.keys())
    indices = [index_map[n] for n in names]

    # Get actual axes from a helper or compute from predicted
    pred_axes = tb_result.predicted_axes
    max_n = max(indices) + 3

    traces = []

    # Regression line
    fit_n = list(range(tb_result.start_index, max_n + 1))
    fit_a = [tb_result.alpha * tb_result.beta**n for n in fit_n]
    traces.append({
        "type": "scatter",
        "mode": "lines",
        "x": fit_n,
        "y": [np.log10(a) for a in fit_a],
        "name": f"Fit: a = {tb_result.alpha:.3f} × {tb_result.beta:.3f}ⁿ",
        "line": {"color": "#58a6ff", "width": 2, "dash": "solid"},
    })

    # Observed planets
    planet_axes = [pred_axes[idx - tb_result.start_index]
                   for idx in indices if idx - tb_result.start_index < len(pred_axes)]
    traces.append({
        "type": "scatter",
        "mode": "markers+text",
        "x": indices,
        "y": [np.log10(a) for a in planet_axes],
        "text": names,
        "textposition": "top center",
        "name": "Known planets",
        "marker": {"size": 12, "color": "#58a6ff", "symbol": "circle"},
        "textfont": {"color": "#e6edf3", "size": 11},
    })

    # Predicted gaps
    if gaps:
        gap_indices = []
        gap_log_a = []
        gap_labels = []
        for g in gaps:
            if not g.is_edge if hasattr(g, 'is_edge') else True:
                idx_est = int(round(np.log(g.predicted_a / tb_result.alpha)
                                   / np.log(tb_result.beta)))
                gap_indices.append(idx_est)
                gap_log_a.append(np.log10(g.predicted_a))
                gap_labels.append(f"{g.predicted_a:.2f} AU")
        if gap_indices:
            traces.append({
                "type": "scatter",
                "mode": "markers+text",
                "x": gap_indices,
                "y": gap_log_a,
                "text": gap_labels,
                "textposition": "bottom center",
                "name": "Predicted gaps",
                "marker": {"size": 14, "color": "#d29922", "symbol": "diamond-open",
                           "line": {"width": 2}},
                "textfont": {"color": "#d29922", "size": 10},
            })

    layout = {
        "title": f"Titius-Bode Fit — {system_name}",
        "xaxis": {"title": "Planet Index n", "dtick": 1},
        "yaxis": {"title": "log₁₀(Semi-major axis / AU)"},
        "showlegend": True,
        "legend": {"x": 0.02, "y": 0.98},
        "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
        "annotations": [{
            "text": f"R² = {tb_result.r_squared:.4f}",
            "x": 0.02, "y": 0.95, "xref": "paper", "yref": "paper",
            "showarrow": False,
            "font": {"color": "#e6edf3", "size": 12},
        }],
        "plot_bgcolor": "#161b22",
        "paper_bgcolor": "#161b22",
        "font": {"color": "#e6edf3"},
    }

    return {"data": traces, "layout": layout}


def system_distribution_plot(system,
                              gaps: Optional[list[GapResult]] = None,
                              system_name: str = "",
                              stellar_mass: float = 1.0) -> dict:
    """2D top-down orbital distribution diagram with known and predicted bodies.

    Pure geometry — no N-body simulation required. Shows known planets as
    colored rings and predicted bodies as dashed rings with diamond markers.
    """
    from stellar_predictor.patterns.stability import StabilityAnalyzer

    planet_data = StabilityAnalyzer.extract_planet_data(system)
    if not planet_data:
        return {"error": "No planet data available"}

    axes = [a for _, a, _ in planet_data]
    names = [n for n, _, _ in planet_data]
    masses_earth = [m * 332946 for _, _, m in planet_data]

    palette = [
        "#8c8c8c", "#e6c35c", "#4a90d9", "#c1440e", "#c88b3a",
        "#e8d080", "#7ecbc4", "#4b70dd", "#f07b6c", "#50c878",
        "#af7ac5", "#f39c12", "#85c1e9", "#d35400", "#2ecc71",
    ]
    max_a = max(axes) if axes else 10.0
    if gaps:
        for g in gaps:
            if g.predicted_a > max_a:
                max_a = g.predicted_a
    limit = max_a * 1.2

    traces = []
    theta = np.linspace(0, 2 * np.pi, 300)

    # --- Central star ---
    # Corona (outermost glow, very large, ultra-faint)
    traces.append({
        "type": "scatter", "mode": "markers",
        "x": [0.0], "y": [0.0],
        "marker": {"size": 90, "color": "rgba(255, 215, 0, 0.06)",
                   "symbol": "circle"},
        "showlegend": False, "hoverinfo": "skip",
    })
    # Inner glow
    traces.append({
        "type": "scatter", "mode": "markers",
        "x": [0.0], "y": [0.0],
        "marker": {"size": 55, "color": "rgba(255, 200, 50, 0.18)",
                   "symbol": "circle"},
        "showlegend": False, "hoverinfo": "skip",
    })
    # Star body
    traces.append({
        "type": "scatter", "mode": "markers+text",
        "x": [0.0], "y": [0.0],
        "text": [f"<b>{system_name}</b><br>{stellar_mass:.2f} M<sub>sun</sub>"],
        "textposition": "top center",
        "name": "Host Star / 主星",
        "marker": {"size": 35, "color": "#ffd700",
                   "symbol": "circle",
                   "line": {"width": 5, "color": "#fffbe6"}},
        "textfont": {"color": "#ffd700", "size": 13, "family": "Arial, sans-serif"},
        "showlegend": True,
    })

    # --- Known planet orbits + large sphere markers ---
    for i, (name, a, mass_e) in enumerate(zip(names, axes, masses_earth)):
        color = PLANET_COLORS.get(name, palette[i % len(palette)])
        x_ring = a * np.cos(theta)
        y_ring = a * np.sin(theta)

        # Orbit ring
        traces.append({
            "type": "scatter", "mode": "lines",
            "x": x_ring.tolist(), "y": y_ring.tolist(),
            "name": name,
            "line": {"color": color, "width": 2},
            "opacity": 0.55,
            "hovertext": f"<b>{name}</b><br>a = {a:.3f} AU<br>mass = {mass_e:.1f} M<sub>earth</sub>",
            "hoverinfo": "text",
            "showlegend": True,
        })

        phi = i * 2 * np.pi / max(1, len(names))
        mx, my = a * np.cos(phi), a * np.sin(phi)

        # Glow halo behind sphere
        traces.append({
            "type": "scatter", "mode": "markers",
            "x": [mx], "y": [my],
            "marker": {"size": 44, "color": _alpha_hex(color, 0.14),
                       "symbol": "circle"},
            "showlegend": False, "hoverinfo": "skip",
        })
        # Planet sphere body (large, visible)
        traces.append({
            "type": "scatter", "mode": "markers",
            "x": [mx], "y": [my],
            "name": name,
            "marker": {"size": 26, "color": color,
                       "symbol": "circle",
                       "line": {"width": 3.5, "color": _lighten_hex(color, 0.4)}},
            "hovertext": f"<b>{name}</b><br>a = {a:.3f} AU<br>mass = {mass_e:.1f} M<sub>earth</sub>",
            "hoverinfo": "text",
            "showlegend": False,
        })

    # --- Predicted bodies (prominently highlighted) ---
    if gaps:
        gap_colors = ["#ff4444", "#ff6b2b", "#ff9500", "#ffb700"]
        for i, gap in enumerate(gaps):
            if gap.predicted_a <= 0:
                continue
            color = gap_colors[min(i, len(gap_colors) - 1)]
            x_gap = gap.predicted_a * np.cos(theta)
            y_gap = gap.predicted_a * np.sin(theta)

            score_label = "HIGH" if gap.combined_score >= 0.5 else ("MED" if gap.combined_score >= 0.3 else "LOW")
            label = (f"Pred #{i+1} / 预测#{i+1} [{score_label}] "
                     f"({gap.predicted_a:.2f} AU, score={gap.combined_score:.2f})")

            # Dashed orbit ring
            dash_width = 5 if i == 0 else 3
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": x_gap.tolist(), "y": y_gap.tolist(),
                "name": label,
                "line": {"color": color, "width": dash_width, "dash": "dash"},
                "opacity": 0.8 if i == 0 else 0.55,
                "showlegend": True,
            })

            phi_d = (i + 0.33) * 2 * np.pi / max(1, len(gaps))
            px, py = gap.predicted_a * np.cos(phi_d), gap.predicted_a * np.sin(phi_d)

            # Outer glow
            glow_size = 50 if i == 0 else 36
            traces.append({
                "type": "scatter", "mode": "markers",
                "x": [px], "y": [py],
                "marker": {"size": glow_size, "color": _alpha_hex(color, 0.15),
                           "symbol": "circle"},
                "showlegend": False, "hoverinfo": "skip",
            })
            # Pulse ring
            traces.append({
                "type": "scatter", "mode": "markers",
                "x": [px], "y": [py],
                "marker": {"size": glow_size * 0.5, "color": "rgba(0,0,0,0)",
                           "symbol": "circle-open",
                           "line": {"width": 3, "color": _alpha_hex(color, 0.45)}},
                "showlegend": False, "hoverinfo": "skip",
            })
            # Large diamond marker (prominent)
            diamond_size = 32 if i == 0 else 22
            traces.append({
                "type": "scatter", "mode": "markers",
                "x": [px], "y": [py],
                "name": f"Pred Body #{i+1}",
                "marker": {"size": diamond_size, "color": color,
                           "symbol": "diamond",
                           "line": {"width": 4, "color": "#ffffff"}},
                "hovertext": f"<b>Predicted #{i+1}</b><br>a = {gap.predicted_a:.3f} AU<br>score = {gap.combined_score:.2f}",
                "hoverinfo": "text",
                "showlegend": False,
            })

    # Scale circles
    for r in range(1, int(np.ceil(limit)) + 1):
        x_s = r * np.cos(theta)
        y_s = r * np.sin(theta)
        traces.append({
            "type": "scatter", "mode": "lines",
            "x": x_s.tolist(), "y": y_s.tolist(),
            "line": {"color": "rgba(139,148,158,0.12)", "width": 0.5},
            "showlegend": False, "hoverinfo": "skip",
        })

    layout = {
        "title": f"天体分布 / System Distribution — {system_name}",
        "xaxis": {"title": "x (AU)", "range": [-limit, limit],
                  "scaleanchor": "y", "scaleratio": 1,
                  "gridcolor": "rgba(48,54,61,0.3)", "zerolinecolor": "rgba(48,54,61,0.5)"},
        "yaxis": {"title": "y (AU)", "range": [-limit, limit],
                  "gridcolor": "rgba(48,54,61,0.3)", "zerolinecolor": "rgba(48,54,61,0.5)"},
        "showlegend": True,
        "legend": {"x": 1.02, "y": 1.0, "xanchor": "left", "yanchor": "top",
                   "font": {"size": 9, "color": "#e6edf3"},
                   "bgcolor": "rgba(22,27,34,0.85)"},
        "margin": {"l": 55, "r": 80, "t": 50, "b": 55},
        "plot_bgcolor": "#161b22",
        "paper_bgcolor": "#161b22",
        "font": {"color": "#e6edf3"},
        "height": 700,
    }

    return {"data": traces, "layout": layout}


def spacing_stability_plot(system, stability_regions,
                           gaps: Optional[list[GapResult]] = None,
                           system_name: str = "") -> dict:
    """Generate Plotly JSON for planetary spacing and stability analysis.

    Shows planets as markers, stability regions as green bands, predicted
    gaps as dark bands with markers at predicted semi-major axes.
    """
    from stellar_predictor.patterns.stability import StabilityAnalyzer
    planet_data = StabilityAnalyzer.extract_planet_data(system)

    traces = []

    # Planet markers
    names = [n for n, a, m in planet_data]
    axes = [a for n, a, m in planet_data]
    masses = [m * 332946 for n, a, m in planet_data]
    traces.append({
        "type": "scatter",
        "mode": "markers+text",
        "x": axes,
        "y": [0] * len(axes),
        "text": [f"{n}<br>{a:.2f} AU<br>{m:.1f} M_Earth" for n, a, m in
                 zip(names, axes, masses)],
        "hoverinfo": "text",
        "name": "Known planets",
        "marker": {"size": 14, "color": "#58a6ff", "symbol": "circle"},
    })

    # Stability regions as colored bands
    for i, sr in enumerate(stability_regions):
        if sr.width_au > 0:
            color = "rgba(63,185,80,0.15)" if sr.gap_ratio >= 1.0 else "rgba(210,153,34,0.05)"
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": [sr.inner_boundary_au, sr.outer_boundary_au],
                "y": [0, 0],
                "name": f"Stable ({sr.neighbor_inner}–{sr.neighbor_outer})"
                        if sr.gap_ratio >= 1.0 else f"Tight gap",
                "line": {"width": 20, "color": color},
                "opacity": 0.5,
                "showlegend": sr.gap_ratio >= 1.0,
            })

    # Predicted gap markers
    if gaps:
        for i, g in enumerate(gaps):
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": [g.predicted_a],
                "y": [0],
                "name": f"Pred: {g.predicted_a:.1f} AU (score={g.combined_score:.2f})",
                "marker": {"size": 10, "color": "#d29922", "symbol": "diamond",
                           "line": {"width": 1}},
            })

    layout = {
        "title": f"Orbital Spacing & Stability — {system_name}",
        "xaxis": {"title": "Semi-major axis (AU)", "type": "log"},
        "yaxis": {"visible": False, "range": [-1, 1]},
        "showlegend": True,
        "legend": {"x": 0.02, "y": 0.98},
        "margin": {"l": 40, "r": 20, "t": 50, "b": 50},
        "plot_bgcolor": "#161b22",
        "paper_bgcolor": "#161b22",
        "font": {"color": "#e6edf3"},
    }

    return {"data": traces, "layout": layout}
