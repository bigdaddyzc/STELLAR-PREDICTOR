"""Derived physical properties for predicted bodies.

Pure analytical formulas — no simulation required. Used to populate the
bilingual prediction report with estimated mass, radius, density,
equilibrium temperature, surface gravity, and Hill sphere radius.
"""

from __future__ import annotations

import numpy as np


def mass_radius_relation(mass_earth: float) -> float:
    """Estimate planetary radius from mass using piecewise M-R relations.

    Returns radius in Earth units (R_earth). Relations from:
    - Zeng et al. (2016) for rocky planets (M < 2 M_earth)
    - Bashi et al. (2017) for Neptune-like transition (2-130 M_earth)
    - Degenerate regime for gas giants (>130 M_earth, ~Jupiter radius)
    """
    if mass_earth < 2.0:
        return mass_earth ** 0.27
    elif mass_earth < 130.0:
        r_at_2 = 2.0 ** 0.27
        return r_at_2 * (mass_earth / 2.0) ** 0.55
    else:
        r_at_2 = 2.0 ** 0.27
        r_at_130 = r_at_2 * (130.0 / 2.0) ** 0.55
        return r_at_130 * (mass_earth / 130.0) ** (-0.04)


def radius_jupiter(mass_earth: float) -> float:
    """Radius in Jupiter units (R_jupiter = 11.21 R_earth)."""
    return mass_radius_relation(mass_earth) / 11.21


def density_gcm3(mass_earth: float, radius_earth: float) -> float:
    """Bulk density in g/cm^3. Earth = 5.51 g/cm^3."""
    return 5.51 * mass_earth / (radius_earth ** 3)


def equilibrium_temperature(
    a_au: float,
    stellar_teff: float,
    stellar_radius_rsun: float = 1.0,
    albedo: float = 0.3,
) -> float:
    """Equilibrium temperature in Kelvin.

    T_eq = T_star * sqrt(R_star / (2 * a)) * (1 - albedo)^(1/4)
    Assumes zero redistribution (dayside-only). For full redistribution,
    replace the factor sqrt(1/2) with (1/4)^(1/4), which yields ~20% lower T.
    """
    r_star_au = stellar_radius_rsun * 0.00465047  # R_sun -> AU
    return stellar_teff * np.sqrt(r_star_au / (2.0 * a_au)) * (1.0 - albedo) ** 0.25


def surface_gravity_ms2(mass_earth: float, radius_earth: float) -> float:
    """Surface gravity in m/s^2. Earth = 9.81 m/s^2."""
    G = 6.6743e-11
    mass_kg = mass_earth * 5.972e24
    radius_m = radius_earth * 6.371e6
    return G * mass_kg / (radius_m ** 2)


def hill_sphere_au(a_au: float, mass_earth: float, stellar_mass: float) -> float:
    """Hill sphere radius in AU. R_H = a * (m / (3M))^(1/3)."""
    mass_sun = mass_earth / 332946.0
    return a_au * (mass_sun / (3.0 * stellar_mass)) ** (1.0 / 3.0)


def classify_planet(mass_earth: float) -> tuple[str, str]:
    """Return (type_zh, type_en) based on mass.

    Thresholds are approximate — informed by Kepler/TESS demographics.
    """
    if mass_earth < 2.0:
        return ("岩石行星", "Rocky")
    elif mass_earth < 10.0:
        return ("超级地球", "Super-Earth")
    elif mass_earth < 50.0:
        return ("海王星类", "Neptune-like")
    elif mass_earth < 130.0:
        return ("亚土星", "Sub-Saturn")
    elif mass_earth < 500.0:
        return ("气态巨行星", "Gas Giant")
    else:
        return ("超级木星", "Super-Jupiter")


def estimate_age_gyr(stellar_teff: float) -> tuple[float, float]:
    """Rough stellar age estimate from effective temperature.

    Returns (min_gyr, max_gyr). Very approximate — real ages require
    rotation or lithium measurements. Based on gyrochronology priors.
    """
    if stellar_teff > 6500:
        return (0.5, 3.0)
    elif stellar_teff > 6000:
        return (1.0, 4.0)
    elif stellar_teff > 5200:
        return (3.0, 7.0)  # Sun-like
    elif stellar_teff > 4000:
        return (3.0, 10.0)
    elif stellar_teff > 3000:
        return (1.0, 10.0)
    else:  # M dwarf — could be very young or very old
        return (0.5, 12.0)


def planet_from_mass(
    mass_earth_low: float,
    mass_earth_high: float,
    predicted_a: float,
    stellar_mass: float,
    stellar_teff: float = 5778.0,
    stellar_radius_rsun: float = 1.0,
) -> dict:
    """Compute complete set of derived parameters for a predicted body.

    Args:
        mass_earth_low, mass_earth_high: estimated mass range in Earth masses
        predicted_a: predicted semi-major axis (AU)
        stellar_mass: host star mass (M_sun)
        stellar_teff: host star effective temperature (K)
        stellar_radius_rsun: host star radius (R_sun)

    Returns:
        Dict with keys for each parameter (label_zh, label_en, value, unit).
    """
    mass_mid = np.sqrt(mass_earth_low * mass_earth_high)
    mass_points = sorted({mass_earth_low, mass_mid, mass_earth_high})

    # Self-consistent (mass, radius) pairs at 3 evaluation points
    r_points = [mass_radius_relation(m) for m in mass_points]
    rho_points = [density_gcm3(m, r) for m, r in zip(mass_points, r_points)]
    g_points = [surface_gravity_ms2(m, r) for m, r in zip(mass_points, r_points)]

    r_low, r_high = min(r_points), max(r_points)
    rho_low, rho_high = min(rho_points), max(rho_points)
    g_low, g_high = min(g_points), max(g_points)

    # Jupiter-radius units
    r_jup_low = r_low / 11.21
    r_jup_high = r_high / 11.21

    # Equilibrium temperature
    t_low = equilibrium_temperature(predicted_a, stellar_teff, stellar_radius_rsun, albedo=0.5)
    t_high = equilibrium_temperature(predicted_a, stellar_teff, stellar_radius_rsun, albedo=0.1)

    # Hill sphere at mid-mass
    hill_au = hill_sphere_au(predicted_a, mass_mid, stellar_mass)
    hill_km = hill_au * 149.6e6

    # Classification at mid-mass
    type_zh, type_en = classify_planet(mass_mid)

    # Age
    age_min, age_max = estimate_age_gyr(stellar_teff)

    period_years = predicted_a ** 1.5 / np.sqrt(stellar_mass)

    def _fmt_val(lo: float, hi: float, decimals: int) -> str:
        """Format a lo–hi range, handling near-zero values gracefully."""
        if lo == hi:
            return f"{lo:.{decimals}f}"
        return f"{lo:.{decimals}f} – {hi:.{decimals}f}"

    return {
        "semi_major_axis": {
            "label_zh": "半长轴",
            "label_en": "Semi-major axis",
            "value": f"{predicted_a:.4f}",
            "unit": "AU",
        },
        "orbital_period": {
            "label_zh": "轨道周期",
            "label_en": "Orbital period",
            "value": f"{period_years:.2f}",
            "unit": f"yr ({period_years * 365.25:.0f} days)",
        },
        "mass": {
            "label_zh": "质量",
            "label_en": "Mass",
            "value": _fmt_val(mass_earth_low, mass_earth_high, 1),
            "unit": "M_earth",
        },
        "radius": {
            "label_zh": "半径",
            "label_en": "Radius",
            "value": _fmt_val(r_low, r_high, 1),
            "unit": f"R_earth ({_fmt_val(r_jup_low, r_jup_high, 2)} R_jup)",
        },
        "density": {
            "label_zh": "密度",
            "label_en": "Density",
            "value": _fmt_val(rho_low, rho_high, 1),
            "unit": "g/cm³",
        },
        "eq_temperature": {
            "label_zh": "平衡温度",
            "label_en": "Equilibrium temp.",
            "value": _fmt_val(t_low, t_high, 0),
            "unit": "K",
        },
        "surface_gravity": {
            "label_zh": "表面重力",
            "label_en": "Surface gravity",
            "value": _fmt_val(g_low, g_high, 1),
            "unit": "m/s² (Earth = 9.81)",
        },
        "hill_sphere": {
            "label_zh": "Hill球半径",
            "label_en": "Hill sphere",
            "value": f"{hill_au:.6f}",
            "unit": f"AU ({hill_km:.0f} km)",
        },
        "planet_type": {
            "label_zh": "行星类型",
            "label_en": "Planet type",
            "value": f"{type_zh} / {type_en}",
            "unit": "",
        },
        "system_age": {
            "label_zh": "系统年龄 (估计)",
            "label_en": "System age (est.)",
            "value": f"{age_min:.1f} – {age_max:.0f}",
            "unit": "Gyr",
        },
    }
