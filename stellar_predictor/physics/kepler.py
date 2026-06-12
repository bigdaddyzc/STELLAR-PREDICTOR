"""Kepler equation solver and orbital element conversions."""

from __future__ import annotations

import numpy as np


def kepler_solve(mean_anomaly: float, eccentricity: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.

    Uses Newton-Raphson iteration.
    """
    M = mean_anomaly % (2 * np.pi)
    e = eccentricity

    # Initial guess
    E = M + e * np.sin(M) if e < 0.8 else np.pi

    for _ in range(100):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break

    return E


def eccentric_to_true_anomaly(E: float, eccentricity: float) -> float:
    """Convert eccentric anomaly to true anomaly."""
    e = eccentricity
    return 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2),
    )


def orbital_elements_to_cartesian(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    longitude_ascending: float,
    argument_perihelion: float,
    true_anomaly: float,
    mu: float = 4 * np.pi**2,  # G*(M_sun) in AU^3/yr^2
) -> tuple[np.ndarray, np.ndarray]:
    """Convert orbital elements to cartesian position and velocity.

    Args:
        semi_major_axis: in AU
        eccentricity: dimensionless
        inclination: radians
        longitude_ascending: radians (Ω)
        argument_perihelion: radians (ω)
        true_anomaly: radians (ν)
        mu: gravitational parameter (default: solar, AU^3/yr^2)

    Returns:
        (position [AU], velocity [AU/yr])
    """
    a, e, i = semi_major_axis, eccentricity, inclination
    Omega, omega, nu = longitude_ascending, argument_perihelion, true_anomaly

    # Distance from focus
    r = a * (1 - e**2) / (1 + e * np.cos(nu))

    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Velocity in orbital plane
    p = a * (1 - e**2)
    h = np.sqrt(mu * p)
    vx_orb = -mu / h * np.sin(nu)
    vy_orb = mu / h * (e + np.cos(nu))

    # Rotation matrix from orbital plane to reference frame
    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_w, sin_w = np.cos(omega), np.sin(omega)
    cos_i, sin_i = np.cos(i), np.sin(i)

    Px = cos_O * cos_w - sin_O * sin_w * cos_i
    Py = sin_O * cos_w + cos_O * sin_w * cos_i
    Pz = sin_w * sin_i

    Qx = -cos_O * sin_w - sin_O * cos_w * cos_i
    Qy = -sin_O * sin_w + cos_O * cos_w * cos_i
    Qz = cos_w * sin_i

    position = np.array([
        Px * x_orb + Qx * y_orb,
        Py * x_orb + Qy * y_orb,
        Pz * x_orb + Qz * y_orb,
    ])

    velocity = np.array([
        Px * vx_orb + Qx * vy_orb,
        Py * vx_orb + Qy * vy_orb,
        Pz * vx_orb + Qz * vy_orb,
    ])

    return position, velocity


def cartesian_to_orbital_elements(
    position: np.ndarray,
    velocity: np.ndarray,
    mu: float = 4 * np.pi**2,
) -> dict:
    """Convert cartesian state vector to orbital elements.

    Returns dict with keys: a, e, i, Omega, omega, nu (true anomaly)
    """
    r_vec = position
    v_vec = velocity
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Node vector
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    energy = v**2 / 2 - mu / r
    a = -mu / (2 * energy) if abs(energy) > 1e-15 else np.inf

    # Inclination
    i = np.arccos(np.clip(h_vec[2] / h, -1, 1))

    # Longitude of ascending node
    if n > 1e-15:
        Omega = np.arccos(np.clip(n_vec[0] / n, -1, 1))
        if n_vec[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0.0

    # Argument of perihelion
    if n > 1e-15 and e > 1e-15:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0

    # True anomaly
    if e > 1e-15:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0

    return {"a": a, "e": e, "i": i, "Omega": Omega, "omega": omega, "nu": nu}
