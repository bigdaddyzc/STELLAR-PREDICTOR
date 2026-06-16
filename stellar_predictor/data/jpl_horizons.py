"""JPL Horizons data fetcher for solar system bodies."""

from __future__ import annotations

import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from stellar_predictor.data.models import CelestialBody, OrbitalElements, StellarSystem

# Mass in solar masses (from JPL)
SOLAR_SYSTEM_MASSES = {
    "Sun": 1.0,
    "Mercury": 1.66012e-7,
    "Venus": 2.44783e-6,
    "Earth": 3.00273e-6,
    "Mars": 3.22715e-7,
    "Jupiter": 9.54786e-4,
    "Saturn": 2.85837e-4,
    "Uranus": 4.36624e-5,
    "Neptune": 5.15138e-5,
    "Pluto": 7.40e-9,
}

# JPL Horizons IDs
HORIZONS_IDS = {
    "Sun": "10",
    "Mercury": "199",
    "Venus": "299",
    "Earth": "399",
    "Mars": "499",
    "Jupiter": "599",
    "Saturn": "699",
    "Uranus": "799",
    "Neptune": "899",
    "Pluto": "999",
}


def fetch_solar_system(
    epoch: str = "2451545.0",  # J2000.0
    bodies: list[str] | None = None,
    exclude: list[str] | None = None,
) -> StellarSystem:
    """Fetch solar system state from JPL Horizons.

    Args:
        epoch: Julian date for the epoch (default J2000.0)
        bodies: List of body names to include (default: all planets + Sun)
        exclude: List of body names to exclude

    Returns:
        StellarSystem with positions and velocities at epoch
    """
    if bodies is None:
        bodies = list(HORIZONS_IDS.keys())

    if exclude:
        bodies = [b for b in bodies if b not in exclude]

    system = StellarSystem(name="Solar System", source="JPL Horizons")
    epoch_time = Time(float(epoch), format="jd")
    system.epoch = epoch_time

    for name in bodies:
        if name == "Sun":
            body = CelestialBody(
                name="Sun",
                mass=1.0,
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
            )
        else:
            body = _fetch_body_state(name, epoch)

        system.add_body(body)

    return system


def _fetch_body_state(name: str, epoch: str) -> CelestialBody:
    """Fetch a single body's state vector from JPL Horizons."""
    obj_id = HORIZONS_IDS[name]

    # Query vectors (position and velocity) relative to Sun
    obj = Horizons(id=obj_id, location="@sun", epochs=float(epoch))
    vectors = obj.vectors()

    position = np.array([
        float(vectors["x"][0]),
        float(vectors["y"][0]),
        float(vectors["z"][0]),
    ])  # AU

    # Horizons gives velocity in AU/day
    velocity = np.array([
        float(vectors["vx"][0]),
        float(vectors["vy"][0]),
        float(vectors["vz"][0]),
    ])  # AU/day

    # Also fetch orbital elements
    elements = obj.elements()
    orbital_elements = OrbitalElements(
        semi_major_axis=float(elements["a"][0]),
        eccentricity=float(elements["e"][0]),
        inclination=float(np.radians(float(elements["incl"][0]))),
        longitude_ascending=float(np.radians(float(elements["Omega"][0]))),
        argument_perihelion=float(np.radians(float(elements["w"][0]))),
        mean_anomaly=float(np.radians(float(elements["M"][0]))),
    )

    return CelestialBody(
        name=name,
        mass=SOLAR_SYSTEM_MASSES.get(name, 0.0),
        position=position,
        velocity=velocity,
        orbital_elements=orbital_elements,
        metadata={"source": "JPL Horizons", "epoch_jd": epoch},
    )


def fetch_observed_positions(
    body_name: str,
    start_date: str,
    end_date: str,
    step: str = "30d",
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch historical positions of a body from JPL Horizons.

    Args:
        body_name: Name of the body
        start_date: Start date (ISO format or JD)
        end_date: End date
        step: Time step (e.g., '30d', '1y')

    Returns:
        (times_jd, positions) - times as JD array, positions as (N, 3) AU
    """
    obj_id = HORIZONS_IDS[body_name]
    obj = Horizons(
        id=obj_id,
        location="@sun",
        epochs={"start": start_date, "stop": end_date, "step": step},
    )
    vectors = obj.vectors()

    times = np.array(vectors["datetime_jd"], dtype=float)
    positions = np.column_stack([
        np.array(vectors["x"], dtype=float),
        np.array(vectors["y"], dtype=float),
        np.array(vectors["z"], dtype=float),
    ])

    return times, positions
