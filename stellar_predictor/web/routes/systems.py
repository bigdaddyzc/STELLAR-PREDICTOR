"""System information endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from stellar_predictor.data.known_systems import SOLAR_SYSTEM_PLANETS
from stellar_predictor.web.schemas import PlanetInfo

router = APIRouter()

# Known multi-planet exoplanet systems (hardcoded from NASA Exoplanet Archive)
# Format: (name, mass_earth, a_au, period_days, eccentricity)
EXOPLANET_SYSTEMS: dict[str, dict] = {
    "trappist1": {
        "name": "TRAPPIST-1",
        "display_name": "TRAPPIST-1",
        "star_mass": 0.089,       # M_sun
        "star_radius": 0.119,     # R_sun
        "star_teff": 2566,        # K
        "planet_count": 7,
        "source": "nasa_exoplanet_archive",
        "has_nbody_support": False,
        "planets": [
            ("TRAPPIST-1 b", 1.374, 0.01154, 1.5109, 0.006),
            ("TRAPPIST-1 c", 1.308, 0.01580, 2.4218, 0.006),
            ("TRAPPIST-1 d", 0.388, 0.02227, 4.0498, 0.008),
            ("TRAPPIST-1 e", 0.692, 0.02925, 6.0990, 0.005),
            ("TRAPPIST-1 f", 1.039, 0.03849, 9.2057, 0.010),
            ("TRAPPIST-1 g", 1.321, 0.04683, 12.3529, 0.002),
            ("TRAPPIST-1 h", 0.326, 0.06189, 18.7676, 0.086),
        ],
    },
    "kepler11": {
        "name": "Kepler-11",
        "display_name": "Kepler-11",
        "star_mass": 0.96,
        "star_radius": 1.06,
        "star_teff": 5663,
        "planet_count": 6,
        "source": "nasa_exoplanet_archive",
        "has_nbody_support": False,
        "planets": [
            ("Kepler-11 b", 1.9, 0.091, 10.304, 0.045),
            ("Kepler-11 c", 2.9, 0.107, 13.024, 0.026),
            ("Kepler-11 d", 7.3, 0.155, 22.684, 0.004),
            ("Kepler-11 e", 8.0, 0.195, 31.999, 0.012),
            ("Kepler-11 f", 2.0, 0.250, 46.688, 0.013),
            ("Kepler-11 g", 0.95, 0.466, 118.380, 0.15),
        ],
    },
    "kepler33": {
        "name": "Kepler-33",
        "display_name": "Kepler-33",
        "star_mass": 1.26,
        "star_radius": 1.58,
        "star_teff": 5904,
        "planet_count": 5,
        "source": "nasa_exoplanet_archive",
        "has_nbody_support": False,
        "planets": [
            ("Kepler-33 b", 0.16, 0.0677, 5.668, 0.0),
            ("Kepler-33 c", 0.29, 0.1189, 13.176, 0.0),
            ("Kepler-33 d", 0.48, 0.1662, 21.776, 0.0),
            ("Kepler-33 e", 0.36, 0.2138, 31.784, 0.0),
            ("Kepler-33 f", 0.40, 0.2535, 41.029, 0.0),
        ],
    },
    "hd219134": {
        "name": "HD 219134",
        "display_name": "HD 219134",
        "star_mass": 0.81,
        "star_radius": 0.778,
        "star_teff": 4699,
        "planet_count": 4,
        "source": "nasa_exoplanet_archive",
        "has_nbody_support": False,
        "planets": [
            ("HD 219134 b", 4.74, 0.0388, 3.093, 0.0),
            ("HD 219134 c", 4.36, 0.0653, 6.765, 0.062),
            ("HD 219134 d", 16.17, 0.237, 46.859, 0.138),
            ("HD 219134 e", 70.90, 2.563, 1842.0, 0.34),
        ],
    },
}


@router.get("/systems")
def list_systems():
    systems = []

    # Solar System
    systems.append({
        "name": "solar_system",
        "display_name": "Solar System",
        "planet_count": 8,
        "source": "jpl_horizons",
        "has_nbody_support": True,
    })

    # Exoplanet systems
    for key, info in EXOPLANET_SYSTEMS.items():
        systems.append({
            "name": key,
            "display_name": info["display_name"],
            "planet_count": info["planet_count"],
            "source": info["source"],
            "has_nbody_support": info["has_nbody_support"],
        })

    return systems


@router.get("/systems/{name}/planets")
def get_system_planets(name: str):
    # Solar System
    if name == "solar_system":
        planets = []
        for pname, mass, a, e, _i, _Om, _om, _M in SOLAR_SYSTEM_PLANETS:
            planets.append(PlanetInfo(
                name=pname,
                semi_major_axis_au=a,
                period_years=a ** 1.5,
                mass_earth=round(mass * 332946, 2),
                eccentricity=e,
                is_star=False,
            ).model_dump())

        return {
            "name": "Solar System",
            "star": {"name": "Sun", "mass_solar": 1.0},
            "planets": planets,
        }

    # Exoplanet system
    if name in EXOPLANET_SYSTEMS:
        info = EXOPLANET_SYSTEMS[name]
        planets = []
        for pname, mass, a, period_days, ecc in info["planets"]:
            planets.append(PlanetInfo(
                name=pname,
                semi_major_axis_au=round(a, 6),
                period_years=round(period_days / 365.25, 4),
                mass_earth=round(mass, 2),
                eccentricity=ecc,
                is_star=False,
            ).model_dump())

        return {
            "name": info["display_name"],
            "star": {
                "name": info["display_name"],
                "mass_solar": info["star_mass"],
            },
            "planets": planets,
        }

    return {"error": f"Unknown system: {name}"}
