"""Unified data fetcher interface."""

from __future__ import annotations

import warnings

from stellar_predictor.data.models import StellarSystem

# Errors that mean "the network source is unavailable", as opposed to a bad
# request. requests' exceptions subclass OSError, as do socket timeouts.
_NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError, ConnectionError, TimeoutError)


def _normalize(system_name: str) -> str:
    return "".join(c for c in system_name.lower() if c not in " _-")


class DataFetcher:
    """Unified interface for fetching stellar system data."""

    def __init__(self, offline_fallback: bool = True):
        """
        Args:
            offline_fallback: When JPL Horizons is unreachable (or astroquery is
                not installed), fall back to the built-in J2000 orbital
                elements instead of raising. A ``RuntimeWarning`` is emitted
                and the returned system's ``source`` says it is a fallback.
        """
        self.offline_fallback = offline_fallback

    def fetch_system(self, system_name: str, **kwargs):
        """Fetch a system by name.

        Supported systems:
        - "solar_system" / "Solar System" / "solar": live from JPL Horizons,
          with an offline fallback to built-in J2000 elements.
        - "trappist1", "kepler11", "kepler33", "hd219134" (dashes/spaces
          ignored): built-in NASA Exoplanet Archive values, no network needed.
        """
        from stellar_predictor.data.known_systems import EXOPLANET_DATA, build_exoplanet

        normalized = _normalize(system_name)

        if normalized in ("solarsystem", "solar"):
            return self._fetch_solar_system(**kwargs)
        if normalized in EXOPLANET_DATA:
            return build_exoplanet(normalized)
        supported = ["solar_system", *EXOPLANET_DATA]
        raise ValueError(
            f"Unknown system: {system_name}. Supported: {', '.join(supported)}"
        )

    def _fetch_solar_system(self, **kwargs) -> StellarSystem:
        try:
            from stellar_predictor.data.jpl_horizons import fetch_solar_system
            return fetch_solar_system(**kwargs)
        except (ImportError, *_NETWORK_ERRORS) as exc:
            if not self.offline_fallback:
                raise
            warnings.warn(
                f"JPL Horizons unavailable ({type(exc).__name__}: {exc}); "
                "using built-in J2000 orbital elements instead.",
                RuntimeWarning,
                stacklevel=3,
            )
            return _offline_solar_system(
                bodies=kwargs.get("bodies"), exclude=kwargs.get("exclude")
            )

    def fetch_observed_positions(
        self,
        body_name: str,
        start_date: str,
        end_date: str,
        step: str = "30d",
    ):
        """Fetch historical observed positions of a body (network required)."""
        from stellar_predictor.data.jpl_horizons import fetch_observed_positions
        return fetch_observed_positions(body_name, start_date, end_date, step)


def _offline_solar_system(bodies: list[str] | None = None,
                          exclude: list[str] | None = None) -> StellarSystem:
    """Built-in J2000 Solar System, filtered like ``fetch_solar_system``."""
    from stellar_predictor.data.known_systems import build_solar_system

    system = build_solar_system()
    keep = {b.lower() for b in bodies} if bodies is not None else None
    drop = {b.lower() for b in exclude} if exclude else set()
    system.bodies = [
        b for b in system.bodies
        if (keep is None or b.name.lower() in keep) and b.name.lower() not in drop
    ]
    system.source = "J2000 orbital elements (offline fallback)"
    return system
