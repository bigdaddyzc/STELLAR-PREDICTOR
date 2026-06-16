"""Unified data fetcher interface."""

from __future__ import annotations

from stellar_predictor.data.models import StellarSystem


class DataFetcher:
    """Unified interface for fetching stellar system data."""

    def fetch_system(self, system_name: str, **kwargs) -> StellarSystem:
        """Fetch a system by name.

        Supported systems:
        - "solar_system" or "Solar System": Solar system from JPL Horizons
        """
        normalized = system_name.lower().replace(" ", "_")

        if normalized in ("solar_system", "solar"):
            from stellar_predictor.data.jpl_horizons import fetch_solar_system
            return fetch_solar_system(**kwargs)
        else:
            raise ValueError(
                f"Unknown system: {system_name}. "
                f"Supported: 'solar_system'"
            )

    def fetch_observed_positions(
        self,
        body_name: str,
        start_date: str,
        end_date: str,
        step: str = "30d",
    ):
        """Fetch historical observed positions of a body."""
        from stellar_predictor.data.jpl_horizons import fetch_observed_positions
        return fetch_observed_positions(body_name, start_date, end_date, step)
