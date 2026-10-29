"""DataFetcher: built-in systems and the offline fallback for JPL outages."""

from __future__ import annotations

import sys
import types

import pytest

from stellar_predictor.data.fetcher import DataFetcher
from stellar_predictor.data.models import ExoplanetSystem, StellarSystem


@pytest.fixture
def jpl_down(monkeypatch):
    """Make jpl_horizons.fetch_solar_system fail as if the network were down."""
    fake = types.ModuleType("stellar_predictor.data.jpl_horizons")

    def fetch_solar_system(**kwargs):
        raise ConnectionError("Horizons unreachable")

    fake.fetch_solar_system = fetch_solar_system
    monkeypatch.setitem(sys.modules, "stellar_predictor.data.jpl_horizons", fake)


@pytest.mark.parametrize("name", ["trappist1", "TRAPPIST-1", "Kepler 11", "kepler33", "HD 219134"])
def test_builtin_exoplanet_systems_need_no_network(name):
    system = DataFetcher().fetch_system(name)
    assert isinstance(system, ExoplanetSystem)
    assert system.num_planets >= 4


def test_unknown_system_lists_supported():
    with pytest.raises(ValueError, match="Supported: solar_system"):
        DataFetcher().fetch_system("Tatooine")


def test_offline_fallback_when_jpl_unreachable(jpl_down):
    with pytest.warns(RuntimeWarning, match="JPL Horizons unavailable"):
        system = DataFetcher().fetch_system("Solar System")
    assert isinstance(system, StellarSystem)
    assert "offline fallback" in system.source
    assert len(system.planets) == 8


def test_offline_fallback_honours_exclude(jpl_down):
    with pytest.warns(RuntimeWarning):
        system = DataFetcher().fetch_system("solar_system", exclude=["Neptune"])
    names = [b.name for b in system.bodies]
    assert "Neptune" not in names and "Sun" in names


def test_offline_fallback_honours_bodies(jpl_down):
    with pytest.warns(RuntimeWarning):
        system = DataFetcher().fetch_system("solar", bodies=["Sun", "Earth", "Mars"])
    assert [b.name for b in system.bodies] == ["Sun", "Earth", "Mars"]


def test_fallback_can_be_disabled(jpl_down):
    with pytest.raises(ConnectionError):
        DataFetcher(offline_fallback=False).fetch_system("solar_system")
