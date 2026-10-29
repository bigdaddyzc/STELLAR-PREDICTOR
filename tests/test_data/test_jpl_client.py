"""Offline tests for the JPL Horizons HTTP client.

The network is never touched: ``requests.Session.get`` is replaced by a
scripted fake so retry, caching and failure handling can be asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
import requests

from stellar_predictor.data import jpl_client
from stellar_predictor.data.jpl_client import JPLClient, _advance_date, _to_jpl_date

VECTORS_TEXT = """header
$$SOE
2451545.000000000, A.D. 2000-Jan-01 00:00:00.0000, -1.771E-01, 9.672E-01, -4.0E-06, -1.720E-02, -3.211E-03, 1.0E-07,
2451546.000000000, A.D. 2000-Jan-02 00:00:00.0000, -1.943E-01, 9.638E-01, -4.1E-06, -1.716E-02, -3.546E-03, 1.0E-07,
$$EOE
footer
"""


class FakeResponse:
    def __init__(self, status: int = 200, text: str = VECTORS_TEXT):
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class ScriptedSession:
    """Returns (or raises) the scripted outcomes in order."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_client(tmp_path, outcomes, max_retries=3):
    client = JPLClient(max_retries=max_retries, cache_dir=tmp_path, backoff_s=0)
    client.session = ScriptedSession(outcomes)
    return client


class TestDateHelpers:
    def test_iso_to_jpl(self):
        assert _to_jpl_date("2000-01-05") == "2000-Jan-05"

    def test_jpl_passthrough(self):
        assert _to_jpl_date("2000-Jan-05") == "2000-Jan-05"
        assert _to_jpl_date("2000-13-05") == "2000-13-05"

    def test_advance_date_crosses_month(self):
        assert _advance_date("2000-Jan-31", 1) == "2000-Feb-01"

    def test_advance_unparseable_is_unchanged(self):
        assert _advance_date("garbage", 3) == "garbage"


class TestJPLClient:
    def test_no_cache_dir_created_at_import_or_init(self, tmp_path):
        target = tmp_path / "nested" / "cache"
        JPLClient(cache_dir=target)
        assert not target.exists()

    def test_parses_ephemeris_and_caches(self, tmp_path):
        client = make_client(tmp_path / "c", [FakeResponse()])
        eph = client.get_ephemeris("Earth", "2000-01-01", "2000-01-02", "1d")
        assert eph["positions_au"].shape == (2, 3)
        assert eph["velocities_au_day"].shape == (2, 3)
        np.testing.assert_allclose(eph["times_jd"], [2451545.0, 2451546.0])
        assert len(list((tmp_path / "c").glob("*.txt"))) == 1

        # Second call is served from the cache: the session is not hit again.
        client.session = ScriptedSession([])
        again = client.get_ephemeris("Earth", "2000-01-01", "2000-01-02", "1d")
        np.testing.assert_allclose(again["positions_au"], eph["positions_au"])

    def test_retries_transient_errors_then_succeeds(self, tmp_path):
        client = make_client(tmp_path, [
            requests.ConnectionError("down"),
            FakeResponse(status=503, text="busy"),
            FakeResponse(),
        ])
        eph = client.get_ephemeris("Mars", "2000-01-01", "2000-01-02")
        assert client.session.calls == 3
        assert eph["body"] == "Mars"

    def test_gives_up_after_max_retries(self, tmp_path):
        client = make_client(tmp_path, [requests.Timeout("slow")] * 2, max_retries=2)
        with pytest.raises(ConnectionError, match="after 2 attempts"):
            client.get_ephemeris("Mars", "2000-01-01", "2000-01-02")
        assert client.session.calls == 2

    def test_client_error_is_not_retried(self, tmp_path):
        client = make_client(tmp_path, [FakeResponse(status=400, text="bad")])
        with pytest.raises(requests.HTTPError):
            client.get_ephemeris("Mars", "2000-01-01", "2000-01-02")
        assert client.session.calls == 1

    def test_date_error_is_not_retried(self, tmp_path):
        client = make_client(tmp_path, [FakeResponse(text="Bad dates -- start must be earlier")])
        with pytest.raises(ValueError, match="date error"):
            client.get_ephemeris("Mars", "2000-01-01", "2000-01-02")
        assert client.session.calls == 1

    def test_response_without_data_is_not_cached(self, tmp_path):
        client = make_client(tmp_path, [FakeResponse(text="No ephemeris for target")])
        with pytest.raises(ValueError, match="No ephemeris data"):
            client.get_ephemeris("Mars", "2000-01-01", "2000-01-02")
        assert list(tmp_path.glob("*.txt")) == []

    def test_corrupt_cache_entry_is_refetched(self, tmp_path):
        client = make_client(tmp_path, [FakeResponse(), FakeResponse()])
        client.get_ephemeris("Venus", "2000-01-01", "2000-01-02")
        (cache_file,) = tmp_path.glob("*.txt")
        cache_file.write_text("")  # simulate an interrupted write
        client.get_ephemeris("Venus", "2000-01-01", "2000-01-02")
        assert client.session.calls == 2

    def test_unwritable_cache_does_not_fail_query(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")  # cache_dir path is a file -> mkdir fails
        client = JPLClient(cache_dir=blocker / "sub", backoff_s=0)
        client.session = ScriptedSession([FakeResponse()])
        eph = client.get_ephemeris("Earth", "2000-01-01", "2000-01-02")
        assert eph["positions_au"].shape == (2, 3)

    def test_unknown_body_raises_value_error(self, tmp_path):
        client = make_client(tmp_path, [])
        with pytest.raises(ValueError, match="Unknown body 'Vulcan'"):
            client.get_ephemeris("Vulcan", "2000-01-01", "2000-01-02")

    def test_invalid_max_retries(self):
        with pytest.raises(ValueError):
            JPLClient(max_retries=0)

    def test_single_epoch_query_advances_stop(self, tmp_path):
        captured = {}

        class Capture(ScriptedSession):
            def get(self, url, params=None, timeout=None):
                captured.update(params)
                return super().get()

        client = JPLClient(cache_dir=tmp_path, backoff_s=0)
        client.session = Capture([FakeResponse()])
        client.get_ephemeris("Earth", "2000-01-01", "2000-01-01")
        assert captured["START_TIME"] == '"2000-Jan-01"'
        assert captured["STOP_TIME"] == '"2000-Jan-02"'

    def test_fetch_system_state_builds_system(self, tmp_path):
        client = make_client(tmp_path, [FakeResponse()])
        system = client.fetch_system_state(bodies=["Sun", "Earth"])
        assert [b.name for b in system.bodies] == ["Sun", "Earth"]
        earth = system.bodies[1]
        assert earth.mass == pytest.approx(jpl_client.SOLAR_SYSTEM_MASSES["Earth"])
        assert earth.position.shape == (3,)
