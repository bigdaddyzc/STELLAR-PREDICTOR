"""HTTP-level tests for the FastAPI routes (systems, analysis, visualizations)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from stellar_predictor.data.known_systems import EXOPLANET_DATA, system_keys
from stellar_predictor.web.app import create_app
from stellar_predictor.web.routes.systems import EXOPLANET_SYSTEMS


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:  # context manager runs the lifespan
        yield c


def _wait_for(client, task_id, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        data = client.get(f"/api/analyze/{task_id}").json()
        if data["status"] in ("complete", "failed"):
            return data
        time.sleep(0.1)
    raise AssertionError(f"task {task_id} did not finish in {timeout_s}s")


def test_list_systems_matches_known_systems(client):
    resp = client.get("/api/systems")
    assert resp.status_code == 200
    assert [s["name"] for s in resp.json()] == system_keys()


@pytest.mark.parametrize("name", system_keys())
def test_planets_endpoint(client, name):
    resp = client.get(f"/api/systems/{name}/planets")
    assert resp.status_code == 200
    planets = resp.json()["planets"]
    assert planets and all(p["semi_major_axis_au"] > 0 for p in planets)


def test_unknown_system_planets_404(client):
    resp = client.get("/api/systems/tatooine/planets")
    assert resp.status_code == 404
    assert "Unknown system" in resp.json()["detail"]


def test_route_planet_table_matches_canonical_data():
    """systems.py keeps its own table (with periods); guard against drift."""
    assert set(EXOPLANET_SYSTEMS) == set(EXOPLANET_DATA)
    for key, info in EXOPLANET_SYSTEMS.items():
        star_mass, canonical = EXOPLANET_DATA[key]
        assert info["star_mass"] == pytest.approx(star_mass)
        assert info["planet_count"] == len(canonical) == len(info["planets"])
        for (n1, m1, a1, _p, e1), (n2, m2, a2, e2) in zip(info["planets"], canonical, strict=True):
            assert (n1, m1, a1, e1) == (n2, m2, a2, e2)


def test_analyze_unknown_system_rejected(client):
    resp = client.post("/api/analyze", json={"system": "tatooine"})
    assert resp.status_code == 422
    assert "Unknown system" in resp.json()["detail"]


def test_unknown_task_404(client):
    assert client.get("/api/analyze/nope").status_code == 404


def test_viz_before_any_task(client):
    data = client.get("/api/viz/distribution/nope").json()
    assert data["error"] == "Task not found"


@pytest.mark.parametrize("system", ["solar_system", "hd219134"])
def test_full_analysis_flow(client, system):
    resp = client.post("/api/analyze", json={"system": system})
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]

    data = _wait_for(client, task_id)
    assert data["status"] == "complete", data.get("error")
    result = data["result"]
    assert result["num_known_planets"] >= 4
    assert isinstance(result["gaps"], list)

    for path in ("distribution", "tb-fit", "spacing"):
        viz = client.get(f"/api/viz/{path}/{task_id}").json()
        assert "error" not in viz, (path, viz)
        assert "data" in viz and "layout" in viz


def test_static_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()
