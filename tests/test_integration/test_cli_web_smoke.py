"""Smoke tests for the CLI and the web task/report path.

These cover entry points that previously had 0% coverage. They are
happy-path smoke tests — they assert the plumbing runs end to end and
surfaces the key fields, not exhaustive behavior.
"""

from __future__ import annotations

from click.testing import CliRunner

from stellar_predictor.cli import main
from stellar_predictor.data.known_systems import STELLAR_INFO, build_system
from stellar_predictor.prediction.pipeline import PredictionPipeline
from stellar_predictor.web.tasks import (
    _format_analysis_result,
    _generate_prediction_report,
)


class TestWebAppImports:
    """Guard against broken route imports (e.g. removed re-exports).

    The app factory imports every route module; a missing symbol here is an
    ImportError that only surfaces at server startup, not in unit tests that
    import task functions directly.
    """

    def test_app_factory_imports_all_routes(self):
        from stellar_predictor.web.app import create_app
        app = create_app()
        assert app is not None

    def test_route_modules_import(self):
        # Importing the package triggers each route module's top-level imports.
        from stellar_predictor.web.routes import (  # noqa: F401
            predictions,
            systems,
            visualizations,
        )


class TestCLI:
    def test_version(self):
        result = CliRunner().invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_help_lists_commands(self):
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        for cmd in ("analyze", "serve"):
            assert cmd in result.output

    def test_analyze_solar_system(self):
        # analyze is registered on the group at import time
        from stellar_predictor.cli import analyze
        result = CliRunner().invoke(analyze, ["--system", "solar_system"])
        assert result.exit_code == 0
        assert "Solar System" in result.output
        assert "Titius-Bode" in result.output

    def test_analyze_unknown_system_errors(self):
        from stellar_predictor.cli import analyze
        result = CliRunner().invoke(analyze, ["--system", "nonexistent"])
        assert result.exit_code == 0  # handled gracefully, not a crash
        assert "Unknown system" in result.output


class TestWebTaskPath:
    """Exercise the report-generation path used by the web task manager."""

    def test_report_contains_reliability_score(self):
        system = build_system("trappist1")
        star = STELLAR_INFO["trappist1"]
        result = PredictionPipeline().analyze(system)
        report = _generate_prediction_report(
            result, "trappist1", star["mass"], star["teff"], star["radius"]
        )
        bodies = report["predicted_bodies"]
        assert len(bodies) > 0
        for b in bodies:
            assert 0.0 <= b["reliability_score"] <= 1.0
            assert b["reliability_grade"]
            assert set(b["reliability_components"]) == {
                "signal", "agreement", "position", "mass", "method"
            }

    def test_format_result_envelope(self):
        system = build_system("kepler11")
        result = PredictionPipeline().analyze(system)
        report = _generate_prediction_report(
            result, "kepler11", 0.96, 5663.0, 1.06
        )
        out = _format_analysis_result(result, report)
        assert out["system_name"]
        assert "gaps" in out and isinstance(out["gaps"], list)
        assert "filter_info" in out
        assert out["filter_info"]["reliable_count"] <= out["filter_info"]["total_gaps"]
        # every reliable gap carries the graded reliability score
        for g in out["gaps"]:
            assert 0.0 <= g["reliability_score"] <= 1.0
