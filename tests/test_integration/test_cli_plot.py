"""CLI output units and the ``analyze --plot`` PNG."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from stellar_predictor.cli import analyze
from stellar_predictor.utils.units import format_au, format_period, sig3

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("system", ["solar_system", "trappist1"])
def test_analyze_plot_writes_png(tmp_path, system):
    out = tmp_path / "plots" / f"{system}.png"
    result = CliRunner().invoke(analyze, ["--system", system, "--plot", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == PNG_MAGIC
    assert "Plot saved to" in result.output


def test_trappist_periods_in_days():
    result = CliRunner().invoke(analyze, ["--system", "trappist1"])
    assert result.exit_code == 0
    assert " d " in result.output
    assert "0.0 yr" not in result.output


def test_solar_axes_three_sig_figs():
    result = CliRunner().invoke(analyze, ["--system", "solar_system"])
    assert result.exit_code == 0
    assert "3.27 AU" in result.output
    assert " yr " in result.output


def test_unit_helpers():
    assert sig3(0.58) == "0.580"
    assert sig3(3.2716) == "3.27"
    assert sig3(0.033512) == "0.0335"
    assert sig3(110.4) == "110"
    assert format_au(46.93) == "46.9 AU"
    assert format_period(0.0166) == "6.06 d"
    assert format_period(5.92) == "5.92 yr"
    assert format_period(0) == "n/a"
