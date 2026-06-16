"""CI regression guards for leave-one-out retrodiction accuracy.

These tests turn the objective accuracy harness
(``stellar_predictor.validation.retrodiction``) into a safety net: any
algorithm change that silently degrades predictive accuracy below the
established floor fails CI. Thresholds sit a margin below the values measured
at the time of writing (v0.6) so normal variation does not flake, but a real
regression does trip them.

Measured at v0.6 (see scripts/validate_accuracy.py):
    overall recall          0.85
    trappist1/kepler11/33   100% recall
    solar_system            83% recall, median pos err 8.4%
"""

from __future__ import annotations

import pytest

from stellar_predictor.validation.retrodiction import (
    evaluate_all_systems,
    leave_one_out_system,
    overall_recall,
)


@pytest.fixture(scope="module")
def results():
    return evaluate_all_systems()


def test_overall_recall_floor(results):
    """Micro-averaged recall across all systems must stay >= 0.80."""
    recall = overall_recall(results)
    assert recall >= 0.80, f"overall recall regressed to {recall:.2%}"


@pytest.mark.parametrize("system", ["trappist1", "kepler11", "kepler33"])
def test_well_behaved_systems_full_recall(results, system):
    """Geometric-progression systems must recover every hidden interior planet."""
    acc = results[system]
    assert acc.n_tested > 0
    assert acc.recall == 1.0, (
        f"{system} recall dropped to {acc.recall:.2%} "
        f"({acc.n_hits}/{acc.n_tested})"
    )


def test_solar_system_position_error(results):
    """Hidden Solar-System planets must be recovered with tight position error."""
    acc = results["solar_system"]
    assert acc.recall >= 0.80
    # Median relative position error over hits — floor is generous vs 8.4%.
    assert acc.median_pos_err <= 0.15, (
        f"solar_system median position error regressed to {acc.median_pos_err:.1%}"
    )


def test_hits_are_within_tolerance(results):
    """Every recorded hit must actually fall within the hit tolerance."""
    for acc in results.values():
        for m in acc.matches:
            if m.is_hit:
                assert m.pos_error is not None and m.pos_error <= 0.25


def test_tighter_tolerance_still_recovers_core(results):
    """At a stricter 10% tolerance, resonant chains still recover most planets.

    Guards the *precision* gain of v0.6 position anchoring, not just recall.
    """
    strict = leave_one_out_system("trappist1", tolerance=0.10)
    assert strict.recall >= 0.8, (
        f"trappist1 precision regressed: only {strict.recall:.0%} within 10%"
    )
