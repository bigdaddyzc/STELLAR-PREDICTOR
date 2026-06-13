"""Tests for the prediction reliability filter."""

import pytest

from stellar_predictor.data.models import GapResult
from stellar_predictor.patterns.reliability import ReliabilityFilter, filter_gaps, filter_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gap(combined=0.70, tb=0.7, stab=0.7, res=0.0, mass=(0.1, 10.0),
              method="titius_bode+stability", inner_a=0.5, outer_a=2.0,
              predicted_a=1.0):
    """Quick fixture for a GapResult with minimal required fields."""
    return GapResult(
        inner_a=inner_a,
        outer_a=outer_a,
        predicted_a=predicted_a,
        predicted_period=predicted_a ** 1.5,
        titius_bode_score=tb,
        stability_score=stab,
        combined_score=combined,
        estimated_mass_range=mass,
        method=method,
        inner_planet="Inner",
        outer_planet="Outer",
        resonance_score=res,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReliabilityFilter:
    """Unit tests for the ReliabilityFilter class."""

    def test_reliable_gap_passes(self):
        """A gap with good scores and reasonable mass range should pass."""
        gap = _make_gap()
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert v.is_reliable, f"Expected reliable but got reasons: {v.reasons}"

    def test_low_combined_score_fails(self):
        """A gap below MIN_COMBINED_SCORE should be filtered."""
        gap = _make_gap(combined=0.10, tb=0.0, stab=0.0, res=0.0)
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("combined_score" in r for r in v.reasons)

    def test_zero_signal_gap_fails(self):
        """A gap with TB=0, Stab=0, Res=0 but above score threshold fails."""
        gap = _make_gap(combined=0.30, tb=0.0, stab=0.0, res=0.0)
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("no supporting signal" in r for r in v.reasons)

    def test_mass_range_absurdity_fails(self):
        """Mass range with extreme width should be filtered."""
        gap = _make_gap(mass=(0.1, 10000.0))
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("mass range too wide" in r for r in v.reasons)

    def test_mass_upper_bound_exceeded_fails(self):
        """Mass upper bound above limit should be filtered."""
        gap = _make_gap(mass=(0.1, 6000.0))
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("exceeds maximum" in r for r in v.reasons)

    def test_outer_edge_inflated_stability_fails(self):
        """Outer-edge gap with high stability but moderate TB flagged."""
        gap = _make_gap(combined=0.82, tb=0.55, stab=0.85, res=0.0,
                        method="tb_extrapolation+stability_edge")
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("may be inflated" in r for r in v.reasons)

    def test_outer_edge_capped_score_fails(self):
        """Outer-edge gap with combined_score > max cap filtered."""
        gap = _make_gap(combined=0.80, tb=0.65, stab=0.75, res=0.0,
                        method="tb_extrapolation+stability_edge")
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("capped" in r for r in v.reasons)

    def test_sub_gap_low_stability_fails(self):
        """Sub-gap with very low stability should be filtered."""
        gap = _make_gap(combined=0.50, tb=0.6, stab=0.05, res=0.0,
                        method="sub_gap_tb_fit")
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("sub-gap" in r for r in v.reasons)

    def test_sub_gap_adequate_stability_passes(self):
        """Sub-gap with adequate stability should pass."""
        gap = _make_gap(combined=0.50, tb=0.6, stab=0.15, res=0.0,
                        method="sub_gap_tb_fit")
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert v.is_reliable

    def test_invalid_predicted_a_fails(self):
        """Invalid predicted_a should be filtered."""
        gap = _make_gap(predicted_a=-1.0)
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("invalid" in r for r in v.reasons)

    def test_gap_beyond_500_au_fails(self):
        """predicted_a > 500 AU should be filtered."""
        gap = _make_gap(predicted_a=600.0)
        flt = ReliabilityFilter()
        v = flt.evaluate(gap)
        assert not v.is_reliable
        assert any("unreasonably large" in r for r in v.reasons)


class TestFilterGapsFunction:
    """Tests for the filter_gaps convenience function."""

    def test_all_reliable(self):
        """All high-quality gaps should pass."""
        gaps = [_make_gap(combined=0.8), _make_gap(combined=0.7)]
        reliable, verdicts = filter_gaps(gaps)
        assert len(reliable) == 2
        assert all(v.is_reliable for v in verdicts)

    def test_mixed_reliability(self):
        """Mix of reliable and unreliable."""
        gaps = [
            _make_gap(combined=0.8),
            _make_gap(combined=0.10, tb=0.0, stab=0.0, res=0.0),  # zero signal
            _make_gap(combined=0.7),
        ]
        reliable, verdicts = filter_gaps(gaps)
        assert len(reliable) == 2
        assert len([v for v in verdicts if v.is_reliable]) == 2
        assert len([v for v in verdicts if not v.is_reliable]) == 1

    def test_all_filtered(self):
        """When all gaps are unreliable, return empty list."""
        gaps = [
            _make_gap(combined=0.10, tb=0.0, stab=0.0, res=0.0),
            _make_gap(combined=0.10, tb=0.0, stab=0.0, res=0.0),
        ]
        reliable, verdicts = filter_gaps(gaps)
        assert len(reliable) == 0

    def test_verdict_index_preserved(self):
        """Verdict gap_index should match original list position."""
        gaps = [
            _make_gap(combined=0.8),
            _make_gap(combined=0.10, tb=0.0, stab=0.0, res=0.0),
        ]
        _, verdicts = filter_gaps(gaps)
        assert verdicts[0].gap_index == 0
        assert verdicts[1].gap_index == 1


class TestFilterSummary:
    """Tests for the filter_summary aggregation function."""

    def test_summary_counts(self):
        """Summary should report correct counts."""
        gaps = [
            _make_gap(combined=0.8),
            _make_gap(combined=0.10, tb=0.0, stab=0.0, res=0.0),
            _make_gap(combined=0.10, tb=0.0, stab=0.0, res=0.0),
        ]
        _, verdicts = filter_gaps(gaps)
        s = filter_summary(verdicts)
        assert s["total_gaps"] == 3
        assert s["reliable_count"] == 1
        assert s["filtered_count"] == 2

    def test_summary_reasons(self):
        """Reasons should be aggregated as a dict."""
        gaps = [
            _make_gap(combined=0.10, tb=0.0, stab=0.0, res=0.0),
            _make_gap(combined=0.30, tb=0.0, stab=0.0, res=0.0),
        ]
        _, verdicts = filter_gaps(gaps)
        s = filter_summary(verdicts)
        assert len(s["filtered_reasons"]) > 0
        # Both gaps should have "combined_score" and "no supporting signal" reasons
        # Each reason appears at least once
        assert sum(s["filtered_reasons"].values()) >= 2
