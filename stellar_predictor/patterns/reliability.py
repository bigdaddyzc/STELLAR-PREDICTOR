"""Prediction reliability evaluation and filtering.

Centralised checks that decide whether a predicted gap is reliable enough
to show in reports, charts, and the API.  Kept separate from the scoring
engine so the judgement logic can evolve without touching physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from stellar_predictor.data.models import GapResult


@dataclass
class ReliabilityVerdict:
    """Outcome of a single gap's reliability evaluation."""

    is_reliable: bool
    reasons: list[str] = field(default_factory=list)
    gap_index: int = -1
    combined_score: float = 0.0
    method: str = ""


# ---------------------------------------------------------------------------
# Default configuration  (also mirrored in config/settings.py)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "min_combined_score": 0.20,
    "require_supporting_signal": True,
    "max_mass_ratio": 1000.0,
    "max_mass_upper": 5000.0,
    "outer_edge_stability_penalty": True,
    "outer_edge_max_score": 0.75,
    "sub_gap_min_stability": 0.10,
}


class ReliabilityFilter:
    """Evaluate a single GapResult against multiple reliability criteria.

    Parameters can be overridden via ``config`` dict at construction time.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.min_combined_score = cfg["min_combined_score"]
        self.require_supporting_signal = cfg["require_supporting_signal"]
        self.max_mass_ratio = cfg["max_mass_ratio"]
        self.max_mass_upper = cfg["max_mass_upper"]
        self.outer_edge_stability_penalty = cfg["outer_edge_stability_penalty"]
        self.outer_edge_max_score = cfg["outer_edge_max_score"]
        self.sub_gap_min_stability = cfg["sub_gap_min_stability"]

    def evaluate(self, gap: GapResult) -> ReliabilityVerdict:
        """Run all checks and return a verdict."""
        reasons: list[str] = []

        # ---- Combined score floor ----
        if gap.combined_score < self.min_combined_score:
            reasons.append(
                f"combined_score {gap.combined_score:.3f} "
                f"< minimum {self.min_combined_score}"
            )

        # ---- At least one dynamical signal must be non-zero ----
        if self.require_supporting_signal:
            has_signal = (
                gap.titius_bode_score > 0.001
                or gap.stability_score > 0.001
                or gap.resonance_score > 0.001
            )
            if not has_signal:
                reasons.append("no supporting signal (TB=0, Stab=0, Res=0)")

        # ---- Mass-range sanity ----
        ml, mh = gap.estimated_mass_range
        if ml <= 0 or mh <= 0:
            reasons.append(f"invalid mass range ({ml}, {mh})")
        else:
            ratio = mh / max(ml, 0.001)
            if ratio > self.max_mass_ratio:
                reasons.append(
                    f"mass range too wide ({ml:.1f}–{mh:.0f} M_Earth, "
                    f"ratio={ratio:.0f})"
                )
            if mh > self.max_mass_upper:
                reasons.append(
                    f"mass upper bound {mh:.0f} exceeds "
                    f"maximum {self.max_mass_upper}"
                )

        # ---- Position sanity ----
        if gap.predicted_a <= 0:
            reasons.append(f"invalid predicted_a ({gap.predicted_a})")
        if gap.inner_a <= 0 or gap.outer_a <= 0:
            reasons.append(f"invalid gap boundaries ({gap.inner_a}, {gap.outer_a})")
        if gap.predicted_a > 500:
            reasons.append(
                f"predicted_a {gap.predicted_a:.1f} AU unreasonably large"
            )

        # ---- Outer-edge inflation check ----
        is_edge = gap.method == "tb_extrapolation+stability_edge"
        if is_edge and self.outer_edge_stability_penalty:
            if gap.stability_score > 0.7 and gap.titius_bode_score < 0.6:
                reasons.append(
                    f"outer-edge: stability_score {gap.stability_score:.2f} "
                    "may be inflated (wide empty space)"
                )
            if gap.combined_score > self.outer_edge_max_score:
                reasons.append(
                    f"outer-edge combined_score {gap.combined_score:.2f} "
                    f"capped at {self.outer_edge_max_score}"
                )

        # ---- Sub-gap stability floor ----
        if gap.method == "sub_gap_tb_fit" and gap.stability_score < self.sub_gap_min_stability:
            reasons.append(
                f"sub-gap with low stability ({gap.stability_score:.3f})"
            )

        return ReliabilityVerdict(
            is_reliable=len(reasons) == 0,
            reasons=reasons,
            combined_score=gap.combined_score,
            method=gap.method,
        )


def filter_gaps(
    gaps: list[GapResult],
    config: Optional[dict] = None,
) -> tuple[list[GapResult], list[ReliabilityVerdict]]:
    """Split *gaps* into reliable / unreliable according to *config*.

    Returns (reliable_gaps, verdicts) where verdicts[i] always corresponds
    to gaps[i] regardless of outcome.
    """
    flt = ReliabilityFilter(config)
    verdicts: list[ReliabilityVerdict] = []
    for idx, g in enumerate(gaps):
        v = flt.evaluate(g)
        v.gap_index = idx
        verdicts.append(v)
    reliable = [g for g, v in zip(gaps, verdicts) if v.is_reliable]
    return reliable, verdicts


def filter_summary(verdicts: list[ReliabilityVerdict]) -> dict:
    """Aggregate filtering statistics for the result envelope."""
    total = len(verdicts)
    reliable = sum(1 for v in verdicts if v.is_reliable)
    filtered = total - reliable

    reason_counts: dict[str, int] = {}
    for v in verdicts:
        if not v.is_reliable:
            for r in v.reasons:
                # Collapse near-identical messages to a category key
                key = r.split("(")[0].strip()
                reason_counts[key] = reason_counts.get(key, 0) + 1

    return {
        "total_gaps": total,
        "reliable_count": reliable,
        "filtered_count": filtered,
        "filtered_reasons": reason_counts,
    }
