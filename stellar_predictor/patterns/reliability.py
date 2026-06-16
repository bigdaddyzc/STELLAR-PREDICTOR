"""Prediction reliability evaluation and filtering.

Centralised checks that decide whether a predicted gap is reliable enough
to show in reports, charts, and the API.  Kept separate from the scoring
engine so the judgement logic can evolve without touching physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from stellar_predictor.data.models import GapResult


@dataclass
class ReliabilityVerdict:
    """Outcome of a single gap's reliability evaluation."""

    is_reliable: bool
    reasons: list[str] = field(default_factory=list)
    gap_index: int = -1
    combined_score: float = 0.0
    method: str = ""
    # Graded reliability (0-1) and its breakdown — distinct from
    # combined_score, which only measures dynamical signal magnitude.
    score: float = 0.0
    grade: str = ""
    components: dict = field(default_factory=dict)


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

    def __init__(self, config: dict | None = None):
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

        score, grade, components = self.reliability_score(gap, is_reliable=len(reasons) == 0)

        return ReliabilityVerdict(
            is_reliable=len(reasons) == 0,
            reasons=reasons,
            combined_score=gap.combined_score,
            method=gap.method,
            score=score,
            grade=grade,
            components=components,
        )

    # ------------------------------------------------------------------
    # Graded reliability score
    # ------------------------------------------------------------------

    def reliability_score(self, gap: GapResult,
                          is_reliable: bool = True) -> tuple[float, str, dict]:
        """Graded 0-1 confidence for a *single* predicted body.

        Unlike ``combined_score`` (raw dynamical signal magnitude), this
        answers "how much should we trust this specific prediction?" by
        combining five independent, interpretable components:

        - ``signal``    (35%): dynamical signal strength (combined_score).
        - ``agreement`` (25%): how many independent signals (TB, stability,
                               resonance) concur — breadth beats a lone spike.
        - ``position``  (20%): positional precision from the predicted-a
                               confidence interval (tighter = better).
        - ``mass``      (10%): how well the mass range is constrained.
        - ``method``    (10%): credibility of the derivation method
                               (interpolated TB+stability > sub-gap >
                               outer-edge extrapolation).

        A gap that fails the hard reliability checks is capped so it can
        never out-rank a gap that passes them.
        """
        # --- signal: dynamical strength, already 0-1 ---
        signal = max(0.0, min(1.0, gap.combined_score))

        # --- agreement: fraction of the three signals that are present,
        # weighted so two strong concurring signals score high ---
        sig_vals = [gap.titius_bode_score, gap.stability_score,
                    gap.resonance_score]
        active = [s for s in sig_vals if s > 0.05]
        if active:
            n_active = len(active)
            mean_active = sum(active) / n_active
            # breadth (n/3) blended with mean strength of active signals
            agreement = (0.5 * (n_active / 3.0) + 0.5 * mean_active)
        else:
            agreement = 0.0
        agreement = max(0.0, min(1.0, agreement))

        # --- position: relative width of the predicted-a CI ---
        lo, hi = gap.predicted_a_lower, gap.predicted_a_upper
        if gap.predicted_a > 0 and hi > lo > 0:
            rel_width = (hi - lo) / gap.predicted_a
            # 0% width -> 1.0, 50% width -> 0.5, >=100% width -> ~0
            position = max(0.0, 1.0 - rel_width)
        else:
            position = 0.4  # unknown CI -> neutral-low

        # --- mass: tightness of the mass range (log-spread) ---
        ml, mh = gap.estimated_mass_range
        if ml > 0 and mh > 0 and mh >= ml:
            decades = math.log10(mh / ml) if mh > ml else 0.0
            # 0 decades -> 1.0, 1 decade (10x) -> 0.5, >=2 decades -> ~0
            mass = max(0.0, 1.0 - 0.5 * decades)
        else:
            mass = 0.0

        # --- method credibility ---
        method_factor = {
            "titius_bode+stability+resonance": 1.0,
            "titius_bode+stability": 0.95,
            "stability_only": 0.6,
            "sub_gap_tb_fit": 0.7,
            "tb_extrapolation+stability_edge": 0.45,
        }.get(gap.method, 0.7)

        components = {
            "signal": round(signal, 3),
            "agreement": round(agreement, 3),
            "position": round(position, 3),
            "mass": round(mass, 3),
            "method": round(method_factor, 3),
        }

        score = (0.35 * signal + 0.25 * agreement + 0.20 * position
                 + 0.10 * mass + 0.10 * method_factor)

        # Unreliable gaps are capped below the reliable band.
        if not is_reliable:
            score = min(score, 0.35)

        score = round(max(0.0, min(1.0, score)), 3)
        return score, self._grade(score), components

    @staticmethod
    def _grade(score: float) -> str:
        """Map a reliability score to a bilingual qualitative grade."""
        if score >= 0.75:
            return "High / 高"
        if score >= 0.55:
            return "Moderate / 中"
        if score >= 0.35:
            return "Low / 低"
        return "Very low / 极低"


def filter_gaps(
    gaps: list[GapResult],
    config: dict | None = None,
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
    reliable = [g for g, v in zip(gaps, verdicts, strict=False) if v.is_reliable]
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
