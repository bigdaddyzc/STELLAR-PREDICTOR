"""Leave-one-out retrodiction: an objective accuracy metric for gap prediction.

For each interior planet in a known system we *hide* it, run the predictor on
the remaining planets, and check whether the model re-predicts a body near the
hidden planet's true position. This is the honest test of predictive power —
unlike in-sample R², it measures generalisation to a genuinely unseen body.

Reported per system:
- recall:     fraction of hidden planets recovered within ``tol`` relative error
- mean_pos_err / median_pos_err: relative |a_pred - a_true| / a_true over hits
- mean_score: mean combined_score of the matched gaps (confidence calibration)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stellar_predictor.data.known_systems import (
    STELLAR_INFO,
    ground_truth_planets,
    system_keys,
)
from stellar_predictor.patterns.predictor import GapPredictor

# Default tolerance: a recovered position within 25% of truth counts as a hit.
DEFAULT_TOLERANCE = 0.25

_EARTH_MASSES_PER_SOLAR = 332946.0


@dataclass
class LOOCVMatch:
    """Outcome of hiding a single planet and trying to re-predict it."""

    system: str
    hidden_planet: str
    true_a: float
    predicted_a: float | None
    pos_error: float | None       # relative |pred - true| / true
    combined_score: float | None
    is_hit: bool


@dataclass
class SystemAccuracy:
    """Aggregated leave-one-out accuracy for one system."""

    system: str
    n_tested: int
    n_hits: int
    matches: list[LOOCVMatch] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.n_hits / self.n_tested if self.n_tested else 0.0

    def _hit_errors(self) -> list[float]:
        return [m.pos_error for m in self.matches
                if m.is_hit and m.pos_error is not None]

    @property
    def mean_pos_err(self) -> float:
        errs = self._hit_errors()
        return sum(errs) / len(errs) if errs else float("nan")

    @property
    def median_pos_err(self) -> float:
        errs = sorted(self._hit_errors())
        if not errs:
            return float("nan")
        mid = len(errs) // 2
        if len(errs) % 2:
            return errs[mid]
        return (errs[mid - 1] + errs[mid]) / 2.0

    @property
    def mean_score(self) -> float:
        scores = [m.combined_score for m in self.matches
                  if m.is_hit and m.combined_score is not None]
        return sum(scores) / len(scores) if scores else float("nan")


def _reduced_tuples(
    planets: list[tuple[str, float, float, float]], skip_index: int
) -> list[tuple[str, float, float, float]]:
    """Build a (name, a, mass_solar, ecc) list with one planet removed."""
    out = []
    for j, (name, a, mass_earth, ecc) in enumerate(planets):
        if j == skip_index:
            continue
        out.append((name, a, mass_earth / _EARTH_MASSES_PER_SOLAR, ecc))
    return out


def leave_one_out_system(
    system_name: str, tolerance: float = DEFAULT_TOLERANCE
) -> SystemAccuracy:
    """Run leave-one-out retrodiction over every interior planet of a system.

    Edge planets (innermost/outermost) are excluded: a hidden edge body has no
    enclosing gap, so it cannot be recovered by interpolation.
    """
    planets = ground_truth_planets(system_name)
    stellar_mass = STELLAR_INFO.get(system_name, {}).get("mass", 1.0)
    acc = SystemAccuracy(system=system_name, n_tested=0, n_hits=0)

    # Only interior planets are recoverable by gap interpolation.
    for k in range(1, len(planets) - 1):
        hidden_name, true_a, _, _ = planets[k]
        inner_name = planets[k - 1][0]
        reduced = _reduced_tuples(planets, k)

        predictor = GapPredictor(stellar_mass=stellar_mass)
        result = predictor.predict(reduced)

        # The hidden planet should reappear in the gap whose inner neighbour is
        # planets[k-1]. Among candidate gaps for that flank, take the closest.
        best = None
        best_err = float("inf")
        for g in result.predicted_gaps:
            if g.inner_planet != inner_name:
                continue
            err = abs(g.predicted_a - true_a) / true_a
            if err < best_err:
                best_err = err
                best = g

        if best is None:
            acc.matches.append(LOOCVMatch(
                system=system_name, hidden_planet=hidden_name, true_a=true_a,
                predicted_a=None, pos_error=None, combined_score=None,
                is_hit=False,
            ))
        else:
            is_hit = best_err <= tolerance
            acc.matches.append(LOOCVMatch(
                system=system_name, hidden_planet=hidden_name, true_a=true_a,
                predicted_a=best.predicted_a, pos_error=best_err,
                combined_score=best.combined_score, is_hit=is_hit,
            ))
            if is_hit:
                acc.n_hits += 1
        acc.n_tested += 1

    return acc


def evaluate_all_systems(
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, SystemAccuracy]:
    """Run leave-one-out retrodiction across all supported systems."""
    return {
        key: leave_one_out_system(key, tolerance=tolerance)
        for key in system_keys()
    }


def overall_recall(results: dict[str, SystemAccuracy]) -> float:
    """Micro-averaged recall across every planet tested in every system."""
    total = sum(a.n_tested for a in results.values())
    hits = sum(a.n_hits for a in results.values())
    return hits / total if total else 0.0
