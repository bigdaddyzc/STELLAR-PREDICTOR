"""Static (Matplotlib, Agg) summary figure for ``stellar-predictor analyze --plot``.

Left panel: top-down orbit diagram (sqrt radial scale so compact inner
orbits stay readable) with known planets and predicted gaps.
Right panel: Titius-Bode fit ``a_n = alpha * beta^n`` on a log axis with the
known planets at their fitted indices and the predicted gaps marked.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: safe in CI / servers  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stellar_predictor.data.models import GapResult  # noqa: E402
from stellar_predictor.patterns.stability import StabilityAnalyzer  # noqa: E402
from stellar_predictor.utils.units import sig3  # noqa: E402

_BG = "#0b1020"
_FG = "#ccd3e0"
_GAP = "#ff5577"
_REJECTED = "#8a8fa3"


def _short(name: str, prefix: str) -> str:
    return name[len(prefix):].strip() if prefix and name.startswith(prefix) else name


def _common_prefix(names: list[str]) -> str:
    """Shared ``'TRAPPIST-1 '``-style prefix to strip from labels."""
    if len(names) < 2:
        return ""
    parts = [n.rsplit(" ", 1) for n in names]
    if all(len(p) == 2 for p in parts) and len({p[0] for p in parts}) == 1:
        return parts[0][0] + " "
    return ""


def _gap_tb_index(gap: GapResult, tb) -> float | None:
    """Fractional TB index of a predicted gap (log a on the fitted law)."""
    if tb is None or tb.alpha <= 0 or tb.beta <= 1.0 or gap.predicted_a <= 0:
        return None
    return math.log(gap.predicted_a / tb.alpha) / math.log(tb.beta)


def plot_analysis(system, result, path: str | Path,
                  gaps: list[GapResult] | None = None,
                  rejected: list[GapResult] | None = None,
                  top_n: int = 4, title: str | None = None) -> Path:
    """Render the orbit diagram + TB fit chart for an analysis and save a PNG.

    Args:
        system: any system accepted by the predictor (StellarSystem,
            ExoplanetSystem or list of tuples).
        result: ``PredictionResult`` from ``GapPredictor.predict`` /
            ``PredictionPipeline.analyze``.
        path: output file (PNG recommended; any Matplotlib format works).
        gaps: reliable gaps to highlight (default: ``result.predicted_gaps``).
        rejected: gaps the reliability filter removed; the best few are drawn
            faintly so a "no reliable gaps" result is still informative.
        top_n: max number of gaps drawn from each list.
    Returns:
        The output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gaps = list(result.predicted_gaps if gaps is None else gaps)[:top_n]
    rejected = list(rejected or [])[: max(0, min(3, top_n))]

    planets = [(n, a) for n, a, _, _ in StabilityAnalyzer.extract_planet_data_full(system)]
    prefix = _common_prefix([n for n, _ in planets])
    tb = result.tb_fit

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7.2),
                                  gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(_BG)
    for a_ in (ax, ax2):
        a_.set_facecolor(_BG)
        a_.tick_params(colors=_FG)
        for sp in a_.spines.values():
            sp.set_color("#445")

    # ---- Left: orbit diagram (sqrt radial scale) ----
    f = np.sqrt
    th = np.linspace(0, 2 * np.pi, 400)
    all_a = [a for _, a in planets] + [g.predicted_a for g in gaps + rejected]
    r_max = f(max(all_a)) * 1.15 if all_a else 1.0
    ax.plot(0, 0, "o", color="#ffcc33", ms=16, zorder=5)
    cols = plt.cm.cool(np.linspace(0, 1, max(len(planets), 1)))
    for i, ((name, a), c) in enumerate(zip(planets, cols, strict=False)):
        ax.plot(f(a) * np.cos(th), f(a) * np.sin(th), color=c, lw=1.1, alpha=0.8)
        ang = 0.6 + i * 0.75
        ax.plot(f(a) * np.cos(ang), f(a) * np.sin(ang), "o", color=c, ms=7)
        ax.text(f(a) * np.cos(ang), f(a) * np.sin(ang) + r_max * 0.03,
                _short(name, prefix), color="white", fontsize=9, ha="center")
    for k, g in enumerate(rejected):
        ax.plot(f(g.predicted_a) * np.cos(th), f(g.predicted_a) * np.sin(th),
                ":", color=_REJECTED, lw=0.9, alpha=0.6)
        ang = 2.6 + k * 0.5
        ax.plot(f(g.predicted_a) * np.cos(ang), f(g.predicted_a) * np.sin(ang),
                "x", color=_REJECTED, ms=8)
    for k, g in enumerate(gaps):
        ax.plot(f(g.predicted_a) * np.cos(th), f(g.predicted_a) * np.sin(th), "--",
                color=_GAP, lw=1.6 if k == 0 else 1.0, alpha=1.0 if k == 0 else 0.55)
        ang = -0.5 - k * 0.55
        x, y = f(g.predicted_a) * np.cos(ang), f(g.predicted_a) * np.sin(ang)
        ax.plot(x, y, "*", color=_GAP, ms=16 if k == 0 else 10, zorder=6)
        ax.text(x + r_max * 0.02, y - r_max * 0.06,
                f"#{k + 1} {sig3(g.predicted_a)} AU\nscore {g.combined_score:.2f}",
                color="#ff99aa", fontsize=8)
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    name = title or result.system_name
    sub = ("red stars = predicted gaps" if gaps
           else "no reliable gap predictions")
    if rejected:
        sub += ", grey x = filtered"
    ax.set_title(f"{name}: orbits (sqrt radial scale)\n{sub}", color="white")

    # ---- Right: Titius-Bode fit ----
    idx_map = tb.index_map if tb is not None and isinstance(tb.index_map, dict) else {}
    if tb is not None and idx_map:
        gap_idx = [i for i in (_gap_tb_index(g, tb) for g in gaps) if i is not None]
        n_hi = max(list(idx_map.values()) + gap_idx) + 1
        n_lo = min(min(idx_map.values()), *(gap_idx or [0])) - 0.5
        n = np.linspace(n_lo, n_hi, 200)
        ax2.semilogy(n, tb.alpha * tb.beta ** n, "-", color="#66ddff",
                     label=(f"a = {tb.alpha:.3g} x {tb.beta:.3f}^n, "
                            f"R2 = {tb.r_squared:.4f}"))
        for (pname, a), c in zip(planets, cols, strict=False):
            i = idx_map.get(pname)
            if i is not None:
                ax2.semilogy(i, a, "o", color=c, ms=9)
                ax2.text(i + 0.12, a, _short(pname, prefix), color="white", fontsize=9)
        for k, g in enumerate(gaps):
            gi = _gap_tb_index(g, tb)
            if gi is None:
                continue
            ax2.semilogy(gi, g.predicted_a, "*", color=_GAP, ms=14 if k == 0 else 10)
            ax2.text(gi + 0.12, g.predicted_a * 0.86, f"#{k + 1}", color="#ff99aa",
                     fontsize=9)
        ax2.legend(facecolor="#151b30", labelcolor="white", fontsize=9, loc="upper left")
        ax2.set_xlabel("TB index n", color=_FG)
    else:
        ax2.semilogy(range(len(planets)), [a for _, a in planets], "o", color="#66ddff")
        ax2.text(0.5, 0.5, "no Titius-Bode fit (too few planets)",
                 transform=ax2.transAxes, color=_FG, ha="center")
        ax2.set_xlabel("planet order", color=_FG)
    ax2.set_ylabel("semi-major axis (AU)", color=_FG)
    ax2.set_title("Titius-Bode spacing law fit + predicted gaps", color="white")

    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
