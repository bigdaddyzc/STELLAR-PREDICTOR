"""Human-friendly formatting of orbital quantities for CLI / report output."""

from __future__ import annotations

import math

DAYS_PER_YEAR = 365.25


def sig3(x: float) -> str:
    """Format ``x`` with 3 significant figures, keeping trailing zeros.

    >>> sig3(0.58), sig3(3.2716), sig3(0.033512), sig3(46.9), sig3(110.4)
    ('0.580', '3.27', '0.0335', '46.9', '110')
    """
    if x is None or not math.isfinite(x):
        return "n/a"
    if x == 0:
        return "0.00"
    s = f"{x:#.3g}"
    if "e" in s:  # very large / very small: plain scientific notation
        return f"{x:.2e}"
    return s.rstrip(".")


def format_au(a_au: float) -> str:
    """Semi-major axis with 3 significant figures, e.g. ``'0.0335 AU'``."""
    return f"{sig3(a_au)} AU"


def format_period(period_yr: float) -> str:
    """Orbital period in days below one year, otherwise in years.

    >>> format_period(0.0166), format_period(5.92)
    ('6.06 d', '5.92 yr')
    """
    if period_yr is None or not math.isfinite(period_yr) or period_yr <= 0:
        return "n/a"
    if period_yr < 1.0:
        return f"{sig3(period_yr * DAYS_PER_YEAR)} d"
    return f"{sig3(period_yr)} yr"
