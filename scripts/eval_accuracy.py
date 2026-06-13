"""Comprehensive prediction accuracy evaluation for all 5 systems.

Evaluates:
- TB fit quality (R², LOOCV RMSE)
- Stability gap detection
- Predicted vs actual known positions (Solar System)
- Cross-gap consistency
- Resonance scoring
- Confidence intervals
- Physical parameter plausibility
"""
import json
import sys
import time
import numpy as np

# ---- Build systems (mirrors web/tasks.py) ----
SOLAR_SYSTEM_PLANETS = [
    ("Mercury", 1.66012e-7, 0.3871, 0.2056),
    ("Venus", 2.44783e-6, 0.7233, 0.0068),
    ("Earth", 3.00273e-6, 1.0000, 0.0167),
    ("Mars", 3.22715e-7, 1.5237, 0.0934),
    ("Jupiter", 9.54786e-4, 5.2034, 0.0484),
    ("Saturn", 2.85837e-4, 9.5371, 0.0542),
    ("Uranus", 4.36624e-5, 19.1913, 0.0472),
    ("Neptune", 5.15138e-5, 30.0690, 0.0086),
]

from stellar_predictor.data.models import CelestialBody, OrbitalElements, StellarSystem, ExoplanetSystem

def build_solar_system():
    system = StellarSystem(name="Solar System", source="J2000 orbital elements")
    system.add_body(CelestialBody("Sun", 1.0, np.zeros(3), np.zeros(3)))
    for name, mass, a, e in SOLAR_SYSTEM_PLANETS:
        system.add_body(CelestialBody(
            name=name, mass=mass,
            orbital_elements=OrbitalElements(a, e, 0.0, 0.0, 0.0, 0.0),
        ))
    return system

EXOPLANET_DATA = {
    "trappist1": (0.089, [
        ("TRAPPIST-1 b", 1.374, 0.01154, 0.006),
        ("TRAPPIST-1 c", 1.308, 0.01580, 0.006),
        ("TRAPPIST-1 d", 0.388, 0.02227, 0.008),
        ("TRAPPIST-1 e", 0.692, 0.02925, 0.005),
        ("TRAPPIST-1 f", 1.039, 0.03849, 0.010),
        ("TRAPPIST-1 g", 1.321, 0.04683, 0.002),
        ("TRAPPIST-1 h", 0.326, 0.06189, 0.086),
    ]),
    "kepler11": (0.96, [
        ("Kepler-11 b", 1.9, 0.091, 0.045),
        ("Kepler-11 c", 2.9, 0.107, 0.026),
        ("Kepler-11 d", 7.3, 0.155, 0.004),
        ("Kepler-11 e", 8.0, 0.195, 0.012),
        ("Kepler-11 f", 2.0, 0.250, 0.013),
        ("Kepler-11 g", 0.95, 0.466, 0.15),
    ]),
    "kepler33": (1.26, [
        ("Kepler-33 b", 0.16, 0.0677, 0.0),
        ("Kepler-33 c", 0.29, 0.1189, 0.0),
        ("Kepler-33 d", 0.48, 0.1662, 0.0),
        ("Kepler-33 e", 0.36, 0.2138, 0.0),
        ("Kepler-33 f", 0.40, 0.2535, 0.0),
    ]),
    "hd219134": (0.81, [
        ("HD 219134 b", 4.74, 0.0388, 0.0),
        ("HD 219134 c", 4.36, 0.0653, 0.062),
        ("HD 219134 d", 16.17, 0.237, 0.138),
        ("HD 219134 e", 70.90, 2.563, 0.34),
    ]),
}

def build_exoplanet(name):
    star_mass, planets = EXOPLANET_DATA[name]
    system = ExoplanetSystem(name=name, stellar_mass=star_mass)
    for pname, mass, a, ecc in planets:
        system.planets.append({
            "name": pname, "a": a, "mass": mass,  # Earth masses (extract_planet_data_full converts to solar)
            "eccentricity": ecc,
        })
    return system

# ---- Run evaluation ----
from stellar_predictor.patterns.predictor import GapPredictor
from stellar_predictor.patterns.stability import StabilityAnalyzer
from stellar_predictor.patterns.titius_bode import TitiusBodeFit
from stellar_predictor.physics.properties import planet_from_mass

ALL_SYSTEMS = [
    ("Solar System", build_solar_system(), {"mass": 1.0, "radius": 1.0, "teff": 5778}),
    ("TRAPPIST-1", build_exoplanet("trappist1"), {"mass": 0.089, "radius": 0.119, "teff": 2566}),
    ("Kepler-11", build_exoplanet("kepler11"), {"mass": 0.96, "radius": 1.06, "teff": 5663}),
    ("Kepler-33", build_exoplanet("kepler33"), {"mass": 1.26, "radius": 1.58, "teff": 5904}),
    ("HD 219134", build_exoplanet("hd219134"), {"mass": 0.81, "radius": 0.778, "teff": 4699}),
]

print("=" * 90)
print("  STELLAR PREDICTOR — COMPREHENSIVE ACCURACY EVALUATION")
print("=" * 90)

all_results = {}

for sys_name, system, star in ALL_SYSTEMS:
    t0 = time.perf_counter()
    predictor = GapPredictor(stellar_mass=star["mass"])
    result = predictor.predict(system)
    elapsed = time.perf_counter() - t0
    all_results[sys_name] = result

    print(f"\n{'─'*90}")
    print(f"  SYSTEM: {sys_name}")
    print(f"  Planets: {result.num_known_planets}")
    print(f"  Execution: {elapsed:.3f}s")

    # ---- TB Fit ----
    if result.tb_fit:
        tb = result.tb_fit
        print(f"\n  ┌─ TITIUS-BODE FIT ──────────────────────────────────────────────")
        print(f"  │  a_n = {tb.alpha:.4f} × {tb.beta:.4f}^n")
        print(f"  │  log_a = ln({tb.alpha:.4f}) + n × ln({tb.beta:.4f})")
        print(f"  │  R^2 = {tb.r_squared:.4f}")
        print(f"  │  LOOCV RMSE = {tb.loocv_rmse:.4f} (log-space)")
        print(f"  │  Start index = {tb.start_index}")
        print(f"  │  N skips = {tb.n_skips}")
        if tb.residuals:
            print(f"  │  Residual std = {float(np.std(tb.residuals)):.4f}")
        if tb.gaps:
            print(f"  │  TB-flagged gaps ({len(tb.gaps)}):")
            for tg in tb.gaps:
                edge = " [edge]" if tg.is_edge else ""
                print(f"  │    #{tg.index}: {tg.inner_planet}→{tg.outer_planet} at {tg.predicted_a:.3f} AU{edge}")
        print(f"  └────────────────────────────────────────────────────────────────")

    # ---- Stability ----
    print(f"\n  ┌─ STABILITY ANALYSIS ──────────────────────────────────────────────")
    sr_count = len(result.stability_regions)
    stable_count = sum(1 for r in result.stability_regions if r.gap_ratio >= 1.0)
    print(f"  │  Regions detected: {sr_count}")
    print(f"  │  Stable (gap_ratio≥1): {stable_count}")
    for i, sr in enumerate(result.stability_regions):
        label = "✅ STABLE" if sr.gap_ratio >= 1.0 else "❌ TIGHT  "
        print(f"  │  [{i}] {sr.neighbor_inner} → {sr.neighbor_outer}")
        print(f"  │      gap_ratio={sr.gap_ratio:.3f}, width={sr.width_au:.4f} AU")
        print(f"  │      boundaries: {sr.inner_boundary_au:.4f} – {sr.outer_boundary_au:.4f} AU")
        print(f"  │      mass range: {sr.min_mass_estimate:.1f} – {sr.max_mass_estimate:.0f} M_Earth  {label}")
    print(f"  └────────────────────────────────────────────────────────────────")

    # ---- Predicted Gaps ----
    print(f"\n  ┌─ PREDICTED GAPS ({len(result.predicted_gaps)}) ───────────────────────────────────────")
    for i, gap in enumerate(result.predicted_gaps):
        print(f"  │")
        print(f"  │  #{i+1}: {gap.inner_planet} ── {gap.outer_planet}")
        print(f"  │     Predicted a = {gap.predicted_a:.4f} AU (range: {gap.predicted_a_lower:.4f} – {gap.predicted_a_upper:.4f})")
        print(f"  │     TB score={gap.titius_bode_score:.3f}, Stability={gap.stability_score:.3f}, Resonance={gap.resonance_score:.3f}")
        print(f"  │     Combined={gap.combined_score:.3f}, Method={gap.method}")
        print(f"  │     Mass range: {gap.estimated_mass_range[0]:.1f} – {gap.estimated_mass_range[1]:.0f} M_Earth")
        # Physical parameters
        try:
            params = planet_from_mass(
                mass_earth_low=gap.estimated_mass_range[0],
                mass_earth_high=gap.estimated_mass_range[1],
                predicted_a=gap.predicted_a,
                stellar_mass=star["mass"],
                stellar_teff=star["teff"],
                stellar_radius_rsun=star["radius"],
            )
            for p in params.values():
                print(f"  │     {p['label_zh']}: {p['value']} {p['unit']}")
        except Exception as e:
            print(f"  │     (params unavailable: {type(e).__name__})")
        # Solar System: check if this gap matches a known body
        if sys_name == "Solar System":
            known_bodies = {
                ("Mars", "Jupiter"): (2.77, 9.4e-7, "Ceres / Asteroid Belt"),
                ("Jupiter", "Saturn"): (0, 0, None),
                ("Saturn", "Uranus"): (0, 0, None),
                ("Uranus", "Neptune"): (0, 0, None),
            }
            key = (gap.inner_planet, gap.outer_planet)
            if key in known_bodies:
                actual_a, actual_mass, actual_name = known_bodies[key]
                if actual_name:
                    err_pct = abs(gap.predicted_a - actual_a) / actual_a * 100
                    print(f"  │     ⚠ KNOWN BODY: {actual_name} at {actual_a:.4f} AU")
                    print(f"  │       Position error: {err_pct:.1f}% ({'good' if err_pct < 20 else 'moderate' if err_pct < 50 else 'poor'})")
                    mass_mid = (gap.estimated_mass_range[0] + gap.estimated_mass_range[1]) / 2
                    actual_mass_earth = actual_mass * 332946
                    print(f"  │       Mass: predicted ~{mass_mid:.0f}, actual {actual_mass_earth:.0f} M_Earth")
                else:
                    print(f"  │     No known major body in this gap")
        print(f"  │")
    print(f"  └────────────────────────────────────────────────────────────────")

    # ---- Warnings ----
    if result.warnings:
        print(f"  ⚠ Warnings:")
        for w in result.warnings:
            print(f"    • {w}")

    print(f"{'─'*90}\n")

# ---- Cross-system summary ----
print("\n" + "=" * 90)
print("  CROSS-SYSTEM SUMMARY")
print("=" * 90)

for sys_name, result in all_results.items():
    tb_r2 = result.tb_fit.r_squared if result.tb_fit else 0
    tb_beta = result.tb_fit.beta if result.tb_fit else 0
    n_gaps = len(result.predicted_gaps)
    warnings = len(result.warnings)
    high_conf = sum(1 for g in result.predicted_gaps if g.combined_score >= 0.6)
    print(f"  {sys_name:15s} | TB R²={tb_r2:.3f} β={tb_beta:.3f} | gaps={n_gaps} | high_conf={high_conf} | warnings={warnings}")

# ---- Edge case analysis ----
print("\n" + "=" * 90)
print("  EDGE CASE / LIMITATION ANALYSIS")
print("=" * 90)

# Solar System: known gaps (Asteroid belt between Mars & Jupiter, Kuiper belt gap)
ss_result = all_results["Solar System"]
print("\n  Solar System known inter-planet gaps:")
ss_gaps = {f"{g.inner_planet}→{g.outer_planet}": g for g in ss_result.predicted_gaps}

# Mars-Jupiter (asteroid belt): Ceres at ~2.77 AU
mj = ss_gaps.get("Mars→Jupiter")
if mj:
    print(f"  • Mars→Jupiter: predicted={mj.predicted_a:.4f} AU, Ceres known at ~2.77 AU")
    if 2.5 < mj.predicted_a < 3.0:
        print(f"    ✅ Predicts Ceres/Vesta region well")
    else:
        print(f"    ⚠ Offset from known asteroid belt")

# Jupiter-Saturn
js = ss_gaps.get("Jupiter→Saturn")
if js:
    print(f"  • Jupiter→Saturn: predicted={js.predicted_a:.4f} AU (no known major planet)")

# TRAPPIST-1: compact resonant chain
tr_result = all_results["TRAPPIST-1"]
print(f"\n  TRAPPIST-1 (known resonant chain):")
print(f"  • TB fit R² = {tr_result.tb_fit.r_squared if tr_result.tb_fit else 0:.4f}")
print(f"  • Gaps: {len(tr_result.predicted_gaps)}")
if tr_result.tb_fit and tr_result.tb_fit.r_squared > 0.98:
    print(f"  ✅ TB fit is excellent for this resonant chain system")
else:
    print(f"  TB fit quality: {'good' if tr_result.tb_fit and tr_result.tb_fit.r_squared > 0.9 else 'moderate'}")

# Kepler-11: closely packed
k11_result = all_results["Kepler-11"]
print(f"\n  Kepler-11 (closely packed):")
print(f"  • TB fit R² = {k11_result.tb_fit.r_squared if k11_result.tb_fit else 0:.4f}")

# Kepler-33: all small planets
k33_result = all_results["Kepler-33"]
print(f"\n  Kepler-33 (uniform small planets):")
print(f"  • TB fit R² = {k33_result.tb_fit.r_squared if k33_result.tb_fit else 0:.4f}")
print(f"  • Gaps: {len(k33_result.predicted_gaps)}")

# HD 219134: uneven, has one far-out planet
hd_result = all_results["HD 219134"]
print(f"\n  HD 219134 (uneven spacing, far-out planet):")
print(f"  • TB fit R² = {hd_result.tb_fit.r_squared if hd_result.tb_fit else 0:.4f}")
print(f"  • Gaps: {len(hd_result.predicted_gaps)}")

# ---- Validation: "would this have predicted known planets" test ----
print("\n" + "=" * 90)
print("  RETRODICTIVE VALIDATION — Would the model predict known bodies?")
print("=" * 90)

# Test with Solar System minus Neptune (simulate pre-1846 knowledge)
print("\n  1. Pre-Neptune Solar System (7 planets, no Neptune):")
ss7_planets = [
    ("Mercury", 0.3871, 1.66012e-7, 0.2056),
    ("Venus", 0.7233, 2.44783e-6, 0.0068),
    ("Earth", 1.0000, 3.00273e-6, 0.0167),
    ("Mars", 1.5237, 3.22715e-7, 0.0934),
    ("Jupiter", 5.2034, 9.54786e-4, 0.0484),
    ("Saturn", 9.5371, 2.85837e-4, 0.0542),
    ("Uranus", 19.1913, 4.36624e-5, 0.0472),
]
predictor_7 = GapPredictor(stellar_mass=1.0)
result_7 = predictor_7.predict(ss7_planets)
if result_7.tb_fit:
    tb7 = result_7.tb_fit
    print(f"  TB fit: a_n = {tb7.alpha:.4f} x {tb7.beta:.4f}^n, R2 = {tb7.r_squared:.4f}")
    neptune_predicted = None
    for g in result_7.predicted_gaps:
        if g.inner_planet == "Uranus":
            neptune_predicted = g.predicted_a
            print(f"  Gap after Uranus: predicted = {g.predicted_a:.2f} AU, confidence = {g.combined_score:.3f}")
    if neptune_predicted:
        print(f"  Actual Neptune: 30.07 AU")
        print(f"  Error: {abs(neptune_predicted - 30.07) / 30.07 * 100:.1f}%")
        if 25 < neptune_predicted < 35:
            print(f"  + Would have predicted Neptune location!")
        else:
            print(f"  - Did not predict Neptune accurately")
    else:
        print(f"  - No gap predicted after Uranus")

# Test with Solar System minus both Uranus & Neptune (pre-1781)
print("\n  2. Pre-Uranus Solar System (6 planets, no Uranus/Neptune):")
ss6_planets = ss7_planets[:5]  # Mercury through Saturn
predictor_6 = GapPredictor(stellar_mass=1.0)
result_6 = predictor_6.predict(ss6_planets)
if result_6.tb_fit:
    tb6 = result_6.tb_fit
    print(f"  TB fit: a_n = {tb6.alpha:.4f} × {tb6.beta:.4f}^n, R² = {tb6.r_squared:.4f}")
    for g in result_6.predicted_gaps:
        actual = "Uranus" if g.inner_planet == "Saturn" else "?"
        print(f"  Gap {g.inner_planet}→{g.outer_planet}: predicted={g.predicted_a:.2f} AU, score={g.combined_score:.3f}")
    uranus_gap = [g for g in result_6.predicted_gaps if g.inner_planet == "Saturn"]
    if uranus_gap:
        u = uranus_gap[0]
        print(f"  Uranus actual: 19.19 AU")
        print(f"  Error: {abs(u.predicted_a - 19.19) / 19.19 * 100:.1f}%")

# ---- Solar System known gaps (asteroid positions for verification) ----
print("\n  3. Known asteroid belt positions in Solar System gaps:")
solar_known_gaps = {
    "Mars→Jupiter": 2.77,  # Ceres (largest asteroid)
    "Jupiter→Saturn": 0.0, # No known major body
    "Saturn→Uranus": 0.0,
    "Uranus→Neptune": 0.0,
}
for gap_name, actual_au in solar_known_gaps.items():
    g = ss_gaps.get(gap_name)
    if g and actual_au > 0:
        err = abs(g.predicted_a - actual_au) / actual_au * 100
        print(f"  • {gap_name}: pred={g.predicted_a:.2f} AU, actual(Ceres)={actual_au:.2f} AU, error={err:.1f}%")

# ---- Overall scoring ----
print("\n" + "=" * 90)
print("  OVERALL ACCURACY ASSESSMENT")
print("=" * 90)

def extract_planet_data(result):
    """Extract planet data from prediction result for residual matching."""
    if hasattr(result.tb_fit, 'alpha_values') and result.tb_fit.alpha_values is not None:
        return []
    return []

# Score each system
scores = {}
for sys_name, result in all_results.items():
    tb_qual = "good" if (result.tb_fit and result.tb_fit.r_squared >= 0.95) else \
              "moderate" if (result.tb_fit and result.tb_fit.r_squared >= 0.8) else \
              "poor" if result.tb_fit else "none"
    n_gaps = len(result.predicted_gaps)
    n_stable = sum(1 for r in result.stability_regions if r.gap_ratio >= 1.0)
    conf_gaps = sum(1 for g in result.predicted_gaps if g.combined_score >= 0.5)
    scores[sys_name] = {
        "tb_quality": tb_qual,
        "tb_r2": round(result.tb_fit.r_squared, 4) if result.tb_fit else 0,
        "n_gaps": n_gaps,
        "n_high_conf": conf_gaps,
        "n_stable_regions": n_stable,
        "n_warnings": len(result.warnings),
    }
    print(f"  {sys_name:15s}: TB={scores[sys_name]['tb_quality']:8s} (R²={scores[sys_name]['tb_r2']:.3f}), "
          f"gaps={n_gaps} ({conf_gaps}≥0.5), stable_regions={n_stable}, warnings={scores[sys_name]['n_warnings']}")

print(f"\n  ╔══ SUMMARY ═══════════════════════════════════════════════════════╗")
print(f"  ║  Systems analyzed: {len(all_results)}")
print(f"  ║  Total predicted gaps: {sum(len(r.predicted_gaps) for r in all_results.values())}")
tb_ok = sum(1 for s in scores.values() if s['tb_quality'] in ('good', 'moderate'))
print(f"  ║  Systems with good/moderate TB fit: {tb_ok}/{len(all_results)}")
print(f"  ║  Retroductive validation (Neptune prediction): ", end="")
# Check from earlier
neptune_ok = False
for g in result_7.predicted_gaps:
    if g.inner_planet == "Uranus" and 25 < g.predicted_a < 35:
        neptune_ok = True
print(f"{'✅' if neptune_ok else '❌'}")

print(f"  ║  Retroductive validation (Uranus prediction): ", end="")
uranus_ok = False
for g in result_6.predicted_gaps:
    if g.inner_planet == "Saturn" and 15 < g.predicted_a < 25:
        uranus_ok = True
print(f"{'✅' if uranus_ok else '❌'}")

print(f"  ╚══════════════════════════════════════════════════════════════════╝")
