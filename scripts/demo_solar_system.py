"""Solar System pattern analysis demo.

Demonstrates genuine pattern-based prediction:
1. Fit Titius-Bode law to the 8 planets
2. Identify the Mars-Jupiter gap (~2.8 AU — Asteroid Belt)
3. Check for outer-system gaps
4. Report stability scores

No "exclude-and-rediscover" — purely pattern-driven prediction.
"""

import numpy as np

from stellar_predictor.data.models import CelestialBody, OrbitalElements, StellarSystem
from stellar_predictor.patterns.predictor import GapPredictor
from stellar_predictor.physics import NBodySimulator


def main():
    print("=" * 65)
    print("  STELLAR PREDICTOR — Pattern Analysis Demo")
    print("  Solar System Orbital Spacing Analysis")
    print("=" * 65)

    # Build Solar System from J2000 orbital elements
    planets = [
        ("Mercury", 1.66012e-7, 0.3871, 0.2056, 0.1222, 0.8436, 0.5088, 4.4026),
        ("Venus", 2.44783e-6, 0.7233, 0.0068, 0.0592, 1.3383, 0.9577, 3.1761),
        ("Earth", 3.00273e-6, 1.0000, 0.0167, 0.0000, -0.1965, 1.7968, 6.2400),
        ("Mars", 3.22715e-7, 1.5237, 0.0934, 0.0323, 0.8653, -1.1951, 0.3381),
        ("Jupiter", 9.54786e-4, 5.2034, 0.0484, 0.0227, 1.7534, 0.2389, 0.3411),
        ("Saturn", 2.85837e-4, 9.5371, 0.0542, 0.0434, 1.9847, 1.6130, 5.5647),
        ("Uranus", 4.36624e-5, 19.1913, 0.0472, 0.0135, 1.2956, 1.6929, 2.4844),
        ("Neptune", 5.15138e-5, 30.0690, 0.0086, 0.0309, 2.2999, -1.4869, 4.4715),
    ]

    system = StellarSystem(name="Solar System", source="J2000 orbital elements")
    system.add_body(CelestialBody("Sun", 1.0, np.zeros(3), np.zeros(3)))
    for name, mass, a, e, i, Om, om, M in planets:
        system.add_body(CelestialBody(
            name=name, mass=mass,
            orbital_elements=OrbitalElements(a, e, i, Om, om, M),
        ))

    # Step 1: Show known planet spacing
    axes = [a for _, _, a, *_ in planets]
    print("\n[1] Known Planet Inter-Planet Spacing:")
    print(f"    {'Gap':>20s}  {'Sep. AU':>10s}  {'Ratio':>8s}")
    print(f"    {'-'*40}")
    for i in range(len(axes) - 1):
        gap = axes[i + 1] - axes[i]
        ratio = axes[i + 1] / axes[i]
        print(f"    {planets[i][0]:>8s} → {planets[i+1][0]:<8s}   {gap:>8.2f} AU   {ratio:>6.2f}x")

    # Step 2: Titius-Bode fit
    print("\n[2] Titius-Bode Law Analysis:")
    predictor = GapPredictor(stellar_mass=1.0, min_known_planets=3)
    result = predictor.predict(system)

    if result.tb_fit:
        tb = result.tb_fit
        print(f"    Log-linear fit: a_n = {tb.alpha:.4f} x {tb.beta:.4f}^n")
        if tb.c > 0:
            print(f"    Classical fit:  a_n = {tb.a0:.4f} + {tb.b:.4f} x {tb.c:.4f}^n")
        print(f"    R^2: {tb.r_squared:.4f}")
        print(f"\n    Planet Index Assignments:")
        for name, idx in sorted(tb.index_map.items(), key=lambda x: x[1]):
            print(f"      {name:<10s} → n={idx}")

    # Step 3: Predicted gaps
    print(f"\n[3] Predicted Gaps (sorted by combined score):")
    print(f"    {'#':<4} {'Gap':>20s} {'Pred a':>8s} {'Period':>8s} "
          f"{'TB':>6s} {'Stab':>6s} {'Comb':>6s}")
    print(f"    {'-'*62}")

    for i, gap in enumerate(result.predicted_gaps):
        tb_label = f"{gap.titius_bode_score:.2f}"
        st_label = f"{gap.stability_score:.2f}"
        comb_label = f"{gap.combined_score:.2f}"
        print(f"    {i+1:<4} {gap.inner_planet:>8s} → {gap.outer_planet:<8s} "
              f"{gap.predicted_a:>7.2f} AU {gap.predicted_period:>7.1f} yr "
              f"{tb_label:>6s} {st_label:>6s} {comb_label:>6s}")

    # Step 4: Highlight best candidate with both TB + stability signals
    print(f"\n[4] Key Finding:")
    # Prefer gaps with TB score > 0 (pattern + stability agreement)
    dual_signal = [g for g in result.predicted_gaps
                   if g.titius_bode_score > 0.1 and g.stability_score > 0.1]
    best = dual_signal[0] if dual_signal else result.predicted_gaps[0]
    print(f"    Top dual-signal prediction: {best.inner_planet} → {best.outer_planet}")
    print(f"    Predicted body at {best.predicted_a:.2f} AU (period {best.predicted_period:.1f} yr)")
    print(f"    TB score: {best.titius_bode_score:.2f}  Stability: {best.stability_score:.2f}  "
          f"Combined: {best.combined_score:.2f}")
    if best.inner_planet == "Mars" and best.outer_planet == "Jupiter":
        print(f"    This IS where the Asteroid Belt exists (2.1–3.3 AU)!")
    print(f"\n    This method identifies gaps that have BOTH:")
    print(f"    - Spacing anomaly vs. Titius-Bode pattern (large ratio excess)")
    print(f"    - Enough Hill-stable orbital room for an additional planet")

    # Step 5: Outer system
    outer_gaps = [g for g in result.predicted_gaps if g.inner_a > 30]
    if outer_gaps:
        print(f"\n[5] Outer System:")
        for og in outer_gaps:
            print(f"    Beyond {og.inner_planet} ({og.inner_a:.1f} AU): "
                  f"predicted gap at {og.predicted_a:.1f} AU")
            print(f"    Score: {og.combined_score:.2f} — "
                  f"{'Possible location for Planet Nine?' if og.combined_score > 0.3 else 'Unlikely'}")

    if result.warnings:
        print(f"\n[!] Warnings: {'; '.join(result.warnings)}")

    print(f"\n    Execution time: {result.execution_time_s:.4f}s")
    print(f"\n{'='*65}")
    print("  This method can now be applied to ANY multi-planet system.")
    print("  CLI: stellar-predictor analyze --system solar_system")
    print("  Web: http://127.0.0.1:8000")
    print("=" * 65)


if __name__ == "__main__":
    main()
