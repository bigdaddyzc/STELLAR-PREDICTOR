"""Command-line interface for stellar-predictor."""

import click


@click.group()
@click.version_option()
def main():
    """Stellar Predictor - Predict unknown planets through orbital pattern analysis."""
    pass


@click.command()
@click.option("--system", default="solar_system", help="System to analyze")
def analyze(system: str):
    """Analyze a planetary system for predicted gaps.

    Fits a generalized Titius-Bode spacing law and Hill-radius stability
    analysis to identify orbital gaps where unknown planets could exist.
    """
    from stellar_predictor.prediction.pipeline import PredictionPipeline

    click.echo(f"Loading system: {system}")
    pipeline = PredictionPipeline()

    from stellar_predictor.data.models import CelestialBody, OrbitalElements, StellarSystem

    # Load Solar System
    if system.lower() in ("solar_system", "solar"):
        full_system = _build_solar_system()
    else:
        click.echo(f"ERROR: Unknown system '{system}'. Try 'solar_system'.", err=True)
        return

    result = pipeline.analyze(full_system)

    click.echo(f"\n{'='*60}")
    click.echo(f"  ANALYSIS: {result.system_name}")
    click.echo(f"  Known planets: {result.num_known_planets}")
    click.echo(f"  Execution time: {result.execution_time_s:.3f}s")
    click.echo(f"{'='*60}")

    if result.tb_fit:
        tb = result.tb_fit
        click.echo(f"\n  Titius-Bode Fit:")
        click.echo(f"    Log-linear: a_n = {tb.alpha:.3f} x {tb.beta:.3f}^n")
        if tb.c > 0:
            click.echo(f"    Classical:  a_n = {tb.a0:.3f} + {tb.b:.3f} x {tb.c:.3f}^n")
        click.echo(f"    R^2: {tb.r_squared:.4f}")
        if tb.outliers:
            click.echo(f"    Outliers: {', '.join(tb.outliers)}")

    click.echo(f"\n  Predicted Gaps (by combined score):")
    click.echo(f"  {'#':<4} {'Inner':>8s} {'Outer':>8s} {'Pred a':>8s} {'Period':>8s} {'TB':>6s} {'Stab':>6s} {'Comb':>6s}")
    click.echo(f"  {'-'*56}")

    for i, gap in enumerate(result.predicted_gaps):
        click.echo(
            f"  {i+1:<4} {gap.inner_planet:>8s} {gap.outer_planet:>8s} "
            f"{gap.predicted_a:>7.2f} AU {gap.predicted_period:>7.1f} yr "
            f"{gap.titius_bode_score:>5.2f} {gap.stability_score:>5.2f} "
            f"{gap.combined_score:>5.2f}"
        )

    if result.warnings:
        click.echo(f"\n  Warnings:")
        for w in result.warnings:
            click.echo(f"    - {w}")

    top = result.predicted_gaps[0] if result.predicted_gaps else None
    if top and top.combined_score > 0.3:
        click.echo(f"\n  Top prediction: gap between {top.inner_planet} and "
                    f"{top.outer_planet} at {top.predicted_a:.2f} AU "
                    f"(score: {top.combined_score:.2f})")
        click.echo(f"  This location COULD harbor an undiscovered planet.")


@click.command()
@click.option("--min-planets", default=3, help="Minimum planets per system")
@click.option("--output", default=None, help="Save report to file")
def survey(min_planets: int, output: str | None):
    """Survey all known multi-planet systems for prediction opportunities."""
    click.echo(f"Surveying multi-planet systems (min {min_planets} planets)...")
    click.echo("(Requires NASA Exoplanet Archive access — coming soon)")
    click.echo("For now, run: stellar-predictor analyze --system solar_system")


@click.command()
@click.option("--system", required=True, help="System to verify")
@click.option("--gap-index", type=int, default=1, help="Which gap to verify (1-indexed)")
def verify(system: str, gap_index: int):
    """Run N-body perturbation verification on a predicted gap."""
    from stellar_predictor.prediction.pipeline import PredictionPipeline
    from stellar_predictor.physics import NBodySimulator

    click.echo(f"Analyzing {system}...")
    pipeline = PredictionPipeline()
    full_system = _build_solar_system()
    result = pipeline.analyze(full_system)

    if not result.predicted_gaps:
        click.echo("No gaps to verify.")
        return

    idx = min(gap_index - 1, len(result.predicted_gaps) - 1)
    gap = result.predicted_gaps[idx]

    click.echo(f"Verifying gap {idx + 1}: {gap.inner_planet} → {gap.outer_planet} "
               f"(predicted a = {gap.predicted_a:.2f} AU)")

    sim = NBodySimulator(full_system)
    sim_result = sim.simulate(t_end=80, n_steps=400)

    from stellar_predictor.verification.perturbation import PerturbationVerifier
    verifier = PerturbationVerifier(
        full_system,
        observed_positions=sim_result.positions,
        times=sim_result.times,
    )
    vr = verifier.verify_gap(gap)

    if vr.error:
        click.echo(f"Verification failed: {vr.error}")
    elif vr.verified:
        click.echo(f"VERIFIED! Improvement ratio: {vr.improvement_ratio:.2f}x")
    else:
        click.echo(f"Not verified. Improvement ratio: {vr.improvement_ratio:.2f}x "
                    f"(threshold: {verifier.verification_threshold})")


@click.command()
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
def serve(host: str, port: int):
    """Launch the web interface."""
    import uvicorn
    from stellar_predictor.web.app import create_app
    app = create_app()
    uvicorn.run(app, host=host, port=port)


def _build_solar_system():
    """Build Solar System model from hardcoded J2000 orbital elements."""
    import numpy as np
    from stellar_predictor.data.models import CelestialBody, OrbitalElements, StellarSystem

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
    return system


main.add_command(analyze)
main.add_command(survey)
main.add_command(verify)
main.add_command(serve)

if __name__ == "__main__":
    main()
