"""Command-line interface for stellar-predictor."""

import click


@click.group()
@click.version_option()
def main():
    """Stellar Predictor - Predict unknown planets through orbital pattern analysis."""
    pass


# Friendly aliases accepted on the command line -> canonical known_systems keys.
_SYSTEM_ALIASES = {
    "solar": "solar_system",
    "sun": "solar_system",
    "trappist-1": "trappist1",
    "kepler-11": "kepler11",
    "kepler-33": "kepler33",
    "hd-219134": "hd219134",
    "hd_219134": "hd219134",
}


def _resolve_system_key(system: str) -> str:
    """Map user input (case/space/dash-insensitive) to a built-in system key.

    Raises click.BadParameter (exit code 2) for unknown systems.
    """
    from stellar_predictor.data.known_systems import system_keys

    raw = system.strip().lower().replace(" ", "_")
    key = _SYSTEM_ALIASES.get(raw, _SYSTEM_ALIASES.get(raw.replace("_", "-"), raw))
    if key not in system_keys():
        raise click.BadParameter(
            f"Unknown system '{system}'. Choose one of: {', '.join(system_keys())}.",
            param_hint="--system",
        )
    return key


@click.command()
@click.option("--system", default="solar_system", show_default=True,
              help="Built-in system: solar_system, trappist1, kepler11, kepler33, hd219134")
@click.option("--plot", "plot_path", default=None, metavar="PATH",
              type=click.Path(dir_okay=False, writable=True),
              help="Save an orbit diagram + Titius-Bode fit chart (PNG) to PATH.")
def analyze(system: str, plot_path: str | None):
    """Analyze a planetary system for predicted gaps.

    Fits a generalized Titius-Bode spacing law and Hill-radius stability
    analysis to identify orbital gaps where unknown planets could exist.
    """
    from stellar_predictor.data.known_systems import build_system
    from stellar_predictor.patterns.reliability import filter_gaps
    from stellar_predictor.prediction.pipeline import PredictionPipeline
    from stellar_predictor.utils.units import format_au, format_period

    key = _resolve_system_key(system)
    click.echo(f"Loading system: {key}")
    pipeline = PredictionPipeline()
    full_system = build_system(key)

    result = pipeline.analyze(full_system)

    # Apply reliability filter
    reliable_gaps, verdicts = filter_gaps(result.predicted_gaps)
    filtered_count = len(result.predicted_gaps) - len(reliable_gaps)

    click.echo(f"\n{'='*60}")
    click.echo(f"  ANALYSIS: {result.system_name}")
    click.echo(f"  Known planets: {result.num_known_planets}")
    click.echo(f"  Execution time: {result.execution_time_s:.3f}s")
    if filtered_count > 0:
        click.echo(f"  ({filtered_count} gap(s) filtered as unreliable)")
    click.echo(f"{'='*60}")

    if result.tb_fit:
        tb = result.tb_fit
        click.echo("\n  Titius-Bode Fit:")
        click.echo(f"    Log-linear: a_n = {tb.alpha:.3g} x {tb.beta:.3f}^n")
        if tb.c > 0:
            click.echo(f"    Classical:  a_n = {tb.a0:.3f} + {tb.b:.3f} x {tb.c:.3f}^n")
        click.echo(f"    R^2: {tb.r_squared:.4f}")
        if tb.outliers:
            click.echo(f"    Outliers: {', '.join(tb.outliers)}")

    if result.warnings:
        click.echo("\n  Warnings:")
        for w in result.warnings:
            click.echo(f"    - {w}")

    if plot_path:
        from stellar_predictor.visualization.analysis_plot import plot_analysis

        rejected = [g for g, v in zip(result.predicted_gaps, verdicts, strict=False)
                    if not v.is_reliable]
        out = plot_analysis(full_system, result, plot_path,
                            gaps=reliable_gaps, rejected=rejected)
        click.echo(f"\n  Plot saved to {out}")

    if not reliable_gaps:
        click.echo("\n  No reliable gap predictions for this system.")
        return

    w_in = max(5, *(len(g.inner_planet) for g in reliable_gaps))
    w_out = max(5, *(len(g.outer_planet) for g in reliable_gaps))
    click.echo("\n  Predicted Gaps (by combined score):")
    header = (f"  {'#':<3} {'Inner':>{w_in}s} {'Outer':>{w_out}s} "
              f"{'Pred a':>10s} {'Period':>9s} {'TB':>5s} {'Stab':>5s} {'Comb':>5s}")
    click.echo(header)
    click.echo(f"  {'-' * (len(header) - 2)}")

    for i, gap in enumerate(reliable_gaps):
        click.echo(
            f"  {i+1:<3} {gap.inner_planet:>{w_in}s} {gap.outer_planet:>{w_out}s} "
            f"{format_au(gap.predicted_a):>10s} {format_period(gap.predicted_period):>9s} "
            f"{gap.titius_bode_score:>5.2f} {gap.stability_score:>5.2f} "
            f"{gap.combined_score:>5.2f}"
        )

    top = reliable_gaps[0] if reliable_gaps else None
    if top and top.combined_score > 0.3:
        click.echo(f"\n  Top prediction: gap between {top.inner_planet} and "
                    f"{top.outer_planet} at {format_au(top.predicted_a)} "
                    f"(score: {top.combined_score:.2f})")
        click.echo("  This location COULD harbor an undiscovered planet.")


@click.command()
@click.option("--min-planets", default=3, help="Minimum planets per system")
@click.option("--output", default=None, help="Save report to file")
def survey(min_planets: int, output: str | None):
    """Survey all known multi-planet systems for prediction opportunities."""
    click.echo(f"Surveying multi-planet systems (min {min_planets} planets)...")
    click.echo("(Requires NASA Exoplanet Archive access — coming soon)")
    click.echo("For now, run: stellar-predictor analyze --system solar_system")


@click.command()
@click.option("--system", default="solar_system", show_default=True,
              help="System to verify (N-body verification supports solar_system only)")
@click.option("--gap-index", type=click.IntRange(min=1), default=1,
              help="Which gap to verify (1-indexed)")
def verify(system: str, gap_index: int):
    """Run N-body perturbation verification on a predicted gap.

    Requires the optional ``experimental`` extra and the ``rebound`` package.
    """
    from stellar_predictor.data.known_systems import build_solar_system
    from stellar_predictor.prediction.pipeline import PredictionPipeline

    key = _resolve_system_key(system)
    if key != "solar_system":
        raise click.BadParameter(
            "N-body verification needs full orbital elements and currently "
            "supports only 'solar_system'.",
            param_hint="--system",
        )
    import importlib.util

    if importlib.util.find_spec("rebound") is None:
        raise click.ClickException(
            "verify requires the 'rebound' package (pip install rebound)."
        )
    try:
        from stellar_predictor.experimental.verification.perturbation import (
            PerturbationVerifier,
        )
    except ImportError as exc:
        raise click.ClickException(
            f"verify requires optional dependencies: {exc}. "
            'Install with: pip install -e ".[experimental]"'
        ) from exc

    click.echo(f"Analyzing {key}...")
    pipeline = PredictionPipeline()
    full_system = build_solar_system()
    result = pipeline.analyze(full_system)

    if not result.predicted_gaps:
        click.echo("No gaps to verify.")
        return

    if gap_index > len(result.predicted_gaps):
        raise click.BadParameter(
            f"only {len(result.predicted_gaps)} gap(s) predicted.",
            param_hint="--gap-index",
        )
    idx = gap_index - 1
    gap = result.predicted_gaps[idx]

    click.echo(f"Verifying gap {idx + 1}: {gap.inner_planet} → {gap.outer_planet} "
               f"(predicted a = {gap.predicted_a:.2f} AU)")

    from stellar_predictor.physics import NBodySimulator

    sim = NBodySimulator(full_system)
    sim_result = sim.simulate(t_end=80, n_steps=400)

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


main.add_command(analyze)
main.add_command(survey)
main.add_command(verify)
main.add_command(serve)

if __name__ == "__main__":
    main()
