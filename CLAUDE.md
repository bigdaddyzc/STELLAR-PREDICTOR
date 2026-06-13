# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stellar Predictor detects orbital gaps in planetary systems that could host unknown bodies, using Titius-Bode law fitting, Hill-radius stability analysis, and cross-gap consistency validation. It also derives physical parameters (mass, radius, density, temperature, surface gravity) for predicted bodies via analytical M-R relations.

A FastAPI web interface provides a system browser, 2D Plotly orbital distribution diagrams, and bilingual (zh/en) prediction reports. Five systems are available: Solar System, TRAPPIST-1, Kepler-11, Kepler-33, HD 219134.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev,notebooks]"

# Run tests
pytest tests/ -v
pytest tests/ -m "not slow"              # skip slow integration tests
pytest tests/test_physics/test_kepler.py  # single test file

# Web interface
stellar-predictor serve --host 127.0.0.1 --port 8000
# Then open http://127.0.0.1:8000

# Run the Neptune prediction demo (compute-intensive, ~minutes)
python scripts/run_neptune_demo.py
```

## Architecture

The pipeline flows: **Data → Patterns → Properties → Visualization**

```
stellar_predictor/
├── data/              # Data models and acquisition
│   ├── models.py          # CelestialBody, StellarSystem, ExoplanetSystem, GapResult, PredictionResult
│   ├── fetcher.py         # Unified DataFetcher interface
│   ├── jpl_horizons.py   # JPL Horizons queries (network-dependent)
│   └── jpl_client.py     # Low-level JPL API client
├── physics/           # Physics engine
│   ├── nbody.py           # REBOUND wrapper (IAS15 integrator)
│   ├── kepler.py          # Kepler equation solver, element ↔ cartesian conversions
│   ├── residuals.py       # Residual computation + Lomb-Scargle periodogram
│   └── properties.py      # Analytical M-R relations, surface gravity, Hill sphere, etc.
├── patterns/          # Orbital pattern analysis (primary prediction method)
│   ├── titius_bode.py     # Log-linear TB fitting, weighted/unweighted, gap scoring
│   ├── stability.py       # Hill radius, mutual Hill separation, stability region detection
│   └── predictor.py       # GapPredictor: TB + stability → GapResult list with confidence scores
├── prediction/        # High-level prediction pipeline
│   └── pipeline.py        # PredictionPipeline.analyze(system) → PredictionResult
├── inference/         # Parameter estimation (for perturbation-based mode)
│   ├── optimizer.py       # Differential evolution optimizer (scipy)
│   └── candidate.py       # CandidateBody with uncertainty tuples (median, lo, hi)
├── detection/         # Perturbation-based detection (legacy orbit-residual method)
│   ├── base.py            # ABC: DetectionMethod, DetectionResult
│   └── orbital_residual.py # Simulate → residuals → periodogram → optimize
├── verification/      # Validation
│   └── perturbation.py    # Cross-validate predicted bodies via perturbation injection
├── visualization/     # Plotly (web) + Matplotlib (CLI) visualization
│   ├── plotly_viz.py      # system_distribution_plot(), titius_bode_plot(), spacing_stability_plot()
│   └── orbit_plot.py      # Matplotlib orbit plotting
├── web/               # FastAPI web interface
│   ├── app.py             # Application factory with lifespan, static file mounting
│   ├── tasks.py           # Background analysis tasks, report generation, 5 system definitions
│   ├── schemas.py         # Pydantic request/response schemas
│   ├── websocket.py       # WebSocket for real-time progress
│   └── routes/
│       ├── systems.py         # GET /api/systems, GET /api/systems/{name}/planets
│       ├── predictions.py     # POST /api/analyze, GET /api/analyze/{task_id}
│       └── visualizations.py  # GET /api/viz/distribution/{id}, /tb-fit/{id}, /spacing/{id}
└── cli.py             # Click CLI entry point (serve, predict, fetch-data)
```

### Primary Prediction Pipeline (Pattern-Based)

1. Extract sorted (name, a_au, mass, ecc) tuples from any system representation
2. Fit Titius-Bode law: a_n = α × β^n (skip-aware log-linear regression, LOOCV diagnostics)
3. Analyze orbital spacing with eccentricity-aware Hill-radius stability criteria
4. For each adjacent planet pair: compute TB score + sigmoid stability score + Gaussian resonance score → combined confidence with dynamic weights
5. Predicted position: geometric mean → resonance-aware bias toward nearest MMR within stable zone
6. Uncertainty propagation from TB residual sigma to mass, period, and combined score
7. Multi-planet-per-wide-gap detection via TB harmonic spacing
8. Outer-edge prediction via TB beta extrapolation (up to 3 steps beyond last planet)
9. Cross-gap consistency pass: amplitude-modulated boost for TB-aligned multi-step spans
10. System-level logistic normalization: score / (score + 0.15)
11. Derive physical parameters for each predicted body (M-R relation, equilibrium temperature, surface gravity, Hill sphere)

### Unit System

All physics uses REBOUND conventions: **AU, years, solar masses**.
- `CelestialBody.mass` — solar masses
- `ExoplanetSystem.planets["mass"]` — Earth masses (converted to solar by `extract_planet_data_full()`)
- Mass estimates in the report use **Earth masses (M_earth)**. Velocities from JPL Horizons arrive in AU/day and are converted to AU/yr when added to REBOUND.

### Data Models

- **GapResult**: inner/outer planet, predicted semi-major axis, TB/stability/combined scores, estimated mass range, uncertainty bounds
- **PredictionResult**: system name, TB fit parameters, stability regions, predicted gaps list, warnings
- **ExoplanetSystem**: name, stellar_mass, planets list (each with name/a/mass/eccentricity dicts)
- `CandidateBody` (inference mode): parameters as `(median, lower_bound, upper_bound)` tuples

### Physical Properties (properties.py)

- `mass_radius_relation()`: piecewise M-R (Zeng+2016 rocky, Bashi+2017 Neptune-like, degenerate gas giant)
- `equilibrium_temperature()`: T_eq = T_star × sqrt(R_star/(2a)) × (1-albedo)^0.25
- `surface_gravity_ms2()`, `hill_sphere_au()`, `density_gcm3()`
- `classify_planet()`: Rocky / Super-Earth / Neptune-like / Sub-Saturn / Gas Giant / Super-Jupiter
- `planet_from_mass()`: returns dict with zh/en labels for 10 parameters, self-consistent ranges

## Key Dependencies

- **rebound** — N-body integrator (IAS15 by default)
- **astroquery** — JPL Horizons data access (requires network)
- **scipy** — `differential_evolution` for optimization, `lombscargle` for periodograms
- **plotly** — Interactive web visualizations (distribution diagrams)
- **fastapi** + **uvicorn** — Web interface
- **numpy** — Array operations throughout

## Web Interface

5 planetary systems available: Solar System (8 planets), TRAPPIST-1 (7), Kepler-11 (6), Kepler-33 (5), HD 219134 (4).

Layout: left panel (system browser + predicted gaps list), right panel (distribution diagram + bilingual prediction report). Analysis runs asynchronously via ThreadPoolExecutor; progress polling at 800ms intervals.

Plotly loads asynchronously with multi-CDN fallback (jsdelivr → unpkg) — the button works before Plotly finishes loading.

## Test Markers

Tests marked `@pytest.mark.slow` run the full detection pipeline with differential evolution (~minutes). Default `pytest` invocation includes them; use `-m "not slow"` to skip.

## Configuration

`config/settings.py` holds default simulation, MCMC, and optimization parameters. These are module-level constants.

## Stellar System Info

STELLAR_INFO dict in `web/tasks.py` contains stellar mass/radius/Teff for each system. Planet data for exoplanet systems is sourced from NASA Exoplanet Archive; Solar System data from JPL J2000 orbital elements.
