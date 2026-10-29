# 🌟 Stellar Predictor

**Detect orbital gaps that could host unknown planets — through Titius-Bode law fitting, Hill-radius stability analysis, and multi-signal pattern fusion.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Overview

Stellar Predictor analyzes known planetary systems to identify orbital gaps where unknown planets could exist. It combines three independent signals into a unified confidence score:

| Signal | Method |
|--------|--------|
| **Titius-Bode** | Skip-aware log-linear regression with LOOCV diagnostics |
| **Stability** | Eccentricity-aware Hill radius mutual separation |
| **Resonance** | Gaussian MMR proximity scoring with order penalty |

**5 supported systems**: Solar System, TRAPPIST-1, Kepler-11, Kepler-33, HD 219134

---

## 🖥️ Web Interface

Run `stellar-predictor serve` and open http://127.0.0.1:8000.

The web interface provides:
- **System browser** — select from 5 planetary systems
- **Orbital distribution diagram** — 2D Plotly chart with known and predicted bodies
- **Titius-Bode fit plot** — regression visualization
- **Spacing & stability plot** — stability region analysis
- **Bilingual prediction report** — complete physical parameters (zh/en)
- **Reliability filtering** — low-confidence predictions are hidden with transparent audit trail

---

## 🚀 Quick Start

```bash
# Install (editable, with dev tools and the web interface)
pip install -e ".[dev,web]"

# Launch web interface
stellar-predictor serve --host 127.0.0.1 --port 8000

# Open http://127.0.0.1:8000
```

### Commands

```bash
# Predict gaps (CLI) — any built-in system:
# solar_system, trappist1, kepler11, kepler33, hd219134
stellar-predictor analyze --system solar_system
stellar-predictor analyze --system trappist1

# Save an orbit diagram + Titius-Bode fit chart with the predicted gaps (PNG)
stellar-predictor analyze --system solar_system --plot solar.png

# N-body perturbation check of a predicted Solar System gap
# (needs: pip install -e ".[experimental]" and rebound)
stellar-predictor verify --gap-index 1

# Run tests
pytest tests/ -v

# Full accuracy evaluation (includes the Neptune retrodiction test)
python scripts/eval_accuracy.py
```

---

## 🧠 How It Works

### Prediction Pipeline (v0.7)

The pipeline flows: **Data → Patterns → Properties → Visualization**

```
1. Extract planet data (sorted by semi-major axis)
2. Fit Titius-Bode law: a_n = α × β^n (skip-aware)
3. Analyze orbital spacing with Hill-radius stability
4. Score each gap: TB "missing-slot" evidence + Stability (sigmoid) + Resonance (Gaussian)
5. Dynamic weights based on TB fit quality (R²), then TB gating (v0.7)
6. Resonance-aware position bias toward nearest MMR
7. Multi-planet detection for wide gaps (TB harmonic spacing)
8. Outer-edge prediction via TB beta extrapolation
9. Cross-gap consistency pass (amplitude-modulated, multiplicative boost)
10. Extrapolation penalty per TB step beyond the outermost planet + resonant-chain down-weighting (v0.7)
11. Logistic system-level normalization: score/(score+0.15)
12. **Reliability filtering** — 6-criteria evaluation per gap
13. Derive physical parameters (M-R relation, T_eq, surface gravity, Hill sphere)
```

### Scoring Formula

```
steps    = ln(a_out / a_in) / ln(β)                       # TB steps spanned by the pair
TB_score = 0.9 × clip((steps − 1.3) / 0.7, 0, 1)
               × (0.5 + 0.5 × exp(−((steps − round(steps)) / 0.3)²))
weighted = (TB_w × TB_score + Stab_w × Stab_score + Res_w × Res_score) / total_w
combined = weighted × (0.15 + 0.85 × TB_score)            # TB gate
               × pair_lock                                 # 1 − 0.7·proximity to 3:2 / 4:3 / 5:4
               × chain                                     # 1 − 0.6·excess locked-pair fraction (> 50 %)
               × extrapolation                             # outer edge: 0.6 × 0.6^(step−1)
normalized = combined / (combined + 0.15)
```

Weights adjust dynamically with TB fit quality. Since v0.7 (scoring v2):

- **TB gating** — stability alone can no longer produce a high score: a gap with
  TB evidence ≈ 0 keeps at most 15 % of its weighted signal.
- **Missing-slot TB evidence** — a gap spanning ~3 TB steps (Mars–Jupiter) scores
  high; an irregular 1.5–1.6-step spacing (Mercury–Venus, Saturn–Uranus) scores low.
- **Resonant chains** — if more than half of adjacent pairs are compact and locked
  near a low-order MMR (TRAPPIST-1: 100 %, Solar System: 29 %), all scores are
  down-weighted; a pair itself locked near 3:2 / 4:3 / 5:4 is penalised further
  because a body in between would break the lock.
- **Extrapolation penalty** — outer-edge candidates lose 40 % per extra TB step, and
  their stability term saturates at 0.7 (empty space is not evidence).
- **Periods** use the stellar mass, P = √(a³ / M★).

### Reliability Filter

All predicted gaps pass through 6 checks before being shown:

| Criterion | Default | Purpose |
|-----------|---------|---------|
| Min combined score | 0.20 | Remove noise |
| Supporting signal | ≥1 non-zero | Remove zero-signal gaps |
| Mass range sanity | ratio ≤ 1000 | Remove unphysical ranges |
| Outer-edge inflation | Penalty + cap at 0.75 | Speculative extrapolations |
| Sub-gap stability | ≥ 0.10 | Remove unsupported sub-gaps |
| Position sanity | a ≤ 500 AU | Remove extreme outliers |

**Result**: 46 raw predictions → 25 reliable (54% reliability rate)

---

## 📊 Accuracy

| System | TB R² | Total | Reliable | Top Prediction |
|--------|-------|-------|----------|----------------|
| Solar System | 0.998 | 11 | 9 | Mars→Jupiter 3.27 AU (0.764) |
| TRAPPIST-1 | 0.994 | 9 | 2 | h→outer edge +1 0.0846 AU (0.306); every interior gap < 0.14 |
| Kepler-11 | 0.997 | 9 | 5 | f→g 0.303 AU (0.796) |
| Kepler-33 | 0.998 | 9 | 5 | b→c 0.0818 AU (0.811) |
| HD 219134 | 0.994 | 7 | 6 | c→d 0.131 AU (0.664) |

**Neptune retrodiction**: 7-planet Solar System → top beyond-Uranus candidate 30.97 AU (3.0% error), ranked #1 of the outer-edge candidates (only the Mars–Jupiter asteroid-belt gap ranks above it overall)

**Uranus retrodiction**: 6-planet Solar System → Uranus is ~1.6 TB steps beyond Saturn, so the +1 slot (13.7 AU) ranks first and the +2 slot (21.2 AU, 10.7% error) second; `tests/test_validation/test_retrodiction_ranking.py` asserts top-2 with this documented tolerance

**TRAPPIST-1 minus e**: the d→f candidate at 0.0279 AU (4.6% error) is ranked #1

**Asteroid Belt**: Mars→Jupiter gap at 3.27 AU vs Ceres at 2.77 AU (18% error; position unchanged by v0.7)

---

## 🏗️ Architecture

```
stellar_predictor/
├── patterns/          # Core prediction engine
│   ├── titius_bode.py     TB fitting (skip-aware, LOOCV)
│   ├── stability.py       Hill-radius stability analysis
│   ├── predictor.py       GapPredictor (multi-signal fusion)
│   └── reliability.py     Reliability filter (v0.5)
├── physics/           # Physical parameter derivation
│   ├── properties.py      M-R relations, T_eq, surface gravity
│   ├── kepler.py          Kepler equation solver
│   └── nbody.py           REBOUND N-body integrator
├── data/              # Data models and sources
│   ├── models.py          CelestialBody, GapResult, ExoplanetSystem
│   ├── known_systems.py   Built-in data for the 5 systems (offline)
│   ├── fetcher.py         DataFetcher (JPL with offline J2000 fallback)
│   └── jpl_client.py      JPL Horizons HTTP client (retries + disk cache)
├── web/               # FastAPI web interface
│   ├── app.py             Application factory
│   ├── tasks.py           Analysis orchestration + report generation
│   ├── routes/            API endpoints
│   └── static/            Frontend (JS + CSS + HTML)
├── visualization/     # Plotly charts
│   └── plotly_viz.py      Distribution, TB fit, stability plots
├── prediction/        # High-level pipeline
│   └── pipeline.py        PredictionPipeline orchestrator
├── experimental/      # Legacy perturbation-based verification (optional extra)
│   └── verification/perturbation.py   N-body cross-validation
└── cli.py             # Click CLI entry point
```

### Key Design Decisions

- **Bilingual UI** — All labels and reports in Chinese and English
- **Cosmic dark theme** — Visual design inspired by space observatory UIs
- **Async analysis** — Gap detection runs in background thread with progress polling
- **Reliability-first** — Low-confidence predictions are hidden, not marked down
- **Unit conventions** — AU, years, solar masses (REBOUND compatible)

---

## 🔧 Configuration

All tuning parameters in `config/settings.py`:

```python
# Scoring weights
TB_BASE_WEIGHT = 0.50
STABILITY_BASE_WEIGHT = 0.40
RESONANCE_BASE_WEIGHT = 0.10

# Resonance catalog (8 ratios)
RESONANCES = [(2,1), (3,2), (4,3), (5,3), (5,4), (3,1), (5,2), (7,3)]

# Reliability thresholds
RELIABILITY_MIN_COMBINED_SCORE = 0.20
RELIABILITY_OUTER_EDGE_MAX_SCORE = 0.75
RELIABILITY_SUB_GAP_MIN_STABILITY = 0.10
```

---

## 🧪 Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/ -v --tb=short         # Verbose failures
pytest tests/ -m "not slow"         # Skip slow integration
pytest tests/test_physics/          # Physics only
pytest tests/test_patterns/         # Pattern analysis only

# Coverage
pytest tests/ --cov=stellar_predictor
```

**183 tests** covering: physics (kepler, nbody, properties), patterns (TB fit, stability, predictor, reliability, input sanitization), data (JPL client retries/caching, offline fallback), web API routes, CLI, and leave-one-out validation. CI (GitHub Actions) runs ruff plus `pytest -m "not slow"` with a coverage report.

The N-body tests are skipped automatically if `rebound` is not installed (it has no prebuilt wheel on some platforms, e.g. Linux aarch64 without a C compiler); everything else runs without it.

---

## 📚 Version History

| Version | Features |
|---------|----------|
| v0.7 | Scoring v2 (TB gating, missing-slot TB evidence, resonant-chain down-weighting, extrapolation penalty), retrodiction ranking tests, adaptive CLI units, `analyze --plot` |
| v0.5 | Reliability filter, frontend filter UI, accuracy report |
| v0.4 | Outer-edge prediction, multi-planet gaps, dynamic weights, sigmoid scoring |
| v0.3 | Resonance scoring, eccentricity-aware stability, LOOCV diagnostics |
| v0.2 | Skip-aware TB fitting, Hill-radius stability, basic predictions |
| v0.1 | Initial prototype: simple OLS TB fit + basic validation |

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

- NASA Exoplanet Archive for exoplanet system data
- JPL Horizons for Solar System orbital elements
- REBOUND N-body integrator
- Plotly for interactive visualization
