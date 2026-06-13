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

![Stellar Predictor Web UI](assets/web_ui.png)

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
# Install (editable with dev tools)
pip install -e ".[dev,notebooks]"

# Launch web interface
stellar-predictor serve --host 127.0.0.1 --port 8000

# Open http://127.0.0.1:8000
```

### Commands

```bash
# Predict gaps for Solar System (CLI)
stellar-predictor analyze --system solar_system

# Run tests
pytest tests/ -v

# Neptune retrodiction demo
python scripts/run_neptune_demo.py

# Full accuracy evaluation
PYTHONIOENCODING=utf-8 python scripts/eval_accuracy.py
```

---

## 🧠 How It Works

### Prediction Pipeline (v0.5)

The pipeline flows: **Data → Patterns → Properties → Visualization**

```
1. Extract planet data (sorted by semi-major axis)
2. Fit Titius-Bode law: a_n = α × β^n (skip-aware)
3. Analyze orbital spacing with Hill-radius stability
4. Score each gap: TB + Stability (sigmoid) + Resonance (Gaussian)
5. Dynamic weights based on TB fit quality (R²)
6. Resonance-aware position bias toward nearest MMR
7. Multi-planet detection for wide gaps (TB harmonic spacing)
8. Outer-edge prediction via TB beta extrapolation
9. Cross-gap consistency pass (amplitude-modulated boost)
10. Logistic system-level normalization: score/(score+0.15)
11. **Reliability filtering** — 6-criteria evaluation per gap
12. Derive physical parameters (M-R relation, T_eq, surface gravity, Hill sphere)
```

### Scoring Formula

```
combined = (TB_w × TB_score + Stab_w × Stab_score + Res_w × Res_score) / total_w
normalized = combined / (combined + 0.15)
```

Weights adjust dynamically with TB fit quality.

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
| Solar System | 0.998 | 11 | 8 | Saturn→Uranus (0.784) |
| TRAPPIST-1 | 0.994 | 9 | 6 | d→e (0.737) |
| Kepler-11 | 0.997 | 10 | 4 | f→g (0.857) |
| Kepler-33 | 0.998 | 9 | 4 | b→c (0.870) |
| HD 219134 | 0.994 | 7 | 3 | c→d (0.806) |

**Neptune retrodiction**: 7-planet Solar System → predicts Neptune at 30.97 AU (3.0% error)

**Asteroid Belt**: Mars→Jupiter gap at 2.91 AU vs Ceres at 2.77 AU (5.2% error)

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
├── data/              # Data models
│   └── models.py          CelestialBody, GapResult, ExoplanetSystem
├── web/               # FastAPI web interface
│   ├── app.py             Application factory
│   ├── tasks.py           Analysis orchestration + report generation
│   ├── routes/            API endpoints
│   └── static/            Frontend (JS + CSS + HTML)
├── visualization/     # Plotly charts
│   └── plotly_viz.py      Distribution, TB fit, stability plots
├── prediction/        # High-level pipeline
│   └── pipeline.py        PredictionPipeline orchestrator
├── verification/      # Perturbation verification
│   └── perturbation.py    N-body cross-validation
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

**47 tests** covering: physics (kepler, nbody), patterns (TB fit, stability, predictor, reliability), integration (solar system).

---

## 📚 Version History

| Version | Features |
|---------|----------|
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
