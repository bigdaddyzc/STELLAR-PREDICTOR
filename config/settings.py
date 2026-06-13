"""Configuration settings for stellar-predictor."""

# Default simulation parameters
DEFAULT_INTEGRATOR = "ias15"
DEFAULT_N_STEPS = 1000

# Unit system (REBOUND)
UNITS = ("AU", "yr", "Msun")

# Data caching
CACHE_DIR = "data/cache"
CACHE_TTL_HOURS = 24 * 7  # 1 week

# MCMC defaults
MCMC_N_WALKERS = 32
MCMC_N_STEPS = 5000
MCMC_BURN_IN_FRACTION = 0.3

# Optimization defaults
DE_MAX_ITER = 200
DE_TOL = 1e-6
DE_SEED = 42

# Pattern analysis defaults
TB_FIT_MIN_PLANETS = 3
STABILITY_CRITICAL_SEP = 10.0
TB_SCORE_WEIGHT = 0.5
STABILITY_SCORE_WEIGHT = 0.5

# Titius-Bode skip-aware fitting (v0.3)
TB_MAX_SKIPS = 3            # max missing-planet slots allowed in the index sequence
TB_SKIP_MIN_R2_GAIN = 0.01  # min R^2 improvement required to prefer a skip fit

# Mean-motion resonance scoring (v0.3 → v0.4)
RESONANCE_SCORE_WEIGHT = 0.15
RESONANCES = [(2, 1), (3, 2), (4, 3), (5, 3), (5, 4), (3, 1), (5, 2), (7, 3)]
RESONANCE_TOLERANCE = 0.05  # relative distance from exact commensurability

# v0.4: Outer-edge prediction
OUTER_EDGE_STEPS = 3            # how many TB steps beyond last planet to check
OUTER_EDGE_MIN_R2 = 0.7         # min TB R^2 required for edge prediction

# v0.4: Multi-planet detection
MULTI_PLANET_MIN_STEPS = 3      # min TB steps in a gap to allow sub-gaps

# v0.4: Scoring parameters
TB_BASE_WEIGHT = 0.50           # base TB weight before R^2 adjustment
STABILITY_BASE_WEIGHT = 0.40    # base stability weight before adjustment
RESONANCE_BASE_WEIGHT = 0.10    # base resonance weight

# Verification
VERIFICATION_THRESHOLD = 1.5
VERIFICATION_TIMEOUT = 600

# v0.5: Reliability filtering for prediction gaps
RELIABILITY_MIN_COMBINED_SCORE = 0.20       # gaps below this are flagged
RELIABILITY_REQUIRE_SUPPORTING_SIGNAL = True   # must have TB>0 OR Stab>0 OR Res>0
RELIABILITY_MAX_MASS_RATIO = 1000.0         # max mass_high / mass_low
RELIABILITY_MAX_MASS_UPPER = 5000.0         # max absolute mass in M_Earth
RELIABILITY_OUTER_EDGE_STABILITY_CHECK = True  # apply outer-edge penalty
RELIABILITY_OUTER_EDGE_MAX_SCORE = 0.75     # cap for outer-edge combined_score
RELIABILITY_SUB_GAP_MIN_STABILITY = 0.10    # sub-gaps need at least this stability
