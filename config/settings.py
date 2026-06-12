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

# Verification
VERIFICATION_THRESHOLD = 1.5
VERIFICATION_TIMEOUT = 600
