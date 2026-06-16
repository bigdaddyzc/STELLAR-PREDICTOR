"""Compatibility redirect: old detection API now lives in verification and patterns."""

from stellar_predictor.experimental.verification.perturbation import (
    PerturbationVerifier,
    VerificationResult,
)
from stellar_predictor.inference.candidate import CandidateBody

__all__ = ["PerturbationVerifier", "VerificationResult", "CandidateBody"]
