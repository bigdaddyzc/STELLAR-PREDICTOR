"""Experimental / legacy perturbation-based detection.

These modules implement the original orbit-residual detection method
(simulate -> residuals -> periodogram -> differential-evolution fit). They
are NOT part of the primary pattern-based prediction path and are kept for
research and the optional ``pipeline.predict(enable_verification=True)`` mode.

Requires the optional ``experimental`` extra:  pip install -e ".[experimental]"
"""
