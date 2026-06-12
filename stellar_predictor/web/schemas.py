"""Pydantic schemas for web API requests and responses."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request to analyze a system for predicted planet gaps."""
    system: str = "solar_system"
    mode: str = "pattern_analysis"  # "pattern_analysis" or "full_prediction"
    include_verification: bool = False


class VerifyRequest(BaseModel):
    """Request to verify a specific gap prediction."""
    system: str
    gap_index: int = 0
    gap_predicted_a: float = 0.0


class PlanetInfo(BaseModel):
    name: str
    semi_major_axis_au: Optional[float] = None
    period_years: Optional[float] = None
    mass_earth: Optional[float] = None
    eccentricity: Optional[float] = None
    is_star: bool = False


class SystemInfo(BaseModel):
    name: str
    display_name: str
    planet_count: int
    source: str = ""
    has_nbody_support: bool = False


class GapInfo(BaseModel):
    """A predicted gap in a planetary system."""
    index: int
    inner_planet: str
    outer_planet: str
    inner_a_au: float
    outer_a_au: float
    predicted_a_au: float
    predicted_period_years: float
    titius_bode_score: float
    stability_score: float
    combined_score: float
    estimated_mass_min: float
    estimated_mass_max: float
    method: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    stage: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
