"""Background task manager for pattern-based predictions."""

from __future__ import annotations

import asyncio
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from stellar_predictor.prediction.pipeline import PredictionPipeline
from stellar_predictor.physics.properties import planet_from_mass
from stellar_predictor.patterns.reliability import filter_gaps, filter_summary
from stellar_predictor.visualization.plotly_viz import (
    system_distribution_plot,
    titius_bode_plot,
    spacing_stability_plot,
)
from stellar_predictor.web.schemas import AnalysisRequest
from stellar_predictor.data.known_systems import (
    EXOPLANET_DATA,
    SOLAR_SYSTEM_PLANETS,
    STELLAR_INFO,
    build_system as _build_known_system,
)

try:
    from config.settings import (
        RELIABILITY_MIN_COMBINED_SCORE,
        RELIABILITY_REQUIRE_SUPPORTING_SIGNAL,
        RELIABILITY_MAX_MASS_RATIO,
        RELIABILITY_MAX_MASS_UPPER,
        RELIABILITY_OUTER_EDGE_STABILITY_CHECK,
        RELIABILITY_OUTER_EDGE_MAX_SCORE,
        RELIABILITY_SUB_GAP_MIN_STABILITY,
    )
except ImportError:
    RELIABILITY_MIN_COMBINED_SCORE = 0.20
    RELIABILITY_REQUIRE_SUPPORTING_SIGNAL = True
    RELIABILITY_MAX_MASS_RATIO = 1000.0
    RELIABILITY_MAX_MASS_UPPER = 5000.0
    RELIABILITY_OUTER_EDGE_STABILITY_CHECK = True
    RELIABILITY_OUTER_EDGE_MAX_SCORE = 0.75
    RELIABILITY_SUB_GAP_MIN_STABILITY = 0.10

# Reliability filter config shared by formatting functions
_RELIABILITY_CONFIG = {
    "min_combined_score": RELIABILITY_MIN_COMBINED_SCORE,
    "require_supporting_signal": RELIABILITY_REQUIRE_SUPPORTING_SIGNAL,
    "max_mass_ratio": RELIABILITY_MAX_MASS_RATIO,
    "max_mass_upper": RELIABILITY_MAX_MASS_UPPER,
    "outer_edge_stability_penalty": RELIABILITY_OUTER_EDGE_STABILITY_CHECK,
    "outer_edge_max_score": RELIABILITY_OUTER_EDGE_MAX_SCORE,
    "sub_gap_min_stability": RELIABILITY_SUB_GAP_MIN_STABILITY,
}

@dataclass
class AnalysisTask:
    id: str
    status: str = "pending"
    progress: float = 0.0
    stage: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    distribution_plot_data: Optional[dict] = None
    tb_plot_data: Optional[dict] = None
    spacing_plot_data: Optional[dict] = None


class TaskManager:
    def __init__(self):
        self.tasks: dict[str, AnalysisTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    def create_task(self, request: AnalysisRequest) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = AnalysisTask(id=task_id)
        return task_id

    def submit(self, task_id: str, request: AnalysisRequest,
               loop: asyncio.AbstractEventLoop):
        loop.run_in_executor(self.executor, self._run_analysis, task_id, request)

    def get_task(self, task_id: str) -> Optional[AnalysisTask]:
        return self.tasks.get(task_id)

    def _run_analysis(self, task_id: str, request: AnalysisRequest):
        task = self.tasks[task_id]
        task.status = "running"

        try:
            def progress(stage: str, pct: float):
                task.stage = stage
                task.progress = pct

            progress("loading_system", 0.05)
            system = _build_system(request.system)
            if system is None:
                raise ValueError(f"Unknown system: {request.system}")

            progress("pattern_analysis", 0.15)
            pipeline = PredictionPipeline(enable_verification=request.include_verification)
            result = pipeline.analyze(system)

            # Get stellar params
            star_info = STELLAR_INFO.get(request.system, {"mass": 1.0, "radius": 1.0, "teff": 5778.0})
            stellar_mass = star_info["mass"]

            progress("generating_visualizations", 0.35)
            # Primary: system distribution diagram
            task.distribution_plot_data = system_distribution_plot(
                system,
                gaps=result.predicted_gaps,
                system_name=system.name,
                stellar_mass=stellar_mass,
            )
            # Supplementary: TB fit plot
            if result.tb_fit:
                task.tb_plot_data = titius_bode_plot(
                    result.tb_fit, system.name,
                    gaps=result.predicted_gaps if result.predicted_gaps else None,
                )
            # Supplementary: stability plot
            task.spacing_plot_data = spacing_stability_plot(
                system, result.stability_regions,
                gaps=result.predicted_gaps,
                system_name=system.name,
            )

            progress("generating_report", 0.75)
            report = _generate_prediction_report(
                result, request.system, stellar_mass,
                star_info["teff"], star_info["radius"],
            )

            progress("formatting_results", 0.9)
            task.result = _format_analysis_result(result, report)
            task.status = "complete"
            task.progress = 1.0
            task.stage = "complete"

        except Exception as e:
            task.status = "failed"
            task.error = f"{type(e).__name__}: {str(e)}"
            task.stage = "error"
            traceback.print_exc()


# ---------------------------------------------------------------------------
# System construction helpers
# ---------------------------------------------------------------------------

def _build_system(system_name: str):
    """Build a supported system by key (delegates to known_systems)."""
    return _build_known_system(system_name)


# ---------------------------------------------------------------------------
# Prediction report generation
# ---------------------------------------------------------------------------

def _generate_prediction_report(result, system_name: str, stellar_mass: float,
                                 stellar_teff: float, stellar_radius: float) -> dict:
    """Generate bilingual (zh/en) parameter report for each predicted gap."""
    star_info = STELLAR_INFO.get(system_name, {"mass": 1.0, "radius": 1.0, "teff": 5778.0})

    # Apply reliability filter (same config as _format_analysis_result)
    reliable_gaps_report, verdicts = filter_gaps(result.predicted_gaps, _RELIABILITY_CONFIG)
    filtered_count = len(result.predicted_gaps) - len(reliable_gaps_report)
    planet_data = []

    for i, gap in enumerate(reliable_gaps_report):
        mass_low = gap.estimated_mass_range[0]
        mass_high = gap.estimated_mass_range[1]

        params = planet_from_mass(
            mass_earth_low=mass_low,
            mass_earth_high=mass_high,
            predicted_a=gap.predicted_a,
            stellar_mass=stellar_mass,
            stellar_teff=stellar_teff,
            stellar_radius_rsun=stellar_radius,
        )

        inner_ref = {
            "name": gap.inner_planet,
            "a_au": round(gap.inner_a, 3),
        }
        outer_ref = {
            "name": gap.outer_planet,
            "a_au": round(gap.outer_a, 3),
        }

        planet_data.append({
            "index": i + 1,
            "params": list(params.values()),
            "inner_planet": inner_ref,
            "outer_planet": outer_ref,
            "combined_score": gap.combined_score,
            "method": gap.method,
        })

    # System reference block
    tb_data = None
    if result.tb_fit:
        tb = result.tb_fit
        tb_data = {
            "alpha": round(tb.alpha, 4),
            "beta": round(tb.beta, 4),
            "r_squared": round(tb.r_squared, 4),
            "start_index": tb.start_index,
        }

    system_ref = {
        "star": {
            "mass": star_info["mass"],
            "radius": star_info["radius"],
            "teff": star_info["teff"],
        },
        "num_known_planets": result.num_known_planets,
        "tb_fit": tb_data,
        "stability_regions": len(result.stability_regions),
        "execution_time_s": round(result.execution_time_s, 4),
        "warnings": result.warnings,
        "total_gaps": len(result.predicted_gaps),
        "filtered_gaps": filtered_count,
    }

    # Add warning about filtered gaps
    if filtered_count > 0:
        warnings = list(result.warnings)
        warnings.append(
            f"{filtered_count} gap(s) filtered as unreliable. "
            "Only showing reliable predictions."
        )
        system_ref["warnings"] = warnings

    return {
        "predicted_bodies": planet_data,
        "system_reference": system_ref,
    }


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_analysis_result(result, report: dict | None = None) -> dict:
    # Apply reliability filter
    reliable_gaps, verdicts = filter_gaps(result.predicted_gaps, _RELIABILITY_CONFIG)
    filtered_verdicts = [v for v in verdicts if not v.is_reliable]

    gaps_data = []
    for i, g in enumerate(reliable_gaps):
        gaps_data.append({
            "index": i + 1,
            "inner_planet": g.inner_planet,
            "outer_planet": g.outer_planet,
            "inner_a_au": round(g.inner_a, 3),
            "outer_a_au": round(g.outer_a, 3),
            "predicted_a_au": g.predicted_a,
            "predicted_period_years": g.predicted_period,
            "titius_bode_score": g.titius_bode_score,
            "stability_score": g.stability_score,
            "combined_score": g.combined_score,
            "estimated_mass_min": round(g.estimated_mass_range[0], 2),
            "estimated_mass_max": round(g.estimated_mass_range[1], 0),
            "method": g.method,
        })

    # Build filter metadata
    fi = filter_summary(verdicts)
    filtered_gaps_info = []
    for v in filtered_verdicts:
        g = result.predicted_gaps[v.gap_index]
        filtered_gaps_info.append({
            "index": v.gap_index + 1,
            "inner_planet": g.inner_planet,
            "outer_planet": g.outer_planet,
            "predicted_a_au": g.predicted_a,
            "combined_score": g.combined_score,
            "method": g.method,
            "reasons": v.reasons,
        })
    fi["filtered_gaps"] = filtered_gaps_info

    tb_data = None
    if result.tb_fit:
        tb = result.tb_fit
        tb_data = {
            "alpha": round(tb.alpha, 4),
            "beta": round(tb.beta, 4),
            "r_squared": round(tb.r_squared, 4),
            "start_index": tb.start_index,
        }

    out = {
        "system_name": result.system_name,
        "num_known_planets": result.num_known_planets,
        "execution_time_s": round(result.execution_time_s, 3),
        "gaps": gaps_data,
        "tb_fit": tb_data,
        "stability_regions_count": len(result.stability_regions),
        "warnings": result.warnings,
        "top_gap": gaps_data[0] if gaps_data else None,
        "filter_info": fi,
    }

    if report:
        out["report"] = report

    return out
