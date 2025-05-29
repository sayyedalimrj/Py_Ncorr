"""
Public façade for the *algorithms* package.
"""

from .seed_manager import (
    _determine_initial_seeds_and_diagram,
    _propagate_seeds_for_step_analysis,
    calculate_all_seeds,
)
from .dic_orchestrator import orchestrate_dic_analysis
from .postprocessing import (
    format_displacement_fields,
    calculate_strain_fields,
)

__all__ = [
    # seed helpers
    "_determine_initial_seeds_and_diagram",
    "_propagate_seeds_for_step_analysis",
    "calculate_all_seeds",
    # main workflow
    "orchestrate_dic_analysis",
    # post-processing
    "format_displacement_fields",
    "calculate_strain_fields",
]
