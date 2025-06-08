"""Public façade for the *algorithms* package."""

from threading import Lock

# Global lock guarding calls into the underlying C++ modules. These
# extensions are not thread-safe because they rely on static buffers.
# Any Python code invoking them must acquire this lock first.
CPP_LOCK = Lock()

from .seed_manager import (  # noqa: E402
    _determine_initial_seeds_and_diagram,
    _propagate_seeds_for_step_analysis,
    calculate_all_seeds,
)
from .dic_orchestrator import orchestrate_dic_analysis  # noqa: E402
from .postprocessing import (  # noqa: E402
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
    "CPP_LOCK",
]
