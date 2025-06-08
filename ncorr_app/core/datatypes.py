"""
Light-weight Python data types used throughout the Ncorr port.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List

__all__ = [
    "OutputState",
    "RegionData",
    "BORDER_INTERP_DEFAULT",
]


class OutputState(enum.IntEnum):
    """Mirror of the C++ `OUT` enumeration."""
    CANCELLED = -1
    FAILED = 0
    SUCCESS = 1


@dataclass(slots=True)
class RegionData:
    """
    Pure-Python description of a connected region inside an ROI mask.
    All co-ordinates are **integer pixels** (MATLAB style, 0-indexed).
    """
    nodelist:       "list[int]"          # flat list of linear indices (x,y pairs)
    noderange:      "list[int]"          # per-row start/end positions
    height_nodelist: int
    width_nodelist:  int
    upperbound:      int
    lowerbound:      int
    leftbound:       int
    rightbound:      int
    totalpoints:     int

    @property
    def bounds_tuple(self) -> tuple[int, int, int, int]:
        return (self.leftbound, self.upperbound,
                self.rightbound, self.lowerbound)

    def as_dict(self) -> dict:
        return self.__dict__


# Handy container types
RegionList = List[RegionData]

# Default border handling for interpolation routines. Matches the MATLAB
# implementation where `3` corresponds to mirror-padding.
BORDER_INTERP_DEFAULT: int = 3
