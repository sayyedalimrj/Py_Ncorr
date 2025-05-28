from enum import Enum

class OutputState(Enum):
    """
    Represents the outcome of an Ncorr operation.
    Mirrors the C++ OUT enum.
    """
    SUCCESS = 1
    FAILED = 0
    CANCELLED = -1

# You can add other simple data structures or enums here if needed later.