"""triverse — lightweight multi-LLM coordinator (Trinity-inspired)."""

from triverse.coordinator import Coordinator
from triverse.types import CompressionConfig, CoordConfig, CoordinationResult, Role, TurnRecord

__all__ = [
    "Coordinator",
    "CoordinationResult",
    "CoordConfig",
    "CompressionConfig",
    "Role",
    "TurnRecord",
]
__version__ = "0.2.0"
