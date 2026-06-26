"""triverse — lightweight multi-LLM coordinator (Trinity-inspired)."""

from triverse.coordinator import Coordinator, CoordinationResult
from triverse.types import CoordConfig, Role, TurnRecord

__all__ = ["Coordinator", "CoordinationResult", "CoordConfig", "Role", "TurnRecord"]
__version__ = "0.1.0"
