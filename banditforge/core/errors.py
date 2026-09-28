"""Error hierarchy for BanditForge.

Codes:
  E1xx  config errors
  E2xx  data errors
  E3xx  policy errors
  E4xx  OPE errors
  E5xx  pipeline errors

Author: 晨星 (CJX0712)
"""
from __future__ import annotations


class BanditForgeError(Exception):
    """Base error with a stable code."""

    code = "E000"

    def __init__(self, message: str) -> None:
        super().__init__(f"[{self.code}] {message}")


class ConfigError(BanditForgeError):
    code = "E100"


class DataError(BanditForgeError):
    code = "E200"


class PolicyError(BanditForgeError):
    code = "E300"


class OPEError(BanditForgeError):
    code = "E400"


class OPEInsufficientOverlapError(OPEError):
    """Raised when overlap is too low for a variance-free estimator."""

    code = "E405"


class PipelineError(BanditForgeError):
    code = "E500"
