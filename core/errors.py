"""core/errors.py — structured error taxonomy (E100–E500).

Every error carries a stable numeric code so pipelines and CI can branch on
failure class rather than parsing message text.
"""

from __future__ import annotations


class ErrorCode:
    CONFIG = 100
    DATA = 200
    NUMERICAL = 300
    ALGORITHM = 400
    RUNTIME = 500


class BanditForgeError(Exception):
    """Base error for all BanditForge failures."""

    code = 0

    def __init__(self, message: str, code: int | None = None):
        self.message = message
        self.code = code if code is not None else self.__class__.code
        super().__init__(f"[E{self.code}] {message}")


class ConfigError(BanditForgeError):
    """Invalid configuration / environment override."""

    code = ErrorCode.CONFIG


class DataError(BanditForgeError):
    """Synthetic data generation or loading problem."""

    code = ErrorCode.DATA


class NumericalError(BanditForgeError):
    """Numerical failure (non-PD matrix, NaN, non-convergence)."""

    code = ErrorCode.NUMERICAL


class AlgorithmError(BanditForgeError):
    """Algorithm misuse (wrong shapes, uninitialized state)."""

    code = ErrorCode.ALGORITHM


class BanditForgeRuntimeError(BanditForgeError):
    """Unexpected runtime condition."""

    code = ErrorCode.RUNTIME
