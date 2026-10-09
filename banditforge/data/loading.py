"""File loading: persist/restore BanditLog as .npz (utf-8 safe, no pandas)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import DataError
from ..core.types import BanditLog


def save_log(log: BanditLog, path: str) -> None:
    np.savez_compressed(path, **log.to_dict())


def load_log(path: str) -> BanditLog:
    p = Path(path)
    if not p.exists():
        raise DataError(f"log file not found: {p}")
    try:
        data = np.load(p)
    except Exception as exc:  # pragma: no cover - corrupted file
        raise DataError(f"cannot load log file {p}: {exc}") from exc
    try:
        return BanditLog.from_dict({k: data[k] for k in data.files})
    except (KeyError, ValueError) as exc:
        raise DataError(f"invalid log file {p}: {exc}") from exc
