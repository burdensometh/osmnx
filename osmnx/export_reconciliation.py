"""Compatibility alias for :mod:`osmnx.reconcile`."""

from __future__ import annotations

from .reconcile import *  # noqa: F403
from . import reconcile as _reconcile

for _alias_name in dir(_reconcile):
    if not _alias_name.startswith("__"):
        globals()[_alias_name] = getattr(_reconcile, _alias_name)
