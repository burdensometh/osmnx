"""Compatibility alias for :mod:`osmnx.reconcile`.

The export reconciliation / manifest / path-cost verification API is
implemented in :mod:`osmnx.reconcile`; this module re-exports it under the
``reconciliation`` name.
"""

from __future__ import annotations

from .reconcile import *  # noqa: F403
from . import reconcile as _reconcile

for _name in dir(_reconcile):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_reconcile, _name)

del _name
