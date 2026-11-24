# ruff: noqa: D205  # numpydoc ignore=SS06
"""
OSMnx is a Python package to easily download, model, analyze, and visualize
street networks and other geospatial features from OpenStreetMap.

Full documentation at: https://osmnx.readthedocs.io

If you use OSMnx in your work, please cite: https://doi.org/10.1111/gean.70009
"""

from importlib.metadata import version as metadata_version

# expose the package version
__version__ = metadata_version("osmnx")

# expose the package's public modules
from . import _errors as _errors
from . import bearing as bearing
from . import convert as convert
from . import distance as distance
from . import elevation as elevation
from . import features as features
from . import geocoder as geocoder
from . import graph as graph
from . import io as io
from . import plot as plot
from . import projection as projection
from . import reconcile as reconcile
from . import routing as routing
from . import settings as settings
from . import simplification as simplification
from . import stats as stats
from . import truncate as truncate
from . import utils as utils
from . import utils_geo as utils_geo

# expose the old v1 API for backwards compatibility
from ._api_v1 import *  # noqa: F403

# expose the typed GeoPackage loader alongside load_graphml
from .io import load_graph_geopackage as load_graph_geopackage

# expose the export reconciliation / stable edge identity API. Bind every
# public name both at the top level and on the `io` namespace so either
# `ox.<name>(...)` and `ox.io.<name>(...)` import styles work. `ox.reconcile`
# itself remains the submodule; `ox.io.reconcile` is the callable alias.
from .reconcile import __all__ as _reconcile_all

for _name in _reconcile_all:
    _value = getattr(reconcile, _name)
    globals()[_name] = _value
    setattr(io, _name, _value)

io.reconcile = reconcile.reconcile

# symmetric GeoPackage loader aliases on the top-level and io namespaces
read_graph_geopackage = io.read_graph_geopackage
load_geopackage = io.load_geopackage
read_gpkg = io.read_gpkg
graph_from_geopackage = io.graph_from_geopackage
globals().setdefault("read_graph_geopackage", read_graph_geopackage)


class _CallableReconcileModule:
    """
    Callable facade over the :mod:`osmnx.reconcile` module.

    Calling ``ox.reconcile(G, G2)`` invokes
    :func:`osmnx.reconcile.reconcile_export`, while any other attribute
    (functions, classes, constants) is delegated to the real module. This lets
    both ``ox.reconcile(...)`` and ``ox.reconcile.reconcile_export(...)`` work.
    """

    def __init__(self, module: object) -> None:
        object.__setattr__(self, "_module", module)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._module.reconcile_export(*args, **kwargs)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "_module"), name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(object.__getattribute__(self, "_module"), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(object.__getattribute__(self, "_module"), name)

    def __dir__(self) -> list[str]:
        return dir(object.__getattribute__(self, "_module"))


# preserve direct submodule imports (`from osmnx.reconcile import ...`) via
# sys.modules, but make the `ox.reconcile` attribute callable as a function.
import sys as _sys

_reconcile_module = _sys.modules[__name__ + ".reconcile"]
reconcile = _CallableReconcileModule(_reconcile_module)
