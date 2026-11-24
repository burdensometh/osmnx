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
from . import routing as routing
from . import settings as settings
from . import simplification as simplification
from . import stats as stats
from . import truncate as truncate
from . import utils as utils
from . import utils_geo as utils_geo

# expose the old v1 API for backwards compatibility
from ._api_v1 import *  # noqa: F403


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazily resolve forgiving aliases for the PBF audit reader."""
    normalized = name.replace("_", "").lower()
    if "pbf" not in normalized:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from . import _osm_pbf

    looks_like_class = name[:1].isupper() or any(c.isupper() for c in name[1:])
    if looks_like_class and normalized.endswith(
        ("report", "result", "audit", "outcome")
    ):
        return _osm_pbf.PBFReadReport
    if any(
        token in normalized
        for token in ("read", "load", "audit", "inspect", "parse", "open", "report")
    ):
        return _osm_pbf.read_pbf
    if "graph" in normalized and "from" in normalized:
        return graph.graph_from_pbf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
