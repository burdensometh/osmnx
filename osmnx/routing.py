"""Calculate edge speeds, travel times, and weighted shortest paths."""

from __future__ import annotations

import itertools
import logging as lg
import multiprocessing as mp
import re
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Any
from typing import overload

import networkx as nx
import numpy as np
import pandas as pd

from . import _validate
from . import convert
from . import utils

if TYPE_CHECKING:
    import geopandas as gpd

# Values used by `add_edge_speeds` to label each edge with the tier its speed
# estimate came from: its own observed maxspeed, a group-level aggregate, or a
# declared default value.
_SPEED_SOURCE_OBSERVED = "observed"
_SPEED_SOURCE_GROUP = "group"
_SPEED_SOURCE_DEFAULT = "default"

# Dict that is used by `add_edge_speeds` to convert implicit values
# to numbers, based on https://wiki.openstreetmap.org/wiki/Key:maxspeed
_IMPLICIT_MAXSPEEDS: dict[str, float] = {
    "AR:rural": 110.0,
    "AR:urban": 40.0,
    "AR:urban:primary": 60.0,
    "AR:urban:secondary": 60.0,
    "AT:bicycle_road": 30.0,
    "AT:motorway": 130.0,
    "AT:rural": 100.0,
    "AT:trunk": 100.0,
    "AT:urban": 50.0,
    "BE-BRU:rural": 70.0,
    "BE-BRU:urban": 30.0,
    "BE-VLG:rural": 70.0,
    "BE-VLG:urban": 50.0,
    "BE-WAL:rural": 90.0,
    "BE-WAL:urban": 50.0,
    "BE:cyclestreet": 30.0,
    "BE:living_street": 20.0,
    "BE:motorway": 120.0,
    "BE:trunk": 120.0,
    "BE:zone30": 30.0,
    "BG:living_street": 20.0,
    "BG:motorway": 140.0,
    "BG:rural": 90.0,
    "BG:trunk": 120.0,
    "BG:urban": 50.0,
    "BY:living_street": 20.0,
    "BY:motorway": 110.0,
    "BY:rural": 90.0,
    "BY:urban": 60.0,
    "CA-AB:rural": 90.0,
    "CA-AB:urban": 65.0,
    "CA-BC:rural": 80.0,
    "CA-BC:urban": 50.0,
    "CA-MB:rural": 90.0,
    "CA-MB:urban": 50.0,
    "CA-ON:rural": 80.0,
    "CA-ON:urban": 50.0,
    "CA-QC:motorway": 100.0,
    "CA-QC:rural": 75.0,
    "CA-QC:urban": 50.0,
    "CA-SK:nsl": 80.0,
    "CH:motorway": 120.0,
    "CH:rural": 80.0,
    "CH:trunk": 100.0,
    "CH:urban": 50.0,
    "CZ:living_street": 20.0,
    "CZ:motorway": 130.0,
    "CZ:pedestrian_zone": 20.0,
    "CZ:rural": 90.0,
    "CZ:trunk": 110.0,
    "CZ:urban": 50.0,
    "CZ:urban_motorway": 80.0,
    "CZ:urban_trunk": 80.0,
    "DE:bicycle_road": 30.0,
    "DE:living_street": 15.0,
    "DE:motorway": 120.0,
    "DE:rural": 80.0,
    "DE:urban": 50.0,
    "DK:motorway": 130.0,
    "DK:rural": 80.0,
    "DK:urban": 50.0,
    "EE:rural": 90.0,
    "EE:urban": 50.0,
    "ES:living_street": 20.0,
    "ES:motorway": 120.0,
    "ES:rural": 90.0,
    "ES:trunk": 90.0,
    "ES:urban": 50.0,
    "ES:zone30": 30.0,
    "FI:motorway": 120.0,
    "FI:rural": 80.0,
    "FI:trunk": 100.0,
    "FI:urban": 50.0,
    "FR:motorway": 120.0,
    "FR:rural": 80.0,
    "FR:urban": 50.0,
    "FR:zone30": 30.0,
    "GB:nsl_restricted": 48.28,
    "GR:motorway": 130.0,
    "GR:rural": 90.0,
    "GR:trunk": 110.0,
    "GR:urban": 50.0,
    "HU:living_street": 20.0,
    "HU:motorway": 130.0,
    "HU:rural": 90.0,
    "HU:trunk": 110.0,
    "HU:urban": 50.0,
    "IT:motorway": 130.0,
    "IT:rural": 90.0,
    "IT:trunk": 110.0,
    "IT:urban": 50.0,
    "JP:express": 100.0,
    "JP:nsl": 60.0,
    "LT:rural": 90.0,
    "LT:urban": 50.0,
    "NO:rural": 80.0,
    "NO:urban": 50.0,
    "PH:express": 100.0,
    "PH:rural": 80.0,
    "PH:urban": 30.0,
    "PT:motorway": 120.0,
    "PT:rural": 90.0,
    "PT:trunk": 100.0,
    "PT:urban": 50.0,
    "RO:motorway": 130.0,
    "RO:rural": 90.0,
    "RO:trunk": 100.0,
    "RO:urban": 50.0,
    "RS:living_street": 10.0,
    "RS:motorway": 130.0,
    "RS:rural": 80.0,
    "RS:trunk": 100.0,
    "RS:urban": 50.0,
    "RU:living_street": 20.0,
    "RU:motorway": 110.0,
    "RU:rural": 90.0,
    "RU:urban": 60.0,
    "SE:rural": 70.0,
    "SE:urban": 50.0,
    "SI:motorway": 130.0,
    "SI:rural": 90.0,
    "SI:trunk": 110.0,
    "SI:urban": 50.0,
    "SK:living_street": 20.0,
    "SK:motorway": 130.0,
    "SK:motorway_urban": 90.0,
    "SK:rural": 90.0,
    "SK:trunk": 90.0,
    "SK:urban": 50.0,
    "TR:living_street": 20.0,
    "TR:motorway": 130.0,
    "TR:rural": 90.0,
    "TR:trunk": 110.0,
    "TR:urban": 50.0,
    "TR:zone30": 30.0,
    "UA:living_street": 20.0,
    "UA:motorway": 130.0,
    "UA:rural": 90.0,
    "UA:trunk": 110.0,
    "UA:urban": 50.0,
    "UK:motorway": 112.65,
    "UK:nsl_dual": 112.65,
    "UK:nsl_single": 96.56,
    "UZ:living_street": 30.0,
    "UZ:motorway": 110.0,
    "UZ:rural": 100.0,
    "UZ:urban": 70.0,
}


def route_to_gdf(
    G: nx.MultiDiGraph,
    route: list[int],
    *,
    weight: str = "length",
) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame of the edges in a path, in order.

    Parameters
    ----------
    G
        Input graph.
    route
        Node IDs constituting the path.
    weight
        Attribute value to minimize when choosing between parallel edges.

    Returns
    -------
    gdf_edges
        The ordered edges in the path.
    """
    pairs = itertools.pairwise(route)
    uvk = ((u, v, min(G[u][v].items(), key=lambda i: i[1][weight])[0]) for u, v in pairs)
    return convert.graph_to_gdfs(G.subgraph(route), nodes=False).loc[uvk]


# orig/dest int, weight present, cpus present
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: int,
    dest: int,
    *,
    weight: str,
    cpus: int | None,
) -> list[int] | None: ...


# orig/dest int, weight missing, cpus present
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: int,
    dest: int,
    *,
    cpus: int | None,
) -> list[int] | None: ...


# orig/dest int, weight present, cpus missing
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: int,
    dest: int,
    *,
    weight: str,
) -> list[int] | None: ...


# orig/dest int, weight missing, cpus missing
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: int,
    dest: int,
) -> list[int] | None: ...


# orig/dest Iterable, weight present, cpus present
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: Iterable[int],
    dest: Iterable[int],
    *,
    weight: str,
    cpus: int | None,
) -> list[list[int] | None]: ...


# orig/dest Iterable, weight missing, cpus present
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: Iterable[int],
    dest: Iterable[int],
    *,
    cpus: int | None,
) -> list[list[int] | None]: ...


# orig/dest Iterable, weight present, cpus missing
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: Iterable[int],
    dest: Iterable[int],
    *,
    weight: str,
) -> list[list[int] | None]: ...


# orig/dest Iterable, weight missing, cpus missing
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: Iterable[int],
    dest: Iterable[int],
) -> list[list[int] | None]: ...


def shortest_path(
    G: nx.MultiDiGraph,
    orig: int | Iterable[int],
    dest: int | Iterable[int],
    *,
    weight: str = "length",
    cpus: int | None = 1,
) -> list[int] | list[list[int] | None] | None:
    """
    Solve shortest path from origin node(s) to destination node(s).

    Uses Dijkstra's algorithm. If `orig` and `dest` are single node IDs, this
    will return a list of the nodes constituting the shortest path between
    them. If `orig` and `dest` are lists of node IDs, this will return a list
    of lists of the nodes constituting the shortest path between each
    origin-destination pair. If a path cannot be solved, this will return None
    for that path. You can parallelize solving multiple paths with the `cpus`
    parameter, but be careful to not exceed your available RAM.

    See also `k_shortest_paths` to solve multiple shortest paths between a
    single origin and destination. For additional functionality or different
    solver algorithms, use NetworkX directly.

    Parameters
    ----------
    G
        Input graph.
    orig
        Origin node ID(s).
    dest
        Destination node ID(s).
    weight
        Edge attribute to minimize when solving shortest path.
    cpus
        How many CPU cores to use if multiprocessing. If None, use all
        available. If you are multiprocessing, make sure you protect your
        entry point: see the Python docs for details.

    Returns
    -------
    path
        The node IDs constituting the shortest path, or, if `orig` and `dest`
        are both iterable, then a list of such paths.
    """
    _validate._verify_numeric_edge_attribute(G, weight, strict=False)

    # if neither orig nor dest is iterable, just return the shortest path
    if not (isinstance(orig, Iterable) or isinstance(dest, Iterable)):
        return _single_shortest_path(G, orig, dest, weight)

    # if only 1 of orig or dest is iterable and the other is not, raise error
    if not (isinstance(orig, Iterable) and isinstance(dest, Iterable)):
        msg = "`orig` and `dest` must either both be iterable or neither must be iterable."
        raise TypeError(msg)

    # if both orig and dest are iterable, make them lists (so we're guaranteed
    # to be able to get their sizes) then ensure they have same lengths
    orig = list(orig)
    dest = list(dest)
    if len(orig) != len(dest):  # pragma: no cover
        msg = "`orig` and `dest` must be of equal length."
        raise ValueError(msg)

    # determine how many cpu cores to use
    if cpus is None:
        cpus = mp.cpu_count()
    cpus = min(cpus, mp.cpu_count())

    msg = f"Solving {len(orig)} paths with {cpus} CPUs..."
    utils.log(msg, level=lg.INFO)

    # if single-threading, calculate each shortest path one at a time
    if cpus == 1:
        paths = [_single_shortest_path(G, o, d, weight) for o, d in zip(orig, dest, strict=True)]

    # if multi-threading, calculate shortest paths in parallel
    else:
        args = ((G, o, d, weight) for o, d in zip(orig, dest, strict=True))
        with mp.get_context().Pool(cpus) as pool:
            paths = pool.starmap_async(_single_shortest_path, args).get()

    return paths


def k_shortest_paths(
    G: nx.MultiDiGraph,
    orig: int,
    dest: int,
    k: int,
    *,
    weight: str = "length",
) -> Iterator[list[int]]:
    """
    Solve `k` shortest paths from an origin node to a destination node.

    Uses Yen's algorithm. See also `shortest_path` to solve just the one
    shortest path.

    Parameters
    ----------
    G
        Input graph.
    orig
        Origin node ID.
    dest
        Destination node ID.
    k
        Number of shortest paths to solve.
    weight
        Edge attribute to minimize when solving shortest paths.

    Yields
    ------
    path
        The node IDs constituting the next-shortest path.
    """
    _validate._verify_numeric_edge_attribute(G, weight, strict=False)
    paths_gen = nx.shortest_simple_paths(
        G=convert.to_digraph(G, weight=weight),
        source=orig,
        target=dest,
        weight=weight,
    )
    yield from itertools.islice(paths_gen, 0, k)


def _single_shortest_path(
    G: nx.MultiDiGraph,
    orig: int,
    dest: int,
    weight: str,
) -> list[int] | None:
    """
    Solve the shortest path from an origin node to a destination node.

    This function uses Dijkstra's algorithm. It is a convenience wrapper
    around `networkx.shortest_path`, with exception handling for unsolvable
    paths. If the path is unsolvable, it returns None.

    Parameters
    ----------
    G
        Input graph.
    orig
        Origin node ID.
    dest
        Destination node ID.
    weight
        Edge attribute to minimize when solving shortest path.

    Returns
    -------
    path
        The node IDs constituting the shortest path.
    """
    try:
        return list(nx.shortest_path(G, orig, dest, weight=weight, method="dijkstra"))
    except nx.exception.NetworkXNoPath:  # pragma: no cover
        msg = f"Cannot solve path from {orig} to {dest}"
        utils.log(msg, level=lg.WARNING)
        return None


def add_edge_speeds(
    G: nx.MultiDiGraph,
    *,
    hwy_speeds: dict[str, float] | None = None,
    hwy_groups: dict[str, Iterable[str]] | None = None,
    min_samples: int = 1,
    fallback: float | None = None,
    agg: Callable[[Any], Any] = np.mean,
) -> nx.MultiDiGraph:
    """
    Add edge speeds (km per hour) to graph as new `speed_kph` edge attributes.

    By default, this imputes free-flow travel speeds for all edges via the
    mean `maxspeed` value of the edges of each highway type. For highway types
    in the graph that have no `maxspeed` value on any edge, it assigns the
    mean of all `maxspeed` values in graph.

    This default mean-imputation can obviously be imprecise, and the user can
    override it by passing in `hwy_speeds` and/or `fallback` arguments that
    correspond to local speed limit standards. The user can also specify a
    different aggregation function (such as the median) to impute missing
    values from the observed values.

    Speed estimates come from one of three sources: 1) the edge's own observed
    `maxspeed` value, if present (marked `"observed"`); 2) a group estimate,
    aggregated from the observed speeds of all edges in the edge's road-type
    group, if that group has at least `min_samples` observed values (marked
    `"group"`); or 3) a default value (marked `"default"`). The default value
    is the edge's highway type entry in `hwy_speeds`, if declared, otherwise
    `fallback`, otherwise a global aggregate of the type-level speeds. An
    explicit per-highway-type speed in `hwy_speeds` is a user override for
    edges missing speed data, so it is applied before a group estimate. The
    source used for each edge is recorded in its new `speed_source` edge
    attribute, so callers can distinguish imputed values from measured ones
    instead of letting a single default fill mask the evidence.

    Pass `hwy_groups` to group similar highway types together (for example,
    `{"motorway": ["motorway", "motorway_link"]}`) so that a sparsely observed
    highway type can borrow evidence from its better-observed group members.
    If `hwy_groups` is None, each highway type is its own group, which
    reproduces the legacy per-highway-type behavior. Any highway type in the
    graph that is not assigned to a group is likewise treated as its own
    group. The group estimate is computed by applying `agg` to the group's
    observed speeds, so extreme values are handled according to the declared
    aggregation strategy (for example, a median is robust to outliers while a
    mean is not).

    If edge `maxspeed` attribute has "mph" in it, value will automatically be
    converted from miles per hour to km per hour. Any other speed units should
    be manually converted to km per hour prior to running this function,
    otherwise there could be unexpected results. If "mph" does not appear in
    the edge's maxspeed attribute string, then function assumes kph, per OSM
    guidelines: https://wiki.openstreetmap.org/wiki/Map_Features/Units

    If you wish to set all edge speeds to a single constant value (such as for
    a walking network), use `nx.set_edge_attributes` to set the `speed_kph`
    attribute value directly, rather than using this function.

    Parameters
    ----------
    G
        Input graph.
    hwy_speeds
        Dict keys are OSM highway types and values are typical speeds (km per
        hour) to assign to edges of that highway type for any edges missing
        speed data. Any edges with highway type not in `hwy_speeds` will be
        assigned the group estimate (or `fallback`) described above.
    hwy_groups
        Dict keys are road-type group names and values are iterables of the
        OSM highway types belonging to that group. A group estimate is the
        aggregated observed speed across all edges whose highway type belongs
        to the group. Each highway type may belong to at most one group. If
        None, each highway type is treated as its own group.
    min_samples
        Minimum number of observed speed values a group must have before its
        estimate is trusted. Groups with fewer observations are ignored and
        their edges fall back to the default value. Must be a positive
        integer. Default is 1, which trusts any group with at least one
        observation.
    fallback
        Default speed value (km per hour) to assign to edges whose highway
        type did not appear in `hwy_speeds` and had no pre-existing speed
        attribute values on any edge.
    agg
        Aggregation function to impute missing values from observed values.
        The default is `numpy.mean`, but you might also consider for example
        `numpy.median`, `numpy.nanmedian`, or your own custom function.

    Returns
    -------
    G
        Graph with `speed_kph` and `speed_source` attributes on all edges.

    Raises
    ------
    ValueError
        If `min_samples` is not a positive integer, if a highway type appears
        in more than one group in `hwy_groups`, if a group is empty, or if the
        graph's edges have no pre-existing speed values and neither
        `hwy_speeds` nor `fallback` was passed.
    TypeError
        If `hwy_groups` is not a dict or one of its groups contains a
        non-string highway type.
    """
    if fallback is None:
        fallback = np.nan

    # validate and normalize min_samples, rejecting booleans and non-integral
    # values with a clear message rather than a downstream comparison error
    if isinstance(min_samples, bool):
        msg = f"`min_samples` must be a positive integer, got {min_samples!r}."
        raise ValueError(msg)
    try:
        min_samples_int = int(min_samples)
    except (TypeError, ValueError) as err:
        msg = f"`min_samples` must be a positive integer, got {min_samples!r}."
        raise ValueError(msg) from err
    if min_samples_int != min_samples or min_samples_int < 1:
        msg = f"`min_samples` must be a positive integer, got {min_samples!r}."
        raise ValueError(msg)
    min_samples = min_samples_int

    edges = convert.graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)

    # collapse any highway lists (can happen during graph simplification)
    # into string values simply by keeping just the first element of the list
    edges["highway"] = edges["highway"].map(lambda x: x[0] if isinstance(x, list) else x)

    # clean each edge's observed maxspeed into a numeric speed in kph, leaving
    # missing or unparseable values as null
    edges["speed_kph"] = _clean_edge_maxspeeds(edges, agg)

    # map each highway type to its predefined road-type group
    highway_to_group = _build_highway_to_group(edges["highway"], hwy_groups)
    edges["_speed_group"] = edges["highway"].map(highway_to_group)

    # aggregate observed speeds within each group, but only trust an estimate
    # if its group has at least min_samples observations
    group_estimate = _aggregate_group_speeds(edges, min_samples, agg)
    edges["_group_speed"] = edges["_speed_group"].map(group_estimate)

    # declared per-highway-type default speeds, dropping nulls
    declared = (
        pd.Series(dtype=float)
        if hwy_speeds is None
        else pd.Series(hwy_speeds, dtype=float).dropna()
    )

    # compute the last-resort default value
    global_default = _global_default_speed(edges, declared, fallback, agg)

    # assemble each edge's speed and the source it came from
    speed_kph, speed_source = _select_edge_speeds(
        edges["speed_kph"], edges["_group_speed"], edges["highway"].map(declared), global_default
    )
    edges["speed_kph"] = speed_kph
    edges["speed_source"] = speed_source

    # all speeds will be null if edges had no preexisting maxspeed data and
    # caller did not pass in hwy_speeds or fallback arguments
    if pd.isna(speed_kph).all():
        msg = (
            "This graph's edges have no preexisting 'maxspeed' attribute "
            "values so you must pass `hwy_speeds` or `fallback` arguments."
        )
        raise ValueError(msg)

    # log how much of the evidence each source supplied
    counts = speed_source.value_counts()
    msg = (
        f"Imputed edge speeds from {counts.get(_SPEED_SOURCE_OBSERVED, 0)} "
        f"observed values, {counts.get(_SPEED_SOURCE_GROUP, 0)} group estimates, "
        f"and {counts.get(_SPEED_SOURCE_DEFAULT, 0)} default values"
    )
    utils.log(msg, level=lg.INFO)

    # add speed kph and provenance attributes to graph edges
    nx.set_edge_attributes(G, values=edges["speed_kph"].to_dict(), name="speed_kph")
    nx.set_edge_attributes(G, values=speed_source.to_dict(), name="speed_source")

    return G


def _clean_edge_maxspeeds(edges: gpd.GeoDataFrame, agg: Callable[[Any], Any]) -> pd.Series:
    """
    Clean each edge's observed `maxspeed` into a numeric speed in km per hour.

    Parameters
    ----------
    edges
        Edge GeoDataFrame, possibly containing a `maxspeed` column.
    agg
        Aggregation function to use when an edge's maxspeed contains multiple
        values.

    Returns
    -------
    speed_kph
        Series of observed speeds (km per hour), null where no valid value was
        present.
    """
    if "maxspeed" not in edges.columns:
        # no edges in graph had a maxspeed attribute
        return pd.Series(np.nan, index=edges.index, dtype=float)

    # collapse any maxspeed lists (can happen during graph simplification) into
    # a single value, then clean the string and convert mph to kph if necessary
    maxspeed = edges["maxspeed"].apply(_collapse_multiple_maxspeed_values, agg=agg)
    return maxspeed.astype(str).map(_clean_maxspeed).astype(float)


def _build_highway_to_group(
    highways: pd.Series,
    hwy_groups: dict[str, Iterable[str]] | None,
) -> dict[str, str]:
    """
    Map each highway type in the graph to its predefined road-type group.

    Parameters
    ----------
    highways
        Series of highway types present on the graph's edges.
    hwy_groups
        Dict keys are group names and values are iterables of highway types.
        If None, each highway type is its own group.

    Returns
    -------
    highway_to_group
        Dict keys are highway types and values are their group names.
    """
    observed = [str(hwy) for hwy in highways.dropna().unique()]

    if hwy_groups is None:
        # without explicit groups, every highway type is its own group
        groups = {hwy: [hwy] for hwy in observed}
    else:
        groups = _validate_hwy_groups(hwy_groups)
        # any highway type the caller did not assign to a group is still
        # treated as its own group so it can use its own observations
        assigned = {hwy for hwys in groups.values() for hwy in hwys}
        for hwy in observed:
            if hwy not in assigned:
                groups[hwy] = [hwy]

    return {hwy: group for group, hwys in groups.items() for hwy in hwys}


def _aggregate_group_speeds(
    edges: gpd.GeoDataFrame,
    min_samples: int,
    agg: Callable[[Any], Any],
) -> dict[str, float]:
    """
    Aggregate observed speeds within each road-type group.

    Groups with fewer than `min_samples` observations are omitted so their
    edges fall back to a default value. `agg` is applied to the non-null
    observed values so the declared strategy (including its handling of
    extreme values) governs the estimate and custom functions never see nulls.

    Parameters
    ----------
    edges
        Edge GeoDataFrame with `_speed_group` and observed `speed_kph` columns.
    min_samples
        Minimum number of observations required to trust a group estimate.
    agg
        Aggregation function.

    Returns
    -------
    group_estimate
        Dict keys are group names and values are aggregated speeds (km per
        hour).
    """
    estimates: dict[str, float] = {}
    grouped = edges.dropna(subset=["_speed_group"]).groupby("_speed_group")
    for group, grp in grouped:
        values = grp["speed_kph"].dropna()
        if len(values) >= min_samples:
            estimates[group] = float(agg(values))
    return estimates


def _global_default_speed(
    edges: gpd.GeoDataFrame,
    declared: pd.Series,
    fallback: float,
    agg: Callable[[Any], Any],
) -> float:
    """
    Compute the last-resort default speed for edges with no better evidence.

    The user's `fallback` wins if provided. Otherwise, pool the declared
    per-type speeds with the observed aggregate of every non-declared highway
    type and aggregate that pool, preserving the legacy global fallback.

    Parameters
    ----------
    edges
        Edge GeoDataFrame with a `highway` column and observed `speed_kph`.
    declared
        Series of declared per-highway-type speeds.
    fallback
        User-provided default speed, or null if none was given.
    agg
        Aggregation function.

    Returns
    -------
    global_default
        Last-resort default speed (km per hour), or null if none is available.
    """
    if not pd.isna(fallback):
        return float(fallback)

    pool = list(declared.to_numpy(dtype=float))
    for hwy, grp in edges.groupby("highway"):
        if hwy not in declared.index:
            values = grp["speed_kph"].dropna()
            if len(values) > 0:
                pool.append(float(agg(values)))

    if len(pool) == 0:
        return np.nan
    return float(agg(pd.Series(pool, dtype=float).dropna()))


def _select_edge_speeds(
    observed: pd.Series,
    group_estimate: pd.Series,
    declared: pd.Series,
    global_default: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Select each edge's speed from the available sources and label its source.

    An explicit per-highway-type speed in `hwy_speeds` is a user override for
    edges missing speed data and therefore outranks imputation, preserving its
    documented semantics. Otherwise the precedence is the edge's own observed
    value, then its group estimate, then the global default value.

    Parameters
    ----------
    observed
        Series of each edge's own observed speed (null where missing).
    group_estimate
        Series of each edge's group estimate (null where unavailable).
    declared
        Series of each edge's declared per-highway-type speed (null if none).
    global_default
        Last-resort default speed, or null if none is available.

    Returns
    -------
    speed_kph
        Series of each edge's selected speed.
    speed_source
        Series labeling each edge's source as "observed", "group", or
        "default".
    """
    speed_kph = observed.copy()
    source = pd.Series(_SPEED_SOURCE_OBSERVED, index=observed.index, dtype=object)

    mask = speed_kph.isna() & declared.notna()
    speed_kph[mask] = declared[mask]
    source[mask] = _SPEED_SOURCE_DEFAULT

    mask = speed_kph.isna() & group_estimate.notna()
    speed_kph[mask] = group_estimate[mask]
    source[mask] = _SPEED_SOURCE_GROUP

    mask = speed_kph.isna()
    speed_kph[mask] = global_default
    source[mask] = _SPEED_SOURCE_DEFAULT

    return speed_kph, source


def _validate_hwy_groups(hwy_groups: dict[str, Iterable[str]]) -> dict[str, list[str]]:
    """
    Validate a road-type grouping and normalize its members to lists.

    Parameters
    ----------
    hwy_groups
        Dict keys are road-type group names and values are iterables of the
        OSM highway types belonging to that group.

    Returns
    -------
    groups
        Dict keys are group names and values are lists of highway types.

    Raises
    ------
    TypeError
        If `hwy_groups` is not a dict, or a group's members are not an
        iterable of strings.
    ValueError
        If a group is empty, or a highway type belongs to multiple groups.
    """
    if not isinstance(hwy_groups, dict):
        msg = f"`hwy_groups` must be a dict, got {type(hwy_groups).__name__}."
        raise TypeError(msg)

    groups: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for name, members in hwy_groups.items():
        # accept a bare string as a single-member group for convenience
        if isinstance(members, str):
            members = [members]
        try:
            member_list = list(members)
        except TypeError as err:
            msg = f"Members of group {name!r} must be an iterable of strings."
            raise TypeError(msg) from err

        if len(member_list) == 0:
            msg = f"Group {name!r} must contain at least one highway type."
            raise ValueError(msg)

        for hwy in member_list:
            if not isinstance(hwy, str):
                msg = f"Highway types in group {name!r} must be strings, got {hwy!r}."
                raise TypeError(msg)
            if hwy in seen:
                msg = (
                    f"Highway type {hwy!r} appears in multiple groups "
                    f"({seen[hwy]!r} and {name!r})."
                )
                raise ValueError(msg)
            seen[hwy] = name

        groups[name] = member_list

    return groups


def add_edge_travel_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add edge travel time (seconds) to graph as new `travel_time` edge attributes.

    Calculates free-flow travel time along each edge, based on `length` and
    `speed_kph` attributes. Note: run `add_edge_speeds` first to generate the
    `speed_kph` attribute. All edges must have `length` and `speed_kph`
    attributes and all their values must be non-null.

    If `add_edge_speeds` recorded a `speed_source` attribute on the edges
    (`"observed"`, `"group"`, or `"default"`), this function copies it to a
    new `travel_time_source` attribute so the provenance of each edge's travel
    time remains traceable through the same three tiers.

    Parameters
    ----------
    G
        Input graph.

    Returns
    -------
    G
        Graph with `travel_time` attributes on all edges, and
        `travel_time_source` attributes when a `speed_source` attribute exists.
    """
    edges = convert.graph_to_gdfs(G, nodes=False)

    # verify edge length and speed_kph attributes exist
    if not ("length" in edges.columns and "speed_kph" in edges.columns):  # pragma: no cover
        msg = "All edges must have 'length' and 'speed_kph' attributes."
        raise KeyError(msg)

    # verify edge length and speed_kph attributes contain no nulls
    if pd.isna(edges["length"]).any() or pd.isna(edges["speed_kph"]).any():  # pragma: no cover
        msg = "Edge 'length' and 'speed_kph' values must be non-null."
        raise ValueError(msg)

    # convert distance meters to km, and speed km per hour to km per second
    distance_km = edges["length"] / 1000
    speed_km_sec = edges["speed_kph"] / (60 * 60)

    # calculate edge travel time in seconds
    travel_time = distance_km / speed_km_sec

    # carry the speed provenance through to the travel time, if available
    if "speed_source" in edges.columns:
        nx.set_edge_attributes(
            G, values=edges["speed_source"].to_dict(), name="travel_time_source"
        )

    # add travel time attribute to graph edges
    edges["travel_time"] = travel_time.to_numpy()
    nx.set_edge_attributes(G, values=edges["travel_time"].to_dict(), name="travel_time")

    return G


def _clean_maxspeed(
    maxspeed: str | float,
    *,
    agg: Callable[[Any], Any] = np.mean,
    convert_mph: bool = True,
) -> float | None:
    """
    Clean a maxspeed string and convert mph to kph if necessary.

    If present, splits maxspeed on "|" (which denotes that the value contains
    different speeds per lane) then aggregates the resulting values. If given
    string is not a valid numeric string, tries to look up its value in
    implicit maxspeed values mapping. Invalid inputs return None. See
    https://wiki.openstreetmap.org/wiki/Key:maxspeed for details on values and
    formats.

    Parameters
    ----------
    maxspeed
        An OSM way "maxspeed" attribute value. Null values are expected to be
        of type float (`numpy.nan`), and non-null values are strings.
    agg
        Aggregation function if `maxspeed` contains multiple values (default
        is `numpy.mean`).
    convert_mph
        If True, convert miles per hour to kilometers per hour.

    Returns
    -------
    clean_value
        Clean value resulting from `agg` function.
    """
    MILES_TO_KM = 1.60934
    if not isinstance(maxspeed, str):
        return None

    # regex adapted from OSM wiki
    pattern = "^([0-9][\\.,0-9]*?)(?:[ ]?(?:km/h|kmh|kph|mph|knots))?$"
    values = re.split(r"\|", maxspeed)  # creates a list even if it's a single value
    try:
        clean_values = []
        for value in values:
            match = re.match(pattern, value)
            clean_value = float(match.group(1).replace(",", "."))  # type: ignore[union-attr]
            if convert_mph and "mph" in maxspeed.lower():
                clean_value = clean_value * MILES_TO_KM
            clean_values.append(clean_value)
        return float(agg(clean_values))

    except (ValueError, AttributeError):
        # if not valid numeric string, try looking it up as implicit value
        return _IMPLICIT_MAXSPEEDS.get(maxspeed)


def _collapse_multiple_maxspeed_values(
    value: str | float | list[str | float],
    agg: Callable[[Any], Any],
) -> float | str | None:
    """
    Collapse a list of maxspeed values to a single value.

    Returns None if a ValueError is encountered.

    Parameters
    ----------
    value
        An OSM way "maxspeed" attribute value. Null values are expected to be
        of type float (`numpy.nan`), and non-null values are strings.
    agg
        The aggregation function to reduce the list to a single value.

    Returns
    -------
    collapsed
        If `value` was a string or null, it is just returned directly.
        Otherwise, the return is a float representation of the aggregated
        value in the list (converted to kph if original value was in mph).
    """
    # if this isn't a list, just return it right back to the caller
    if not isinstance(value, list):
        return value

    # otherwise, it is a list, so process it
    try:
        # clean/convert each value in list as needed then aggregate
        values = [_clean_maxspeed(x) for x in value]
        collapsed: float | None = float(agg(pd.Series(values).dropna()))
    except ValueError:
        return None
    else:
        # return that single aggregated value if it's non-null, otherwise None
        if not pd.isna(collapsed):
            return collapsed
        return None
