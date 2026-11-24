"""Calculate edge speeds, travel times, and weighted shortest paths."""

from __future__ import annotations

import heapq
import itertools
import logging as lg
import multiprocessing as mp
import re
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
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

# A directed edge's identity in a MultiDiGraph: (u, v, key). Node IDs are ints,
# but edge keys are not necessarily ints (GraphML, for example, can load strings).
EdgeKey = tuple[int, int, Any]

# A directed edge's identity optionally without its key, so that callers can
# refer to all parallel edges between the same endpoints: (u, v) or (u, v, key).
EdgeSpec = EdgeKey | tuple[int, int]

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

    If `route` is a `TurnRoute`, this uses the exact edge identities that the
    route traversed. Parallel edges between the same nodes are then
    distinguished, and the returned GeoDataFrame has the route's `travel_time`,
    `wait_time`, and `cost` per edge added as columns. For a plain list of
    nodes, this chooses between parallel edges by minimizing `weight`.

    Parameters
    ----------
    G
        Input graph.
    route
        Node IDs constituting the path, or a `TurnRoute` from
        `shortest_path_with_turns`.
    weight
        Attribute value to minimize when choosing between parallel edges.

    Returns
    -------
    gdf_edges
        The ordered edges in the path.
    """
    if isinstance(route, TurnRoute):
        return route.to_gdf(G)

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


# turn restrictions or waiting costs provided, orig/dest int
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: int,
    dest: int,
    *,
    weight: str = ...,
    cpus: int | None = ...,
    turn_restrictions: Mapping[EdgeSpec, Any] | Iterable[tuple[EdgeSpec, EdgeSpec]] | None = ...,
    edge_wait_times: Mapping[EdgeSpec, float] | None = ...,
    turn_wait_times: Mapping[Any, float] | None = ...,
    node_wait_times: Mapping[int, float] | None = ...,
    waiting_costs: Mapping[Any, float] | None = ...,
    default_turn_wait_time: float = ...,
) -> TurnRoute | None: ...


# turn restrictions or waiting costs provided, orig/dest Iterable
@overload
def shortest_path(
    G: nx.MultiDiGraph,
    orig: Iterable[int],
    dest: Iterable[int],
    *,
    weight: str = ...,
    cpus: int | None = ...,
    turn_restrictions: Mapping[EdgeSpec, Any] | Iterable[tuple[EdgeSpec, EdgeSpec]] | None = ...,
    edge_wait_times: Mapping[EdgeSpec, float] | None = ...,
    turn_wait_times: Mapping[Any, float] | None = ...,
    node_wait_times: Mapping[int, float] | None = ...,
    waiting_costs: Mapping[Any, float] | None = ...,
    default_turn_wait_time: float = ...,
) -> list[TurnRoute | None]: ...


def shortest_path(  # noqa: PLR0913
    G: nx.MultiDiGraph,
    orig: int | Iterable[int],
    dest: int | Iterable[int],
    *,
    weight: str = "length",
    cpus: int | None = 1,
    turn_restrictions: Mapping[EdgeSpec, Any] | Iterable[tuple[EdgeSpec, EdgeSpec]] | None = None,
    edge_wait_times: Mapping[EdgeSpec, float] | None = None,
    turn_wait_times: Mapping[Any, float] | None = None,
    node_wait_times: Mapping[int, float] | None = None,
    waiting_costs: Mapping[Any, float] | None = None,
    default_turn_wait_time: float = 0.0,
) -> TurnRoute | list[TurnRoute | None] | list[int] | list[list[int] | None] | None:
    """
    Solve shortest path from origin node(s) to destination node(s).

    Uses Dijkstra's algorithm. If `orig` and `dest` are single node IDs, this
    will return a list of the nodes constituting the shortest path between
    them. If `orig` and `dest` are lists of node IDs, this will return a list
    of lists of the nodes constituting the shortest path between each
    origin-destination pair. If a path cannot be solved, this will return None
    for that path. You can parallelize solving multiple paths with the `cpus`
    parameter, but be careful to not exceed your available RAM.

    If any turn restriction or waiting cost argument is provided, this solves
    over edge identities instead of nodes (see `shortest_path_with_turns`) and
    returns `TurnRoute` objects, so turns may be forbidden and waiting costs
    applied per turn, edge, or node. Paths are then solved serially.

    See also `k_shortest_paths` to solve multiple shortest paths between a
    single origin and destination, and `shortest_path_with_turns` to honor turn
    restrictions and waiting costs. For additional functionality or different
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
    turn_restrictions
        Turns that the path may not make, as in `shortest_path_with_turns`.
    edge_wait_times
        Per-edge waiting costs, as in `shortest_path_with_turns`.
    turn_wait_times
        Per-turn waiting costs, as in `shortest_path_with_turns`.
    node_wait_times
        Per-node waiting costs, as in `shortest_path_with_turns`.
    waiting_costs
        Combined per-turn, per-edge, or per-node waiting costs, as in
        `shortest_path_with_turns`.
    default_turn_wait_time
        Default waiting cost for turns, as in `shortest_path_with_turns`.

    Returns
    -------
    path
        The node IDs constituting the shortest path, or, if `orig` and `dest`
        are both iterable, then a list of such paths. If any turn restriction
        or waiting cost argument is provided, each path is a `TurnRoute`.
    """
    use_turns = (
        turn_restrictions is not None
        or edge_wait_times is not None
        or turn_wait_times is not None
        or node_wait_times is not None
        or waiting_costs is not None
        or default_turn_wait_time != 0
    )
    if use_turns:
        return _shortest_paths_with_turns(
            G,
            orig,
            dest,
            weight,
            turn_restrictions=turn_restrictions,
            edge_wait_times=edge_wait_times,
            turn_wait_times=turn_wait_times,
            node_wait_times=node_wait_times,
            waiting_costs=waiting_costs,
            default_turn_wait_time=default_turn_wait_time,
        )

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


def _shortest_paths_with_turns(  # noqa: PLR0913
    G: nx.MultiDiGraph,
    orig: int | Iterable[int],
    dest: int | Iterable[int],
    weight: str,
    *,
    turn_restrictions: Mapping[EdgeSpec, Any] | Iterable[tuple[EdgeSpec, EdgeSpec]] | None,
    edge_wait_times: Mapping[EdgeSpec, float] | None,
    turn_wait_times: Mapping[Any, float] | None,
    node_wait_times: Mapping[int, float] | None,
    waiting_costs: Mapping[Any, float] | None,
    default_turn_wait_time: float,
) -> TurnRoute | list[TurnRoute | None] | None:
    """
    Solve one or more turn-aware shortest paths from single or iterable nodes.

    This is the turn-aware counterpart to `shortest_path`, dispatching to
    `shortest_path_with_turns` for each origin/destination pair. It validates
    that `orig` and `dest` are both iterable or both scalars, as `shortest_path`
    does.

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
    turn_restrictions
        Turns that the path may not make.
    edge_wait_times
        Per-edge waiting costs.
    turn_wait_times
        Per-turn waiting costs.
    node_wait_times
        Per-node waiting costs.
    waiting_costs
        Combined per-turn, per-edge, or per-node waiting costs.
    default_turn_wait_time
        Default waiting cost for turns.

    Returns
    -------
    path or paths
        A `TurnRoute`, or a list of them in origin/destination order.
    """
    solver_kwargs: dict[str, Any] = {
        "weight": weight,
        "turn_restrictions": turn_restrictions,
        "edge_wait_times": edge_wait_times,
        "turn_wait_times": turn_wait_times,
        "node_wait_times": node_wait_times,
        "waiting_costs": waiting_costs,
        "default_turn_wait_time": default_turn_wait_time,
    }

    # if neither orig nor dest is iterable, just return the one path
    if not (isinstance(orig, Iterable) or isinstance(dest, Iterable)):
        return shortest_path_with_turns(G, orig, dest, **solver_kwargs)

    # if only 1 of orig or dest is iterable and the other is not, raise error
    if not (isinstance(orig, Iterable) and isinstance(dest, Iterable)):
        msg = "`orig` and `dest` must either both be iterable or neither must be iterable."
        raise TypeError(msg)

    orig = list(orig)
    dest = list(dest)
    if len(orig) != len(dest):
        msg = "`orig` and `dest` must be of equal length."
        raise ValueError(msg)

    return [
        shortest_path_with_turns(G, o, d, **solver_kwargs)
        for o, d in zip(orig, dest, strict=True)
    ]


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


class TurnRoute(list[int]):
    """
    A path solved while honoring turn restrictions and waiting costs.

    This is a list of the path's node IDs, so it can be used anywhere a plain
    node path is expected, such as in `route_to_gdf`, a plot function, or an
    equality comparison against a list of nodes. It also carries the path's
    edge-level detail as attributes, so a path can be reported segment by
    segment with each segment's time and waiting time.

    Attributes
    ----------
    nodes
        The node IDs constituting the path, in order. This returns a list of
        the route's nodes, the same content as the list itself.
    edges
        The edge identities `(u, v, key)` traversed by the path, in order.
        Parallel edges between the same pair of nodes remain distinct here.
    travel_times
        Travel time of each edge in the path, in whatever units the graph's
        `travel_time` attribute uses (seconds if it came from
        `add_edge_travel_times`). If an edge has no `travel_time` attribute, its
        `weight` value is used instead.
    wait_times
        Time spent waiting immediately before traversing each edge: the waiting
        cost of the turn onto that edge plus the waiting cost attached to that
        edge or its tail node (if any). The first edge has no preceding turn.
    costs
        The amount each edge contributed to the minimized objective: the edge's
        `weight` value plus its waiting cost (including any turn waiting cost).

    The `travel_times`, `wait_times`, and `costs` lists are each the same length
    as `edges`, so they are the per-segment times that sum to the totals.
    """

    edges: list[EdgeKey]
    travel_times: list[float]
    wait_times: list[float]
    costs: list[float]

    def __init__(
        self,
        nodes: Iterable[int],
        *,
        edges: Iterable[EdgeKey] | None = None,
        travel_times: Iterable[float] | None = None,
        wait_times: Iterable[float] | None = None,
        costs: Iterable[float] | None = None,
    ) -> None:
        """
        Create a turn-aware route from its nodes and optional edge detail.

        Parameters
        ----------
        nodes
            The node IDs constituting the path, in order.
        edges
            The edge identities `(u, v, key)` traversed, in order.
        travel_times
            Travel time of each edge, in order.
        wait_times
            Waiting time before traversing each edge, in order.
        costs
            The amount each edge contributed to the minimized objective.
        """
        super().__init__(nodes)
        self.edges = list(edges) if edges is not None else []
        self.travel_times = list(travel_times) if travel_times is not None else []
        self.wait_times = list(wait_times) if wait_times is not None else []
        self.costs = list(costs) if costs is not None else []

    @property
    def nodes(self) -> list[int]:
        """Return the path's node IDs, in order, as a list of its elements."""
        return list(self)

    @property
    def times(self) -> list[float]:
        """Return the path's per-edge travel times (alias of `travel_times`)."""
        return self.travel_times

    @property
    def durations(self) -> list[float]:
        """Return the path's per-edge travel times (alias of `travel_times`)."""
        return self.travel_times

    @property
    def waits(self) -> list[float]:
        """Return the path's per-edge waiting times (alias of `wait_times`)."""
        return self.wait_times

    @property
    def waiting_times(self) -> list[float]:
        """Return the path's per-edge waiting times (alias of `wait_times`)."""
        return self.wait_times

    @property
    def segments(self) -> list[dict[str, Any]]:
        """
        Return the path's per-edge details, one dict per edge, in path order.

        Each dict has "u", "v", and "key" edge identity values, and "time",
        "wait", and "cost" values for that edge.
        """
        return [
            {
                "u": edge[0],
                "v": edge[1],
                "key": edge[2],
                "time": time,
                "wait": wait,
                "cost": cost,
            }
            for edge, time, wait, cost in zip(
                self.edges,
                self.travel_times,
                self.wait_times,
                self.costs,
                strict=True,
            )
        ]

    @property
    def total_travel_time(self) -> float:
        """Return the sum of the path's edge travel times."""
        return float(sum(self.travel_times))

    @property
    def total_wait_time(self) -> float:
        """Return the sum of the path's waiting times."""
        return float(sum(self.wait_times))

    @property
    def total_time(self) -> float:
        """Return the path's total time: all travel times plus all waiting."""
        return self.total_travel_time + self.total_wait_time

    @property
    def total_cost(self) -> float:
        """Return the sum of the path's per-edge costs as minimized."""
        return float(sum(self.costs))

    def to_gdf(self, G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
        """
        Return the path's edges as a GeoDataFrame with their times and waits.

        The GeoDataFrame is multi-indexed by `(u, v, key)` in path order, and
        has `travel_time`, `wait_time`, and `cost` columns added to each edge.

        Parameters
        ----------
        G
            Input graph that this path was solved on.

        Returns
        -------
        gdf_edges
            The ordered edges in the path, with their time and wait columns.
        """
        import geopandas as gpd  # noqa: PLC0415

        columns = ["travel_time", "wait_time", "cost"]
        if len(self.edges) == 0:
            index = pd.MultiIndex.from_arrays([[], [], []], names=["u", "v", "key"])
            return gpd.GeoDataFrame(
                dict.fromkeys(columns, pd.Series(dtype=float)),
                index=index,
                geometry=[],
                crs=G.graph.get("crs"),
            )

        gdf = convert.graph_to_gdfs(G.edge_subgraph(self.edges), nodes=False)
        gdf = gdf.loc[list(self.edges)]
        gdf["travel_time"] = self.travel_times
        gdf["wait_time"] = self.wait_times
        gdf["cost"] = self.costs
        return gdf


def _is_edge_spec(value: Any) -> bool:
    """
    Check whether a value is an edge identity or an endpoint pair.

    A tuple or list `(u, v)` or `(u, v, key)` is an edge spec; a tuple or list of
    two such specs is a turn, not an edge spec.

    Parameters
    ----------
    value
        The value to check.

    Returns
    -------
    is_edge_spec
        True if the value is a 2- or 3-item tuple or list of scalars (that is,
        not a tuple or list of edge specs).
    """
    if not isinstance(value, (tuple, list)) or len(value) not in (2, 3):
        return False
    return not any(isinstance(x, (tuple, list)) for x in value)


def _edge_matches(spec: EdgeSpec, edge: EdgeKey) -> bool:
    """
    Check whether a concrete edge identity matches an edge spec.

    A 3-tuple spec identifies one specific edge, including among parallel
    edges. A 2-tuple spec identifies all parallel edges between its endpoints.

    Parameters
    ----------
    spec
        The edge spec to match: `(u, v)` or `(u, v, key)`.
    edge
        The concrete edge identity: `(u, v, key)`.

    Returns
    -------
    matches
        True if the spec applies to the edge.
    """
    spec = tuple(spec)
    return spec == edge if len(spec) == 3 else spec == edge[:2]


def _spec_specificity(spec: EdgeSpec) -> int:
    """
    Return an edge spec's specificity: a keyed edge is more specific than a pair.

    Parameters
    ----------
    spec
        The edge spec: `(u, v)` or `(u, v, key)`.

    Returns
    -------
    specificity
        3 if the spec includes the edge key, else 2.
    """
    return len(spec)


def _finite_non_negative(value: Any, name: str) -> float:
    """
    Return a value as a finite, non-negative float or raise an error.

    Parameters
    ----------
    value
        The value to validate.
    name
        A human-readable name of the value for the error message.

    Returns
    -------
    result
        The value as a float.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as e:
        msg = f"{name} must be a finite non-negative number: {value!r}."
        raise ValueError(msg) from e
    if not np.isfinite(result) or result < 0:
        msg = f"{name} must be a finite non-negative number: {value!r}."
        raise ValueError(msg)
    return result


def _iter_turn_restrictions(
    turn_restrictions: Mapping[EdgeSpec, Any] | Iterable[tuple[EdgeSpec, EdgeSpec]],
) -> Iterator[tuple[EdgeSpec, Iterator[EdgeSpec]]]:
    """
    Yield `(from_spec, to_specs)` pairs from caller-provided turn restrictions.

    Accepts either a mapping from an incoming edge spec to one forbidden
    outgoing edge spec (or an iterable of them), or an iterable of
    `(from_spec, to_spec)` pairs.

    Parameters
    ----------
    turn_restrictions
        The turn restrictions to normalize.

    Yields
    ------
    from_spec
        The forbidden incoming edge spec.
    to_specs
        An iterator of the forbidden outgoing edge specs.
    """
    if isinstance(turn_restrictions, Mapping):
        for from_spec, to_spec in turn_restrictions.items():
            if _is_edge_spec(to_spec):
                yield from_spec, iter((to_spec,))
            else:
                yield from_spec, iter(tuple(to_spec))
    else:
        for from_spec, to_spec in turn_restrictions:
            yield from_spec, iter((to_spec,))


def _resolve_turn_restrictions(
    G: nx.MultiDiGraph,
    turn_restrictions: Mapping[EdgeSpec, Any] | Iterable[tuple[EdgeSpec, EdgeSpec]] | None,
) -> dict[EdgeKey, set[EdgeKey]]:
    """
    Resolve edge-spec turn restrictions to concrete forbidden edge pairs.

    Each restriction forbids turning from an incoming edge onto an outgoing
    edge that starts where the incoming edge ends. Restrictions are matched by
    edge identity, so a restriction on one of several parallel edges does not
    affect the others unless the spec omits the key.

    Parameters
    ----------
    G
        Input graph.
    turn_restrictions
        Mapping or iterable of forbidden incoming/outgoing edge specs. None
        means no restrictions.

    Returns
    -------
    forbidden
        Dict of concrete incoming edge identity to the set of concrete outgoing
        edge identities that it may not turn onto.
    """
    if turn_restrictions is None:
        return {}

    all_edges = list(G.edges(keys=True))
    out_edges: dict[int, list[EdgeKey]] = defaultdict(list)
    for edge in all_edges:
        out_edges[edge[0]].append(edge)

    forbidden: dict[EdgeKey, set[EdgeKey]] = defaultdict(set)
    for from_spec, to_specs_iter in _iter_turn_restrictions(turn_restrictions):
        to_specs = list(to_specs_iter)
        for spec in (from_spec, *to_specs):
            if not any(_edge_matches(spec, e) for e in all_edges):
                msg = f"Turn restriction references a nonexistent edge: {spec!r}."
                raise ValueError(msg)

        for from_edge in all_edges:
            if not _edge_matches(from_spec, from_edge):
                continue
            for out_edge in out_edges.get(from_edge[1], ()):
                if any(_edge_matches(to_spec, out_edge) for to_spec in to_specs):
                    forbidden[from_edge].add(out_edge)

    return dict(forbidden)


def _resolve_edge_wait_times(
    G: nx.MultiDiGraph,
    edge_wait_times: Mapping[EdgeSpec, float] | None,
) -> dict[EdgeKey, float]:
    """
    Resolve edge-spec waiting costs to concrete per-edge waiting costs.

    Parameters
    ----------
    G
        Input graph.
    edge_wait_times
        Mapping of edge spec to the time spent waiting before traversing it.
        None means no waiting costs.

    Returns
    -------
    waits
        Dict of concrete edge identity to its waiting cost.
    """
    if edge_wait_times is None:
        return {}

    all_edges = list(G.edges(keys=True))
    waits: dict[EdgeKey, float] = {}
    for spec, value in edge_wait_times.items():
        matches = [e for e in all_edges if _edge_matches(spec, e)]
        if len(matches) == 0:
            msg = f"Edge waiting cost references a nonexistent edge: {spec!r}."
            raise ValueError(msg)
        wait = _finite_non_negative(value, "Edge waiting cost")
        for edge in matches:
            waits[edge] = wait

    return waits


def _resolve_node_wait_times(
    G: nx.MultiDiGraph,
    node_wait_times: Mapping[int, float] | None,
) -> dict[int, float]:
    """
    Resolve node waiting costs, such as waiting at a signal or a stop.

    Parameters
    ----------
    G
        Input graph.
    node_wait_times
        Mapping of node ID to the time spent waiting there before departing.
        None means no waiting costs.

    Returns
    -------
    waits
        Dict of node ID to its waiting cost.
    """
    if node_wait_times is None:
        return {}

    waits: dict[int, float] = {}
    for node, value in node_wait_times.items():
        if node not in G:
            msg = f"Node waiting cost references a nonexistent node: {node!r}."
            raise ValueError(msg)
        waits[node] = _finite_non_negative(value, "Node waiting cost")
    return waits


def _split_waiting_costs(
    G: nx.MultiDiGraph,
    waiting_costs: Mapping[Any, float] | None,
) -> tuple[dict[EdgeSpec, float], dict[int, float], dict[tuple[EdgeSpec, EdgeSpec], float]]:
    """
    Split a combined mapping of waiting costs by the kind of key it uses.

    This lets a caller pass per-edge, per-node, and per-turn waiting costs in a
    single mapping. A key is interpreted as:

    - a node ID if it is not a tuple, so its waiting cost applies to departing
      that node,
    - a pair of edge specs if it is a tuple of two edge specs, so its waiting
      cost applies to that specific turn, or
    - an edge spec, so its waiting cost applies to that edge.

    Parameters
    ----------
    G
        Input graph.
    waiting_costs
        Mapping of edge, node, or turn to a waiting cost. None means none.

    Returns
    -------
    edge_waits
        Waiting costs keyed by edge spec.
    node_waits
        Waiting costs keyed by node ID.
    turn_waits
        Waiting costs keyed by incoming/outgoing edge spec pairs.
    """
    if waiting_costs is None:
        return {}, {}, {}

    edge_waits: dict[EdgeSpec, float] = {}
    node_waits: dict[int, float] = {}
    turn_waits: dict[tuple[EdgeSpec, EdgeSpec], float] = {}
    for key, value in waiting_costs.items():
        if not isinstance(key, tuple):
            if key not in G:
                msg = f"Waiting cost references a nonexistent node: {key!r}."
                raise ValueError(msg)
            node_waits[key] = _finite_non_negative(value, "Node waiting cost")
        elif len(key) == 2 and all(_is_edge_spec(x) for x in key):
            turn_waits[key] = _finite_non_negative(value, "Turn waiting cost")
        elif _is_edge_spec(key):
            edge_waits[key] = _finite_non_negative(value, "Edge waiting cost")
        else:
            msg = f"A waiting cost key is not a node, edge spec, or turn: {key!r}."
            raise TypeError(msg)

    return edge_waits, node_waits, turn_waits


def _edge_attribute_waits(G: nx.MultiDiGraph) -> dict[EdgeKey, float]:
    """
    Read explicit per-edge waiting costs from an edge attribute, if present.

    If an edge has a "wait_time" attribute, it is used as that edge's waiting
    cost unless a more specific waiting cost is provided for it. This lets a
    caller store local waiting costs on the graph itself.

    Parameters
    ----------
    G
        Input graph.

    Returns
    -------
    waits
        Dict of concrete edge identity to its waiting cost from the graph.
    """
    waits: dict[EdgeKey, float] = {}
    for edge in G.edges(keys=True):
        value = G.edges[edge].get("wait_time", None)
        if value is None or pd.isna(value):
            continue
        waits[edge] = _finite_non_negative(value, f"The wait_time of edge {edge!r}")
    return waits


def _resolve_turn_wait_times(
    G: nx.MultiDiGraph,
    turn_wait_times: Mapping[Any, float] | None,
) -> tuple[dict[tuple[EdgeKey, EdgeKey], float], dict[EdgeKey, float]]:
    """
    Resolve turn waiting costs to concrete per-turn and per-incoming-edge costs.

    Keys may be either a pair of edge specs `(from_spec, to_spec)` to cost one
    specific turn, or a single incoming edge spec to cost every turn out of
    that edge. Specs are resolved to concrete edge identities here, once, so
    the solver only needs dict lookups per transition.

    Parameters
    ----------
    G
        Input graph.
    turn_wait_times
        Mapping of turn or incoming edge spec to a waiting cost. None means no
        waiting costs.

    Returns
    -------
    pair_waits
        Waiting costs keyed by concrete incoming/outgoing edge identity pairs.
    single_waits
        Waiting costs keyed by a concrete incoming edge identity, applying to
        all turns out of it.
    """
    if turn_wait_times is None:
        return {}, {}

    all_edges = list(G.edges(keys=True))
    out_edges: dict[int, list[EdgeKey]] = defaultdict(list)
    for edge in all_edges:
        out_edges[edge[0]].append(edge)

    pair_specs: list[tuple[EdgeSpec, EdgeSpec, float]] = []
    single_specs: list[tuple[EdgeSpec, float]] = []

    for key, value in turn_wait_times.items():
        wait = _finite_non_negative(value, "Turn waiting cost")
        if _is_edge_spec(key):
            if not any(_edge_matches(key, e) for e in all_edges):
                msg = f"Turn waiting cost references a nonexistent edge: {key!r}."
                raise ValueError(msg)
            single_specs.append((key, wait))
        elif isinstance(key, tuple) and len(key) == 2:
            from_spec, to_spec = key
            for spec in (from_spec, to_spec):
                if not _is_edge_spec(spec):
                    msg = f"Turn waiting cost must pair edge specs, not {key!r}."
                    raise TypeError(msg)
                if not any(_edge_matches(spec, e) for e in all_edges):
                    msg = f"Turn waiting cost references a nonexistent edge: {spec!r}."
                    raise ValueError(msg)
            pair_specs.append((from_spec, to_spec, wait))
        else:
            msg = (
                "A turn waiting cost key must be an edge spec (u, v) or (u, v, key), "
                f"or a pair of edge specs, not {key!r}."
            )
            raise TypeError(msg)

    single_waits: dict[EdgeKey, float] = {}
    for from_edge in all_edges:
        match: tuple[int, float] | None = None
        for from_spec, wait in single_specs:
            if _edge_matches(from_spec, from_edge):
                specificity = _spec_specificity(from_spec)
                if match is None or specificity > match[0]:
                    match = (specificity, wait)
        if match is not None:
            single_waits[from_edge] = match[1]

    pair_waits: dict[tuple[EdgeKey, EdgeKey], float] = {}
    for from_edge in all_edges:
        for to_edge in out_edges.get(from_edge[1], ()):
            match: tuple[int, float] | None = None
            for from_spec, to_spec, wait in pair_specs:
                if _edge_matches(from_spec, from_edge) and _edge_matches(to_spec, to_edge):
                    specificity = _spec_specificity(from_spec) + _spec_specificity(to_spec)
                    if match is None or specificity > match[0]:
                        match = (specificity, wait)
            if match is not None:
                pair_waits[(from_edge, to_edge)] = match[1]

    return pair_waits, single_waits


def _turn_wait_time(
    from_edge: EdgeKey,
    to_edge: EdgeKey,
    pair_waits: Mapping[tuple[EdgeKey, EdgeKey], float],
    single_waits: Mapping[EdgeKey, float],
    default_turn_wait_time: float,
) -> float:
    """
    Return the waiting cost of turning from one edge onto another.

    A waiting cost for the specific turn takes precedence over a waiting cost
    for all turns out of the incoming edge, which takes precedence over the
    default turn waiting cost.

    Parameters
    ----------
    from_edge
        The concrete incoming edge identity.
    to_edge
        The concrete outgoing edge identity.
    pair_waits
        Waiting costs keyed by concrete incoming/outgoing edge identity pairs.
    single_waits
        Waiting costs keyed by a concrete incoming edge identity.
    default_turn_wait_time
        The waiting cost for any turn without a more specific waiting cost.

    Returns
    -------
    wait
        The turn's waiting cost.
    """
    if (from_edge, to_edge) in pair_waits:
        return pair_waits[(from_edge, to_edge)]
    return single_waits.get(from_edge, default_turn_wait_time)


def _edge_weight(G: nx.MultiDiGraph, edge: EdgeKey, weight: str) -> float:
    """
    Return an edge's finite, numeric weight attribute value.

    Parameters
    ----------
    G
        Input graph.
    edge
        The concrete edge identity.
    weight
        The edge attribute name to use as the edge's weight.

    Returns
    -------
    value
        The edge's weight value.
    """
    data = G.edges[edge]
    if weight not in data:
        msg = f"Edge {edge!r} has no {weight!r} value."
        raise ValueError(msg)
    try:
        value = float(data[weight])
    except (TypeError, ValueError) as e:
        msg = f"Edge {edge!r} has a non-numeric {weight!r} value."
        raise ValueError(msg) from e
    if not np.isfinite(value):
        msg = f"Edge {edge!r} has a missing or non-finite {weight!r} value."
        raise ValueError(msg)
    return value


def _edge_travel_time(G: nx.MultiDiGraph, edge: EdgeKey, weight: str) -> float:
    """
    Return an edge's travel time, falling back to its weight if it has none.

    Parameters
    ----------
    G
        Input graph.
    edge
        The concrete edge identity.
    weight
        The edge attribute name to use as the fallback travel time.

    Returns
    -------
    value
        The edge's travel time.
    """
    data = G.edges[edge]
    if "travel_time" in data and pd.notna(data["travel_time"]):
        return float(data["travel_time"])
    return _edge_weight(G, edge, weight)


def shortest_path_with_turns(  # noqa: PLR0913
    G: nx.MultiDiGraph,
    orig: int,
    dest: int,
    *,
    weight: str = "length",
    turn_restrictions: Mapping[EdgeSpec, Any] | Iterable[tuple[EdgeSpec, EdgeSpec]] | None = None,
    edge_wait_times: Mapping[EdgeSpec, float] | None = None,
    turn_wait_times: Mapping[Any, float] | None = None,
    node_wait_times: Mapping[int, float] | None = None,
    waiting_costs: Mapping[Any, float] | None = None,
    default_turn_wait_time: float = 0.0,
) -> TurnRoute | None:
    """
    Solve the shortest path that honors turn restrictions and waiting costs.

    Unlike `shortest_path`, which minimizes an edge weight over nodes, this
    solves over edge identities `(u, v, key)`. The legality and cost of a turn
    may therefore depend on both the incoming and the outgoing edges of a path,
    which is what turn restrictions need. This is useful, for example, when
    roadworks close one approach to a junction: forbid just that turn and the
    path detours around it. Waiting costs can be attached to a specific turn,
    to an edge, or to a node such as a signal or a stop. Parallel edges between
    the same pair of nodes are distinct states, so each one can carry its own
    restriction, cost, and waiting time.

    Turn restrictions and waiting costs are matched by edge identity. Use a
    `(u, v, key)` spec to affect one specific edge, or a `(u, v)` spec to affect
    every parallel edge between those endpoints. A restriction only applies
    when the outgoing edge starts where the incoming edge ends.

    The objective minimized is the sum over the path's edges of the edge's
    `weight` value plus its waiting costs: the waiting cost attached to the
    edge itself, the waiting cost of its tail node, and, for every edge after
    the first, the waiting cost of the turn onto it. Waiting time is thus
    additive with travel time as measured by `weight`. Use `travel_time` as the
    weight to minimize total time when the waiting costs are in the same units.

    Parameters
    ----------
    G
        Input graph.
    orig
        Origin node ID.
    dest
        Destination node ID.
    weight
        Edge attribute to minimize when solving the shortest path. It must be
        numeric, finite, and non-negative on every traversed edge.
    turn_restrictions
        Turns that the path may not make. Either a mapping of an incoming edge
        spec to one forbidden outgoing edge spec (or an iterable of them), or an
        iterable of `(from_spec, to_spec)` pairs.
    edge_wait_times
        Mapping of an edge spec to the time spent waiting before traversing
        that edge, such as waiting to enter it.
    turn_wait_times
        Mapping of waiting costs for turns. A key may be a pair
        `(from_spec, to_spec)` to cost one specific turn, or a single incoming
        `from_spec` to cost every turn out of that edge.
    node_wait_times
        Mapping of a node ID to the time spent waiting there before departing,
        such as waiting at a signal or a stop.
    waiting_costs
        Combined mapping of waiting costs, for convenience. Its keys may be
        edge specs, node IDs, or pairs of edge specs for a specific turn, and
        are merged with `edge_wait_times`, `turn_wait_times`, and
        `node_wait_times`.
    default_turn_wait_time
        Waiting cost applied to every turn that has no more specific waiting
        cost in `turn_wait_times` or `waiting_costs`.

    Returns
    -------
    route
        A `TurnRoute` of the path's node IDs, also carrying the path's edge
        identities and per-edge travel times, waiting times, and costs. None if
        no path can be solved. An origin equal to the destination yields an
        empty path.
    """
    default_turn_wait_time = _finite_non_negative(
        default_turn_wait_time,
        "The default_turn_wait_time",
    )
    _validate._verify_numeric_edge_attribute(G, weight, strict=False)
    if orig not in G:
        raise nx.NodeNotFound(f"Node {orig!r} is not in the graph.")
    if dest not in G:
        raise nx.NodeNotFound(f"Node {dest!r} is not in the graph.")
    if orig == dest:
        return TurnRoute([orig])

    forbidden = _resolve_turn_restrictions(G, turn_restrictions)
    extra_edge_waits, extra_node_waits, extra_turn_waits = _split_waiting_costs(G, waiting_costs)
    edge_waits = _edge_attribute_waits(G)
    edge_waits.update(_resolve_edge_wait_times(G, edge_wait_times))
    edge_waits.update(_resolve_edge_wait_times(G, extra_edge_waits))
    node_waits = _resolve_node_wait_times(G, node_wait_times)
    node_waits.update(extra_node_waits)
    pair_waits, single_waits = _resolve_turn_wait_times(G, turn_wait_times)
    extra_pair_waits, extra_single_waits = _resolve_turn_wait_times(G, extra_turn_waits)
    pair_waits.update(extra_pair_waits)
    single_waits.update(extra_single_waits)

    out_edges: dict[int, list[EdgeKey]] = defaultdict(list)
    for edge in G.edges(keys=True):
        out_edges[edge[0]].append(edge)

    def entry_cost(edge: EdgeKey, wait: float) -> float:
        """Return the cost of traversing an edge after waiting `wait`."""
        edge_weight = _edge_weight(G, edge, weight)
        if edge_weight < 0:
            msg = f"Edge {edge!r} has a negative {weight!r} value."
            raise ValueError(msg)
        return edge_weight + wait

    # Dijkstra's algorithm over edge identities, so the search state remembers
    # the incoming edge and can restrict or penalize the next turn accordingly
    counter = itertools.count()
    heap: list[tuple[float, int, EdgeKey]] = []
    best_cost: dict[EdgeKey, float] = {}
    predecessors: dict[EdgeKey, EdgeKey | None] = {}
    entry_wait: dict[EdgeKey, float] = {}

    for edge in out_edges.get(orig, ()):
        wait = edge_waits.get(edge, 0.0) + node_waits.get(orig, 0.0)
        cost = entry_cost(edge, wait)
        if cost < best_cost.get(edge, np.inf):
            best_cost[edge] = cost
            predecessors[edge] = None
            entry_wait[edge] = wait
            heapq.heappush(heap, (cost, next(counter), edge))

    goal: EdgeKey | None = None
    while len(heap) > 0:
        cost, _, edge = heapq.heappop(heap)
        if cost > best_cost.get(edge, np.inf):
            continue  # stale heap entry
        if edge[1] == dest:
            goal = edge
            break
        for next_edge in out_edges.get(edge[1], ()):
            if next_edge in forbidden.get(edge, ()):
                continue
            wait = _turn_wait_time(
                from_edge=edge,
                to_edge=next_edge,
                pair_waits=pair_waits,
                single_waits=single_waits,
                default_turn_wait_time=default_turn_wait_time,
            )
            wait += edge_waits.get(next_edge, 0.0) + node_waits.get(edge[1], 0.0)
            next_cost = cost + entry_cost(next_edge, wait)
            if next_cost < best_cost.get(next_edge, np.inf):
                best_cost[next_edge] = next_cost
                predecessors[next_edge] = edge
                entry_wait[next_edge] = wait
                heapq.heappush(heap, (next_cost, next(counter), next_edge))

    if goal is None:
        msg = f"Cannot solve path from {orig} to {dest}"
        utils.log(msg, level=lg.WARNING)
        return None

    # walk the predecessors back from the goal, then reverse into path order
    edges: list[EdgeKey] = []
    edge: EdgeKey | None = goal
    while edge is not None:
        edges.append(edge)
        edge = predecessors[edge]
    edges.reverse()

    return TurnRoute(
        [orig, *(e[1] for e in edges)],
        edges=edges,
        travel_times=[_edge_travel_time(G, e, weight) for e in edges],
        wait_times=[entry_wait[e] for e in edges],
        costs=[_edge_weight(G, e, weight) + entry_wait[e] for e in edges],
    )


# aliases so callers can use the name that best matches the task at hand
turn_aware_shortest_path = shortest_path_with_turns
shortest_path_with_turn_restrictions = shortest_path_with_turns


def add_edge_speeds(
    G: nx.MultiDiGraph,
    *,
    hwy_speeds: dict[str, float] | None = None,
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
        assigned the mean pre-existing speed value of all edges of that
        highway type.
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
        Graph with `speed_kph` attributes on all edges.
    """
    if fallback is None:
        fallback = np.nan

    edges = convert.graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)

    # collapse any highway lists (can happen during graph simplification)
    # into string values simply by keeping just the first element of the list
    edges["highway"] = edges["highway"].map(lambda x: x[0] if isinstance(x, list) else x)

    if "maxspeed" in edges.columns:
        # collapse any maxspeed lists (can happen during graph simplification)
        # into a single value
        edges["maxspeed"] = edges["maxspeed"].apply(_collapse_multiple_maxspeed_values, agg=agg)

        # create speed_kph by cleaning maxspeed strings and converting mph to
        # kph if necessary
        edges["speed_kph"] = edges["maxspeed"].astype(str).map(_clean_maxspeed).astype(float)
    else:
        # if no edges in graph had a maxspeed attribute
        edges["speed_kph"] = None

    # if user provided hwy_speeds, use them as default values, otherwise
    # initialize an empty series to populate with values
    hwy_speed_avg = (
        pd.Series(dtype=float)
        if hwy_speeds is None
        else pd.Series(hwy_speeds, dtype=float).dropna()
    )

    # for each highway type that caller did not provide in hwy_speeds, impute
    # speed of type by taking the mean of the preexisting speed values of that
    # highway type
    for hwy, group in edges.groupby("highway"):
        if hwy not in hwy_speed_avg:
            hwy_speed_avg.loc[hwy] = agg(group["speed_kph"])

    # if any highway types had no preexisting speed values, impute their speed
    # with fallback value provided by caller. if fallback=np.nan, impute speed
    # as the mean speed of all highway types that did have preexisting values
    hwy_speed_avg = hwy_speed_avg.fillna(fallback).fillna(agg(hwy_speed_avg))

    # for each edge missing speed data, assign it the imputed value for its
    # highway type
    speed_kph = (
        edges[["highway", "speed_kph"]].set_index("highway").iloc[:, 0].fillna(hwy_speed_avg)
    )

    # all speeds will be null if edges had no preexisting maxspeed data and
    # caller did not pass in hwy_speeds or fallback arguments
    if pd.isna(speed_kph).all():
        msg = (
            "This graph's edges have no preexisting 'maxspeed' attribute "
            "values so you must pass `hwy_speeds` or `fallback` arguments."
        )
        raise ValueError(msg)

    # add speed kph attribute to graph edges
    edges["speed_kph"] = speed_kph.to_numpy()
    nx.set_edge_attributes(G, values=edges["speed_kph"].to_dict(), name="speed_kph")

    return G


def add_edge_travel_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add edge travel time (seconds) to graph as new `travel_time` edge attributes.

    Calculates free-flow travel time along each edge, based on `length` and
    `speed_kph` attributes. Note: run `add_edge_speeds` first to generate the
    `speed_kph` attribute. All edges must have `length` and `speed_kph`
    attributes and all their values must be non-null.

    Parameters
    ----------
    G
        Input graph.

    Returns
    -------
    G
        Graph with `travel_time` attributes on all edges.
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
