"""Truncate or clip graph by distance, bounding box, or polygon."""

from __future__ import annotations

import logging as lg
from collections import Counter
from itertools import chain
from typing import Any

import networkx as nx
from shapely import Geometry
from shapely import GeometryCollection
from shapely import LineString
from shapely import MultiLineString
from shapely import MultiPolygon
from shapely import Polygon

from . import convert
from . import distance
from . import projection
from . import utils
from . import utils_geo


def truncate_graph_dist(
    G: nx.MultiDiGraph,
    source_node: int,
    dist: float,
    *,
    weight: str = "length",
) -> nx.MultiDiGraph:
    """
    Remove from a graph every node beyond some network distance from a node.

    This function must calculate shortest path distances between `source_node`
    and every other node, which can be slow on large graphs.

    Parameters
    ----------
    G
        Input graph.
    source_node
        Node from which to measure network distances to all other nodes.
    dist
        Remove every node in the graph that is greater than `dist` distance
        (in same units as `weight` attribute) along the network from
        `source_node`.
    weight
        Graph edge attribute to use to measure distance.

    Returns
    -------
    G
        The truncated graph.
    """
    # get the shortest distance between the node and every other node
    distances = nx.shortest_path_length(G, source=source_node, weight=weight)

    # then identify every node further than dist away
    distant_nodes = {k for k, v in distances.items() if v > dist}
    unreachable_nodes = G.nodes - distances.keys()

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    G.remove_nodes_from(distant_nodes | unreachable_nodes)

    msg = f"Truncated graph by {weight}-weighted network distance"
    utils.log(msg, level=lg.INFO)
    return G


def truncate_graph_bbox(
    G: nx.MultiDiGraph,
    bbox: tuple[float, float, float, float],
    *,
    truncate_by_edge: bool = False,
) -> nx.MultiDiGraph:
    """
    Remove from a graph every node that falls outside a bounding box.

    Parameters
    ----------
    G
        Input graph.
    bbox
        Bounding box as `(left, bottom, right, top)`.
    truncate_by_edge
        If True, retain nodes outside bounding box if at least one of node's
        neighbors is within the bounding box.

    Returns
    -------
    G
        The truncated graph.
    """
    # convert bounding box to a polygon, then truncate
    polygon = utils_geo.bbox_to_poly(bbox=bbox)
    G = truncate_graph_polygon(G, polygon, truncate_by_edge=truncate_by_edge)

    msg = "Truncated graph by bounding box"
    utils.log(msg, level=lg.INFO)
    return G


def truncate_graph_polygon(
    G: nx.MultiDiGraph,
    polygon: Polygon | MultiPolygon,
    *,
    truncate_by_edge: bool = False,
) -> nx.MultiDiGraph:
    """
    Remove from a graph every node that falls outside a (Multi)Polygon.

    Parameters
    ----------
    G
        Input graph.
    polygon
        Only retain nodes in graph that lie within this geometry.
    truncate_by_edge
        If True, retain nodes outside boundary polygon if at least one of
        node's neighbors is within the polygon.

    Returns
    -------
    G
        The truncated graph.
    """
    msg = "Identifying all nodes that lie outside the polygon..."
    utils.log(msg, level=lg.INFO)

    # first identify all nodes whose point geometries lie within the polygon
    gs_nodes = convert.graph_to_gdfs(G, edges=False)["geometry"]
    to_keep = utils_geo._intersect_index_quadrats(gs_nodes, polygon)

    if len(to_keep) == 0:
        # no graph nodes within the polygon: can't create a graph from that
        msg = "Found no graph nodes within the requested polygon."
        raise ValueError(msg)

    # now identify all nodes whose point geometries lie outside the polygon
    gs_nodes_outside_poly = gs_nodes[~gs_nodes.index.isin(to_keep)]
    nodes_outside_poly = set(gs_nodes_outside_poly.index)

    if truncate_by_edge:
        # retain nodes outside boundary polygon if at least one of node's
        # neighbors is within the polygon
        nodes_to_remove = set()
        for node in nodes_outside_poly:
            # if all the neighbors of this node also lie outside polygon, then
            # mark this node for removal
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            if neighbors.issubset(nodes_outside_poly):
                nodes_to_remove.add(node)
    else:
        nodes_to_remove = nodes_outside_poly

    # now remove from the graph all those nodes that lie outside the polygon
    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    G.remove_nodes_from(nodes_to_remove)
    msg = f"Removed {len(nodes_to_remove):,} nodes outside polygon"
    utils.log(msg, level=lg.INFO)

    msg = "Truncated graph by polygon"
    utils.log(msg, level=lg.INFO)
    return G


def clip_graph_bbox(
    G: nx.MultiDiGraph,
    bbox: tuple[float, float, float, float],
) -> nx.MultiDiGraph:
    """
    Clip a graph's edge geometries to a bounding box.

    This function converts the bounding box to a rectangle polygon then calls
    `clip_graph_polygon`. See that function's documentation for details of
    how edges and nodes are clipped.

    Parameters
    ----------
    G
        Input graph.
    bbox
        Bounding box as `(left, bottom, right, top)`.

    Returns
    -------
    G
        The clipped graph.
    """
    polygon = utils_geo.bbox_to_poly(bbox=bbox)
    G = clip_graph_polygon(G, polygon)
    msg = "Clipped graph by bounding box"
    utils.log(msg, level=lg.INFO)
    return G


def clip_graph_polygon(
    G: nx.MultiDiGraph,
    polygon: Polygon | MultiPolygon,
) -> nx.MultiDiGraph:
    """
    Clip a graph's edge geometries to a (Multi)Polygon.

    Unlike `truncate_graph_polygon`, which only inspects node locations and
    therefore drops roads whose endpoints both lie outside the polygon even
    if the road passes through it, this function intersects every edge's
    line geometry with the polygon and retains the portions lying within it.
    Edges that cross the polygon boundary are split at their intersection
    points, and new synthetic boundary nodes (flagged with a `clipped=True`
    node attribute) are created there so the clipping points remain
    explicit, reviewable nodes in the graph.

    Edge direction is always preserved: each resulting edge runs in the
    same from-to order as its source edge. Each resulting edge receives a
    `geometry` attribute (the clipped `LineString`) and a recomputed
    `length` attribute in meters. For geographic (unprojected) graphs,
    lengths are calculated as the sum of great-circle distances between
    the clipped geometry's consecutive vertices; for projected graphs,
    lengths are taken directly from the clipped geometry's length.

    MultiPolygons, boundary tangencies, and edges missing a `geometry`
    attribute are all handled consistently: MultiPolygons are intersected
    as a single geometry so an edge weaving in and out produces one clipped
    segment per polygon part; edges that merely touch the boundary at a
    point without entering produce point-only intersections and are
    discarded; and edges without a `geometry` attribute are reconstructed
    as the straight line between their endpoint nodes (matching
    `convert.graph_to_gdfs`).

    Parameters
    ----------
    G
        Input graph.
    polygon
        Clip the graph's edge geometries to this geometry.

    Returns
    -------
    G
        The clipped graph.
    """
    # verify the polygon is a valid Polygon or MultiPolygon before proceeding
    if not polygon.is_valid:
        msg = "The geometry of `polygon` is invalid."
        raise ValueError(msg)
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        msg = (
            "Geometry must be a shapely Polygon or MultiPolygon. See OSMnx "
            "documentation for details."
        )
        raise TypeError(msg)

    msg = "Clipping graph edge geometries to polygon..."
    utils.log(msg, level=lg.INFO)

    if len(G.edges) == 0:
        msg = "Graph contains no edges to clip."
        raise ValueError(msg)

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()

    # remember whether the input graph tracked street_count on every node:
    # must be evaluated before any nodes are rebuilt below
    had_street_count = all("street_count" in d for _, d in G.nodes(data=True))

    # node coords lookup, used both to reconstruct missing edge geometries
    # and to resolve clipped endpoints back onto existing graph nodes
    node_coords = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    # reverse lookup from coordinate to existing node ID
    coord_to_node = {coord: n for n, coord in node_coords.items()}

    # whether lengths must be computed as great-circle distances (geographic
    # CRS) or can be read directly from geometry lengths (projected CRS)
    geographic = not projection.is_projected(G.graph["crs"])

    # synthetic boundary nodes are assigned negative integer IDs, starting
    # below any existing integer node ID so they can never collide
    int_ids = [n for n in G.nodes if isinstance(n, int)]
    next_new_id = (min(int_ids) - 1) if int_ids else -1
    new_nodes: dict[tuple[float, float], int] = {}

    def get_or_create_node(xy: tuple[float, float]) -> int:
        # reuse an existing node if one sits exactly at this coordinate...
        if xy in coord_to_node:
            return coord_to_node[xy]
        # ...otherwise create (and remember) a synthetic boundary node
        if xy not in new_nodes:
            nonlocal next_new_id
            new_id = next_new_id
            next_new_id -= 1
            G.add_node(new_id, x=xy[0], y=xy[1], clipped=True)
            new_nodes[xy] = new_id
            coord_to_node[xy] = new_id
        return new_nodes[xy]

    # gather all edges first: we will rebuild the graph from clipped results
    input_edges = tuple(G.edges(keys=True, data=True))
    G.remove_edges_from(list(G.edges))

    n_clipped_edges = 0
    for u, v, _key, data in input_edges:
        # reconstruct geometry if missing, matching graph_to_gdfs convention
        geom = data.get("geometry")
        if geom is None or geom.is_empty:
            geom = LineString((node_coords[u], node_coords[v]))

        # intersect the edge geometry with the clipping polygon. shapely
        # preserves the original coordinate ordering (thus edge direction)
        # and natively handles MultiPolygons
        clipped = geom.intersection(polygon)
        if clipped.is_empty:
            continue

        # normalize the intersection result into a list of LineStrings,
        # ignoring point-only contacts (tangencies) and other non-linear
        # intersection artifacts (including ones nested in GeometryCollections)
        parts = _line_parts(clipped)
        parts = [p for p in parts if not p.is_empty and len(p.coords) >= 2]
        if not parts:
            continue

        # one source edge may yield several clipped segments (e.g. weaving
        # in and out of a MultiPolygon)
        for part in parts:
            coords = list(part.coords)
            new_u = get_or_create_node(coords[0])
            new_v = get_or_create_node(coords[-1])

            new_data: dict[str, Any] = dict(data)
            new_data["geometry"] = part
            new_data["length"] = _geometry_length_m(part, geographic=geographic)

            # drop any stale geometry-derived keys then add the new edge,
            # letting networkx assign the next free parallel key
            new_data.pop("u_original", None)
            new_data.pop("v_original", None)
            G.add_edge(new_u, new_v, **new_data)
            n_clipped_edges += 1

    # remove original nodes that no longer participate in any edge (they lay
    # fully outside the polygon and no clipped segment reaches them)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    if len(G.nodes) == 0:
        msg = "Found no graph edges intersecting the requested polygon."
        raise ValueError(msg)

    # recompute street_count when the input graph tracked it, so the new
    # boundary nodes and altered connectivity stay internally consistent
    if had_street_count:
        counts = _count_streets_per_node(G)
        nx.set_node_attributes(G, counts, name="street_count")

    n_new = len(new_nodes)
    msg = (
        f"Clipped graph to polygon: retained {len(G.nodes):,} nodes "
        f"({n_new:,} new boundary nodes) and {n_clipped_edges:,} edges"
    )
    utils.log(msg, level=lg.INFO)
    return G


def _line_parts(geom: Geometry) -> list[LineString]:
    """
    Extract all `LineString` components from an intersection result.

    A line-polygon intersection may yield a `LineString`, a
    `MultiLineString`, or (for example when an edge both crosses and merely
    touches another part of a MultiPolygon) a `GeometryCollection` mixing
    lines and points. Only the linear components can become graph edges, so
    the point-only contacts (boundary tangencies) are discarded.

    Parameters
    ----------
    geom
        The intersection geometry returned by shapely.

    Returns
    -------
    lines
        Every line geometry contained within `geom`, in their original
        coordinate order.
    """
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        lines: list[LineString] = []
        for part in geom.geoms:
            lines.extend(_line_parts(part))
        return lines
    return []


def _geometry_length_m(geom: LineString, *, geographic: bool) -> float:
    """
    Calculate a line geometry's length in meters.

    For geographic (unprojected) coordinates, sum the great-circle distances
    between consecutive vertices. For projected coordinates, return the
    geometry's native length, which is already in the CRS's linear units
    (typically meters).

    Parameters
    ----------
    geom
        The line geometry to measure.
    geographic
        If True, treat coordinates as unprojected lon/lat and compute
        great-circle distances; otherwise use the geometry's own length.

    Returns
    -------
    length
        The geometry's length in meters.
    """
    if not geographic:
        return float(geom.length)

    coords = list(geom.coords)
    total = 0.0
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:], strict=True):
        total += float(distance.great_circle(y1, x1, y2, x2))
    return total


def _count_streets_per_node(G: nx.MultiDiGraph) -> dict[Any, int]:
    """
    Count undirected street segments incident on each node.

    Mirrors `stats.count_streets_per_node` but is implemented locally to keep
    the truncate module free of a circular import through the stats module.

    Parameters
    ----------
    G
        Input graph.

    Returns
    -------
    streets_per_node
        Counts of how many physical streets connect to each node.
    """
    Gu = G.to_undirected(reciprocal=False, as_view=True)
    self_loop_edges = set(nx.selfloop_edges(Gu, keys=False))
    non_self_loop_edges = [e for e in Gu.edges(keys=False) if e not in self_loop_edges]
    all_unique_edges = non_self_loop_edges + list(self_loop_edges)
    counts = Counter(chain.from_iterable(all_unique_edges))
    return {node: counts[node] for node in G.nodes}


def largest_component(G: nx.MultiDiGraph, *, strongly: bool = False) -> nx.MultiDiGraph:
    """
    Return `G`'s largest weakly or strongly connected component as a graph.

    Parameters
    ----------
    G
        Input graph.
    strongly
        If True, return the largest strongly connected component. Otherwise
        return the largest weakly connected component.

    Returns
    -------
    G
        The largest connected component subgraph of the original graph.
    """
    if strongly:
        kind = "strongly"
        is_connected = nx.is_strongly_connected
        connected_components = nx.strongly_connected_components
    else:
        kind = "weakly"
        is_connected = nx.is_weakly_connected
        connected_components = nx.weakly_connected_components

    if not is_connected(G):
        # get all the connected components in graph then identify the largest
        largest_cc = max(connected_components(G), key=len)
        n = len(G)

        # induce (frozen) subgraph then unfreeze it by making new MultiDiGraph
        G = nx.MultiDiGraph(G.subgraph(largest_cc))

        msg = f"Got largest {kind} connected component ({len(G):,} of {n:,} total nodes)"
        utils.log(msg, level=lg.INFO)

    return G
