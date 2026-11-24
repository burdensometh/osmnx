"""
Detect changes between two road network graphs of the same region.

The two graphs represent the same region at two points in time (for example,
a city's two monthly road map releases) and must preserve the source data's
original identifiers: nodes are indexed by their OSM IDs and edges carry an
``osmid`` attribute, as produced by OSMnx's graph creation functions.

The module distinguishes three kinds of changes:

- ``added``: an identifier present in the new graph but not in the old one
- ``closed``: an identifier present in the old graph but not in the new one
- ``modified``: an identifier present in both, whose attributes, direction, or
  geometry/length genuinely changed

Comparisons are order-insensitive: list-valued OSM tags (such as ``osmid``,
``highway``, and ``name``) and parallel-edge key assignment are canonicalized
before comparison, so differences that only reflect an internal iteration
order are ignored. Real changes in direction (reversal of a directed edge) or
length (geometry changes beyond a tolerance) are always reported with
evidence (old value, new value, and, where relevant, their difference).

Ambiguous segment relationships are never force-matched. For example, if one
OSM ID maps to several parallel edges in one release but not the other, or a
simplified segment's ID sets cannot be aligned, the candidates are routed to
a review queue instead of being guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from . import distance
from . import utils

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

# node attributes derived from geometry rather than source tags
_NODE_GEOM_ATTRS = {"x", "y", "street_count"}

# edge attributes derived from geometry rather than source tags
_EDGE_GEOM_DERIVED_ATTRS = {"length", "geometry"}


@dataclass(frozen=True)
class Change:
    """
    A single detected change to a node or edge.

    Parameters
    ----------
    element_type
        Either ``"node"`` or ``"edge"``.
    change_type
        One of ``"added"``, ``"closed"``, or ``"modified"``.
    element_id
        Stable identifier of the element: a node's OSM ID, or a tuple
        ``(osmid, u, v, key)`` for an edge.
    changes
        Mapping of attribute name to ``(old_value, new_value)`` for modified
        elements. Empty for added/closed elements; geometry/direction changes
        are recorded under the ``"geometry"`` and ``"direction"`` keys and
        length changes under ``"length"``.
    evidence
        Extra human- and machine-readable evidence supporting the change,
        such as recomputed lengths and their difference.
    """

    element_type: str
    change_type: str
    element_id: Any
    changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewItem:
    """
    An ambiguous element relationship that requires manual review.

    Parameters
    ----------
    reason
        Machine-readable reason code explaining why the relationship could
        not be resolved automatically.
    element_type
        Either ``"node"`` or ``"edge"``.
    old_candidates
        Identifiers of candidate elements in the old graph.
    new_candidates
        Identifiers of candidate elements in the new graph.
    detail
        Human-readable explanation of the ambiguity.
    """

    reason: str
    element_type: str
    old_candidates: tuple[Any, ...]
    new_candidates: tuple[Any, ...]
    detail: str


@dataclass
class GraphDiff:
    """
    Result of diffing two road network graphs.

    Parameters
    ----------
    node_changes
        Added, closed, and modified nodes.
    edge_changes
        Added, closed, and modified edges.
    review
        Ambiguous relationships routed for manual review rather than
        force-matched.
    """

    node_changes: list[Change] = field(default_factory=list)
    edge_changes: list[Change] = field(default_factory=list)
    review: list[ReviewItem] = field(default_factory=list)

    @property
    def changes(self) -> list[Change]:
        """All node and edge changes, nodes first."""
        return [*self.node_changes, *self.edge_changes]

    def summary(self) -> dict[str, dict[str, int]]:
        """
        Count changes by element type and change type.

        Returns
        -------
        counts
            Nested mapping ``{element_type: {change_type: count}}`` plus the
            number of items in the review queue under the ``"review"`` key.
        """
        counts: dict[str, dict[str, int]] = {
            "node": {"added": 0, "closed": 0, "modified": 0},
            "edge": {"added": 0, "closed": 0, "modified": 0},
            "review": {"items": len(self.review)},
        }
        for change in self.node_changes:
            counts["node"][change.change_type] += 1
        for change in self.edge_changes:
            counts["edge"][change.change_type] += 1
        return counts


def diff_graphs(
    G_old: nx.MultiDiGraph,
    G_new: nx.MultiDiGraph,
    *,
    node_attrs: Iterable[str] | None = None,
    edge_attrs: Iterable[str] | None = None,
    length_tolerance: float = 1.0,
    coord_precision: int = 7,
) -> GraphDiff:
    """
    Diff two same-region road network graphs that retain OSM identifiers.

    Nodes are matched by their OSM ID (the graph node index). Edges are
    matched by their source OSM way ID together with their directed
    endpoints and a deterministically assigned parallel-edge key. No spatial
    or fuzzy matching is performed: ambiguous relationships go to the
    returned review queue.

    List-valued attributes are compared as order-insensitive canonical
    values, so re-ordering multi-valued OSM tags (or edges' internal
    iteration order) does not generate spurious changes. A directed edge
    whose endpoints swap between releases is reported as a real direction
    change, and geometry changes whose recomputed length difference exceeds
    `length_tolerance` meters are reported as real length changes.

    Parameters
    ----------
    G_old
        The earlier graph.
    G_new
        The later graph.
    node_attrs
        Node attribute names to compare. If None, compare every node
        attribute present in either graph (excluding derived geometry
        fields such as ``x`` and ``y``, whose changes are detected via node
        coordinates instead).
    edge_attrs
        Edge attribute names to compare. If None, compare every edge
        attribute present in either graph (excluding the derived
        ``length`` and ``geometry`` fields, which are compared
        geometrically).
    length_tolerance
        Minimum absolute length difference, in meters, for a geometry change
        to be reported as a real length change. Smaller differences are
        considered coordinate noise.
    coord_precision
        Decimal places to round node coordinates and geometry vertices to
        before comparison, making the diff robust to floating-point
        re-serialization noise.

    Returns
    -------
    diff
        The detected changes and review items.
    """
    if not isinstance(G_old, nx.MultiDiGraph) or not isinstance(G_new, nx.MultiDiGraph):
        msg = "Both graphs must be networkx.MultiDiGraph instances."
        raise TypeError(msg)

    if length_tolerance < 0:
        msg = "`length_tolerance` must be non-negative."
        raise ValueError(msg)

    result = GraphDiff()

    _require_osm_identifiers(G_old, "G_old")
    _require_osm_identifiers(G_new, "G_new")

    old_nodes = _index_nodes(G_old)
    new_nodes = _index_nodes(G_new)
    _diff_nodes(old_nodes, new_nodes, node_attrs, coord_precision, result)

    old_edges = _index_edges(G_old)
    new_edges = _index_edges(G_new)
    _diff_edges(
        G_old,
        G_new,
        old_edges,
        new_edges,
        edge_attrs,
        length_tolerance,
        coord_precision,
        result,
    )

    msg = (
        f"Diffed graphs: {len(result.node_changes)} node changes, "
        f"{len(result.edge_changes)} edge changes, {len(result.review)} review items."
    )
    utils.log(msg)
    return result


def _require_osm_identifiers(G: nx.MultiDiGraph, graph_name: str) -> None:
    """Ensure every edge carries the source ``osmid`` identifier."""
    missing = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if "osmid" not in data
    ]
    if missing:
        msg = (
            f"{graph_name} contains {len(missing)} edge(s) without an `osmid` attribute; "
            "diffing requires graphs that retain their original OSM identifiers."
        )
        raise ValueError(msg)


def _index_nodes(G: nx.MultiDiGraph) -> dict[Any, tuple[Any, dict[str, Any]]]:
    """Map each node's OSM ID to its id and attribute dict."""
    return {node: (node, dict(data)) for node, data in G.nodes(data=True)}


def _diff_nodes(
    old_nodes: dict[Any, tuple[Any, dict[str, Any]]],
    new_nodes: dict[Any, tuple[Any, dict[str, Any]]],
    node_attrs: Iterable[str] | None,
    coord_precision: int,
    result: GraphDiff,
) -> None:
    """Classify nodes as added, closed, or modified."""
    old_ids = set(old_nodes)
    new_ids = set(new_nodes)

    for node_id in sorted(new_ids - old_ids, key=_stable_id):
        result.node_changes.append(Change("node", "added", node_id))

    for node_id in sorted(old_ids - new_ids, key=_stable_id):
        result.node_changes.append(Change("node", "closed", node_id))

    if node_attrs is None:
        attrs = set()
        for _, data in old_nodes.values():
            attrs.update(data)
        for _, data in new_nodes.values():
            attrs.update(data)
        attrs.difference_update(_NODE_GEOM_ATTRS)
        attr_names: Sequence[str] = sorted(attrs)
    else:
        attr_names = tuple(node_attrs)

    for node_id in sorted(old_ids & new_ids, key=_stable_id):
        _, old_data = old_nodes[node_id]
        _, new_data = new_nodes[node_id]
        changes: dict[str, tuple[Any, Any]] = {}

        # coordinate movement is real geometry evidence, not tag noise
        old_xy = _canonical_xy(old_data, coord_precision)
        new_xy = _canonical_xy(new_data, coord_precision)
        if old_xy != new_xy:
            changes["location"] = (old_xy, new_xy)

        for attr in attr_names:
            old_val = _canonical_value(old_data.get(attr))
            new_val = _canonical_value(new_data.get(attr))
            if old_val != new_val:
                changes[attr] = (old_val, new_val)

        if changes:
            result.node_changes.append(Change("node", "modified", node_id, changes))


def _index_edges(
    G: nx.MultiDiGraph,
) -> dict[Any, list[tuple[tuple[int, int, int], dict[str, Any]]]]:
    """
    Group edges by their source OSM way identifier.

    Within each group, parallel edges receive a deterministic key derived
    from their directed endpoints and canonical geometry, so the index does
    not depend on graph iteration order.
    """
    grouped: dict[Any, list[tuple[tuple[int, int, int], dict[str, Any]]]] = {}
    for u, v, _key, data in G.edges(keys=True, data=True):
        osmid = _canonical_value(data.get("osmid"))
        grouped.setdefault(osmid, []).append(((u, v, _key), dict(data)))

    # deterministic ordering inside each group, independent of iteration order
    for entries in grouped.values():
        entries.sort(key=lambda item: (item[0][0], item[0][1], _geometry_signature(item[1])))
    return grouped


def _diff_edges(
    G_old: nx.MultiDiGraph,
    G_new: nx.MultiDiGraph,
    old_edges: dict[Any, list[tuple[tuple[int, int, int], dict[str, Any]]]],
    new_edges: dict[Any, list[tuple[tuple[int, int, int], dict[str, Any]]]],
    edge_attrs: Iterable[str] | None,
    length_tolerance: float,
    coord_precision: int,
    result: GraphDiff,
) -> None:
    """Classify edges as added, closed, modified, or ambiguous."""
    old_ids = set(old_edges)
    new_ids = set(new_edges)
    added_ids = new_ids - old_ids
    closed_ids = old_ids - new_ids

    # simplified edges are keyed by a tuple of constituent osmids: a changed
    # composition (re-split or merged segment) cannot be aligned automatically.
    # unmatched tuple keys whose osmid sets overlap are routed to review rather
    # than force-matched; fully disjoint ways remain genuine added/closed.
    reviewed_ids = _review_ambiguous_segments(
        old_edges, new_edges, closed_ids, added_ids, result
    )

    for osmid in sorted(added_ids - reviewed_ids, key=_stable_id):
        for uvk, _ in new_edges[osmid]:
            result.edge_changes.append(Change("edge", "added", _edge_id(osmid, uvk)))

    for osmid in sorted(closed_ids - reviewed_ids, key=_stable_id):
        for uvk, _ in old_edges[osmid]:
            result.edge_changes.append(Change("edge", "closed", _edge_id(osmid, uvk)))

    if edge_attrs is None:
        attrs = set()
        for entries in old_edges.values():
            for _, data in entries:
                attrs.update(data)
        for entries in new_edges.values():
            for _, data in entries:
                attrs.update(data)
        attrs.difference_update(_EDGE_GEOM_DERIVED_ATTRS)
        attr_names: Sequence[str] = tuple(sorted(attrs))
    else:
        attr_names = tuple(edge_attrs)

    for osmid in sorted(old_ids & new_ids, key=_stable_id):
        old_entries = old_edges[osmid]
        new_entries = new_edges[osmid]
        _diff_edge_group(
            G_old,
            G_new,
            osmid,
            old_entries,
            new_entries,
            attr_names,
            length_tolerance,
            coord_precision,
            result,
        )


def _review_ambiguous_segments(
    old_edges: dict[Any, list[tuple[tuple[int, int, int], dict[str, Any]]]],
    new_edges: dict[Any, list[tuple[tuple[int, int, int], dict[str, Any]]]],
    closed_ids: set[Any],
    added_ids: set[Any],
    result: GraphDiff,
) -> set[Any]:
    """
    Route partially-overlapping simplified-segment ids to the review queue.

    Only tuple-valued keys (simplified edges carrying multiple constituent
    osmids) are considered, and only when at least one constituent osmid is
    shared across the two releases. Everything else stays a plain
    added/closed change. Returns every id consumed by a review item so the
    caller does not double-report it.
    """
    unmatched_old = [osmid for osmid in closed_ids if isinstance(osmid, tuple)]
    unmatched_new = [osmid for osmid in added_ids if isinstance(osmid, tuple)]
    if not unmatched_old or not unmatched_new:
        return set()

    # map each constituent way id to the simplified groups that contain it
    old_members: dict[Any, list[Any]] = {}
    new_members: dict[Any, list[Any]] = {}
    for osmid in unmatched_old:
        for member in osmid:
            old_members.setdefault(member, []).append(osmid)
    for osmid in unmatched_new:
        for member in osmid:
            new_members.setdefault(member, []).append(osmid)

    shared = set(old_members) & set(new_members)
    if not shared:
        return set()

    reviewed: set[Any] = set()
    # pair each old group with every new group sharing a constituent way
    seen_pairs: set[tuple[Any, Any]] = set()
    for member in sorted(shared, key=_stable_id):
        for old_id in old_members[member]:
            for new_id in new_members[member]:
                pair = (old_id, new_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                reviewed.update((old_id, new_id))
                old_candidates = tuple(_edge_id(old_id, uvk) for uvk, _ in old_edges[old_id])
                new_candidates = tuple(_edge_id(new_id, uvk) for uvk, _ in new_edges[new_id])
                result.review.append(
                    ReviewItem(
                        reason="ambiguous_simplified_segment",
                        element_type="edge",
                        old_candidates=old_candidates,
                        new_candidates=new_candidates,
                        detail=(
                            f"Simplified segments {old_id!r} and {new_id!r} share constituent "
                            f"way id {member!r} but have different compositions; segment "
                            "correspondence requires manual review."
                        ),
                    )
                )
    return reviewed


def _diff_edge_group(
    G_old: nx.MultiDiGraph,
    G_new: nx.MultiDiGraph,
    osmid: Any,
    old_entries: list[tuple[tuple[int, int, int], dict[str, Any]]],
    new_entries: list[tuple[tuple[int, int, int], dict[str, Any]]],
    attr_names: Sequence[str],
    length_tolerance: float,
    coord_precision: int,
    result: GraphDiff,
) -> None:
    """Compare the edge(s) sharing one source OSM way identifier."""
    old_candidates = tuple(_edge_id(osmid, uvk) for uvk, _ in old_entries)
    new_candidates = tuple(_edge_id(osmid, uvk) for uvk, _ in new_entries)

    # one way id on multiple parallel edges in only one release: the mapping
    # between releases cannot be resolved without guessing, so route to review
    if len(old_entries) != len(new_entries):
        result.review.append(
            ReviewItem(
                reason="ambiguous_parallel_edges",
                element_type="edge",
                old_candidates=old_candidates,
                new_candidates=new_candidates,
                detail=(
                    f"OSM id {osmid!r} maps to {len(old_entries)} edge(s) in the old graph "
                    f"and {len(new_entries)} in the new graph; relationships not force-matched."
                ),
            )
        )
        return

    if len(old_entries) == 1:
        pairs = [(old_entries[0], new_entries[0])]
    else:
        # multiple parallel edges: pair explicitly by directed endpoints plus
        # geometry signature rather than by list position. if the two releases
        # do not contain the same set of signatures, do not guess: route to review
        old_by_sig = {_parallel_signature(uvk, data): (uvk, data) for uvk, data in old_entries}
        new_by_sig = {_parallel_signature(uvk, data): (uvk, data) for uvk, data in new_entries}
        if set(old_by_sig) != set(new_by_sig):
            result.review.append(
                ReviewItem(
                    reason="ambiguous_parallel_edges",
                    element_type="edge",
                    old_candidates=old_candidates,
                    new_candidates=new_candidates,
                    detail=(
                        f"Parallel edges for OSM id {osmid!r} cannot be aligned by their "
                        "directed endpoints and geometry; relationships not force-matched."
                    ),
                )
            )
            return
        pairs = [
            (old_by_sig[sig], new_by_sig[sig]) for sig in sorted(old_by_sig, key=_stable_id)
        ]

    for (old_uvk, old_data), (new_uvk, new_data) in pairs:
        _compare_edge(
            G_old,
            G_new,
            osmid,
            old_uvk,
            new_uvk,
            old_data,
            new_data,
            attr_names,
            length_tolerance,
            coord_precision,
            result,
        )


def _parallel_signature(
    uvk: tuple[int, int, int], data: dict[str, Any]
) -> tuple[int, int, str]:
    """
    Build an order-independent matching signature for a parallel edge.

    The directed endpoints distinguish opposite-direction edges of the same
    way, and the geometry WKT distinguishes distinct parallel centerlines.
    """
    u, v, _ = uvk
    return u, v, _geometry_signature(data)


def _compare_edge(
    G_old: nx.MultiDiGraph,
    G_new: nx.MultiDiGraph,
    osmid: Any,
    old_uvk: tuple[int, int, int],
    new_uvk: tuple[int, int, int],
    old_data: dict[str, Any],
    new_data: dict[str, Any],
    attr_names: Sequence[str],
    length_tolerance: float,
    coord_precision: int,
    result: GraphDiff,
) -> None:
    """Compare a matched pair of edges and record any real changes."""
    changes: dict[str, tuple[Any, Any]] = {}
    evidence: dict[str, Any] = {}

    old_u, old_v, _ = old_uvk
    new_u, new_v, _ = new_uvk

    # direction: swapped directed endpoints on the same way is a real reversal
    if (old_u, old_v) != (new_u, new_v):
        if (old_u, old_v) == (new_v, new_u):
            changes["direction"] = (
                f"{old_u}->{old_v}",
                f"{new_u}->{new_v}",
            )
            evidence["direction"] = "edge reversed between releases"
        else:
            # same way id but endpoints changed without a clean reversal:
            # the topology correspondence is ambiguous
            result.review.append(
                ReviewItem(
                    reason="ambiguous_endpoint_change",
                    element_type="edge",
                    old_candidates=(_edge_id(osmid, old_uvk),),
                    new_candidates=(_edge_id(osmid, new_uvk),),
                    detail=(
                        f"OSM id {osmid!r} endpoints changed from {old_u}->{old_v} "
                        f"to {new_u}->{new_v} without a clean reversal."
                    ),
                )
            )
            return

    # tag attributes, compared order-insensitively
    for attr in attr_names:
        old_val = _canonical_value(old_data.get(attr))
        new_val = _canonical_value(new_data.get(attr))
        if old_val != new_val:
            changes[attr] = (old_val, new_val)

    # geometry and length: recompute from canonical geometry so stored-length
    # rounding and iteration order can never manufacture a change
    old_geom = _edge_geometry(G_old, old_uvk, old_data, coord_precision)
    new_geom = _edge_geometry(G_new, new_uvk, new_data, coord_precision)
    if old_geom != new_geom:
        old_len = float(_geometry_length(G_old, old_geom))
        new_len = float(_geometry_length(G_new, new_geom))
        delta = new_len - old_len
        evidence["old_length_m"] = old_len
        evidence["new_length_m"] = new_len
        evidence["length_delta_m"] = delta
        if abs(delta) >= length_tolerance:
            changes["length"] = (old_len, new_len)
            changes["geometry"] = (
                _geometry_wkt(old_geom),
                _geometry_wkt(new_geom),
            )
            evidence["length_change"] = f"{abs(delta):.3f} m exceeds tolerance {length_tolerance} m"
        else:
            # geometry vertices differ but only at noise level: keep evidence
            # without flagging a length change
            evidence["geometry"] = "vertex-level change within length tolerance"

    if changes:
        result.edge_changes.append(
            Change(
                "edge",
                "modified",
                _edge_id(osmid, new_uvk),
                changes,
                evidence,
            )
        )


def _canonical_value(value: Any) -> Any:
    """
    Return an order-insensitive canonical form of an attribute value.

    Lists (OSM multi-valued tags, or lists of osmids from simplification)
    become sorted tuples so that re-ordering alone never counts as a change.
    Scalars pass through unchanged.
    """
    if value is None:
        return None
    if isinstance(value, list):
        try:
            return tuple(sorted(value, key=_stable_id))
        except TypeError:
            return tuple(value)
    if isinstance(value, tuple):
        try:
            return tuple(sorted(value, key=_stable_id))
        except TypeError:
            return value
    return value


def _canonical_xy(data: dict[str, Any], coord_precision: int) -> tuple[float, float] | None:
    """Return a node's rounded (x, y) coordinates, or None if unavailable."""
    if "x" not in data or "y" not in data:
        return None
    return round(float(data["x"]), coord_precision), round(float(data["y"]), coord_precision)


def _edge_geometry(
    G: nx.MultiDiGraph,
    uvk: tuple[int, int, int],
    data: dict[str, Any],
    coord_precision: int,
) -> tuple[tuple[float, float], ...]:
    """
    Return an edge's geometry as a rounded tuple of (x, y) vertices.

    Uses the stored ``geometry`` attribute when present, otherwise the
    straight line between the edge's endpoint nodes.
    """
    geom = data.get("geometry")
    if geom is None:
        u, v, _ = uvk
        geom = LineString(
            [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
        )
    return tuple(
        (round(x, coord_precision), round(y, coord_precision)) for x, y in geom.coords
    )


def _geometry_length(G: nx.MultiDiGraph, coords: tuple[tuple[float, float], ...]) -> float:
    """
    Compute an edge's length in meters from canonical geometry vertices.

    Uses projected (euclidean) distances when the graph has a projected CRS,
    otherwise great-circle distances on geographic coordinates.
    """
    is_projected = _is_projected(G)
    total = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        if is_projected:
            total += distance.euclidean(y1, x1, y2, x2)
        else:
            total += distance.great_circle(y1, x1, y2, x2)
    return total


def _is_projected(G: nx.MultiDiGraph) -> bool:
    """Check whether the graph's CRS is projected (False for geographic CRS)."""
    crs = G.graph.get("crs")
    if crs is None:
        return False
    try:
        from pyproj import CRS

        return bool(CRS.from_user_input(crs).is_projected)
    except (ValueError, TypeError):  # pragma: no cover
        # fall back to a heuristic if pyproj cannot parse the CRS
        text = str(crs).lower()
        return not any(token in text for token in ("4326", "epsg:4326", "wgs", "lon", "lat"))


def _geometry_signature(data: dict[str, Any]) -> str:
    """Stable sort signature for parallel edges, based on geometry if present."""
    geom = data.get("geometry")
    if geom is None:
        return ""
    return geom.wkt


def _geometry_wkt(coords: tuple[tuple[float, float], ...]) -> str:
    """Serialize canonical geometry vertices as WKT."""
    return LineString(coords).wkt


def _edge_id(osmid: Any, uvk: tuple[int, int, int]) -> tuple[Any, int, int, int]:
    """Build the stable external identifier of an edge."""
    u, v, k = uvk
    return osmid, u, v, k


def _stable_id(value: Any) -> tuple[int, Any]:
    """Sort key that works across mixed-type OSM id collections."""
    return (0, value) if isinstance(value, (int, float, str)) else (1, str(value))


def diff_to_gdfs(diff: GraphDiff) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Convert a `GraphDiff` to node and edge change GeoDataFrames.

    Each row is one change. Added/closed elements carry no ``old_value``/
    ``new_value`` payload; modified elements carry one row per changed
    attribute (exploded from the `Change.changes` mapping), and review items
    are included under change type ``"review"``.

    Parameters
    ----------
    diff
        The output of `diff_graphs`.

    Returns
    -------
    gdf_nodes, gdf_edges
        GeoDataFrames of node-level and edge-level changes. The node frame
        lists node changes and node review items; the edge frame lists edge
        changes and edge review items.
    """
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for change in diff.node_changes:
        if not change.changes:
            node_rows.append(
                {
                    "change_type": change.change_type,
                    "element_id": change.element_id,
                    "attribute": None,
                    "old_value": None,
                    "new_value": None,
                    "evidence": change.evidence,
                }
            )
        else:
            for attr, (old_val, new_val) in change.changes.items():
                node_rows.append(
                    {
                        "change_type": change.change_type,
                        "element_id": change.element_id,
                        "attribute": attr,
                        "old_value": old_val,
                        "new_value": new_val,
                        "evidence": change.evidence,
                    }
                )

    for change in diff.edge_changes:
        if not change.changes:
            edge_rows.append(
                {
                    "change_type": change.change_type,
                    "element_id": change.element_id,
                    "attribute": None,
                    "old_value": None,
                    "new_value": None,
                    "evidence": change.evidence,
                }
            )
        else:
            for attr, (old_val, new_val) in change.changes.items():
                edge_rows.append(
                    {
                        "change_type": change.change_type,
                        "element_id": change.element_id,
                        "attribute": attr,
                        "old_value": old_val,
                        "new_value": new_val,
                        "evidence": change.evidence,
                    }
                )

    for item in diff.review:
        target = edge_rows if item.element_type == "edge" else node_rows
        target.append(
            {
                "change_type": "review",
                "element_id": (item.old_candidates, item.new_candidates),
                "attribute": item.reason,
                "old_value": item.old_candidates,
                "new_value": item.new_candidates,
                "evidence": {"detail": item.detail},
            }
        )

    gdf_nodes = gpd.GeoDataFrame(node_rows)
    gdf_edges = gpd.GeoDataFrame(edge_rows)
    return gdf_nodes, gdf_edges
