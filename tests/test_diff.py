# ruff: noqa: PLR2004, S101
"""
Tests for the road network change-detection module (`osmnx.diff`).

The fixtures are synthetic same-region graphs built with OSMnx-style
identifiers (nodes indexed by OSM IDs, edges carrying ``osmid`` tags), so the
tests require no network access.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest
from shapely.geometry import LineString

import osmnx as ox
from osmnx import diff as oxdiff

CRS = "EPSG:4326"


def _make_graph(
    edges: list[tuple[Any, Any, dict[str, Any]]],
    *,
    nodes: dict[Any, tuple[float, float]] | None = None,
    crs: str = CRS,
) -> nx.MultiDiGraph:
    """
    Build an OSMnx-style MultiDiGraph.

    Each edge is ``(u, v, data)``. Node coordinates are derived from integer
    ids unless an explicit ``nodes`` mapping of ``id -> (x, y)`` is given.
    """
    G = nx.MultiDiGraph(crs=crs)
    used_nodes: set[Any] = set()
    for u, v, _ in edges:
        used_nodes.update((u, v))
    all_nodes: set[Any] = set(nodes) if nodes else set()
    all_nodes.update(used_nodes)
    for node in all_nodes:
        if nodes and node in nodes:
            x, y = nodes[node]
        else:
            x, y = float(node), float(node)
        G.add_node(node, osmid=node, x=x, y=y)
    for u, v, data in edges:
        G.add_edge(u, v, **data)
    return G


def _edge(osmid: Any, u: Any, v: Any, **extra: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Edge data with a default one-way tag and straight-line geometry."""
    data: dict[str, Any] = {"osmid": osmid, "oneway": True, "highway": "residential"}
    data.update(extra)
    return u, v, data


@pytest.fixture
def base_graph() -> nx.MultiDiGraph:
    return _make_graph(
        [
            _edge(101, 1, 2, name="Oak St"),
            _edge(102, 2, 3, name="Pine Ave"),
            _edge(103, 3, 4, name="Elm Rd"),
        ]
    )


def test_no_changes_on_identical_graphs(base_graph) -> None:
    result = ox.diff.diff_graphs(base_graph, base_graph.copy())
    assert result.changes == []
    assert result.review == []
    assert result.summary()["edge"]["modified"] == 0


def test_iteration_order_does_not_produce_changes() -> None:
    # same elements, inserted in a different order with shuffled list-tag order
    edges_g1 = [
        _edge(101, 1, 2, name=["Oak St", "Old Oak Rd"], highway=["residential", "tertiary"]),
        _edge(102, 2, 3, name="Pine Ave"),
    ]
    edges_g2 = [
        _edge(102, 2, 3, name="Pine Ave"),
        _edge(101, 1, 2, name=["Old Oak Rd", "Oak St"], highway=["tertiary", "residential"]),
    ]
    result = ox.diff.diff_graphs(_make_graph(edges_g1), _make_graph(edges_g2))
    assert result.edge_changes == []
    assert result.review == []


def test_added_and_closed_edges(base_graph) -> None:
    g_new = _make_graph(
        [
            _edge(101, 1, 2, name="Oak St"),
            _edge(103, 3, 4, name="Elm Rd"),
            _edge(201, 4, 5, name="New Blvd"),  # added
        ]
    )
    result = ox.diff.diff_graphs(base_graph, g_new)
    added = [c for c in result.edge_changes if c.change_type == "added"]
    closed = [c for c in result.edge_changes if c.change_type == "closed"]
    assert len(added) == 1
    assert added[0].element_id[0] == 201
    assert len(closed) == 1
    assert closed[0].element_id[0] == 102
    summary = result.summary()
    assert summary["edge"] == {"added": 1, "closed": 1, "modified": 0}


def test_attribute_modification_is_detected_order_insensitively(base_graph) -> None:
    g_new = _make_graph(
        [
            _edge(101, 1, 2, name="Oak St", highway="primary"),  # reclassified
            _edge(102, 2, 3, name="Pine Ave"),
            _edge(103, 3, 4, name="Elm Rd"),
        ]
    )
    result = ox.diff.diff_graphs(base_graph, g_new)
    mods = [c for c in result.edge_changes if c.change_type == "modified"]
    assert len(mods) == 1
    assert mods[0].element_id[0] == 101
    assert mods[0].changes["highway"] == ("residential", "primary")


def test_direction_reversal_is_real_change() -> None:
    g_old = _make_graph([_edge(101, 1, 2, oneway=True)])
    g_new = _make_graph([_edge(101, 2, 1, oneway=True)])
    result = ox.diff.diff_graphs(g_old, g_new)
    mods = [c for c in result.edge_changes if c.change_type == "modified"]
    assert len(mods) == 1
    assert "direction" in mods[0].changes
    assert mods[0].changes["direction"] == ("1->2", "2->1")
    assert "direction" in mods[0].evidence


def test_length_change_keeps_evidence() -> None:
    # move the intermediate geometry vertex far enough to exceed tolerance
    geom_old = LineString([(0.0, 0.0), (0.0, 0.01)])
    geom_new = LineString([(0.0, 0.0), (0.005, 0.01)])
    g_old = _make_graph([_edge(101, 1, 2, geometry=geom_old, length=1100.0)])
    g_new = _make_graph([_edge(101, 1, 2, geometry=geom_new, length=1300.0)])
    result = ox.diff.diff_graphs(g_old, g_new, length_tolerance=1.0)
    mod = next(c for c in result.edge_changes if c.change_type == "modified")
    assert "length" in mod.changes
    assert "geometry" in mod.changes
    # recomputed from geometry, so the stale stored length values are ignored
    assert mod.evidence["length_delta_m"] != 200.0
    assert mod.evidence["new_length_m"] > mod.evidence["old_length_m"]


def test_subthreshold_geometry_noise_is_not_a_change() -> None:
    geom_old = LineString([(0.0, 0.0), (0.0, 0.001)])
    geom_new = LineString([(0.0, 0.0), (1e-9, 0.001)])
    g_old = _make_graph([_edge(101, 1, 2, geometry=geom_old)])
    g_new = _make_graph([_edge(101, 1, 2, geometry=geom_new)])
    result = ox.diff.diff_graphs(g_old, g_new, length_tolerance=1.0)
    assert result.edge_changes == []


def test_node_add_closed_and_moved() -> None:
    g_old = _make_graph(
        [_edge(101, 1, 2)],
        nodes={1: (0.0, 0.0), 2: (1.0, 1.0), 9: (5.0, 5.0)},
    )
    g_new = _make_graph(
        [_edge(101, 1, 2)],
        nodes={1: (0.0, 0.0), 2: (1.5, 1.5), 8: (6.0, 6.0)},
    )
    result = ox.diff.diff_graphs(g_old, g_new)
    types = {c.element_id: c.change_type for c in result.node_changes}
    assert types[9] == "closed"
    assert types[8] == "added"
    moved = next(c for c in result.node_changes if c.element_id == 2)
    assert moved.change_type == "modified"
    assert "location" in moved.changes


def test_parallel_edge_ambiguity_goes_to_review() -> None:
    # one way id with two parallel edges in old release, one in new release
    g_old = _make_graph(
        [
            (1, 2, {"osmid": 101, "oneway": True, "geometry": LineString([(0, 0), (1, 1)])}),
            (1, 2, {"osmid": 101, "oneway": True, "geometry": LineString([(0, 0), (0, 1), (1, 1)])}),
        ]
    )
    g_new = _make_graph([_edge(101, 1, 2)])
    result = ox.diff.diff_graphs(g_old, g_new)
    assert result.edge_changes == []
    assert len(result.review) == 1
    item = result.review[0]
    assert item.reason == "ambiguous_parallel_edges"
    assert len(item.old_candidates) == 2
    assert len(item.new_candidates) == 1


def test_endpoint_change_without_reversal_goes_to_review() -> None:
    g_old = _make_graph([_edge(101, 1, 2)])
    g_new = _make_graph([_edge(101, 3, 4)])
    result = ox.diff.diff_graphs(g_old, g_new)
    assert result.edge_changes == []
    assert len(result.review) == 1
    assert result.review[0].reason == "ambiguous_endpoint_change"


def test_simplified_segment_composition_change_goes_to_review() -> None:
    g_old = _make_graph([_edge([101, 102], 1, 3)])
    g_new = _make_graph([_edge([101, 103], 1, 3)])
    result = ox.diff.diff_graphs(g_old, g_new)
    assert result.edge_changes == []
    assert len(result.review) == 1
    assert result.review[0].reason == "ambiguous_simplified_segment"


def test_parallel_edges_with_deterministic_matching() -> None:
    # identical parallel edges in both releases, insertion order shuffled
    geom_a = LineString([(0, 0), (1, 1)])
    geom_b = LineString([(0, 0), (0, 1), (1, 1)])
    g_old = _make_graph(
        [
            (1, 2, {"osmid": 101, "oneway": True, "geometry": geom_a}),
            (1, 2, {"osmid": 101, "oneway": True, "geometry": geom_b}),
        ]
    )
    g_new = _make_graph(
        [
            (1, 2, {"osmid": 101, "oneway": True, "geometry": geom_b}),
            (1, 2, {"osmid": 101, "oneway": True, "geometry": geom_a}),
        ]
    )
    result = ox.diff.diff_graphs(g_old, g_new)
    assert result.edge_changes == []
    assert result.review == []


def test_parallel_edge_geometry_change_goes_to_review() -> None:
    # parallel edges exist in both releases, but their geometry signatures no
    # longer align: the tool must not guess which old edge matches which new one
    geom_a = LineString([(0, 0), (1, 1)])
    geom_b = LineString([(0, 0), (0, 1), (1, 1)])
    g_old = _make_graph(
        [
            (1, 2, {"osmid": 101, "oneway": True, "geometry": geom_a}),
            (1, 2, {"osmid": 101, "oneway": True, "geometry": geom_b}),
        ]
    )
    g_new = _make_graph(
        [
            (1, 2, {"osmid": 101, "oneway": True, "geometry": geom_a}),
            (
                1,
                2,
                {"osmid": 101, "oneway": True, "geometry": LineString([(0, 0), (0, 2), (1, 1)])},
            ),
        ]
    )
    result = ox.diff.diff_graphs(g_old, g_new)
    assert result.edge_changes == []
    assert len(result.review) == 1
    assert result.review[0].reason == "ambiguous_parallel_edges"


def test_edge_without_osmid_raises() -> None:
    g_old = _make_graph([(1, 2, {"oneway": True})])
    g_new = _make_graph([_edge(101, 1, 2)])
    with pytest.raises(ValueError, match="osmid"):
        ox.diff.diff_graphs(g_old, g_new)


def test_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="MultiDiGraph"):
        ox.diff.diff_graphs(nx.Graph(), nx.MultiDiGraph())
    with pytest.raises(ValueError, match="length_tolerance"):
        ox.diff.diff_graphs(
            nx.MultiDiGraph(crs=CRS), nx.MultiDiGraph(crs=CRS), length_tolerance=-1
        )


def test_multivalue_tag_content_change_is_detected() -> None:
    # same ordering pattern, but the tag set itself genuinely changed
    g_old = _make_graph([_edge(101, 1, 2, highway=["residential", "tertiary"])])
    g_new = _make_graph([_edge(101, 1, 2, highway=["residential", "primary"])])
    result = ox.diff.diff_graphs(g_old, g_new)
    mod = next(c for c in result.edge_changes if c.change_type == "modified")
    assert mod.changes["highway"] == (("residential", "tertiary"), ("primary", "residential"))


def test_coord_precision_suppresses_tiny_node_movement() -> None:
    g_old = _make_graph([_edge(101, 1, 2)], nodes={1: (0.0, 0.0), 2: (1.0, 1.0)})
    g_new = _make_graph([_edge(101, 1, 2)], nodes={1: (0.0, 0.0), 2: (1.00000001, 1.0)})
    strict = ox.diff.diff_graphs(g_old, g_new, coord_precision=9)
    rounded = ox.diff.diff_graphs(g_old, g_new, coord_precision=7)
    assert any(c.element_id == 2 for c in strict.node_changes)
    assert rounded.node_changes == []


def test_review_items_appear_in_gdfs() -> None:
    g_old = _make_graph(
        [
            (1, 2, {"osmid": 101, "oneway": True, "geometry": LineString([(0, 0), (1, 1)])}),
            (
                1,
                2,
                {"osmid": 101, "oneway": True, "geometry": LineString([(0, 0), (0, 1), (1, 1)])},
            ),
        ]
    )
    g_new = _make_graph([_edge(101, 1, 2)])
    result = ox.diff.diff_graphs(g_old, g_new)
    _, gdf_edges = oxdiff.diff_to_gdfs(result)
    review_rows = gdf_edges[gdf_edges["change_type"] == "review"]
    assert len(review_rows) == 1
    assert review_rows.iloc[0]["attribute"] == "ambiguous_parallel_edges"


def test_diff_to_gdfs(base_graph) -> None:
    g_new = _make_graph(
        [
            _edge(101, 1, 2, name="Oak St", highway="primary"),
            _edge(103, 3, 4, name="Elm Rd"),
            _edge(201, 4, 5, name="New Blvd"),
        ]
    )
    result = ox.diff.diff_graphs(base_graph, g_new)
    gdf_nodes, gdf_edges = oxdiff.diff_to_gdfs(result)
    assert {"added", "closed", "modified"} <= set(gdf_edges["change_type"])
    # modified edge yields one row per changed attribute
    attr_rows = gdf_edges[gdf_edges["change_type"] == "modified"]
    assert set(attr_rows["attribute"]) == {"highway"}


def test_projected_graph_uses_euclidean_length() -> None:
    geom_old = LineString([(0.0, 0.0), (0.0, 100.0)])
    geom_new = LineString([(0.0, 0.0), (50.0, 100.0)])
    g_old = _make_graph(
        [_edge(101, 1, 2, geometry=geom_old)], crs="EPSG:3857"
    )
    g_new = _make_graph(
        [_edge(101, 1, 2, geometry=geom_new)], crs="EPSG:3857"
    )
    result = ox.diff.diff_graphs(g_old, g_new, length_tolerance=1.0)
    mod = next(c for c in result.edge_changes if c.change_type == "modified")
    # euclidean: 100 m vs sqrt(50^2 + 100^2) ~ 111.8 m
    assert mod.evidence["old_length_m"] == pytest.approx(100.0)
    assert mod.evidence["new_length_m"] == pytest.approx(111.8, abs=0.1)
