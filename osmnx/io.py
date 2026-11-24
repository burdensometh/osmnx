"""File I/O functions to save/load graphs to/from files on disk."""

from __future__ import annotations

import ast
import contextlib
import logging as lg
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import networkx as nx
import pandas as pd
from shapely import wkt

from . import _osm_xml
from . import convert
from . import settings
from . import utils

if TYPE_CHECKING:
    import geopandas as gpd


def save_graph_geopackage(
    G: nx.MultiDiGraph,
    filepath: str | Path | None = None,
    *,
    directed: bool = False,
    encoding: str = "utf-8",
    manifest: bool | str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """
    Save graph nodes and edges to disk as layers in a GeoPackage file.

    Parameters
    ----------
    G
        The graph to save.
    filepath
        Path to the GeoPackage file including extension. If None, use default
        `settings.data_folder/graph.gpkg`.
    directed
        If False, save one edge for each undirected edge in the graph but
        retain original oneway and to/from information as edge attributes. If
        True, save one edge for each directed edge in the graph.
    encoding
        The character encoding of the saved GeoPackage file.
    manifest
        Optional export manifest. If True, write a
        ``<filepath>.manifest.json`` reconciliation manifest describing field
        types and unsupported fields. If a path, write the manifest there and
        also return it. Default None writes nothing.
    **kwargs
        Accepts ``write_manifest``/``with_manifest`` aliases for `manifest`.

    Returns
    -------
    manifest : ExportManifest or None
        The manifest object only when `manifest` is truthy, otherwise None.
    """
    manifest = _manifest_kwarg(manifest, kwargs)
    # default filepath if none was provided
    filepath = Path(settings.data_folder) / "graph.gpkg" if filepath is None else Path(filepath)

    # if save folder does not already exist, create it
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # convert graph to gdfs and stringify non-numeric columns
    if directed:
        gdf_nodes, gdf_edges = convert.graph_to_gdfs(G)
    else:
        gdf_nodes, gdf_edges = convert.graph_to_gdfs(convert.to_undirected(G))
    gdf_nodes = _stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = _stringify_nonnumeric_cols(gdf_edges)

    # save the nodes and edges as GeoPackage layers
    gdf_nodes.to_file(filepath, layer="nodes", driver="GPKG", index=True, encoding=encoding)
    gdf_edges.to_file(filepath, layer="edges", driver="GPKG", index=True, encoding=encoding)

    msg = f"Saved graph as GeoPackage at {str(filepath)!r}"
    utils.log(msg, level=lg.INFO)
    return _maybe_write_manifest(G, filepath, "geopackage", manifest, kwargs)


def _manifest_kwarg(
    manifest: bool | str | Path | None,
    kwargs: dict[str, Any],
) -> bool | str | Path | None:
    """Resolve the manifest option from its keyword aliases."""
    if manifest is not None:
        return manifest
    for alias in ("write_manifest", "with_manifest", "export_manifest"):
        if kwargs.get(alias):
            return kwargs[alias]
    return None


def _maybe_write_manifest(
    G: nx.MultiDiGraph,
    filepath: Path,
    fmt: str,
    manifest: bool | str | Path,
    kwargs: dict[str, Any],
) -> Any:
    """Build and optionally write a reconciliation manifest after saving."""
    if not manifest:
        return None
    from .reconcile import build_manifest

    edge_id_attr = kwargs.get("edge_id_attr", "edge_id")
    manifest_obj = build_manifest(G, fmt=fmt, path=filepath, edge_id_attr=edge_id_attr)
    if isinstance(manifest, str | Path):
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            # interpret a relative manifest name next to the export file
            manifest_path = filepath.parent / manifest_path
        manifest_obj.write_json(manifest_path)
    else:
        manifest_obj.write_json(filepath.with_suffix(filepath.suffix + ".manifest.json"))
    return manifest_obj


def load_graph_geopackage(
    filepath: str | Path | None = None,
    *,
    nodes_layer: str = "nodes",
    edges_layer: str = "edges",
    node_dtypes: dict[str, Any] | None = None,
    edge_dtypes: dict[str, Any] | None = None,
    graph_attrs: dict[str, Any] | None = None,
    restore_types: bool = True,
    empty_as_missing: bool = True,
    **kwargs: Any,
) -> nx.MultiDiGraph:
    """
    Load an OSMnx-saved GeoPackage file from disk into a MultiDiGraph.

    This is the symmetric read counterpart to `save_graph_geopackage`. Reading
    a GeoPackage layer back through plain `geopandas.read_file` and
    `convert.graph_from_gdfs` preserves the `(u, v, key)` MultiIndex, but
    non-numeric/mixed columns were stringified on save, so multi-value tags
    become `"['a', 'b']"` text and booleans in mixed columns can become
    `"True"`/`"False"` text. This function restores those logical types using
    the same default dtype map as `load_graphml` (overridable via
    `node_dtypes`/`edge_dtypes`), so parallel edges, multi-value tags, and
    boolean direction attributes survive a GeoPackage round trip.

    Parameters
    ----------
    filepath
        Path to the GeoPackage file. May also be passed as `path` or `fp`.
    nodes_layer
        Name of the nodes layer.
    edges_layer
        Name of the edges layer.
    node_dtypes
        Dict of node attribute names:types/converter callables, overriding the
        defaults. List-valued attributes have the converter applied per item.
    edge_dtypes
        Dict of edge attribute names:types/converter callables, overriding the
        defaults. List-valued attributes have the converter applied per item.
    graph_attrs
        Graph-level attributes passed to `convert.graph_from_gdfs`. If None,
        the edges layer CRS is used as the only graph-level attribute.
    restore_types
        If True, parse stringified containers and convert attribute dtypes.
    empty_as_missing
        If True, treat empty strings (written from null values by
        `save_graph_geopackage`) as absent attributes.
    **kwargs
        Accepts `path`/`fp` aliases for `filepath`.

    Returns
    -------
    G
        The loaded MultiDiGraph, with `(u, v, key)` edges intact.
    """
    filepath = _resolve_filepath(filepath, kwargs)

    # geopandas is a core dependency, but import it lazily to match the
    # module-level TYPE_CHECKING import convention above.
    import geopandas as gpd

    gdf_nodes = gpd.read_file(filepath, layer=nodes_layer)
    gdf_edges = gpd.read_file(filepath, layer=edges_layer)

    gdf_nodes = _prepare_gpkg_nodes(gdf_nodes)
    gdf_edges = _prepare_gpkg_edges(gdf_edges)

    G = convert.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=graph_attrs)

    if restore_types:
        merged_node_dtypes = _gpkg_default_node_dtypes()
        merged_edge_dtypes = _gpkg_default_edge_dtypes()
        if node_dtypes is not None:
            merged_node_dtypes.update(node_dtypes)
        if edge_dtypes is not None:
            merged_edge_dtypes.update(edge_dtypes)
        _coerce_graph_dtypes(
            G,
            node_dtypes=merged_node_dtypes,
            edge_dtypes=merged_edge_dtypes,
            empty_as_missing=empty_as_missing,
        )

    msg = (
        f"Loaded graph with {len(G)} nodes and {len(G.edges)} edges "
        f"from GeoPackage {str(filepath)!r}"
    )
    utils.log(msg, level=lg.INFO)
    return G


# concise aliases for the GeoPackage loader, bound to the implementation below
read_graph_geopackage = load_graph_geopackage
load_geopackage = load_graph_geopackage
read_gpkg = load_graph_geopackage
graph_from_geopackage = load_graph_geopackage


def _resolve_filepath(
    filepath: str | Path | None,
    kwargs: dict[str, Any],
) -> Path:
    """Resolve a filepath from the positional arg or common keyword aliases."""
    if filepath is None:
        for alias in ("path", "fp", "file", "filename"):
            if alias in kwargs and kwargs[alias] is not None:
                filepath = kwargs[alias]
                break
    if filepath is None:
        msg = "You must pass a filepath (or `path`/`fp`)."
        raise ValueError(msg)
    return Path(filepath)


def _prepare_gpkg_nodes(gdf_nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure nodes GeoDataFrame is indexed by `osmid`."""
    if "osmid" in gdf_nodes.columns:
        gdf_nodes = gdf_nodes.set_index("osmid")
    gdf_nodes.index.name = "osmid"
    return gdf_nodes


def _prepare_gpkg_edges(gdf_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure edges GeoDataFrame has a `(u, v, key)` MultiIndex."""
    if {"u", "v", "key"}.issubset(gdf_edges.columns):
        gdf_edges = gdf_edges.set_index(["u", "v", "key"])
    elif list(gdf_edges.index.names) != ["u", "v", "key"]:
        msg = "GeoPackage edges layer must contain u, v, and key columns."
        raise ValueError(msg)
    return gdf_edges


def _gpkg_default_node_dtypes() -> dict[str, Any]:
    """Default node converters for a GeoPackage readback."""
    return {
        "elevation": float,
        "elevation_res": float,
        "osmid": int,
        "street_count": int,
        "x": float,
        "y": float,
    }


def _gpkg_default_edge_dtypes() -> dict[str, Any]:
    """Default edge converters for a GeoPackage readback."""
    return {
        "bearing": float,
        "grade": float,
        "grade_abs": float,
        "length": float,
        "oneway": _coerce_bool,
        "osmid": int,
        "reversed": _coerce_bool,
        "speed_kph": float,
        "travel_time": float,
    }


def _coerce_bool(value: Any) -> bool:  # noqa: FBT001
    """
    Robustly coerce a bool, numpy bool, 0/1, or true/false literal to bool.

    Unlike `_convert_bool_string`, this also accepts numpy booleans and numeric
    0/1 values as returned by GeoPackage drivers. Raises ValueError for values
    that do not unambiguously represent a boolean.
    """
    if isinstance(value, bool):
        return value

    # handle numpy.bool_ without importing numpy at module load
    if type(value).__name__ == "bool_":
        return bool(value)

    if isinstance(value, int | float) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False

    if isinstance(value, str):
        literal = value.strip()
        if literal in {"True", "true", "TRUE", "1"}:
            return True
        if literal in {"False", "false", "FALSE", "0"}:
            return False

    msg = f"Invalid literal for boolean: {value!r}."
    raise ValueError(msg)


def _maybe_eval_literal(value: Any) -> Any:
    """Parse a stringified list/dict/set/tuple, else return value unchanged."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if len(stripped) >= 2 and (
        (stripped[0] in "[{" and stripped[-1] in "]}")
        or (stripped[0] == "(" and stripped[-1] == ")")
    ):
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            try:
                import json

                return json.loads(stripped)
            except (ValueError, TypeError):
                return value
    return value


def _coerce_graph_dtypes(
    G: nx.MultiDiGraph,
    *,
    node_dtypes: dict[str, Any] | None = None,
    edge_dtypes: dict[str, Any] | None = None,
    empty_as_missing: bool = True,
) -> nx.MultiDiGraph:
    """
    Restore container literals and attribute dtypes after a tabular read.

    Works in place on `G` and returns it. Container strings are parsed with
    `ast.literal_eval` (and JSON as a fallback), then any converter in the
    dtype maps is applied (per item for list/tuple values). Empty strings can
    be removed because `save_graph_geopackage` writes nulls as `""`.
    """
    node_dtypes = node_dtypes or {}
    edge_dtypes = edge_dtypes or {}

    for _, data in G.nodes(data=True):
        _coerce_attr_dict(data, node_dtypes, empty_as_missing)

    for _, _, data in G.edges(keys=False, data=True):
        _coerce_attr_dict(data, edge_dtypes, empty_as_missing)

    return G


def _coerce_attr_dict(
    data: dict[str, Any],
    dtypes: dict[str, Any],
    empty_as_missing: bool,
) -> None:
    """Coerce one node/edge attribute dict in place."""
    if empty_as_missing:
        empty_keys = [k for k, v in data.items() if isinstance(v, str) and v == ""]
        for k in empty_keys:
            data.pop(k, None)

    for attr in list(data.keys()):
        data[attr] = _maybe_eval_literal(data[attr])

    for attr in data.keys() & dtypes.keys():
        converter = dtypes[attr]
        value = data[attr]
        if isinstance(value, list | tuple):
            data[attr] = [converter(item) for item in value]
        elif value is None:
            continue
        else:
            with contextlib.suppress(ValueError, TypeError):
                data[attr] = converter(value)


def save_graphml(
    G: nx.MultiDiGraph,
    filepath: str | Path | None = None,
    *,
    gephi: bool = False,
    encoding: str = "utf-8",
    manifest: bool | str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """
    Save graph to disk as GraphML file.

    Parameters
    ----------
    G
        The graph to save as.
    filepath
        Path to the GraphML file including extension. If None, use default
        `settings.data_folder/graph.graphml`.
    gephi
        If True, give each edge a unique key/id for compatibility with Gephi's
        interpretation of the GraphML specification.
    encoding
        The character encoding of the saved GraphML file.
    manifest
        Optional export manifest. If True, write a
        ``<filepath>.manifest.json`` reconciliation manifest; if a path,
        write it there and return it. Default None writes nothing.
    **kwargs
        Accepts ``write_manifest``/``with_manifest`` aliases and
        ``edge_id_attr`` for the manifest.

    Returns
    -------
    manifest : ExportManifest or None
        The manifest object only when `manifest` is truthy.
    """
    manifest = _manifest_kwarg(manifest, kwargs)
    # default filepath if none was provided
    filepath = Path(settings.data_folder) / "graph.graphml" if filepath is None else Path(filepath)

    # if save folder does not already exist, create it
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # preserve the original-typed graph for any manifest built below, then
    # make a copy to not mutate original graph object caller passed in
    source_G = G
    G = G.copy()

    if gephi:
        # for gephi compatibility, each edge's key must be unique as an id
        uvkd = [(u, v, k, d) for k, (u, v, d) in enumerate(G.edges(keys=False, data=True))]
        G.clear_edges()
        G.add_edges_from(uvkd)

    # stringify all the graph attribute values
    for attr, value in G.graph.items():
        G.graph[attr] = str(value)

    # stringify all the node attribute values
    for _, data in G.nodes(data=True):
        for attr, value in data.items():
            data[attr] = str(value)

    # stringify all the edge attribute values
    for _, _, data in G.edges(keys=False, data=True):
        for attr, value in data.items():
            data[attr] = str(value)

    nx.write_graphml(G, path=filepath, encoding=encoding)
    msg = f"Saved graph as GraphML file at {str(filepath)!r}"
    utils.log(msg, level=lg.INFO)
    return _maybe_write_manifest(source_G, filepath, "graphml", manifest, kwargs)


def load_graphml(
    filepath: str | Path | None = None,
    *,
    graphml_str: str | None = None,
    node_dtypes: dict[str, Any] | None = None,
    edge_dtypes: dict[str, Any] | None = None,
    graph_dtypes: dict[str, Any] | None = None,
) -> nx.MultiDiGraph:
    """
    Load an OSMnx-saved GraphML file from disk or GraphML string.

    This function converts node, edge, and graph-level attributes (serialized
    as strings) to their appropriate data types. These can be customized as
    needed by passing in dtypes arguments providing types or custom converter
    functions. For example, if you want to convert some attribute's values to
    `bool`, consider using the built-in `ox.io._convert_bool_string` function
    to properly handle "True"/"False" string literals as True/False booleans:
    `ox.load_graphml(fp, node_dtypes={my_attr: ox.io._convert_bool_string})`.

    If you manually configured the `all_oneway=True` setting, you may need to
    manually specify here that edge `oneway` attributes should be type `str`.

    Note that you must pass one and only one of `filepath` or `graphml_str`.
    If passing `graphml_str`, you may need to decode the bytes read from your
    file before converting to string to pass to this function.

    Parameters
    ----------
    filepath
        Path to the GraphML file.
    graphml_str
        Valid and decoded string representation of a GraphML file's contents.
    node_dtypes
        Dict of node attribute names:types to convert values' data types. The
        type can be a type or a custom string converter function.
    edge_dtypes
        Dict of edge attribute names:types to convert values' data types. The
        type can be a type or a custom string converter function.
    graph_dtypes
        Dict of graph-level attribute names:types to convert values' data
        types. The type can be a type or a custom string converter function.

    Returns
    -------
    G
        The loaded MultiDiGraph.
    """
    if (filepath is None and graphml_str is None) or (
        filepath is not None and graphml_str is not None
    ):  # pragma: no cover
        msg = "You must pass one and only one of `filepath` or `graphml_str`."
        raise ValueError(msg)

    # specify default graph/node/edge attribute values' data types
    default_graph_dtypes = {
        "consolidated": _convert_bool_string,
        "simplified": _convert_bool_string,
    }
    default_node_dtypes = {
        "elevation": float,
        "elevation_res": float,
        "osmid": int,
        "street_count": int,
        "x": float,
        "y": float,
    }
    default_edge_dtypes = {
        "bearing": float,
        "grade": float,
        "grade_abs": float,
        "length": float,
        "oneway": _convert_bool_string,
        "osmid": int,
        "reversed": _convert_bool_string,
        "speed_kph": float,
        "travel_time": float,
    }

    # override default graph/node/edge attr types with user-passed types, if any
    if graph_dtypes is not None:
        default_graph_dtypes.update(graph_dtypes)
    if node_dtypes is not None:
        default_node_dtypes.update(node_dtypes)
    if edge_dtypes is not None:
        default_edge_dtypes.update(edge_dtypes)

    if filepath is not None:
        # read the graphml file from disk
        source = filepath
        G = nx.read_graphml(
            Path(filepath),
            node_type=default_node_dtypes["osmid"],
            force_multigraph=True,
        )
    else:
        # parse the graphml string
        source = "string"
        G = nx.parse_graphml(
            graphml_str,
            node_type=default_node_dtypes["osmid"],
            force_multigraph=True,
        )

    # convert graph/node/edge attribute data types
    msg = "Converting node, edge, and graph-level attribute data types"
    utils.log(msg, level=lg.INFO)
    G = _convert_graph_attr_types(G, default_graph_dtypes)
    G = _convert_node_attr_types(G, default_node_dtypes)
    G = _convert_edge_attr_types(G, default_edge_dtypes)

    msg = f"Loaded graph with {len(G)} nodes and {len(G.edges)} edges from {str(source)!r}"
    utils.log(msg, level=lg.INFO)
    return G


def save_graph_xml(
    G: nx.MultiDiGraph,
    filepath: str | Path | None = None,
    *,
    way_tag_aggs: dict[str, Any] | None = None,
    encoding: str = "utf-8",
) -> None:
    """
    Save graph to disk as an OSM XML file.

    This function exists only to allow serialization to the OSM XML format
    for applications that require it, and has constraints to conform to that.
    As such, it has a limited use case which does not include saving/loading
    graphs for subsequent OSMnx analysis. To save/load graphs to/from disk for
    later use in OSMnx, use the `io.save_graphml` and `io.load_graphml`
    functions instead. To load a graph from an OSM XML file that you have
    downloaded or generated elsewhere, use the `graph.graph_from_xml`
    function.

    Use the `settings` module's `useful_tags_node` and `useful_tags_way`
    settings to configure which tags your graph is created and saved with.
    This function merges graph edges such that each OSM way has one entry in
    the XML output, with the way's nodes topologically sorted. `G` must be
    unsimplified to save as OSM XML: otherwise, one edge could comprise
    multiple OSM ways, making it impossible to group and sort edges in way.
    `G` should also have been created with `ox.settings.all_oneway=True` for
    this function to behave properly.

    Parameters
    ----------
    G
        Unsimplified, unprojected graph to save as an OSM XML file.
    filepath
        Path to the saved file including extension. If None, use default
        `settings.data_folder/graph.osm`.
    way_tag_aggs
        Keys are OSM way tag keys and values are aggregation functions
        (anything accepted as an argument by pandas.agg). Allows user to
        aggregate graph edge attribute values into single OSM way values. If
        None, or if some tag's key does not exist in the dict, the way
        attribute will be assigned the value of the first edge of the way.
    encoding
        The character encoding of the saved OSM XML file.
    """
    _osm_xml._save_graph_xml(G, filepath, way_tag_aggs, encoding)


def _convert_graph_attr_types(G: nx.MultiDiGraph, dtypes: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Convert graph-level attributes using a dict of data types.

    Parameters
    ----------
    G
        Graph to convert the graph-level attributes of.
    dtypes
        Dict of graph-level attribute names:types.

    Returns
    -------
    G
        The graph with its graph-level attributes' types converted.
    """
    # remove node_default and edge_default metadata keys if they exist
    G.graph.pop("node_default", None)
    G.graph.pop("edge_default", None)

    for attr in G.graph.keys() & dtypes.keys():
        G.graph[attr] = dtypes[attr](G.graph[attr])

    return G


def _convert_node_attr_types(G: nx.MultiDiGraph, dtypes: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Convert graph nodes' attributes using a dict of data types.

    Parameters
    ----------
    G
        Graph to convert the node attributes of.
    dtypes
        Dict of node attribute names:types.

    Returns
    -------
    G
        The graph with its nodes' attributes' types converted.
    """
    for _, data in G.nodes(data=True):
        # first, eval stringified lists, dicts, or sets to convert them to objects
        # lists, dicts, or sets would be custom attribute types added by a user
        for attr, value in data.items():
            if (value.startswith("[") and value.endswith("]")) or (
                value.startswith("{") and value.endswith("}")
            ):
                with contextlib.suppress(SyntaxError, ValueError):
                    data[attr] = ast.literal_eval(value)

        for attr in data.keys() & dtypes.keys():
            data[attr] = dtypes[attr](data[attr])
    return G


def _convert_edge_attr_types(G: nx.MultiDiGraph, dtypes: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Convert graph edges' attributes using a dict of data types.

    Parameters
    ----------
    G
        Graph to convert the edge attributes of.
    dtypes
        Dict of edge attribute names:types.

    Returns
    -------
    G
        The graph with its edges' attributes' types converted.
    """
    # for each edge in the graph, eval attribute value lists and convert types
    for _, _, data in G.edges(data=True, keys=False):
        # remove extraneous "id" attribute added by graphml saving
        data.pop("id", None)

        # first, eval stringified lists, dicts, or sets to convert them to objects
        # edge attributes might have a single value, or a list if simplified
        # dicts or sets would be custom attribute types added by a user
        for attr, value in data.items():
            if (value.startswith("[") and value.endswith("]")) or (
                value.startswith("{") and value.endswith("}")
            ):
                with contextlib.suppress(SyntaxError, ValueError):
                    data[attr] = ast.literal_eval(value)

        # next, convert attribute value types if attribute appears in dtypes
        for attr in data.keys() & dtypes.keys():
            if isinstance(data[attr], list):
                # if it's a list, eval it then convert each item
                data[attr] = [dtypes[attr](item) for item in data[attr]]
            else:
                # otherwise, just convert the single value
                data[attr] = dtypes[attr](data[attr])

        # if "geometry" attr exists, convert its well-known text to LineString
        if "geometry" in data:
            data["geometry"] = wkt.loads(data["geometry"])

    return G


def _convert_bool_string(value: bool | str) -> bool:  # noqa: FBT001
    """
    Convert a "True" or "False" string literal to corresponding boolean type.

    This is necessary because Python will otherwise parse the string "False"
    to the boolean value True, that is, `bool("False") == True`. This function
    raises a ValueError if a value other than "True" or "False" is passed.

    If the value is already a boolean, this function just returns it, to
    accommodate usage when the value was originally inside a stringified list.

    Parameters
    ----------
    value
        The string to convert to bool.

    Returns
    -------
    bool_value
        The boolean equivalent of the string literal.
    """
    if isinstance(value, bool):
        return value

    if value in {"True", "False"}:
        return value == "True"

    # otherwise the value is not a valid boolean
    msg = f"Invalid literal for boolean: {value!r}."
    raise ValueError(msg)


def _stringify_nonnumeric_cols(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make every non-numeric GeoDataFrame column (besides geometry) a string.

    This allows proper serializing via Fiona of GeoDataFrames with mixed types
    such as strings and ints in the same column.

    Parameters
    ----------
    gdf
        GeoDataFrame to stringify non-numeric columns of.

    Returns
    -------
    gdf
        GeoDataFrame with non-numeric columns stringified.
    """
    # stringify every non-numeric column other than geometry column
    for col in (c for c in gdf.columns if c != "geometry"):
        if not pd.api.types.is_numeric_dtype(gdf[col]):
            gdf[col] = gdf[col].fillna("").astype(str)

    return gdf
