"""Add node elevations from raster files or web APIs, and calculate edge grades."""

from __future__ import annotations

import hashlib
import logging as lg
import multiprocessing as mp
import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import requests

from . import _http
from . import convert
from . import settings
from . import utils
from ._errors import InsufficientResponseError

if TYPE_CHECKING:
    from collections.abc import Iterable

# rasterio is an optional dependency for raster querying
try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

# names of the node and graph attributes that record where elevations came from
ELEVATION_SRC_ATTR = "elevation_src"
ELEVATION_VERSIONS_ATTR = "elevation_versions"

# how many bytes to read at a time when hashing raster files
_CHECKSUM_CHUNK_SIZE = 8 * 1024 * 1024


def add_edge_grades(G: nx.MultiDiGraph, *, add_absolute: bool = True) -> nx.MultiDiGraph:
    """
    Calculate and add `grade` attributes to all graph edges.

    Vectorized function to calculate the directed grade (i.e., rise over run)
    for each edge in the graph and add it to the edge as an attribute. Nodes
    must already have `elevation` and `length` attributes before using this
    function.

    A grade can only be calculated when both of an edge's endpoint nodes have
    valid (i.e., not null) `elevation` values. For any edge whose endpoint
    nodes do not both have valid elevations, any pre-existing `grade` and
    `grade_abs` attributes are removed instead. Likewise, when `add_absolute`
    is False, any pre-existing `grade_abs` attributes are removed so the
    returned graph never retains stale grades that no longer match the current
    node elevations.

    See also the `add_node_elevations_raster` and `add_node_elevations_google`
    functions.

    Parameters
    ----------
    G
        Graph with `elevation` node attributes.
    add_absolute
        If True, also add absolute value of grade as `grade_abs` attribute.

    Returns
    -------
    G
        Graph with `grade` (and optionally `grade_abs`) attributes on the
        edges.
    """
    elev_lookup = G.nodes(data="elevation")
    uvk = tuple(G.edges(keys=True))
    if not uvk:
        msg = "Added grade attributes to all edges"
        utils.log(msg, level=lg.INFO)
        return G

    lengths = tuple(G.edges[edge]["length"] for edge in uvk)

    # calculate edges' elevation changes from u to v then divide by lengths;
    # missing or null node elevations become NaN and invalidate the edge grade
    elevs = np.array(
        [
            [np.nan if elev_lookup[node] is None else elev_lookup[node] for node in edge[:2]]
            for edge in uvk
        ],
        dtype=float,
    )
    valid = ~np.isnan(elevs).any(axis=1)
    grades = np.full(len(uvk), np.nan)
    grades[valid] = (elevs[valid, 1] - elevs[valid, 0]) / np.array(lengths)[valid]

    # set grades on edges with valid endpoint elevations; revoke stale grades
    # on edges that lost a valid elevation at either endpoint
    grades_dict = {
        edge: grade
        for edge, grade in zip(uvk, grades, strict=True)
        if np.isfinite(grade)
    }
    nx.set_edge_attributes(G, grades_dict, name="grade")

    # optionally add grade absolute value; always revoke stale grade/grade_abs
    # attributes on invalid edges so nothing retained contradicts the elevations
    invalid_edges = [edge for edge, is_valid in zip(uvk, valid, strict=True) if not is_valid]
    for edge in invalid_edges:
        G.edges[edge].pop("grade", None)
        G.edges[edge].pop("grade_abs", None)
    if add_absolute:
        grade_abs_dict = {edge: abs(grade) for edge, grade in grades_dict.items()}
        nx.set_edge_attributes(G, grade_abs_dict, name="grade_abs")
    else:
        for edge in uvk:
            G.edges[edge].pop("grade_abs", None)

    msg = "Added grade attributes to all edges"
    utils.log(msg, level=lg.INFO)
    return G


def _query_raster(
    nodes: pd.DataFrame,
    filepath: str | Path,
    band: int,
) -> Iterable[tuple[int, Any]]:
    """
    Query a raster file for values at coordinates in DataFrame x/y columns.

    Raster nodata values are returned as `numpy.nan` so they can never
    overwrite a valid elevation sampled from a higher-priority source.

    Parameters
    ----------
    nodes
        DataFrame indexed by node ID and with two columns representing x and y
        coordinates.
    filepath
        Path to the raster file to query.
    band
        Which raster band to query.

    Returns
    -------
    nodes_values
        Zip of node IDs and corresponding raster values.
    """
    # must open raster file here: cannot pickle it to pass in multiprocessing
    with rasterio.open(filepath) as raster:
        values = np.array(tuple(raster.sample(nodes.to_numpy(), band)), dtype=float).reshape(-1)
        values[values == raster.nodata] = np.nan
        return zip(nodes.index, values, strict=True)


def _raster_checksum(filepath: str | Path) -> str:
    """
    Calculate a SHA-256 checksum of a raster file's contents.

    The checksum identifies the actual file version, so two deliveries of
    same-named rasters (including a file overwritten in place) are still
    distinguishable by their content.

    Parameters
    ----------
    filepath
        Path to the raster file to checksum.

    Returns
    -------
    checksum
        The file's hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(filepath).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHECKSUM_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_node_elevations_raster(
    G: nx.MultiDiGraph,
    filepath: str | Path | Iterable[str | Path],
    *,
    band: int = 1,
    cpus: int | None = None,
) -> nx.MultiDiGraph:
    """
    Add `elevation` attributes to all nodes from local raster file(s).

    If `filepath` is an iterable of paths, they are treated as an ordered
    sequence of elevation sources in descending priority: the first raster in
    the sequence that provides a valid sample at a node determines that
    node's elevation. Nodata pixels and pixels outside a raster's extent never
    overwrite a valid value, so where a high-priority raster is nodata the
    node falls back to the next raster that samples it validly. This makes the
    result reproducible: recomputing from scratch with an ordered list of
    rasters yields the same node elevations (and therefore the same directed
    edge grades) as applying successive deliveries incrementally.

    Each node's contributing source is recorded in its `elevation_src`
    attribute as the source raster's path. A graph-level `elevation_versions`
    attribute maps each source path to the SHA-256 checksum of the file
    actually sampled, so elevations can be traced back to the exact file
    version even when different deliveries share the same filename. Nodes that
    no raster samples validly receive a null elevation.

    See also the `add_edge_grades` function.

    Parameters
    ----------
    G
        Graph in same CRS as raster.
    filepath
        The path(s) to the raster file(s) to query, ordered from highest to
        lowest priority.
    band
        Which raster band to query.
    cpus
        How many CPU cores to use if multiprocessing. If None, use all
        available. If you are multiprocessing, make sure you protect your
        entry point: see the Python docs for details.

    Returns
    -------
    G
        Graph with `elevation` attributes on the nodes.
    """
    if rasterio is None:  # pragma: no cover
        msg = "rasterio must be installed as an optional dependency to query rasters."
        raise ImportError(msg)

    # normalize to an ordered, de-duplicated list of raster file paths
    if isinstance(filepath, (str, Path)):
        filepaths = [Path(filepath)]
    else:
        filepaths = list(dict.fromkeys(Path(path) for path in filepath))

    if cpus is None:
        cpus = mp.cpu_count()
    cpus = min(cpus, mp.cpu_count())
    msg = f"Attaching elevations with {cpus} CPUs..."
    utils.log(msg, level=lg.INFO)

    # record the checksum of every sampled file so each elevation can be
    # traced to the exact file version, including same-named deliveries
    versions = {str(path): _raster_checksum(path) for path in filepaths}

    nodes = convert.graph_to_gdfs(G, edges=False, node_geometry=False)[["x", "y"]]

    # sample every source at every node (serially or in chunks across a
    # process pool), then merge the samples in priority order so each node
    # keeps the first valid value and falls back past nodata samples
    tasks = [(nodes, path) for path in filepaths]
    sampled = _sample_tasks(tasks, band, cpus)
    elevations = pd.Series(np.nan, index=nodes.index, dtype=float)
    sources = pd.Series(np.nan, index=nodes.index, dtype=object)
    for path, values in zip(filepaths, sampled, strict=True):
        for node_id, value in values.items():
            if pd.isna(elevations.loc[node_id]) and pd.notna(value):
                elevations.loc[node_id] = value
                sources.loc[node_id] = str(path)

    # null elevations mark nodes that no source sampled validly; their source
    # is left unset so nothing claims a nodata delivery. Remove any pre-existing
    # provenance first so nodes that lost their valid sample don't keep stale
    # source claims from an earlier delivery.
    for node_data in G.nodes(data=False):
        G.nodes[node_data].pop(ELEVATION_SRC_ATTR, None)
    nx.set_node_attributes(G, elevations.to_dict(), name="elevation")
    nx.set_node_attributes(G, sources.dropna().to_dict(), name=ELEVATION_SRC_ATTR)
    G.graph[ELEVATION_VERSIONS_ATTR] = versions

    msg = "Added elevation data from raster to all nodes"
    utils.log(msg, level=lg.INFO)
    return G


def _sample_tasks(
    tasks: list[tuple[pd.DataFrame, Path]],
    band: int,
    cpus: int,
) -> list[dict[int, float]]:
    """
    Sample each task's nodes from its raster file.

    With one CPU each task is queried directly. With multiple CPUs every
    task's nodes are split into equal-sized chunks and all chunks are sampled
    in a single process pool.

    Parameters
    ----------
    tasks
        Ordered list of (nodes DataFrame, raster path) sampling tasks.
    band
        Which raster band to query.
    cpus
        How many CPU cores to use if multiprocessing.

    Returns
    -------
    sampled
        For each task, a dict of node ID to sampled value (nodata becomes
        `numpy.nan`), in the same order as `tasks`.
    """
    if cpus == 1:
        return [dict(_query_raster(task_nodes, path, band)) for task_nodes, path in tasks]

    chunked: list[tuple[pd.DataFrame, Path, int]] = []
    chunk_counts = []
    for task_nodes, path in tasks:
        size = max(int(np.ceil(len(task_nodes) / cpus)), 1)
        starts = range(0, len(task_nodes), size)
        chunk_counts.append(len(starts))
        for i in starts:
            chunked.append((task_nodes.iloc[i : i + size], path, band))

    # nothing to sample (e.g., an empty graph): skip spinning up a process pool
    if not chunked:
        return [{} for _ in tasks]

    with mp.get_context().Pool(cpus) as pool:
        results = pool.starmap_async(_query_raster, chunked).get()

    # recombine each task's chunks back into a single dict
    sampled: list[dict[int, float]] = []
    result_iter = iter(results)
    for n_chunks in chunk_counts:
        values: dict[int, float] = {}
        for _ in range(n_chunks):
            values.update(next(result_iter))
        sampled.append(values)
    return sampled


def add_node_elevations_google(
    G: nx.MultiDiGraph,
    *,
    api_key: str | None = None,
    batch_size: int = 512,
    pause: float = 0,
) -> nx.MultiDiGraph:
    """
    Add `elevation` (meters) attributes to all nodes using a web API.

    By default this uses the Google Maps Elevation API, but you could instead
    use any equivalent API with the same interface and response format (such
    as the Open Topo Data API or the Open-Elevation API) via the `settings`
    module's `elevation_url_template`. Adjust the `batch_size` and `pause`
    arguments as needed for the provider. The Google Maps Elevation API
    requires an API key but other providers may not. You can find more
    information about the Google Maps Elevation API interface and format at:
    https://developers.google.com/maps/documentation/elevation

    For a free local alternative see the `add_node_elevations_raster`
    function. See also the `add_edge_grades` function.

    Parameters
    ----------
    G
        Graph to add elevation data to.
    api_key
        A valid API key. Can be None if the API does not require a key.
    batch_size
        Max number of coordinate pairs to submit in each request (depends on
        provider's limits). Google's limit is 512.
    pause
        How long to pause in seconds between API calls, which can be increased
        if you get rate limited.

    Returns
    -------
    G
        Graph with `elevation` attributes on the nodes.
    """
    # make a pandas series of all the nodes' coordinates as "lat,lon" and
    # round coordinates to 6 decimal places (approx 5 to 10 cm resolution)
    node_points = pd.Series({n: f"{d['y']:.6f},{d['x']:.6f}" for n, d in G.nodes(data=True)})
    n_calls = int(np.ceil(len(node_points) / batch_size))
    hostname = _http._hostname_from_url(settings.elevation_url_template)

    msg = f"Requesting node elevations from {hostname!r} in {n_calls} request(s)"
    utils.log(msg, level=lg.INFO)

    # break the series of coordinates into chunks of batch_size
    # API format is locations=lat,lon|lat,lon|lat,lon|lat,lon...
    results = []
    for i in range(0, len(node_points), batch_size):
        chunk = node_points.iloc[i : i + batch_size]
        locations = "|".join(chunk)
        url = settings.elevation_url_template.format(locations=locations, key=api_key)

        # download and append these elevation results to list of all results
        response_json = _elevation_request(url, pause)
        if "results" in response_json and len(response_json["results"]) > 0:
            results.extend(response_json["results"])
        else:
            raise InsufficientResponseError(str(response_json))

    # sanity check that all our vectors have the same number of elements
    msg = f"Graph has {len(G):,} nodes and we received {len(results):,} results from {hostname!r}"
    utils.log(msg, level=lg.INFO)
    if not (len(results) == len(G) == len(node_points)):  # pragma: no cover
        err_msg = f"{msg}\n{response_json}"
        raise InsufficientResponseError(err_msg)

    # add elevation as an attribute to the nodes
    df_elev = pd.DataFrame(node_points, columns=["node_points"])
    df_elev["elevation"] = [result["elevation"] for result in results]
    nx.set_node_attributes(G, name="elevation", values=df_elev["elevation"].to_dict())
    msg = f"Added elevation data from {hostname!r} to all nodes."
    utils.log(msg, level=lg.INFO)

    return G


def _elevation_request(url: str, pause: float) -> dict[str, Any]:
    """
    Send a HTTP GET request to a Google Maps-style elevation API.

    Parameters
    ----------
    url
        URL of API endpoint, populated with request data.
    pause
        How long to pause in seconds before request.

    Returns
    -------
    response_json
        The elevation API's response.
    """
    # check if request already exists in cache
    cached_response_json = _http._retrieve_from_cache(url)
    if isinstance(cached_response_json, dict):
        return cached_response_json

    # pause then request this URL
    hostname = _http._hostname_from_url(url)
    msg = f"Pausing {pause} second(s) before making HTTP GET request to {hostname!r}"
    utils.log(msg, level=lg.INFO)
    time.sleep(pause)

    # transmit the HTTP GET request
    msg = f"Get {url} with timeout={settings.requests_timeout}"
    utils.log(msg, level=lg.INFO)
    response = requests.get(
        url,
        timeout=settings.requests_timeout,
        headers=_http._get_http_headers(),
        **settings.requests_kwargs,
    )

    response_json = _http._parse_response(response)
    if not isinstance(response_json, dict):  # pragma: no cover
        msg = "Elevation API did not return a dict of results."
        raise InsufficientResponseError(msg)
    _http._save_to_cache(url, response_json, response.ok)
    return response_json
