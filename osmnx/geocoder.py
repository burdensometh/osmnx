"""
Geocode place names or addresses or retrieve OSM elements by place name or ID.

This module uses the Nominatim API's "search" and "lookup" endpoints. For more
details see https://wiki.openstreetmap.org/wiki/Elements and
https://nominatim.org/.

A place name can match several OSM elements (for example, several same-named
districts), so a search can return several boundary polygons. Callers can
select one or more of those results by their position in the importance-sorted
result list, or retrieve them directly by OSM ID. Each returned row records the
query it came from and the identity of the OSM element it represents.
"""

from __future__ import annotations

import logging as lg
import re
from collections import OrderedDict
from typing import Any

import geopandas as gpd
import pandas as pd

from . import _nominatim
from . import settings
from . import utils
from ._errors import InsufficientResponseError

# geometry types that describe a place boundary rather than a point or line
_POLYGON_TYPES = {"Polygon", "MultiPolygon"}

# patterns matching caller-provided OSM ID strings: "R1234" or "relation/1234"
_OSM_ID_PATTERNS = (
    re.compile(r"^(?P<type>[nwrNWR])(?P<id>\d+)$"),
    re.compile(r"^(?P<type>[a-zA-Z]+)\s*/\s*(?P<id>\d+)$"),
)

# columns that describe the provenance of each geocoding result
_PROVENANCE_COLUMNS = (
    "name",
    "display_name",
    "osm_type",
    "osm_id",
    "place_id",
    "query",
    "which_result",
)

# a region identifier selects a specific OSM element: an OSM type + ID string
# such as "R1234" (or "relation/1234"), or a dict with "osm_type"/"osm_id" or
# "place_id" keys. A selection is such an identifier, an int position (1-based)
# in the importance-sorted result list, None to auto-select the first
# (Multi)Polygon, or a list of any of these to select several sub-areas.
_Identifier = str | dict[str, Any]
_Selection = int | None | _Identifier | list[int | None | _Identifier]

# OSM type aliases accepted in an identifier, mapped to Nominatim's osm_type
_OSM_TYPE_ALIASES = {
    "n": "node",
    "node": "node",
    "w": "way",
    "way": "way",
    "r": "relation",
    "relation": "relation",
}


class _NoPolygonError(TypeError):
    """Exception raised when a query has no boundary polygon result."""

    def __init__(
        self,
        query: str | dict[str, str],
        results: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize the error with the query that lacked a boundary polygon.

        Parameters
        ----------
        query
            The query that resolved to no boundary polygon.
        results
            The Nominatim results for the query, used to name the place that
            lacked a boundary.
        """
        self.query = query
        self.results = results or []
        super().__init__(query)

    @property
    def label(self) -> str:
        """
        Return a human-readable label naming the place that lacks a boundary.

        Returns
        -------
        label
            The place's name if Nominatim identified one, else its display
            name, else the query itself.
        """
        for result in self.results:
            name = result.get("name")
            if name:
                return repr(name)
        for result in self.results:
            display_name = result.get("display_name")
            if display_name:
                return repr(display_name)
        return repr(self.query)


def geocode(query: str) -> tuple[float, float]:
    """
    Geocode place names or addresses to `(lat, lon)` with the Nominatim API.

    This geocodes the query via the Nominatim "search" endpoint.

    Parameters
    ----------
    query
        The query string to geocode.

    Returns
    -------
    point
        The `(lat, lon)` coordinates returned by the geocoder.
    """
    # define the parameters
    params: OrderedDict[str, int | str] = OrderedDict()
    params["format"] = "json"
    params["limit"] = 1
    params["dedupe"] = 0  # prevent deduping to get precise number of results
    params["q"] = query
    response_json = _nominatim._nominatim_request(params=params)

    # if results were returned, parse lat and lon out of the result
    if response_json and "lat" in response_json[0] and "lon" in response_json[0]:
        lat = float(response_json[0]["lat"])
        lon = float(response_json[0]["lon"])
        point = (lat, lon)

        msg = f"Geocoded {query!r} to {point}"
        utils.log(msg, level=lg.INFO)
        return point

    # otherwise we got no results back
    msg = f"Nominatim could not geocode query {query!r}."
    raise InsufficientResponseError(msg)


def geocode_to_gdf(
    query: str | dict[str, str] | list[str | dict[str, str]],
    *,
    which_result: int | str | dict[str, Any] | list[Any] | None = None,
    by_osmid: bool = False,
) -> gpd.GeoDataFrame:
    """
    Retrieve OSM elements by place name or OSM ID with the Nominatim API.

    If searching by place name, the `query` argument can be a string or
    structured dict, or a list of such strings/dicts to send to the geocoder.
    This uses the Nominatim "search" endpoint to geocode the place name to the
    best-matching OSM element(s), then returns those elements and their
    attribute data.

    A place name can match several OSM elements, such as several same-named
    districts. You can select one or more of the results with `which_result`:
    pass an int to select a single result by its position in the
    importance-sorted result list, pass a region identifier (an OSM type + ID
    string like `"R1234"`, or a dict with `osm_type`/`osm_id` or `place_id`
    keys) to select a specific element regardless of its position, or pass a
    list of any of these to select and combine several results (several valid
    sub-areas) for one query. Each selected result becomes its own row, so
    after combining their geometries you can still trace each one's `name` and
    source identity (`osm_type`, `osm_id`, `place_id`, `display_name`). The
    returned `query` and `which_result` columns record where each row came
    from.

    You can instead query by OSM ID by passing `by_osmid=True`. This uses the
    Nominatim "lookup" endpoint to retrieve the OSM element(s) with that ID. The
    `query` argument is then an OSM ID or a comma-separated string of them,
    prepended with their types: node (N), way (W), or relation (R) in
    accordance with the Nominatim API format. For example,
    `query="R2192363,N240109189,W427818536"` returns one row per element.

    If `query` is a list, then `which_result` must be either an int or a list
    with the same length as `query`. The queries you provide must be
    resolvable to elements in the Nominatim database. The resulting
    GeoDataFrame's geometry column contains place boundaries if they exist.

    If no query resolves to a boundary polygon, a `TypeError` is raised that
    names every place lacking a boundary. If a selected result resolves only
    to an approximate point (or another non-boundary geometry), a warning is
    logged that names the place and its missing boundary.

    Parameters
    ----------
    query
        The query string(s) or structured dict(s) to geocode.
    which_result
        Which search result(s) to return. If None, auto-select the first
        (Multi)Polygon or raise an error if OSM doesn't return one. To get
        the top match (sorted by importance) regardless of geometry type, set
        `which_result=1`. Pass a region identifier (`"R1234"`, `"relation/1234"`,
        or a dict with `osm_type`/`osm_id` or `place_id` keys) to select a
        specific element by identity. Pass a list of ints and/or identifiers
        to select and combine several results for a query (one row each). If
        `query` is a list, pass either an int/None/identifier (applied to
        every query) or a list of the same length whose elements are each an
        int, None, identifier, or a list of these. Ignored if `by_osmid=True`.
    by_osmid
        If True, treat query as an OSM ID lookup rather than text search.

    Returns
    -------
    gdf
        GeoDataFrame with one row for each selected query result.
    """
    q_list, wr_list = _normalize_queries_and_results(query, which_result)

    # geocode each query, collecting any queries that lack a boundary polygon
    frames: list[gpd.GeoDataFrame] = []
    missing: list[str] = []
    for q, wr in zip(q_list, wr_list, strict=True):
        try:
            frames.append(_geocode_query_to_gdf(q, wr, by_osmid))
        except _NoPolygonError as e:
            missing.append(e.label)

    # if any queries resolved only to points, report every one of their names
    if len(missing) > 0:
        labels = ", ".join(missing)
        msg = (
            "Nominatim did not return a boundary polygon of type (Multi)Polygon "
            f"for these place queries: {labels}."
        )
        utils.log(msg, level=lg.WARNING)
        raise TypeError(msg)

    if len(frames) == 0:  # pragma: no cover
        msg = "No queries were provided to geocode."
        raise InsufficientResponseError(msg)

    # concat as GeoDataFrame rows, set the CRS, then restore numeric columns
    gdf = pd.concat(frames, ignore_index=True).set_crs(settings.default_crs)
    for col in ("lat", "lon", "bbox_north", "bbox_south", "bbox_east", "bbox_west"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(float)

    # ensure the provenance columns exist even if Nominatim omitted some
    for col in _PROVENANCE_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = None

    msg = f"Created GeoDataFrame with {len(gdf)} rows from {len(q_list)} queries"
    utils.log(msg, level=lg.INFO)
    return gdf


def _normalize_queries_and_results(
    query: str | dict[str, str] | list[str | dict[str, str]],
    which_result: Any,
) -> tuple[list[str | dict[str, str]], list[_Selection]]:
    """
    Normalize the query and which_result arguments to parallel lists.

    Parameters
    ----------
    query
        The query string(s) or structured dict(s) to geocode.
    which_result
        Which search result(s) to return.

    Returns
    -------
    q_list
        The queries as a list.
    wr_list
        One selection per query: an int, None, an identifier, or a list of
        these.
    """
    if isinstance(query, list):
        q_list = list(query)
        is_query_list = True
    else:
        q_list = [query]
        is_query_list = False

    if which_result is None or _is_scalar_selection(which_result):
        selection = _normalize_selection(which_result)
        return q_list, [selection] * len(q_list)

    if not isinstance(which_result, list):  # pragma: no cover
        msg = "`which_result` must be an int, identifier, list, or None."
        raise TypeError(msg)

    if is_query_list:
        # one selection per query, each of which may itself be a list
        if len(which_result) != len(q_list):
            msg = "`which_result` length must equal `query` length."
            raise ValueError(msg)
        wr_list: list[_Selection] = [_normalize_selection(wr) for wr in which_result]
        return q_list, wr_list

    # for a single query, a list selects several results to combine
    return q_list, [_normalize_selection(which_result)]


def _is_scalar_selection(selection: Any) -> bool:
    """
    Determine whether a selection applies to a query as a whole.

    A scalar selection is an int, None, an OSM ID string, or an identifier
    dict. A list always means per-query selections (or several sub-areas for
    one query), so it is not scalar.

    Parameters
    ----------
    selection
        The selection to test.

    Returns
    -------
    is_scalar
        True if the selection applies to a query as a whole.
    """
    return selection is None or isinstance(selection, (int, str, dict))


def _normalize_selection(selection: Any) -> _Selection:
    """
    Normalize a single query's result selection.

    Parameters
    ----------
    selection
        An int, None, identifier, or list of these.

    Returns
    -------
    selection
        The normalized selection.
    """
    if selection is None or (isinstance(selection, int) and not isinstance(selection, bool)):
        return selection

    if isinstance(selection, str) or isinstance(selection, dict):
        return _normalize_identifier(selection)

    if isinstance(selection, list):
        normalized: list[int | None | _Identifier] = []
        for item in selection:
            if item is None or (isinstance(item, int) and not isinstance(item, bool)):
                normalized.append(item)
            elif isinstance(item, (str, dict)):
                normalized.append(_normalize_identifier(item))
            else:  # pragma: no cover
                msg = (
                    "Each element of a `which_result` list must be an int, "
                    "None, or a region identifier."
                )
                raise TypeError(msg)
        if len(normalized) == 0:
            msg = "`which_result` cannot be an empty list."
            raise ValueError(msg)
        return normalized

    msg = "Each `which_result` selection must be an int, None, identifier, or list."
    raise TypeError(msg)


def _normalize_identifier(identifier: _Identifier) -> dict[str, Any]:
    """
    Normalize a caller-provided region identifier to a lookup-key dict.

    An OSM ID string such as "R1234" or "relation/1234" is parsed into its
    type and ID. A dict must contain an "osm_type" and "osm_id" pair, or a
    "place_id". The returned dict always names the identifier in a "label"
    key for provenance and error messages.

    Parameters
    ----------
    identifier
        The identifier to normalize.

    Returns
    -------
    identifier
        The normalized identifier.
    """
    if isinstance(identifier, str):
        parsed = _parse_osm_id(identifier)
        if parsed is None:
            msg = (
                f"Region identifier {identifier!r} is not an OSM ID. Use a string "
                "like 'R1234' or 'relation/1234', or a dict with 'osm_type' and "
                "'osm_id', or 'place_id'."
            )
            raise ValueError(msg)
        osm_type, osm_id = parsed
        return {
            "label": f"{osm_type}/{osm_id}",
            "osm_type": osm_type,
            "osm_id": osm_id,
        }

    # identifier is a dict: accept an osm_type/osm_id pair or a place_id
    if "place_id" in identifier and identifier["place_id"] is not None:
        place_id = int(identifier["place_id"])
        return {"label": f"place/{place_id}", "place_id": place_id}

    if "osm_type" in identifier and "osm_id" in identifier:
        osm_type = _OSM_TYPE_ALIASES.get(str(identifier["osm_type"]).lower())
        if osm_type is None:
            msg = f"Region identifier has invalid `osm_type` {identifier['osm_type']!r}."
            raise ValueError(msg)
        osm_id = int(identifier["osm_id"])
        return {
            "label": f"{osm_type}/{osm_id}",
            "osm_type": osm_type,
            "osm_id": osm_id,
        }

    msg = (
        "Region identifier dicts must contain either 'place_id', or both "
        "'osm_type' and 'osm_id'."
    )
    raise ValueError(msg)


def _parse_osm_id(identifier: str) -> tuple[str, int] | None:
    """
    Parse an OSM ID string into its type and ID.

    Parameters
    ----------
    identifier
        A string like "R1234", "r1234", or "relation/1234".

    Returns
    -------
    parsed
        A (type, ID) tuple, or None if the string is not an OSM ID.
    """
    text = identifier.strip()
    for pattern in _OSM_ID_PATTERNS:
        match = pattern.match(text)
        if match is not None:
            osm_type = _OSM_TYPE_ALIASES.get(match.group("type").lower())
            if osm_type is not None:
                return osm_type, int(match.group("id"))
    return None


def _geocode_query_to_gdf(
    query: str | dict[str, str],
    which_result: _Selection,
    by_osmid: bool,  # noqa: FBT001
) -> gpd.GeoDataFrame:
    """
    Geocode a single place query to a GeoDataFrame.

    Parameters
    ----------
    query
        Query string or structured dict to geocode.
    which_result
        Which search result(s) to return. If None, auto-select the first
        (Multi)Polygon or raise an error if OSM doesn't return one. To get
        the top match regardless of geometry type, set `which_result=1`. A
        list selects several results. Ignored if `by_osmid=True`.
    by_osmid
        If True, treat query as an OSM ID lookup rather than text search.

    Returns
    -------
    gdf
        GeoDataFrame with one row per selected geocoding result.
    """
    limit = _result_limit(which_result)
    results = _nominatim._download_nominatim_element(query, by_osmid=by_osmid, limit=limit)

    # choose the right result(s) from the JSON response
    if len(results) == 0:
        # if no results were returned, raise error
        msg = f"Nominatim geocoder returned 0 results for query {query!r}."
        raise InsufficientResponseError(msg)

    if by_osmid:
        # a lookup can return several results for a comma-separated ID string
        results = _dedupe_results(results)
        indices = list(range(len(results)))
    else:
        # ensure geocoder results are sorted from most to least important
        results = sorted(results, key=lambda x: x.get("importance", 0), reverse=True)
        indices = _resolve_result_indices(results, which_result, query)

    # build one GeoJSON feature per selected result, retaining its provenance
    features = [
        _build_feature(results[idx], query, which_result=idx + 1) for idx in indices
    ]

    # create and return the GeoDataFrame
    return gpd.GeoDataFrame.from_features(features)


def _result_limit(which_result: _Selection) -> int:
    """
    Determine how many search results to request from the Nominatim API.

    Parameters
    ----------
    which_result
        Which search result(s) to return.

    Returns
    -------
    limit
        The maximum number of results to request.
    """
    selections = which_result if isinstance(which_result, list) else [which_result]

    # an identifier lookup may match any position, so request a large pool
    needs_pool = any(item is None or isinstance(item, (str, dict)) for item in selections)
    if needs_pool:
        return 50

    # otherwise request at least as many results as the highest position
    positions = [item for item in selections if isinstance(item, int)]
    return max([1, *positions])


def _resolve_result_indices(
    results: list[dict[str, Any]],
    which_result: _Selection,
    query: str | dict[str, str],
) -> list[int]:
    """
    Resolve a selection to the 0-based indices of the chosen results.

    Parameters
    ----------
    results
        Results from the Nominatim API, sorted by importance.
    which_result
        Which search result(s) to return.
    query
        The query, used in error messages.

    Returns
    -------
    indices
        The 0-based indices of the chosen results.
    """
    selections = which_result if isinstance(which_result, list) else [which_result]

    indices: list[int] = []
    for selection in selections:
        if selection is None:
            index = _get_first_polygon_index(results, query)
        elif isinstance(selection, (str, dict)):
            index = _resolve_identifier_index(results, selection, query)
        elif isinstance(selection, int):
            index = _validate_result_index(results, selection, query)
        else:  # pragma: no cover
            msg = "Each `which_result` selection must be an int, None, or an identifier."
            raise TypeError(msg)
        # keep order but do not return the same sub-area twice
        if index not in indices:
            indices.append(index)
    return indices


def _resolve_identifier_index(
    results: list[dict[str, Any]],
    identifier: _Identifier,
    query: str | dict[str, str],
) -> int:
    """
    Resolve a region identifier to the index of the matching result.

    Parameters
    ----------
    results
        Results from the Nominatim API, sorted by importance.
    identifier
        The region identifier to resolve.
    query
        The query, used in error messages.

    Returns
    -------
    index
        The 0-based index of the matching result.
    """
    if isinstance(identifier, str):
        # identifiers were normalized by the caller: an unnormalized string
        # only reaches here if it was passed directly to this internal helper
        target = _normalize_identifier(identifier)
    else:
        target = identifier

    for index, result in enumerate(results):
        if _result_matches_identifier(result, target):
            return index

    label = target.get("label", identifier)
    msg = (
        f"Nominatim returned no result matching region identifier {label!r} "
        f"for query {query!r}."
    )
    raise InsufficientResponseError(msg)


def _result_matches_identifier(result: dict[str, Any], identifier: dict[str, Any]) -> bool:
    """
    Determine whether a Nominatim result matches a region identifier.

    Parameters
    ----------
    result
        A result from the Nominatim API.
    identifier
        The normalized region identifier.

    Returns
    -------
    matches
        True if the result identifies the same OSM element.
    """
    if "place_id" in identifier:
        return str(result.get("place_id")) == str(identifier["place_id"])

    osm_type = str(result.get("osm_type", "")).lower()
    osm_id = str(result.get("osm_id"))
    return osm_type == identifier["osm_type"] and osm_id == str(identifier["osm_id"])


def _validate_result_index(
    results: list[dict[str, Any]],
    which_result: int,
    query: str | dict[str, str],
) -> int:
    """
    Validate a 1-based result index and return its 0-based index.

    Parameters
    ----------
    results
        Results from the Nominatim API.
    which_result
        The 1-based result index to validate.
    query
        The query, used in error messages.

    Returns
    -------
    index
        The 0-based result index.
    """
    if which_result < 1 or len(results) < which_result:
        msg = (
            f"Nominatim returned {len(results)} result(s) for query {query!r} "
            f"but `which_result={which_result}` was requested."
        )
        raise InsufficientResponseError(msg)
    return which_result - 1


def _get_first_polygon_index(
    results: list[dict[str, Any]],
    query: str | dict[str, str],
) -> int:
    """
    Return the index of the first result of type (Multi)Polygon.

    Parameters
    ----------
    results
        Results from the Nominatim API.
    query
        The query, used in error messages.

    Returns
    -------
    index
        The 0-based index of the chosen result.
    """
    for index, result in enumerate(results):
        geojson = result.get("geojson")
        if isinstance(geojson, dict) and geojson.get("type") in _POLYGON_TYPES:
            return index

    # if we never found a polygon, raise an error that identifies this query
    raise _NoPolygonError(query, results)


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    De-duplicate results by OSM identity, retaining their original order.

    Parameters
    ----------
    results
        Results from the Nominatim API.

    Returns
    -------
    results
        The de-duplicated results.
    """
    seen: set[tuple[Any, Any]] = set()
    deduped: list[dict[str, Any]] = []
    for result in results:
        key = (result.get("osm_type"), result.get("osm_id"))
        if key not in seen:
            seen.add(key)
            deduped.append(result)
    return deduped


def _build_feature(
    result: dict[str, Any],
    query: str | dict[str, str],
    *,
    which_result: int,
) -> dict[str, Any]:
    """
    Build a GeoJSON feature from a Nominatim result, retaining its provenance.

    Parameters
    ----------
    result
        A result from the Nominatim API.
    query
        The query that produced this result.
    which_result
        The 1-based position of this result in the response.

    Returns
    -------
    feature
        The GeoJSON feature.
    """
    geojson = result.get("geojson")
    if not isinstance(geojson, dict) or "type" not in geojson:
        msg = f"Nominatim returned no geometry for query {query!r}."
        raise InsufficientResponseError(msg)

    # if we got a non (Multi)Polygon geometry (like a point), log a warning
    # that names the place and explicitly notes its missing boundary
    geom_type = geojson["type"]
    if geom_type not in _POLYGON_TYPES:
        kind = "an approximate point" if geom_type == "Point" else f"a {geom_type}"
        msg = (
            f"Nominatim resolved query {query!r} only to {kind} rather than a "
            "boundary polygon: this place has no (Multi)Polygon boundary in OSM."
        )
        utils.log(msg, level=lg.WARNING)

    # build the GeoJSON feature from the chosen result
    bottom, top, left, right = result.get("boundingbox", [None, None, None, None])
    properties: dict[str, Any] = {
        "bbox_west": left,
        "bbox_south": bottom,
        "bbox_east": right,
        "bbox_north": top,
        "query": query,
        "which_result": which_result,
    }

    # add the other attributes we retrieved
    for attr, value in result.items():
        if attr not in {"address", "boundingbox", "geojson", "icon", "licence"}:
            properties[attr] = value

    return {"type": "Feature", "geometry": geojson, "properties": properties}


def _boundary_polygon_and_places(
    gdf: gpd.GeoDataFrame,
) -> tuple[Any, list[dict[str, Any]]]:
    """
    Combine the polygonal rows of a geocoded GeoDataFrame into one geometry.

    Rows that resolved only to approximate points (or other non-boundary
    geometries) cannot be combined into the query area. They are excluded
    from the returned geometry, logged by name, and still reported in the
    place records so their provenance remains traceable.

    Parameters
    ----------
    gdf
        A GeoDataFrame returned by the geocode_to_gdf function.

    Returns
    -------
    polygon
        The union of all (Multi)Polygon rows.
    places
        One provenance record per row, with its name and OSM identity.
    """
    places = [_place_record(row) for _, row in gdf.iterrows()]

    # keep only the rows whose geometry is a boundary polygon
    is_polygonal = gdf.geometry.geom_type.isin(_POLYGON_TYPES).fillna(False)
    polygonal = gdf[is_polygonal]
    if len(polygonal) == 0:
        labels = ", ".join(_place_labels(places))
        msg = (
            "Nominatim did not return a boundary polygon of type (Multi)Polygon "
            f"for these place queries: {labels}."
        )
        utils.log(msg, level=lg.WARNING)
        raise TypeError(msg)

    # account for any selected sub-areas lacking a boundary polygon
    if len(polygonal) < len(gdf):
        point_places = [p for p in places if p["geometry_type"] not in _POLYGON_TYPES]
        labels = ", ".join(_place_labels(point_places))
        msg = (
            "These place queries resolved only to approximate points rather than "
            f"boundary polygons and were excluded from the query area: {labels}."
        )
        utils.log(msg, level=lg.WARNING)

    return polygonal.union_all(), places


def _place_record(row: pd.Series) -> dict[str, Any]:
    """
    Build a provenance record for one geocoded place.

    Parameters
    ----------
    row
        A row of a GeoDataFrame returned by the geocode_to_gdf function.

    Returns
    -------
    record
        The place provenance record.
    """
    record: dict[str, Any] = {}
    for col in _PROVENANCE_COLUMNS:
        value = row[col] if col in row.index else None
        record[col] = _to_python_scalar(value)
    geometry = row["geometry"] if "geometry" in row.index else None
    record["geometry_type"] = None if geometry is None else geometry.geom_type
    return record


def _to_python_scalar(value: Any) -> Any:
    """
    Convert a pandas/numpy scalar to a plain Python value.

    Parameters
    ----------
    value
        The value to convert.

    Returns
    -------
    value
        The converted value.
    """
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):  # pragma: no cover
            return value
    return value


def _place_labels(places: list[dict[str, Any]]) -> list[str]:
    """
    Return a readable name for each place record.

    Parameters
    ----------
    places
        The place provenance records.

    Returns
    -------
    labels
        A label for each record, preferring its name over its display name.
    """
    return [repr(p.get("name") or p.get("display_name") or p.get("query")) for p in places]
