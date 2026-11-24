"""Tools to work with the Overpass API."""

from __future__ import annotations

import datetime as dt
import logging as lg
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError

from . import _http
from . import projection
from . import settings
from . import utils
from . import utils_geo
from ._errors import IncompleteCoverageError
from ._errors import InsufficientResponseError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from shapely import MultiPolygon
    from shapely import Polygon

# how many times to attempt each shard request before marking it failed: the
# first attempt plus this many retries, each preceded by the service-mandated
# wait (or a conservative fallback when the status endpoint is unreachable)
_MAX_REQUEST_ATTEMPTS = 3

# fallback pause (seconds) before retrying when the status endpoint is
# unreachable or the server cannot be contacted, matching the pause used for
# retryable HTTP status codes
_NETWORK_ERROR_PAUSE = 55


# define built-in network exclusions once for both Overpass and PBF matching
_DEFAULT_PBF_ACCESS = "private"
_DEFAULT_OVERPASS_ACCESS = f'["access"!~"{_DEFAULT_PBF_ACCESS}"]'
_NETWORK_EXCLUDED_PATTERNS: dict[str, dict[str, str]] = {
    "drive": {
        "area": "yes",
        "access": _DEFAULT_PBF_ACCESS,
        "highway": (
            "abandoned|bridleway|bus_guideway|construction|corridor|cycleway|"
            "elevator|escalator|footway|no|path|pedestrian|planned|platform|"
            "proposed|raceway|razed|rest_area|service|services|steps|track"
        ),
        "motor_vehicle": "no",
        "motorcar": "no",
        "service": ("alley|driveway|emergency_access|parking|parking_aisle|private"),
    },
    "drive_service": {
        "area": "yes",
        "access": _DEFAULT_PBF_ACCESS,
        "highway": (
            "abandoned|bridleway|bus_guideway|construction|corridor|cycleway|"
            "elevator|escalator|footway|no|path|pedestrian|planned|platform|"
            "proposed|raceway|razed|rest_area|services|steps|track"
        ),
        "motor_vehicle": "no",
        "motorcar": "no",
        "service": "emergency_access|parking|parking_aisle|private",
    },
    "walk": {
        "area": "yes",
        "access": _DEFAULT_PBF_ACCESS,
        "highway": (
            "abandoned|bus_guideway|construction|cycleway|motor|no|planned|"
            "platform|proposed|raceway|razed|rest_area|services"
        ),
        "foot": "no",
        "service": "private",
        "sidewalk": "separate",
        "sidewalk:both": "separate",
        "sidewalk:left": "separate",
        "sidewalk:right": "separate",
    },
    "bike": {
        "area": "yes",
        "access": _DEFAULT_PBF_ACCESS,
        "highway": (
            "abandoned|bus_guideway|construction|corridor|elevator|escalator|"
            "footway|motor|no|planned|platform|proposed|raceway|razed|"
            "rest_area|services|steps"
        ),
        "bicycle": "no",
        "service": "private",
    },
    "all_public": {
        "area": "yes",
        "access": _DEFAULT_PBF_ACCESS,
        "highway": (
            "abandoned|construction|no|planned|platform|proposed|raceway|razed|rest_area|services"
        ),
        "service": "private",
    },
    "all": {
        "area": "yes",
        "highway": (
            "abandoned|construction|no|planned|platform|proposed|raceway|razed|rest_area|services"
        ),
    },
}


def _get_network_filter(network_type: str) -> str:
    """
    Create a filter to query Overpass for the specified network type.

    The filter queries Overpass for every OSM way with a "highway" tag but
    excludes ways that are incompatible with the requested network type. You
    can choose from the following types:

    "all" retrieves all public and private-access ways currently in use,
    excluding those that represent areas either explicitly (area=yes) or by
    convention (rest_area, services).

    "all_public" retrieves all public ways currently in use.

    "bike" retrieves public bikeable ways and excludes foot ways, motor ways,
    and anything tagged biking=no.

    "drive" retrieves public drivable streets and excludes service roads,
    anything tagged motor=no, and certain non-service roads tagged as
    providing certain services (such as alleys or driveways).

    "drive_service" retrieves public drivable streets including service roads
    but excludes certain services (such as parking or emergency access).

    "walk" retrieves public walkable ways and excludes cycle ways, motor ways,
    and anything tagged foot=no. It includes service roads like parking lot
    aisles and alleys that you can walk on even if they are unpleasant walks.

    Parameters
    ----------
    network_type
        {"all", "all_public", "bike", "drive", "drive_service", "walk"}
        What type of street network to retrieve.

    Returns
    -------
    way_filter
        The Overpass query filter.
    """
    if network_type not in _NETWORK_EXCLUDED_PATTERNS:  # pragma: no cover
        msg = f"Unrecognized network_type {network_type!r}."
        raise ValueError(msg)

    # require highway ways, then render excluded values as Overpass clauses
    excluded = _NETWORK_EXCLUDED_PATTERNS[network_type]
    filters = ['["highway"]']
    for key, pattern in excluded.items():
        if key == "access":
            # preserve the configurable Overpass access clause
            filters.append(settings.default_access)
        else:
            filters.append(f'["{key}"!~"{pattern}"]')
    return "".join(filters)


def _network_filter_matches(network_type: str, tags: dict[str, str]) -> bool:
    """
    Return whether a way's tags match a built-in network type preset.

    Parameters
    ----------
    network_type
        Network type preset to match.
    tags
        OSM way tags.

    Returns
    -------
    matches
        Whether the tags satisfy every predicate in the preset.
    """
    if network_type not in _NETWORK_EXCLUDED_PATTERNS:
        msg = f"Unrecognized network_type {network_type!r}."
        raise ValueError(msg)

    if "highway" not in tags:
        return False

    # reject values that match an exclusion pattern
    for key, pattern in _NETWORK_EXCLUDED_PATTERNS[network_type].items():
        if key in tags and re.search(pattern, tags[key]):
            return False
    return True


def _get_overpass_pause(
    base_endpoint: str,
    *,
    recursion_pause: float = 5,
    default_pause: float = 60,
) -> float:
    """
    Retrieve a pause duration from the Overpass API status endpoint.

    Check the Overpass API status endpoint to determine how long to wait until
    the next slot is available. You can disable this via the `settings`
    module's `overpass_rate_limit` setting.

    Parameters
    ----------
    base_endpoint
        Base Overpass API URL (without "/status" at the end).
    recursion_pause
        How long to wait between recursive calls if the server is currently
        running a query.
    default_pause
        If a fatal error occurs, fall back on this liberal pause duration.

    Returns
    -------
    pause
        The current pause duration specified by the Overpass status endpoint.
    """
    # if overpass rate limiting is False, then there is zero pause
    if not settings.overpass_rate_limit:
        return 0

    url = base_endpoint.rstrip("/") + "/status"

    # try to retrieve the URL
    try:
        response = requests.get(
            url,
            headers=_http._get_http_headers(),
            timeout=settings.requests_timeout,
            **settings.requests_kwargs,
        )
        response_text = response.text
    except RequestsConnectionError as e:  # pragma: no cover
        # cannot reach status endpoint: log error and return default duration
        msg = f"Unable to reach {url}, {e}"
        utils.log(msg, level=lg.ERROR)
        return default_pause

    # try to parse the output
    try:
        status = response_text.split("\n")[4]
        status_first_part = status.split(" ")[0]
    except (AttributeError, IndexError, ValueError):  # pragma: no cover
        # cannot parse output: log error and return default duration
        msg = f"Unable to parse {url} response: {response_text}"
        utils.log(msg, level=lg.ERROR)
        return default_pause

    # determine the current status of the server
    try:
        # if first token is numeric, it's how many slots you have available,
        # no wait required
        _ = int(status_first_part)  # number of available slots
        pause: float = 0

    except ValueError:  # pragma: no cover
        # if first token is 'Slot', it tells you when your slot will be free
        if status_first_part == "Slot":
            utc_time_str = status.split(" ")[3]
            pattern = "%Y-%m-%dT%H:%M:%SZ,"
            utc_time = dt.datetime.strptime(utc_time_str, pattern).replace(tzinfo=dt.UTC)
            utc_now = dt.datetime.now(tz=dt.UTC)
            seconds = int(np.ceil((utc_time - utc_now).total_seconds()))
            pause = max(seconds, 1)

        # if first token is 'Currently', it is currently running a query so
        # check back in recursion_pause seconds
        elif status_first_part == "Currently":
            time.sleep(recursion_pause)
            pause = _get_overpass_pause(base_endpoint)

        # any other status is unrecognized: log error, return default duration
        else:
            msg = f"Unrecognized server status: {status!r}"
            utils.log(msg, level=lg.ERROR)
            return default_pause

    return pause


def _make_overpass_settings() -> str:
    """
    Make settings string to send in Overpass query.

    Returns
    -------
    overpass_settings
        The `settings.overpass_settings` string formatted with "timeout" and
        "maxsize" values.
    """
    maxsize = "" if settings.overpass_memory is None else f"[maxsize:{settings.overpass_memory}]"
    return settings.overpass_settings.format(timeout=settings.requests_timeout, maxsize=maxsize)


@dataclass
class ShardReceipt:
    """
    Plan and outcome of a single shard request in a subdivided query.

    Parameters
    ----------
    index
        Zero-based position of this shard among the planned shards.
    polygon_coord_str
        The lat-lon exterior coordinate string sent to Overpass.
    bounds
        The shard's expected extent as `(left, bottom, right, top)` in
        unprojected latitude-longitude degrees (EPSG:4326).
    response_json
        The most recent successful Overpass API response for this shard,
        otherwise None.
    attempts
        Number of HTTP attempts made on the most recent request.
    error
        The exception raised on the most recent failed attempt, or None.
    request_count
        Number of distinct requests planned for this shard (more than one
        when a list of custom way filters is used).
    success_count
        Number of planned requests that returned successfully.
    """

    index: int
    polygon_coord_str: str
    bounds: tuple[float, float, float, float]
    response_json: dict[str, Any] | None = None
    attempts: int = 0
    error: Exception | None = None
    request_count: int = 0
    success_count: int = 0

    @property
    def succeeded(self) -> bool:
        """Whether every request planned for this shard returned successfully."""
        return self.request_count > 0 and self.success_count == self.request_count

    @property
    def failed(self) -> bool:
        """Whether at least one request planned for this shard is unconfirmed."""
        return not self.succeeded


def _make_overpass_shards(polygon: Polygon | MultiPolygon) -> list[ShardReceipt]:
    """
    Subdivide query polygon and plan one `ShardReceipt` per sub-polygon.

    Project to UTM, divide `polygon` up into sub-polygons if area exceeds a
    max size (in meters), project back to lat-lon, then build a receipt for
    each sub-polygon recording its expected extent and exterior coordinates.
    Interior ("holes") coordinates are ignored.

    Parameters
    ----------
    polygon
        The (Multi)Polygon to subdivide into query shards.

    Returns
    -------
    shards
        Planned shard receipts, one per subdivided constituent polygon.
    """
    # subdivide the polygon if its area exceeds max size
    # this results in a multipolygon of 1+ constituent polygons
    poly_proj, crs_proj = projection.project_geometry(polygon)
    multi_poly_proj = utils_geo._consolidate_subdivide_geometry(poly_proj)
    multi_poly, _ = projection.project_geometry(multi_poly_proj, crs=crs_proj, to_latlong=True)

    # extract each shard's extent and exterior coords in the string format
    # Overpass expects, rounding lats and lons to 6 decimals (approx 5 to 10
    # cm resolution) so we can hash and cache URL strings consistently
    shards = []
    for index, geom in enumerate(multi_poly.geoms):
        x, y = geom.exterior.xy
        coord_list = [f"{xy[1]:.6f}{' '}{xy[0]:.6f}" for xy in zip(x, y, strict=True)]
        polygon_coord_str = " ".join(coord_list)
        left, bottom, right, top = geom.bounds
        shards.append(
            ShardReceipt(
                index=index,
                polygon_coord_str=polygon_coord_str,
                bounds=(left, bottom, right, top),
            )
        )

    return shards


def _make_overpass_polygon_coord_strs(polygon: Polygon | MultiPolygon) -> list[str]:
    """
    Subdivide query polygon and return list of coordinate strings.

    Project to UTM, divide `polygon` up into sub-polygons if area exceeds a
    max size (in meters), project back to lat-lon, then get a list of
    polygon(s) exterior coordinates. Ignore interior ("holes") coordinates.

    Parameters
    ----------
    polygon
        The (Multi)Polygon to convert to exterior coordinate strings.

    Returns
    -------
    coord_strs
        Exterior coordinates of polygon(s).
    """
    return [shard.polygon_coord_str for shard in _make_overpass_shards(polygon)]


def _create_overpass_features_query(  # noqa: PLR0912
    polygon_coord_str: str,
    tags: dict[str, bool | str | list[str]],
) -> str:
    """
    Create an Overpass features query string based on tags.

    Parameters
    ----------
    polygon_coord_str
        The lat lon coordinates.
    tags
        Tags used for finding elements in the search area.

    Returns
    -------
    query
        The Overpass features query.
    """
    # create overpass settings string
    overpass_settings = _make_overpass_settings()

    # make sure every value in dict is bool, str, or list of str
    err_msg = "`tags` must be a dict with values of bool, str, or list of str."
    if not isinstance(tags, dict):  # pragma: no cover
        raise TypeError(err_msg)

    tags_dict: dict[str, bool | str | list[str]] = {}
    for key, value in tags.items():
        if isinstance(value, bool):
            tags_dict[key] = value

        elif isinstance(value, str):
            tags_dict[key] = [value]

        elif isinstance(value, list):
            if not all(isinstance(s, str) for s in value):  # pragma: no cover
                raise TypeError(err_msg)
            tags_dict[key] = value

        else:  # pragma: no cover
            raise TypeError(err_msg)

    # convert the tags dict into a list of {tag:value} dicts
    tags_list: list[dict[str, bool | str | list[str]]] = []
    for key, value in tags_dict.items():
        if isinstance(value, bool):
            tags_list.append({key: value})
        else:
            for value_item in value:
                tags_list.append({key: value_item})  # noqa: PERF401

    # add node/way/relation query components one at a time
    components = []
    for d in tags_list:
        for key, value in d.items():
            if isinstance(value, bool):
                # if bool (ie, True) just pass the key, no value
                tag_str = f"[{key!r}](poly:{polygon_coord_str!r});(._;>;);"
            else:
                # otherwise, pass "key"="value"
                tag_str = f"[{key!r}={value!r}](poly:{polygon_coord_str!r});(._;>;);"

            for kind in ("node", "way", "relation"):
                components.append(f"({kind}{tag_str});")  # noqa: PERF401

    # finalize query and return
    components_str = "".join(components)
    return f"{overpass_settings};({components_str});out;"


def _download_overpass_network(
    polygon: Polygon | MultiPolygon,
    network_type: str,
    custom_filter: str | list[str] | None,
) -> Iterator[dict[str, Any]]:
    """
    Retrieve networked ways and nodes within boundary from the Overpass API.

    The boundary is subdivided into planned shards. Every shard must return
    successfully for the full-area result to be considered complete: if any
    shard still fails after all allowed retries, an `IncompleteCoverageError`
    is raised identifying the failed shards. Responses for successful shards
    are cached, so re-running after a failure resumes from the failed shards.

    Parameters
    ----------
    polygon
        The boundary to fetch the network ways/nodes within.
    network_type
        What type of street network to get if `custom_filter` is None.
    custom_filter
        A custom "ways" filter to be used instead of `network_type` presets.

    Yields
    ------
    response_json
        JSON response from the Overpass server, one per successful shard and
        way filter.
    """
    # create filter(s) to exclude certain kinds of ways based on the requested
    # network_type, if provided, otherwise use custom_filter
    way_filters = []
    if isinstance(custom_filter, list):
        way_filters = custom_filter
    elif isinstance(custom_filter, str):
        way_filters = [custom_filter]
    else:
        way_filters = [_get_network_filter(network_type)]

    # create overpass settings string
    overpass_settings = _make_overpass_settings()

    # subdivide query polygon into planned shards with expected extents
    shards = _plan_shards(polygon)

    # request each shard, one at a time. the '>' makes it recurse so we get
    # ways and the ways' nodes. each shard is fully resolved (all its way
    # filters) before it counts as covered: a single failed filter leaves
    # `shard.failed` True, so nothing is yielded for an incomplete shard.
    for shard in shards:
        shard_responses: list[dict[str, Any]] = []
        for way_filter in way_filters:
            query_str = f"{overpass_settings};(way{way_filter}(poly:{shard.polygon_coord_str!r});>;);out;"
            response_json = _request_shard(shard, OrderedDict(data=query_str))
            if response_json is not None:
                shard_responses.append(response_json)
        if shard.succeeded:
            yield from shard_responses

    _verify_complete_coverage(shards)


def _download_overpass_features(
    polygon: Polygon,
    tags: dict[str, bool | str | list[str]],
) -> Iterator[dict[str, Any]]:
    """
    Retrieve OSM features within some boundary polygon from the Overpass API.

    The boundary is subdivided into planned shards. Every shard must return
    successfully for the full-area result to be considered complete: if any
    shard still fails after all allowed retries, an `IncompleteCoverageError`
    is raised identifying the failed shards. Responses for successful shards
    are cached, so re-running after a failure resumes from the failed shards.

    Parameters
    ----------
    polygon
        Boundary to retrieve elements within.
    tags
        Tags used for finding elements in the selected area.

    Yields
    ------
    response_json
        JSON response from the Overpass server, one per successful shard.
    """
    # subdivide query polygon into planned shards with expected extents
    shards = _plan_shards(polygon)

    # pass exterior coordinates of each shard to API, one at a time
    for shard in shards:
        query_str = _create_overpass_features_query(shard.polygon_coord_str, tags)
        response_json = _request_shard(shard, OrderedDict(data=query_str))
        if response_json is not None:
            yield response_json

    _verify_complete_coverage(shards)


def _plan_shards(polygon: Polygon | MultiPolygon) -> list[ShardReceipt]:
    """
    Subdivide a polygon into planned query shards and log their extents.

    Parameters
    ----------
    polygon
        The boundary to subdivide into query shards.

    Returns
    -------
    shards
        Planned shard receipts, each recording its index and expected extent.
    """
    shards = _make_overpass_shards(polygon)
    msg = f"Requesting data from API in {len(shards)} request(s)"
    utils.log(msg, level=lg.INFO)
    for shard in shards:
        left, bottom, right, top = shard.bounds
        msg = (
            f"Planned shard {shard.index + 1} of {len(shards)} with extent "
            f"(left={left:.6f}, bottom={bottom:.6f}, right={right:.6f}, top={top:.6f})"
        )
        utils.log(msg, level=lg.INFO)
    return shards


def _request_shard(
    shard: ShardReceipt,
    data: OrderedDict[str, Any],
) -> dict[str, Any] | None:
    """
    Execute one shard request, retrying transient network failures.

    Mutates `shard` in place to record the plan, attempts, last error, and a
    successful response. Successful responses are written to the local HTTP
    cache by `_overpass_request`, so a later re-run of the same query resumes
    from failed shards instead of re-downloading the confirmed coverage.

    Parameters
    ----------
    shard
        The planned shard receipt to request.
    data
        Key-value pairs of parameters for the Overpass request.

    Returns
    -------
    response_json
        The Overpass API response, or None if every attempt failed. A None
        result leaves `shard.failed` True so the caller never treats the
        full-area result as complete.
    """
    shard.request_count += 1
    for attempt in range(1, _MAX_REQUEST_ATTEMPTS + 1):
        shard.attempts = attempt
        try:
            response_json = _overpass_request(data)

        except RequestsConnectionError as e:
            # transient network failure: honor the service's prescribed wait
            # (or a conservative fallback), then retry until attempts exhaust
            shard.error = e
            if attempt < _MAX_REQUEST_ATTEMPTS:
                pause = _get_retry_pause(settings.overpass_url, repr(e))
                msg = (
                    f"Shard {shard.index + 1} attempt {attempt} of "
                    f"{_MAX_REQUEST_ATTEMPTS} failed ({e!r}): retrying in {pause} secs"
                )
                utils.log(msg, level=lg.WARNING)
                time.sleep(pause)
            continue

        # success: record the receipt and log the confirmed coverage
        shard.response_json = response_json
        shard.error = None
        shard.success_count += 1
        left, bottom, right, top = shard.bounds
        msg = (
            f"Received shard {shard.index + 1} response covering extent "
            f"(left={left:.6f}, bottom={bottom:.6f}, right={right:.6f}, top={top:.6f}) "
            f"on attempt {attempt}"
        )
        utils.log(msg, level=lg.INFO)
        return response_json

    # all attempts failed: the shard stays marked failed and returns no data
    msg = (
        f"Shard {shard.index + 1} failed after {shard.attempts} attempt(s): "
        f"{shard.error!r}"
    )
    utils.log(msg, level=lg.ERROR)
    return None


def _get_retry_pause(base_endpoint: str, reason: str) -> float:
    """
    Determine how long to wait before retrying a failed shard request.

    Use the pause mandated by the Overpass status endpoint when reachable,
    otherwise fall back on a conservative default because a transient network
    failure prevented reading the server's slot timing.

    Parameters
    ----------
    base_endpoint
        Base Overpass API URL (without "/status" at the end).
    reason
        Description of the failure that triggered the retry.

    Returns
    -------
    pause
        Seconds to wait before the next attempt.
    """
    pause = _get_overpass_pause(base_endpoint, default_pause=_NETWORK_ERROR_PAUSE)
    if pause <= 0:
        # status endpoint reported a free slot, but the request itself failed:
        # still back off briefly before retrying
        msg = f"Status endpoint reports no wait after {reason}; backing off {_NETWORK_ERROR_PAUSE} secs"
        utils.log(msg, level=lg.WARNING)
        pause = _NETWORK_ERROR_PAUSE
    return pause


def _verify_complete_coverage(shards: list[ShardReceipt]) -> None:
    """
    Confirm every planned shard returned successfully, or raise if not.

    Never report a full-area result when any shard is missing. The caller can
    fix the network/service problem and re-run the same query: successful
    shards are served from the local cache and only the failed shards are
    requested again.

    Parameters
    ----------
    shards
        Shard receipts covering the whole subdivided query area.

    Raises
    ------
    IncompleteCoverageError
        If one or more planned shards do not have a successful response.
    """
    failed = [shard for shard in shards if shard.failed]
    if failed:
        descriptions = [
            f"shard {shard.index + 1} (extent {shard.bounds}, "
            f"attempts {shard.attempts}, error {shard.error!r})"
            for shard in failed
        ]
        msg = (
            f"Incomplete coverage: {len(failed)} of {len(shards)} planned "
            f"shard(s) did not return successfully: {'; '.join(descriptions)}. "
            "Successful shards are cached; re-run the same query to resume "
            "from the failed shards."
        )
        raise IncompleteCoverageError(msg)

    msg = f"All {len(shards)} planned shard(s) confirmed: full-area coverage complete"
    utils.log(msg, level=lg.INFO)


def _overpass_request(data: OrderedDict[str, Any]) -> dict[str, Any]:
    """
    Send a HTTP POST request to the Overpass API and return response.

    Retryable HTTP status codes (429, 502, 504) are retried indefinitely
    after pausing, matching the server's slot management. Connection-level
    failures (e.g. the network is temporarily unavailable) are propagated to
    the caller, where the shard retrier handles them so that exhausted
    shards can be tracked and resumed rather than silently dropped.

    Parameters
    ----------
    data
        Key-value pairs of parameters.

    Returns
    -------
    response_json
        The Overpass API's response.
    """
    # prepare the Overpass API URL and see if request already exists in cache
    url = settings.overpass_url.rstrip("/") + "/interpreter"
    prepared_url = str(requests.Request("GET", url, params=data).prepare().url)
    cached_response_json = _http._retrieve_from_cache(prepared_url)
    if isinstance(cached_response_json, dict):
        return cached_response_json

    # resolve url to same IP even if there is server round-robin redirecting
    _http._config_dns(settings.overpass_url)

    # pause then request this URL
    pause = _get_overpass_pause(settings.overpass_url)
    hostname = _http._hostname_from_url(url)
    msg = f"Pausing {pause} second(s) before making HTTP POST request to {hostname!r}"
    utils.log(msg, level=lg.INFO)
    time.sleep(pause)

    # transmit the HTTP POST request
    msg = f"Post {prepared_url} with timeout={settings.requests_timeout}"
    utils.log(msg, level=lg.INFO)
    response = requests.post(
        url,
        data=data,
        timeout=settings.requests_timeout,
        headers=_http._get_http_headers(),
        **settings.requests_kwargs,
    )

    # handle 429, 502, and 504 errors by pausing then recursively re-trying request
    if response.status_code in {429, 502, 504}:  # pragma: no cover
        error_pause = 55
        msg = (
            f"{hostname!r} responded {response.status_code} {response.reason}: "
            f"we'll retry in {error_pause} secs"
        )
        utils.log(msg, level=lg.WARNING)
        time.sleep(error_pause)
        return _overpass_request(data)

    response_json = _http._parse_response(response)
    if not isinstance(response_json, dict):  # pragma: no cover
        msg = "Overpass API did not return a dict of results."
        raise InsufficientResponseError(msg)
    _http._save_to_cache(prepared_url, response_json, response.ok)
    return response_json
