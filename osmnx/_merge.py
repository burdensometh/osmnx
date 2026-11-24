"""
Merge duplicate elements returned by multiple API sub-queries.

OSMnx subdivides a large query geometry into several smaller parcels and
queries the API once per parcel. Elements that straddle a parcel boundary are
returned by every parcel that touches them, so the raw responses contain the
same element several times. This module merges those duplicates *before* graph
or feature construction. Duplicates are identified by element type and
original OSM ID, so a node and a way that happen to share an ID are never
confused.

Conflicts are resolved with explicit version facts. OSM attaches a version
number and timestamp to every element; the observation with the greatest
`(version, timestamp)` pair wins. If the winning version facts tie, the
observations must agree. A tie whose geometry (coordinates) differs is an
unresolvable conflict: it raises `SourceConflictError` identifying the element
and reporting each observation's coordinates and source parcel. A tie whose
only differences are non-geometry attributes (for example, tags) is resolved
deterministically so the merged output never depends on the order in which the
parcels were returned.

Every merged element records the source parcels it was observed in. Parcel
identifiers are sorted, so permuting the responses can change neither the
merged elements nor the provenance they carry.
"""

from __future__ import annotations

import hashlib
import json
import logging as lg
from typing import TYPE_CHECKING
from typing import Any

from . import utils
from ._errors import SourceConflictError

if TYPE_CHECKING:
    from collections.abc import Iterable

# response key used to annotate a response with the parcel it came from
SOURCE_KEY = "_source"

# element attribute used to record the source parcels an element came from
PROVENANCE_ATTR = "source_parcels"

# element types in the canonical order they are emitted from a merge
_TYPE_ORDER = {"node": 0, "way": 1, "relation": 2}


def _element_key(element: dict[str, Any]) -> tuple[str, int]:
    """
    Build an element's merge key from its type and original OSM ID.

    Parameters
    ----------
    element
        An OSM element from an API response.

    Returns
    -------
    key
        The `(element_type, osmid)` merge key.
    """
    return (element["type"], element["id"])


def _version_fact(element: dict[str, Any]) -> tuple[int, str]:
    """
    Extract an element's explicit version fact.

    OSM attaches a monotonically increasing version number and a timestamp to
    every element. Together these identify which of several observations of
    the same element is the most recent. Missing or malformed version facts
    sort below any explicit one.

    Parameters
    ----------
    element
        An OSM element from an API response.

    Returns
    -------
    fact
        The `(version, timestamp)` fact used to order conflicting observations.
    """
    try:
        version = int(element.get("version", 0))
    except (TypeError, ValueError):  # pragma: no cover
        version = 0
    timestamp = str(element.get("timestamp", ""))
    return (version, timestamp)


def _content_payload(element: dict[str, Any]) -> str:
    """
    Serialize an element's content canonically for equality and ordering.

    The merge's own provenance attribute and a response's source annotation
    are excluded, so they cannot influence conflict resolution.

    Parameters
    ----------
    element
        An OSM element from an API response.

    Returns
    -------
    payload
        A canonical JSON string of the element's content.
    """
    content = {k: v for k, v in element.items() if k not in {SOURCE_KEY, PROVENANCE_ATTR}}
    return json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)


def _geometry_payload(element: dict[str, Any]) -> tuple[Any, ...]:
    """
    Extract the geometry-defining content of an element as a hashable tuple.

    Parameters
    ----------
    element
        An OSM element from an API response.

    Returns
    -------
    geometry
        Coordinates for a node, node references for a way, or member
        references and roles for a relation.
    """
    element_type = element["type"]
    if element_type == "node":
        return (element_type, element.get("lat"), element.get("lon"))
    if element_type == "way":
        # node order is topological content, so a reversed order is a real
        # difference rather than the same geometry
        return (element_type, tuple(element.get("nodes", ())))
    members = tuple(
        (member.get("type"), member.get("ref"), member.get("role"))
        for member in element.get("members", ())
    )
    return (element_type, members)


def _response_signature(response: dict[str, Any]) -> str:
    """
    Compute a content signature for a response lacking an explicit source.

    The signature depends only on the response's elements, sorted canonically,
    so it is independent of both element ordering and response ordering.

    Parameters
    ----------
    response
        An API response JSON object.

    Returns
    -------
    signature
        The response's content signature.
    """
    elements = sorted(
        response.get("elements", ()),
        key=lambda element: (
            _TYPE_ORDER.get(element.get("type", ""), len(_TYPE_ORDER)),
            element.get("id", 0),
        ),
    )
    payload = json.dumps(elements, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()  # noqa: S324


def _conflict_message(
    key: tuple[str, int],
    observations: list[tuple[str, dict[str, Any]]],
) -> str:
    """
    Build a message that locates an unresolvable coordinate conflict.

    Parameters
    ----------
    key
        The conflicting element's `(element_type, osmid)` merge key.
    observations
        The conflicting `(parcel_label, element)` observations.

    Returns
    -------
    message
        A message identifying the element, each observation's geometry, and
        the source parcel that reported it.
    """
    details = "; ".join(
        f"parcel {label!r} reported {_geometry_payload(element)!r}"
        for label, element in observations
    )
    return (
        f"Unresolvable coordinate conflict for {key[0]} {key[1]} from multiple "
        f"source parcels: tied version facts but differing geometry ({details})."
    )


def _resolve(
    key: tuple[str, int],
    observations: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    """
    Select the observation to retain and the parcels it came from.

    The observation with the greatest version fact wins. Observations whose
    winning version facts tie must agree: a tie with differing geometry raises
    a `SourceConflictError`, while a tie with differing non-geometry content
    is resolved to the canonically smallest payload.

    Parameters
    ----------
    key
        The element's `(element_type, osmid)` merge key.
    observations
        The `(parcel_label, element)` observations.

    Returns
    -------
    element
        The selected observation.
    parcels
        The sorted labels of every parcel that observed this element.
    """
    parcels = sorted({label for label, _ in observations})

    # fast path: a single observation is trivially the winner
    if len(observations) == 1:
        element = dict(observations[0][1])
        element.pop(PROVENANCE_ATTR, None)
        return element, parcels

    facts = [_version_fact(element) for _, element in observations]
    best_fact = max(facts)
    winners = [
        (label, element)
        for (label, element), fact in zip(observations, facts, strict=True)
        if fact == best_fact
    ]

    # fast path: one observation has the strictly greatest version fact
    if len(winners) == 1:
        element = dict(winners[0][1])
        element.pop(PROVENANCE_ATTR, None)
        return element, parcels

    payloads = {_content_payload(element) for _, element in winners}
    if len(payloads) > 1:
        geometries = {_geometry_payload(element) for _, element in winners}
        if len(geometries) > 1:
            raise SourceConflictError(_conflict_message(key, winners))

        # non-geometry content differs but version facts tie: pick the
        # canonically smallest payload so the result cannot depend on order
        winners = [min(winners, key=lambda winner: _content_payload(winner[1]))]
        msg = (
            f"Conflicting non-geometry attributes for {key[0]} {key[1]} with "
            f"tied version facts; retained the canonically smallest payload "
            f"observed in parcels {parcels}."
        )
        utils.log(msg, level=lg.WARNING)

    element = dict(winners[0][1])
    element.pop(PROVENANCE_ATTR, None)
    return element, parcels


def merge_osm_elements(
    responses: Iterable[dict[str, Any]],
    *,
    sources: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[str]], int]:
    """
    Merge duplicate elements across API responses.

    Duplicates are identified by `(element_type, osmid)`. The observation with
    the greatest explicit version fact wins, unresolvable coordinate conflicts
    raise `SourceConflictError`, and non-geometry ties are resolved canonically.
    The returned elements are sorted canonically: nodes, then ways, then
    relations, and within each type by OSM ID. The result therefore never
    depends on the order in which the responses were returned.

    Parameters
    ----------
    responses
        Iterable of API response JSON objects. A response may carry an
        optional `SOURCE_KEY` entry naming its source parcel. When absent, a
        content signature derived from the response's elements is used.
    sources
        Optional sequence of parcel identifiers, parallel to `responses`. If
        provided, it overrides any `SOURCE_KEY` annotations.

    Returns
    -------
    elements
        The merged elements, canonically ordered. Each element's source
        parcels are returned separately, in `provenance`.
    provenance
        Keys are `(element_type, osmid)` merge keys and values are sorted
        lists of the source parcels the element was observed in.
    response_count
        The number of responses consumed.
    """
    responses = list(responses)
    if sources is not None and len(sources) != len(responses):  # pragma: no cover
        msg = "`sources` must be parallel to `responses`."
        raise ValueError(msg)
    if sources is None:
        sources = [
            str(response.get(SOURCE_KEY) or _response_signature(response))
            for response in responses
        ]

    observations: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = {}
    for label, response in zip(sources, responses, strict=True):
        for element in response.get("elements", ()):
            observations.setdefault(_element_key(element), []).append((label, element))

    elements: list[dict[str, Any]] = []
    provenance: dict[tuple[str, int], list[str]] = {}
    for key in sorted(
        observations,
        key=lambda key: (_TYPE_ORDER.get(key[0], len(_TYPE_ORDER)), key[1]),
    ):
        element, parcels = _resolve(key, observations[key])
        elements.append(element)
        provenance[key] = parcels

    msg = f"Merged {len(responses)} response(s) into {len(elements):,} unique elements"
    utils.log(msg, level=lg.INFO)
    return elements, provenance, len(responses)


def format_provenance(parcels: list[str]) -> str:
    """
    Render a provenance parcel list as a serialization-safe string.

    Graph and feature attributes must survive simplification (which takes a
    `set` of values), GraphML round-trips, and GeoPackage export. A sorted,
    comma-joined string satisfies all three while remaining human-readable.

    Parameters
    ----------
    parcels
        Sorted labels of the source parcels an element was observed in.

    Returns
    -------
    value
        A comma-joined string of the parcels.
    """
    return ",".join(parcels)


def merge_provenance(values: Iterable[str]) -> str:
    """
    Combine several provenance strings into one, preserving their union.

    When simplification merges edges or consolidation merges nodes, each
    contributing element carries its own provenance string. This function
    unions their parcels and returns a single sorted, comma-joined string.

    Parameters
    ----------
    values
        Provenance strings to combine.

    Returns
    -------
    value
        A sorted, comma-joined string of every unique parcel.
    """
    parcels = {parcel for value in values for parcel in str(value).split(",") if parcel}
    return format_provenance(sorted(parcels))


def has_multiple_sources(provenance: dict[tuple[str, int], list[str]]) -> bool:
    """
    Determine whether a merged result was observed in more than one parcel.

    Provenance is only meaningful when the input was actually subdivided into
    multiple parcels. This helper lets callers skip annotating elements from a
    single-parcel or single-file input.

    Parameters
    ----------
    provenance
        Keys are `(element_type, osmid)` merge keys and values are the source
        parcels each element was observed in.

    Returns
    -------
    multiple
        True if the elements were observed in more than one distinct parcel.
    """
    parcels = {parcel for values in provenance.values() for parcel in values}
    return len(parcels) > 1
