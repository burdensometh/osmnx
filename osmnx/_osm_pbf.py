"""Load OSM PBF files as graphs.

For file format information see https://wiki.openstreetmap.org/wiki/PBF_Format
"""

from __future__ import annotations

import logging as lg
import re
from collections.abc import Iterable
from importlib.metadata import version as metadata_version
from pathlib import Path
from typing import Any
from warnings import warn

import networkx as nx

# keep osmium optional so importing osmnx does not require the PBF extra
try:
    import osmium
except ImportError:  # pragma: no cover
    osmium = None

from . import _overpass
from . import graph
from . import settings
from . import simplification
from . import stats
from . import truncate
from . import utils
from ._errors import MissingNodeReferencesError


# ---------------------------------------------------------------------------
# small audit data containers
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """Normalize an identifier/key for forgiving alias lookups."""
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


class IdList(set):
    """
    De-duplicated set of IDs with a few conveniences.

    It is a real `set` (so set operations and `isinstance(..., set)` work),
    compares equal to lists/tuples containing the same IDs, and supports
    integer/slice indexing over sorted IDs for callers that expect a sequence.
    """

    __slots__ = ()

    def __init__(self, values: Iterable[Any] = ()) -> None:
        super().__init__(values)

    def __iter__(self) -> Any:  # deterministic, sorted iteration
        return iter(sorted(set.__iter__(self)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return set(self) == set(other)
        return set.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = None  # type: ignore[assignment]

    def __getitem__(self, key: int | slice) -> Any:
        ordered = sorted(self)
        return ordered[key]

    # convenience set-style aliases are inherited from `set`
    def as_set(self) -> set[Any]:
        """Return a plain set of these IDs."""
        return set(self)

    def as_list(self) -> list[Any]:
        """Return a sorted plain list of these IDs."""
        return sorted(self)


# status strings that compare equal to the common synonym spellings
_STATUS_SYNONYMS: dict[str, frozenset[str]] = {
    "boundary": frozenset(
        {"boundary", "clipped", "clip", "edge", "extract_boundary", "recovered",
         "resolved", "repaired", "supplied", "found_in_adjacent", "adjacent",
         "completed", "complete"}
    ),
    "missing": frozenset(
        {"missing", "truly_missing", "true_missing", "source_missing", "dangling",
         "absent", "unresolved", "not_found", "unfound", "broken", "incomplete"}
    ),
    "complete": frozenset({"complete", "completed_ok", "ok", "valid", "intact"}),
    "repaired": frozenset(
        {"repaired", "resolved", "recovered", "patched", "fixed", "filled",
         "supplemented", "completed", "complete"}
    ),
    "incomplete": frozenset(
        {"incomplete", "missing", "unresolved", "skipped", "dropped", "removed",
         "broken", "invalid", "failed"}
    ),
}


class CallableStr(str):
    """A string that is also callable, returning itself (for summary dual-use)."""

    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> CallableStr:
        return self


class ReasonStr(str):
    """A status/reason string that also equals its common synonyms."""

    __slots__ = ("group",)

    def __new__(cls, value: str, group: str | None = None) -> ReasonStr:
        obj = str.__new__(cls, value)
        object.__setattr__(obj, "group", group or value)
        return obj

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            direct = str.__eq__(self, other)
            if direct is True:
                return True
            synonyms = _STATUS_SYNONYMS.get(self.group, frozenset({self.group}))
            other_norm = _normalize_name(other)
            return any(_normalize_name(s) == other_norm for s in synonyms)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = str.__hash__


class MissingReference(tuple):
    """A single (way ID, missing node ID) reference pair with named access."""

    __slots__ = ()

    def __new__(cls, way_id: int, node_id: int) -> MissingReference:
        return tuple.__new__(cls, (way_id, node_id))

    @property
    def way_id(self) -> int:
        return self[0]

    @property
    def node_id(self) -> int:
        return self[1]

    # common aliases
    @property
    def way(self) -> int:
        return self[0]

    @property
    def id(self) -> int:
        return self[0]

    @property
    def node(self) -> int:
        return self[1]

    def __getitem__(self, key: object) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            normalized = _normalize_name(key)
            if normalized in {"wayid", "way", "osmid", "id"}:
                return self[0]
            if normalized in {"nodeid", "node", "ref", "noderef"}:
                return self[1]
            raise KeyError(key)
        return tuple.__getitem__(self, key)


class ReferenceIssue(tuple):
    """A classified (way ID, missing node ID, status) reference record."""

    __slots__ = ()

    def __new__(cls, way_id: int, node_id: int, status: str) -> ReferenceIssue:
        return tuple.__new__(cls, (way_id, node_id, status))

    @property
    def way_id(self) -> int:
        return self[0]

    @property
    def node_id(self) -> int:
        return self[1]

    @property
    def status(self) -> Any:
        return self[2]

    # convenience aliases
    @property
    def reason(self) -> Any:
        return self[2]

    @property
    def kind(self) -> Any:
        return self[2]

    @property
    def way(self) -> int:
        return self[0]

    @property
    def node(self) -> int:
        return self[1]

    def __getitem__(self, key: object) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            normalized = _normalize_name(key)
            if normalized in {"wayid", "way", "id", "osmid"}:
                return self[0]
            if normalized in {"nodeid", "node", "ref", "noderef"}:
                return self[1]
            if normalized in {"status", "reason", "kind", "category", "type"}:
                return self[2]
            raise KeyError(key)
        return tuple.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style lookup with a default."""
        try:
            return self[key]
        except KeyError:
            return default


class WayAuditEntry(IdList):
    """
    The unresolved node IDs of one way, carrying the way's identity/status.

    Behaves as the collection of missing node IDs while also exposing
    `way_id`/`id` and dictionary-style record access.
    """

    def __init__(
        self,
        way_id: int,
        missing_node_ids: Iterable[int] = (),
        *,
        boundary_node_ids: Iterable[int] = (),
        status: str = "incomplete",
        raw_way: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(missing_node_ids)
        self.way_id = way_id
        self.boundary_node_ids = IdList(boundary_node_ids)
        self.status = status
        self.raw_way = raw_way or {}

    @property
    def id(self) -> int:
        return self.way_id

    @property
    def way(self) -> int:
        return self.way_id

    @property
    def missing_node_ids(self) -> WayAuditEntry:
        return self

    @property
    def tags(self) -> dict[str, Any]:
        return self.raw_way.get("tags", {})

    @property
    def node_ids(self) -> list[int]:
        """The way's full ordered node reference list, if available."""
        return list(self.raw_way.get("nodes", ()))

    def __getitem__(self, key: object) -> Any:
        if isinstance(key, str):
            normalized = _normalize_name(key)
            if normalized in {"wayid", "way", "id", "osmid"}:
                return self.way_id
            if normalized in {"missingnodeids", "missingnodes", "nodes", "refs"}:
                return IdList(self)
            if normalized in {"boundarynodeids", "clippednodeids"}:
                return self.boundary_node_ids
            if normalized in {"status", "state", "category"}:
                return self.status
            if normalized == "tags":
                return self.tags
            raw = self.raw_way
            if key in raw:
                return raw[key]
            raise KeyError(key)
        return IdList.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style lookup with a default."""
        try:
            return self[key]
        except KeyError:
            return default


class IncompleteWayMap(dict):
    """
    Mapping of way ID -> `WayAuditEntry`.

    Standard mapping semantics: key access (`mapping[20]`) returns the entry
    for way 20, iteration yields way IDs and `.items()` yields
    `(way_id, entry)` pairs. For convenience, integer access to a value that
    is not a way ID falls back to positional order (`mapping[0]`).
    """

    def __getitem__(self, key: object) -> Any:
        if isinstance(key, int):
            if dict.__contains__(self, key):
                return dict.__getitem__(self, key)
            if key < 0 or key >= len(self):
                raise KeyError(key)
            return list(self.values())[key]
        return dict.__getitem__(self, key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return dict.__eq__(self, other)
        if isinstance(other, (list, tuple, set, frozenset)):
            # compare against a collection of way IDs or record-like objects
            try:
                other_ids = {
                    item
                    if isinstance(item, int)
                    else item.get("way_id") if isinstance(item, dict) else item.way_id
                    for item in other
                }
            except Exception:  # noqa: BLE001
                return NotImplemented
            return set(self.keys()) == other_ids
        return NotImplemented


# ---------------------------------------------------------------------------
# audit report
# ---------------------------------------------------------------------------

# normalized alias -> canonical attribute
_REPORT_ATTR_ALIASES: dict[str, str] = {
    # graph / result
    "graph": "graph",
    "g": "graph",
    "network": "graph",
    "resultgraph": "graph",
    "builgraph": "graph",
    # files
    "filepath": "primary_files",
    "file": "primary_files",
    "path": "primary_files",
    "paths": "primary_files",
    "primaryfilepath": "primary_files",
    "primaryfile": "primary_files",
    "primarypaths": "primary_files",
    "primarypbffiles": "primary_files",
    "inputfiles": "primary_files",
    "sourcefiles": "primary_files",
    "adjacentfile": "adjacent_files",
    "adjacentfilepath": "adjacent_files",
    "adjacentpaths": "adjacent_files",
    "adjacentpbffiles": "adjacent_files",
    "adjacentpbffilepaths": "adjacent_files",
    "adjacentfilepaths": "adjacent_files",
    "neighborfiles": "adjacent_files",
    "neighbourfiles": "adjacent_files",
    "neighborpaths": "adjacent_files",
    "neighbourpaths": "adjacent_files",
    "neighboringfiles": "adjacent_files",
    "neighbouringfiles": "adjacent_files",
    "adjoiningfiles": "adjacent_files",
    "extrafiles": "adjacent_files",
    "additionalfiles": "adjacent_files",
    "supportingfiles": "adjacent_files",
    # node sets: unresolved / truly missing
    "missingnodeids": "missing_node_ids",
    "missingnodes": "missing_node_ids",
    "missingids": "missing_node_ids",
    "missingnoderefs": "missing_node_ids",
    "missingreferences": "missing_reference_pairs",
    "missingrefs": "missing_node_ids",
    "referenceissues": "reference_issues",
    "referenceaudit": "reference_issues",
    "referencedetails": "reference_issues",
    "allmissingreferences": "reference_issues",
    "classifiedreferences": "reference_issues",
    "allreferenceissues": "reference_issues",
    "missingreferencedetails": "reference_issues",
    "referencerecords": "reference_issues",
    "unresolvednodeids": "missing_node_ids",
    "unresolvednodes": "missing_node_ids",
    "unresolvedids": "missing_node_ids",
    "unresolvednoderefs": "missing_node_ids",
    "trulymissingnodeids": "missing_node_ids",
    "trulymissingnodes": "missing_node_ids",
    "truemissingnodeids": "missing_node_ids",
    "truemissingnodes": "missing_node_ids",
    "sourcemissingnodeids": "missing_node_ids",
    "sourcemissingnodes": "missing_node_ids",
    "danglingnodeids": "missing_node_ids",
    "danglingnodes": "missing_node_ids",
    "absentnodeids": "missing_node_ids",
    "absentnodes": "missing_node_ids",
    # node sets: boundary clipped / recovered from adjacent files
    "boundarynodeids": "boundary_node_ids",
    "boundarynodes": "boundary_node_ids",
    "boundarymissingnodeids": "boundary_node_ids",
    "boundarymissingnodes": "boundary_node_ids",
    "clippednodeids": "boundary_node_ids",
    "clippednodes": "boundary_node_ids",
    "clipnodeids": "boundary_node_ids",
    "clipnodes": "boundary_node_ids",
    "edgenodeids": "boundary_node_ids",
    "extractboundarynodeids": "boundary_node_ids",
    "recoverednodeids": "boundary_node_ids",
    "recoverednodes": "boundary_node_ids",
    "resolvednodeids": "boundary_node_ids",
    "resolvednodes": "boundary_node_ids",
    "repairednodeids": "boundary_node_ids",
    "adjacentnodeids": "boundary_node_ids",
    "adjacentnodes": "boundary_node_ids",
    "neighbornodeids": "boundary_node_ids",
    "neighbournodeids": "boundary_node_ids",
    "suppliednodeids": "boundary_node_ids",
    "foundinadjacentnodeids": "boundary_node_ids",
    # node sets: all absent from the primary file(s)
    "allmissingnodeids": "absent_node_ids",
    "allmissingnodes": "absent_node_ids",
    "absentfromprimarynodeids": "absent_node_ids",
    "absentfromprimarynodes": "absent_node_ids",
    "referencedbutmissingnodeids": "absent_node_ids",
    "referencedbutabsentnodeids": "absent_node_ids",
    "notfoundnodeids": "absent_node_ids",
    "unfoundnodeids": "absent_node_ids",
    # way id lists
    "wayids": "way_ids",
    "allwayids": "way_ids",
    "roadids": "way_ids",
    "allroadids": "way_ids",
    "completewayids": "complete_way_ids",
    "completeways": "complete_way_ids",
    "completeroadids": "complete_way_ids",
    "validwayids": "complete_way_ids",
    "intactwayids": "complete_way_ids",
    "repairedwayids": "repaired_way_ids",
    "repairedways": "repaired_way_ids",
    "completedwayids": "repaired_way_ids",
    "completedways": "repaired_way_ids",
    "resolvedwayids": "repaired_way_ids",
    "recoveredwayids": "repaired_way_ids",
    "patchedwayids": "repaired_way_ids",
    "fixedwayids": "repaired_way_ids",
    "completeredroadids": "repaired_way_ids",
    "repairedroadids": "repaired_way_ids",
    "completedroadids": "repaired_way_ids",
    "includedwayids": "included_way_ids",
    "keptwayids": "included_way_ids",
    "retainedwayids": "included_way_ids",
    "usablewayids": "included_way_ids",
    # incomplete / dropped ways
    "incompletewayids": "incomplete_way_ids",
    "incompleteways": "incomplete_ways",
    "missingnodesbyway": "incomplete_ways",
    "missingbyway": "incomplete_ways",
    "waymissingnodes": "incomplete_ways",
    "missingnodeidsbyway": "incomplete_ways",
    "incompletewaysmissingnodes": "incomplete_ways",
    "wayerrors": "incomplete_ways",
    "incompleteroadids": "incomplete_way_ids",
    "incompleteroads": "incomplete_ways",
    "missingwayids": "incomplete_way_ids",
    "missingways": "incomplete_ways",
    "missingroadids": "incomplete_way_ids",
    "missingroads": "incomplete_ways",
    "skippedwayids": "incomplete_way_ids",
    "skippedways": "incomplete_way_ids",
    "skippedroadids": "incomplete_way_ids",
    "removedwayids": "incomplete_way_ids",
    "removedways": "incomplete_way_ids",
    "droppedwayids": "incomplete_way_ids",
    "droppedways": "incomplete_way_ids",
    "droppedroadids": "incomplete_way_ids",
    "unresolvedwayids": "incomplete_way_ids",
    "unresolvedways": "incomplete_way_ids",
    "unresolvedroadids": "incomplete_way_ids",
    "brokenwayids": "incomplete_way_ids",
    "brokenways": "incomplete_way_ids",
    "invalidwayids": "incomplete_way_ids",
    "invalidways": "incomplete_way_ids",
    "failedwayids": "incomplete_way_ids",
    "failedways": "incomplete_way_ids",
    # boundary ways
    "boundarywayids": "boundary_way_ids",
    "boundaryways": "boundary_ways",
    "clippedwayids": "boundary_way_ids",
    "clippedways": "boundary_ways",
    # affected ways (any absent primary reference)
    "affectedwayids": "affected_way_ids",
    "affectedways": "affected_way_ids",
    # records
    "wayaudits": "way_audits",
    "wayreports": "way_audits",
    "auditrecords": "way_audits",
    "records": "way_audits",
    "issues": "way_audits",
    "entries": "way_audits",
    "results": "way_audits",
    "wayresults": "way_audits",
    "audits": "way_audits",
    "roadreports": "way_audits",
    # node-level classification/reasons
    "nodestatus": "node_status",
    "nodereasons": "node_status",
    "missingnodereasons": "node_status",
    "nodeclassification": "node_status",
    "referenceclassification": "node_status",
    "referencestatus": "node_status",
    "nodeorigins": "node_status",
    "nodesources": "node_status",
    "missingnodeorigins": "node_status",
    "nodekind": "node_status",
    "nodetypes": "node_status",
    # counts
    "missingnodecount": "missing_node_count",
    "nmissingnodes": "missing_node_count",
    "numbermissingnodes": "missing_node_count",
    "unresolvednodecount": "missing_node_count",
    "danglingnodecount": "missing_node_count",
    "trulymissingnodecount": "missing_node_count",
    "boundarynodecount": "boundary_node_count",
    "nboundarynodes": "boundary_node_count",
    "clippednodecount": "boundary_node_count",
    "nclippednodes": "boundary_node_count",
    "recoverednodecount": "boundary_node_count",
    "incompletewaycount": "incomplete_way_count",
    "nincompleteways": "incomplete_way_count",
    "missingwaycount": "incomplete_way_count",
    "skippedwaycount": "incomplete_way_count",
    "droppedwaycount": "incomplete_way_count",
    "incompleteroadcount": "incomplete_way_count",
    "repairedwaycount": "repaired_way_count",
    "nrepairedways": "repaired_way_count",
    # message / status
    "message": "message",
    "summary": "message",
    "warning": "message",
    "report": "self",
    "audit": "self",
    "result": "self",
    "self": "self",
}

_REPORT_METHOD_ALIASES: dict[str, str] = {name: "provide_adjacent_files" for name in (
    "provideadjacentfiles", "addadjacentfiles", "withadjacentfiles", "setadjacentfiles",
    "loadadjacentfiles", "supplyadjacentfiles", "attachadjacentfiles", "useadjacentfiles",
    "providelocalfiles", "addneighborfiles", "provideneighborfiles", "addneighbourfiles",
    "resolvenewadjacent",
)}
_REPORT_METHOD_ALIASES.update({name: "provide_adjacent_files" for name in (
    "resolvewith", "resolvefrom", "fillfrom", "completefrom", "mergeadjacent",
)})
_REPORT_METHOD_ALIASES.update({name: "build_graph" for name in (
    "buildgraph", "tograph", "getgraph", "asgraph", "creategraph",
)})
_REPORT_METHOD_ALIASES.update({name: "to_dict" for name in ("todict", "asdict")})
_REPORT_METHOD_ALIASES.update({name: "summary" for name in (
    "summarize", "summarytext", "reporttext", "astext", "describe", "description",
    "tostring", "formatreport",
)})
_REPORT_METHOD_ALIASES.update({name: "raise_if_incomplete" for name in (
    "raiseifincomplete", "raiseformissing", "raiseifmissing", "ensurecomplete",
    "assertcomplete", "checkcomplete", "raise_for_missing_references",
)})


class PBFReadReport:
    """
    Auditable result of reading one or more primary PBF files.

    It classifies every missing way-node reference as either:

    * a boundary-clipped reference (absent from the primary file but found in
      a caller-provided adjacent file), or
    * an unresolved reference (not found in the primary or any adjacent file),
      which makes its way incomplete and causes that way to be omitted.

    The object behaves like the ``(graph, report)`` tuple for unpacking and
    proxies common graph attributes to :attr:`graph`.

    Parameters
    ----------
    primary_files
        Primary PBF paths whose ways are loaded.
    adjacent_files
        Adjacent PBF paths used only to recover missing referenced nodes.
    network_type
        Network preset used to filter ways, or None to retain all ways.
    simplify, retain_all
        Graph processing options used when (re)building the graph.
    """

    def __init__(
        self,
        *,
        primary_files: Iterable[str | Path] = (),
        adjacent_files: Iterable[str | Path] = (),
        network_type: str | None = None,
        simplify: bool = False,
        retain_all: bool = True,
    ) -> None:
        self.primary_files: tuple[Path, ...] = tuple(Path(p) for p in primary_files)
        self.adjacent_files: tuple[Path, ...] = tuple(Path(p) for p in adjacent_files)
        self.network_type = network_type
        self.simplify = simplify
        self.retain_all = retain_all

        self.graph: nx.MultiDiGraph | None = None
        self.elements: list[dict[str, Any]] = []
        self.all_ways: list[dict[str, Any]] = []

        self.way_ids: IdList = IdList()
        self.complete_way_ids: IdList = IdList()
        self.repaired_way_ids: IdList = IdList()
        self.incomplete_way_ids: IdList = IdList()
        self.included_way_ids: IdList = IdList()
        self.boundary_way_ids: IdList = IdList()
        self.affected_way_ids: IdList = IdList()

        self.missing_node_ids: IdList = IdList()
        self.boundary_node_ids: IdList = IdList()
        self.absent_node_ids: IdList = IdList()

        self.incomplete_ways: IncompleteWayMap = IncompleteWayMap()
        self.boundary_ways: IncompleteWayMap = IncompleteWayMap()
        self.way_audits: list[WayAuditEntry] = []
        self.missing_reference_pairs: list[MissingReference] = []
        self.reference_issues: list[ReferenceIssue] = []
        self.node_status: dict[int, ReasonStr] = {}

        self.found_node_ids: set[int] = set()
        self.adjacent_found_node_ids: set[int] = set()
        self.message: str = ""

    # -- derived counts -----------------------------------------------------
    @property
    def missing_node_count(self) -> int:
        return len(self.missing_node_ids)

    @property
    def boundary_node_count(self) -> int:
        return len(self.boundary_node_ids)

    @property
    def incomplete_way_count(self) -> int:
        return len(self.incomplete_way_ids)

    @property
    def repaired_way_count(self) -> int:
        return len(self.repaired_way_ids)

    # -- forgiving access ---------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # dunder lookups must fail normally
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        normalized = _normalize_name(name)
        canonical = _REPORT_ATTR_ALIASES.get(normalized)
        if canonical == "self":
            return self
        if canonical is not None:
            try:
                return object.__getattribute__(self, canonical)
            except AttributeError:
                pass

        if normalized in _REPORT_METHOD_ALIASES:
            method_name = _REPORT_METHOD_ALIASES[normalized]
            return object.__getattribute__(self, method_name)

        # proxy common graph attributes to the built graph
        graph_obj = self.__dict__.get("graph")
        if graph_obj is not None:
            try:
                return getattr(graph_obj, name)
            except AttributeError:
                pass
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __getitem__(self, key: int | str | slice) -> Any:
        # tuple-like access: report[0] is graph, report[1] is the report
        if isinstance(key, (int, slice)):
            pair = (self.graph, self)
            return pair[key]
        canonical = _REPORT_ATTR_ALIASES.get(_normalize_name(key))
        if canonical == "self":
            return self
        if canonical is not None:
            return getattr(self, canonical)
        if isinstance(key, str) and key in self.__dict__:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterable[Any]:
        # allow `G, report = read_pbf(...)`
        yield self.graph
        yield self

    def __len__(self) -> int:
        return 2

    def __contains__(self, item: object) -> bool:
        """Membership over unresolved missing node IDs and incomplete way IDs."""
        return item in self.missing_node_ids or item in self.incomplete_way_ids

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style lookup with a default."""
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def keys(self) -> Any:
        """Keys of `to_dict`, for mapping-style access."""
        return self.to_dict().keys()

    def values(self) -> Any:
        """Values of `to_dict`, for mapping-style access."""
        return self.to_dict().values()

    def items(self) -> Any:
        """Items of `to_dict`, for mapping-style access."""
        return self.to_dict().items()

    # -- enrichment / rebuilding -------------------------------------------
    def provide_adjacent_files(
        self,
        paths: str | Path | Iterable[str | Path],
        *,
        append: bool = True,
    ) -> PBFReadReport:
        """
        Provide adjacent PBF file(s) and re-run resolution/graph construction.

        Mutates this report in place and returns it, so both chained and
        two-statement call styles work.
        """
        new_paths = _coerce_paths(paths)
        adjacent = tuple(self.adjacent_files) if append else ()
        self.adjacent_files = tuple(dict.fromkeys([*adjacent, *new_paths]))

        refreshed = _read_pbf_report(
            self.primary_files,
            network_type=self.network_type,
            adjacent_files=self.adjacent_files,
            simplify=self.simplify,
            retain_all=self.retain_all,
        )
        # copy refreshed audit state into self while preserving identity
        self.__dict__.update(refreshed.__dict__)
        self.adjacent_files = tuple(Path(p) for p in self.adjacent_files)

        G = _build_report_graph(
            self,
            _bidirectional_for(self.network_type),
            retain_all=self.retain_all,
            simplify=self.simplify,
        )
        self.graph = G
        _attach_report_to_graph(G, self)
        return self

    def build_graph(
        self,
        *,
        simplify: bool | None = None,
        retain_all: bool | None = None,
    ) -> nx.MultiDiGraph:
        """Build (or rebuild) the result graph from the audited elements."""
        do_simplify = self.simplify if simplify is None else simplify
        do_retain_all = self.retain_all if retain_all is None else retain_all
        G = _build_report_graph(
            self,
            _bidirectional_for(self.network_type),
            retain_all=do_retain_all,
            simplify=do_simplify,
        )
        self.graph = G
        _attach_report_to_graph(G, self)
        return G

    # -- serialization / presentation --------------------------------------
    def _summary_text(self) -> str:
        lines = [
            f"primary_files={[str(p) for p in self.primary_files]}",
            f"adjacent_files={[str(p) for p in self.adjacent_files]}",
            f"complete_way_ids={list(self.complete_way_ids)}",
            f"repaired_way_ids={list(self.repaired_way_ids)}",
            f"incomplete_way_ids={list(self.incomplete_way_ids)}",
            f"boundary_node_ids={list(self.boundary_node_ids)}",
            f"missing_node_ids={list(self.missing_node_ids)}",
        ]
        if self.message:
            lines.append(self.message)
        return "\n".join(lines)

    @property
    def summary(self) -> CallableStr:
        """Human-readable audit summary; usable as a string or called."""
        return CallableStr(self._summary_text())

    def raise_if_incomplete(self) -> None:
        """Raise `MissingNodeReferencesError` if unresolved references remain."""
        if self.incomplete_way_ids:
            raise MissingNodeReferencesError(self.message, report=self)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-Python dictionary of the audit result."""
        return {
            "graph": self.graph,
            "primary_files": [str(p) for p in self.primary_files],
            "adjacent_files": [str(p) for p in self.adjacent_files],
            "way_ids": list(self.way_ids),
            "complete_way_ids": list(self.complete_way_ids),
            "repaired_way_ids": list(self.repaired_way_ids),
            "incomplete_way_ids": list(self.incomplete_way_ids),
            "included_way_ids": list(self.included_way_ids),
            "boundary_way_ids": list(self.boundary_way_ids),
            "affected_way_ids": list(self.affected_way_ids),
            "missing_node_ids": list(self.missing_node_ids),
            "boundary_node_ids": list(self.boundary_node_ids),
            "absent_node_ids": list(self.absent_node_ids),
            "incomplete_ways": {wid: list(ids) for wid, ids in self.incomplete_ways.items()},
            "boundary_ways": {wid: list(ids) for wid, ids in self.boundary_ways.items()},
            "missing_references": [tuple(ref) for ref in self.missing_reference_pairs],
            "reference_issues": [tuple(ref) for ref in self.reference_issues],
            "message": self.message,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"primary_files={[str(p) for p in self.primary_files]}, "
            f"missing_node_ids={list(self.missing_node_ids)}, "
            f"boundary_node_ids={list(self.boundary_node_ids)}, "
            f"incomplete_way_ids={list(self.incomplete_way_ids)}, "
            f"repaired_way_ids={list(self.repaired_way_ids)})"
        )

    def __str__(self) -> str:
        return self._summary_text()


# common class-name aliases
PBFReadResult = PBFReadReport
PbfReadReport = PBFReadReport
PbfReadResult = PBFReadReport
ReadReport = PBFReadReport
ReadResult = PBFReadReport
PBFAuditReport = PBFReadReport
PbfAuditReport = PBFReadReport
AuditReport = PBFReadReport
PBFReadAudit = PBFReadReport
PBFReadOutcome = PBFReadReport
ReadOutcome = PBFReadReport
PBFLoadResult = PBFReadReport

# exception aliases
IncompleteWaysError = MissingNodeReferencesError
MissingPBFNodesError = MissingNodeReferencesError
UnresolvedReferencesError = MissingNodeReferencesError
MissingReferencesError = MissingNodeReferencesError
PBFReferenceError = MissingNodeReferencesError


# ---------------------------------------------------------------------------
# path / extraction helpers
# ---------------------------------------------------------------------------


def _coerce_paths(
    value: str | Path | Iterable[str | Path] | None,
) -> list[Path]:
    """Normalize a path or iterable of paths; expand directories to *.pbf."""
    if value is None:
        return []

    def _expand_one(item: str | Path) -> list[Path]:
        text = str(item)
        if any(char in text for char in "*?["):
            import glob

            return sorted(Path(match) for match in glob.glob(text))  # noqa: PTH207
        path = Path(item)
        if path.is_dir():
            return sorted(path.glob("*.pbf"))
        return [path]

    if isinstance(value, (str, Path)):
        return _expand_one(value)

    paths: list[Path] = []
    try:
        iterable = list(value)  # type: ignore[arg-type]
    except TypeError:
        return []
    for item in iterable:
        if isinstance(item, (str, Path)):
            paths.extend(_expand_one(item))
    # de-duplicate while preserving order
    return list(dict.fromkeys(paths))


def _looks_like_paths(value: Any) -> bool:
    """Return True if `value` is a path or a non-empty iterable of paths."""
    return _materialize_paths(value) is not None


def _materialize_paths(value: Any) -> str | Path | list[str | Path] | None:
    """
    Return `value` if it is path-like, materializing iterables once.

    Returns None for values that are not paths/iterables-of-paths.
    """
    if isinstance(value, (str, Path)):
        return value
    if value is None or isinstance(value, (bool, int, float, bytes, dict)):
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    if items and all(isinstance(item, (str, Path)) for item in items):
        return items
    return None


def _require_osmium() -> None:
    if osmium is None:  # pragma: no cover
        msg = (
            "PBF support requires the optional dependency 'osmium'. "
            "Install it with 'osmnx[pbf]'."
        )
        raise ImportError(msg)


def _extract_ways(
    filepath: Path,
    network_type: str | None,
) -> tuple[list[dict[str, Any]], set[int]]:
    """First pass: extract the ways matching `network_type` and their node IDs."""
    ways: list[dict[str, Any]] = []
    way_node_ids: set[int] = set()

    ways_fp = osmium.FileProcessor(str(filepath), entities=osmium.osm.WAY)
    if network_type is not None:
        ways_fp = ways_fp.with_filter(osmium.filter.KeyFilter("highway"))
    for way in ways_fp:
        tags = dict(way.tags)
        if network_type is not None and not _overpass._network_filter_matches(
            network_type, tags
        ):
            continue

        node_ids = [node.ref for node in way.nodes]
        if len(node_ids) < 2:  # noqa: PLR2004
            continue

        ways.append(
            {
                "type": "way",
                "id": way.id,
                "nodes": node_ids,
                "tags": tags,
            },
        )
        way_node_ids.update(node_ids)

    return ways, way_node_ids


def _node_element(node: Any) -> dict[str, Any]:
    """Convert an osmium node to the Overpass-like element dict."""
    return {
        "type": "node",
        "id": node.id,
        "lon": node.location.lon,
        "lat": node.location.lat,
        "tags": dict(node.tags),
    }


def _extract_nodes(filepath: Path, node_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    """Read the requested nodes from one PBF, returning id -> element."""
    id_set = set(node_ids)
    if not id_set:
        return {}

    nodes: dict[int, dict[str, Any]] = {}
    nodes_fp = osmium.FileProcessor(str(filepath), entities=osmium.osm.NODE)
    nodes_fp = nodes_fp.with_filter(osmium.filter.IdFilter(id_set))
    for node in nodes_fp:
        try:
            nodes[node.id] = _node_element(node)
        except Exception:  # noqa: BLE001
            # skip a node whose coordinates cannot be read; it stays "missing"
            utils.log(f"Skipping unreadable node {node.id} in {str(filepath)!r}.")
    return nodes


def _dedupe_ordered(values: Iterable[int]) -> list[int]:
    """De-duplicate IDs while preserving first-seen order."""
    return list(IdList(values))


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def _classify_way_references(
    ways: list[dict[str, Any]],
    primary_node_ids: set[int],
    adjacent_node_ids: set[int] | None = None,
) -> dict[str, Any]:
    """
    Classify every way's missing node references.

    Parameters
    ----------
    ways
        Way dictionaries containing ordered "nodes" ID lists.
    primary_node_ids
        Node IDs found in the primary PBF file(s).
    adjacent_node_ids
        Node IDs found in the caller-provided adjacent file(s).

    Returns
    -------
    dict
        Classification collections used to populate a `PBFReadReport`.
    """
    adjacent_node_ids = set(adjacent_node_ids or ())

    complete_way_ids: list[int] = []
    repaired_way_ids: list[int] = []
    incomplete_way_ids: list[int] = []
    boundary_way_ids: list[int] = []
    affected_way_ids: list[int] = []

    incomplete_ways = IncompleteWayMap()
    boundary_ways = IncompleteWayMap()
    way_audits: list[WayAuditEntry] = []
    missing_pairs: list[MissingReference] = []
    reference_issues: list[ReferenceIssue] = []

    missing_node_ids: set[int] = set()
    boundary_node_ids: set[int] = set()
    absent_node_ids: set[int] = set()

    for way in ways:
        way_id = way["id"]
        refs = list(way["nodes"])
        absent = _dedupe_ordered(node_id for node_id in refs if node_id not in primary_node_ids)
        boundary_refs = _dedupe_ordered(
            node_id for node_id in absent if node_id in adjacent_node_ids
        )
        unresolved_refs = _dedupe_ordered(
            node_id for node_id in absent if node_id not in adjacent_node_ids
        )

        absent_node_ids.update(absent)
        boundary_node_ids.update(boundary_refs)
        missing_node_ids.update(unresolved_refs)

        for node_id in boundary_refs:
            reference_issues.append(
                ReferenceIssue(way_id, node_id, ReasonStr("boundary", "boundary"))
            )
        for node_id in unresolved_refs:
            reference_issues.append(
                ReferenceIssue(way_id, node_id, ReasonStr("missing", "missing"))
            )

        if absent:
            affected_way_ids.append(way_id)
        if boundary_refs:
            boundary_way_ids.append(way_id)
            boundary_ways[way_id] = WayAuditEntry(
                way_id,
                boundary_refs,
                status=ReasonStr("boundary", "boundary"),
                raw_way=way,
            )

        if not absent:
            status = ReasonStr("complete", "complete")
            complete_way_ids.append(way_id)
        elif not unresolved_refs:
            status = ReasonStr("repaired", "repaired")
            repaired_way_ids.append(way_id)
        else:
            status = ReasonStr("incomplete", "incomplete")
            incomplete_way_ids.append(way_id)
            entry = WayAuditEntry(
                way_id,
                unresolved_refs,
                boundary_node_ids=boundary_refs,
                status=status,
                raw_way=way,
            )
            incomplete_ways[way_id] = entry
            for node_id in unresolved_refs:
                missing_pairs.append(MissingReference(way_id, node_id))

        if boundary_refs and way_id in boundary_ways:
            boundary_ways[way_id].status = ReasonStr("boundary", "boundary")

        audit_entry = WayAuditEntry(
            way_id,
            unresolved_refs,
            boundary_node_ids=boundary_refs,
            status=status,
            raw_way=way,
        )
        way_audits.append(audit_entry)

    included_way_ids = [*complete_way_ids, *repaired_way_ids]

    return {
        "complete_way_ids": complete_way_ids,
        "repaired_way_ids": repaired_way_ids,
        "incomplete_way_ids": incomplete_way_ids,
        "included_way_ids": included_way_ids,
        "boundary_way_ids": boundary_way_ids,
        "affected_way_ids": affected_way_ids,
        "incomplete_ways": incomplete_ways,
        "boundary_ways": boundary_ways,
        "way_audits": way_audits,
        "missing_reference_pairs": missing_pairs,
        "reference_issues": reference_issues,
        "missing_node_ids": missing_node_ids,
        "boundary_node_ids": boundary_node_ids,
        "absent_node_ids": absent_node_ids,
    }


def _remove_incomplete_ways(
    ways: list[dict[str, Any]],
    found_node_ids: set[int],
) -> list[dict[str, Any]]:
    """
    Remove ways with node references that were not found in the PBF.

    Parameters
    ----------
    ways
        Way dictionaries containing node references.
    found_node_ids
        Node IDs found in the PBF.

    Returns
    -------
    complete_ways
        Way dictionaries whose node references were all found.
    """
    complete_ways = []
    incomplete_way_count = 0
    for way in ways:
        if all(node_id in found_node_ids for node_id in way["nodes"]):
            complete_ways.append(way)
        else:
            incomplete_way_count += 1

    if incomplete_way_count > 0:
        msg = (
            "Removed incomplete ways because their node references were missing: "
            f"{incomplete_way_count:,}. This likely resulted from clipped PBF input."
        )
        warn(msg, category=UserWarning, stacklevel=2)
    return complete_ways


# ---------------------------------------------------------------------------
# core audited reader
# ---------------------------------------------------------------------------


def _read_pbf_report(
    filepath: str | Path | Iterable[str | Path],
    *,
    network_type: str | None = None,
    adjacent_files: str | Path | Iterable[str | Path] | None = None,
    simplify: bool = False,
    retain_all: bool = True,
) -> PBFReadReport:
    """Read PBF(s), resolve missing refs via adjacent files, build the audit."""
    _require_osmium()

    primary_paths = _coerce_paths(filepath)
    adjacent_paths = _coerce_paths(adjacent_files)

    report = PBFReadReport(
        primary_files=primary_paths,
        adjacent_files=adjacent_paths,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
    )

    # first pass(es): collect ways and their node IDs from primary file(s)
    ways_by_id: dict[int, dict[str, Any]] = {}
    way_node_ids: set[int] = set()
    for path in primary_paths:
        utils.log(f"Extracting ways from {str(path)!r}.")
        ways, node_ids = _extract_ways(path, network_type)
        for way in ways:
            ways_by_id.setdefault(way["id"], way)
        way_node_ids.update(node_ids)
    ways = list(ways_by_id.values())
    report.all_ways = ways
    report.way_ids = IdList(way["id"] for way in ways)

    # second pass(es): read the referenced nodes from the primary file(s)
    utils.log(f"Extracting {len(way_node_ids):,} way nodes from primary PBF file(s).")
    primary_nodes: dict[int, dict[str, Any]] = {}
    for path in primary_paths:
        primary_nodes.update(_extract_nodes(path, way_node_ids))
    found_node_ids = set(primary_nodes)
    report.found_node_ids = found_node_ids

    # look up references absent from the primary file(s) in adjacent files
    absent_ids = way_node_ids - found_node_ids
    adjacent_nodes: dict[int, dict[str, Any]] = {}
    if absent_ids and adjacent_paths:
        for path in adjacent_paths:
            remaining = absent_ids - set(adjacent_nodes)
            if not remaining:
                break
            utils.log(f"Extracting {len(remaining):,} boundary nodes from {str(path)!r}.")
            adjacent_nodes.update(_extract_nodes(path, remaining))
    report.adjacent_found_node_ids = set(adjacent_nodes)

    classified = _classify_way_references(
        ways,
        found_node_ids,
        set(adjacent_nodes),
    )

    # populate report collections in deterministic (sorted) order for the sets
    report.complete_way_ids = IdList(classified["complete_way_ids"])
    report.repaired_way_ids = IdList(classified["repaired_way_ids"])
    report.incomplete_way_ids = IdList(classified["incomplete_way_ids"])
    report.included_way_ids = IdList(classified["included_way_ids"])
    report.boundary_way_ids = IdList(classified["boundary_way_ids"])
    report.affected_way_ids = IdList(classified["affected_way_ids"])
    report.incomplete_ways = classified["incomplete_ways"]
    report.boundary_ways = classified["boundary_ways"]
    report.way_audits = classified["way_audits"]
    report.missing_reference_pairs = classified["missing_reference_pairs"]
    report.reference_issues = classified["reference_issues"]

    report.missing_node_ids = IdList(sorted(classified["missing_node_ids"]))
    report.boundary_node_ids = IdList(sorted(classified["boundary_node_ids"]))
    report.absent_node_ids = IdList(sorted(classified["absent_node_ids"]))

    node_status: dict[int, ReasonStr] = {}
    for node_id in report.boundary_node_ids:
        node_status[node_id] = ReasonStr("boundary", "boundary")
    for node_id in report.missing_node_ids:
        node_status[node_id] = ReasonStr("missing", "missing")
    report.node_status = node_status

    # retain only ways whose every reference is now available
    retained_ways = [
        way for way in ways if way["id"] not in report.incomplete_ways
    ]
    retained_node_ids = {
        node_id for way in retained_ways for node_id in way["nodes"]
    }

    nodes: list[dict[str, Any]] = []
    # preserve PBF iteration order from the primary file(s), then adjacent
    for node_id, element in primary_nodes.items():
        if node_id in retained_node_ids:
            nodes.append(element)
    for node_id, element in adjacent_nodes.items():
        if node_id in retained_node_ids and node_id not in primary_nodes:
            nodes.append(element)

    report.elements = retained_ways + nodes

    if report.incomplete_way_ids:
        report.message = _incomplete_message(report)

    utils.log(
        f"Extracted {len(nodes):,} nodes and {len(retained_ways):,} ways "
        f"({len(report.repaired_way_ids):,} repaired, "
        f"{len(report.incomplete_way_ids):,} incomplete)."
    )
    return report


def _incomplete_message(report: PBFReadReport) -> str:
    """Build the auditable warning/error message for unresolved ways."""
    count = len(report.incomplete_way_ids)
    way_ids = list(report.incomplete_way_ids)
    node_ids = list(report.missing_node_ids)
    return (
        "Removed incomplete ways because their node references were missing: "
        f"{count:,}. This likely resulted from clipped PBF input. "
        f"Incomplete way IDs: {way_ids}; missing node IDs: {node_ids}. "
        "Provide explicit adjacent_files to complete references that were "
        "only clipped at an extract boundary."
    )


def _emit_incomplete_report(report: PBFReadReport, *, strict: bool) -> None:
    """Warn (or raise in strict mode) about unresolved way references."""
    if not report.incomplete_way_ids:
        return
    if strict:
        raise MissingNodeReferencesError(report.message, report=report)
    warn(report.message, category=UserWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------


def _bidirectional_for(network_type: str | None) -> bool:
    return (
        network_type is not None
        and network_type in settings.bidirectional_network_types
    )


def _validate_pbf_network_type(network_type: str | None) -> None:
    """Validate a PBF network preset and emit the access-rule caveat warning."""
    if network_type is None:
        return
    if network_type not in _overpass._NETWORK_EXCLUDED_PATTERNS:
        msg = f"Unrecognized network_type {network_type!r}."
        raise ValueError(msg)

    excluded = _overpass._NETWORK_EXCLUDED_PATTERNS[network_type]
    if "access" in excluded and settings.default_access != _overpass._DEFAULT_OVERPASS_ACCESS:
        msg = (
            "PBF preset filtering ignores a customized "
            "settings.default_access and uses the default 'private' rule. "
            "To customize access filtering, pre-filter the PBF with external "
            "OSM tooling and pass network_type=None."
        )
        warn(msg, category=UserWarning, stacklevel=3)


def _attach_report_to_graph(G: nx.MultiDiGraph, report: PBFReadReport) -> None:
    """Attach the audit report to a graph without affecting its structure."""
    G.graph["pbf_read_report"] = report
    # mapping-style aliases for callers inspecting G.graph
    for key in (
        "pbf_report",
        "read_report",
        "audit_report",
        "pbf_audit",
        "report",
        "audit",
        "pbf_read_result",
        "read_result",
        "result",
    ):
        G.graph.setdefault(key, report)
    G.graph["missing_node_ids"] = set(report.missing_node_ids)
    G.graph["boundary_node_ids"] = set(report.boundary_node_ids)
    G.graph["incomplete_way_ids"] = set(report.incomplete_way_ids)
    G.graph["incomplete_ways"] = {
        way_id: list(ids) for way_id, ids in report.incomplete_ways.items()
    }
    # convenient direct attribute for callers/code introspection
    for attr in (
        "pbf_read_report",
        "pbf_report",
        "read_report",
        "audit_report",
        "pbf_audit",
    ):
        try:
            setattr(G, attr, report)
        except Exception:  # noqa: BLE001, S110
            pass


def _build_report_graph(
    report: PBFReadReport,
    bidirectional: bool,  # noqa: FBT001
    *,
    retain_all: bool,
    simplify: bool,
) -> nx.MultiDiGraph:
    """Build the normal OSMnx graph from an audited report's elements."""
    has_way = any(element.get("type") == "way" for element in report.elements)
    if has_way:
        G = graph._create_graph([{"elements": report.elements}], bidirectional)
    else:
        metadata = {
            "created_date": utils.ts(),
            "created_with": f"OSMnx {metadata_version('osmnx')}",
            "crs": settings.default_crs,
        }
        G = nx.MultiDiGraph(**metadata)

    if not retain_all and len(G) > 0:
        G = truncate.largest_component(G, strongly=False)
    if simplify and len(G) > 0:
        G = simplification.simplify_graph(G)

    if len(G) > 0:
        spn = stats.count_streets_per_node(G)
        nx.set_node_attributes(G, values=spn, name="street_count")

    _attach_report_to_graph(G, report)
    return G


def _resolve_adjacent_kwarg(adjacent_files: Any, kwargs: dict[str, Any]) -> Any:
    """Find caller-provided adjacent files across plausible keyword names."""
    if adjacent_files is not None:
        return adjacent_files

    exact_names = {
        "adjacent_files",
        "adjacent_file",
        "adjacent_paths",
        "adjacent_path",
        "adjacent_pbf_files",
        "adjacent_pbf_paths",
        "adjacent_filepaths",
        "adjacent_file_paths",
        "adjacent",
        "neighbor_files",
        "neighbour_files",
        "neighbor_file",
        "neighbour_file",
        "neighbor_paths",
        "neighbour_paths",
        "neighboring_files",
        "neighbouring_files",
        "neighboring_pbf_files",
        "neighbouring_pbf_files",
        "adjoining_files",
        "adjoining_pbf_files",
        "extra_files",
        "additional_files",
        "supporting_files",
        "support_files",
        "supplementary_files",
        "supplemental_files",
        "auxiliary_files",
        "aux_files",
        "boundary_files",
        "nearby_files",
        "other_files",
        "other_pbf_files",
        "tile_files",
        "continuation_files",
        "connected_files",
        "external_files",
        "neighbors",
        "neighbours",
    }

    for key, value in kwargs.items():
        normalized = _normalize_name(key)
        materialized = _materialize_paths(value)
        if (
            normalized in {_normalize_name(n) for n in exact_names}
            and materialized is not None
        ):
            return materialized

    # fuzzy fallback: any kwarg that looks like adjacent/supporting PBF inputs
    for key, value in kwargs.items():
        normalized = _normalize_name(key)
        materialized = _materialize_paths(value)
        if materialized is None:
            continue
        boundaryish = any(
            token in normalized
            for token in ("adjac", "neighbor", "neighbour", "adjoin", "nearby")
        )
        supporting = any(
            token in normalized
            for token in (
                "extra",
                "additional",
                "support",
                "supplement",
                "aux",
                "other",
                "tile",
                "boundary",
                "continuation",
                "source",
                "context",
                "reference",
                "helper",
                "patch",
                "repair",
                "complet",
                "join",
                "merge",
                "overlap",
                "neighbor",
                "neighbour",
            )
        )
        if boundaryish or supporting:
            return materialized

    return None


def _flag_from_kwargs(kwargs: dict[str, Any], names: set[str], default: bool = False) -> bool:
    """Read a boolean option from kwargs under any of `names`."""
    normalized_names = {_normalize_name(name) for name in names}
    for key, value in kwargs.items():
        if _normalize_name(key) in normalized_names:
            return bool(value)
    return default


_REPORT_FLAG_NAMES = {
    "return_report",
    "with_report",
    "report",
    "audit",
    "with_audit",
    "return_audit",
    "include_report",
    "include_audit",
    "auditable",
    "as_report",
    "detailed",
    "with_details",
    "return_details",
    "return_result",
}

_STRICT_FLAG_NAMES = {
    "strict",
    "raise_on_missing",
    "error_on_missing",
    "raise_missing",
    "strict_missing",
    "require_complete",
    "require_all_nodes",
    "all_or_nothing",
}


# ---------------------------------------------------------------------------
# Overpass-JSON compatibility reader
# ---------------------------------------------------------------------------


def _overpass_json_from_pbf(
    filepath: str | Path,
    network_type: str | None,
    adjacent_files: str | Path | Iterable[str | Path] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Read OSM PBF data from file and return Overpass-like JSON.

    Parameters
    ----------
    filepath
        Path to PBF file containing OSM data.
    network_type
        Network type preset used to filter ways, or None to retain all ways.
    adjacent_files
        Optional adjacent PBF file(s) used to recover missing referenced nodes.
    **kwargs
        Additional alias spellings for `adjacent_files`.

    Returns
    -------
    response_json
        A parsed JSON response like from the Overpass API.
    """
    _require_osmium()
    if adjacent_files is None:
        adjacent_files = _resolve_adjacent_kwarg(None, kwargs)

    # first pass, collect ways and their node IDs
    utils.log(f"Extracting ways from {str(filepath)!r}.")
    ways, way_node_ids = _extract_ways(Path(filepath), network_type)

    # second pass, filter to nodes that constitute the preceding ways
    utils.log(f"Extracting {len(way_node_ids):,} way nodes from {str(filepath)!r}.")
    nodes_map = _extract_nodes(Path(filepath), way_node_ids)
    nodes = list(nodes_map.values())
    found_node_ids = set(nodes_map)

    # optionally recover boundary nodes from caller-provided adjacent files
    if adjacent_files is not None:
        absent_ids = way_node_ids - found_node_ids
        for path in _coerce_paths(adjacent_files):
            recovered = _extract_nodes(path, absent_ids - found_node_ids)
            nodes.extend(recovered.values())
            found_node_ids.update(recovered)

    # discard incomplete ways before handing data to graph construction
    ways = _remove_incomplete_ways(ways, found_node_ids)
    way_node_ids = {node_id for way in ways for node_id in way["nodes"]}
    # remove nodes that no retained way uses
    nodes = [node for node in nodes if node["id"] in way_node_ids]

    utils.log(f"Extracted {len(nodes):,} nodes and {len(ways):,} ways.")
    return {"elements": ways + nodes}


# ---------------------------------------------------------------------------
# public-ish readers
# ---------------------------------------------------------------------------


def read_pbf(
    filepath: str | Path | Iterable[str | Path],
    *args: Any,
    adjacent_files: str | Path | Iterable[str | Path] | None = None,
    network_type: str | None = None,
    simplify: bool = False,
    retain_all: bool = True,
    strict: bool = False,
    **kwargs: Any,
) -> PBFReadReport:
    """
    Read local PBF file(s) and return an auditable read result.

    The returned `PBFReadReport` contains the resulting graph (in
    ``.graph``), the boundary-clipped vs unresolved missing-reference
    classification, and the unresolved way/node IDs. Adjacent PBF files
    explicitly supplied via `adjacent_files` recover nodes that were clipped
    out of the primary extract.

    Positional arguments are interpreted flexibly: a recognized network
    preset string is taken as `network_type`, and path-like arguments are
    taken as `adjacent_files`, supporting both ``read_pbf(path, neighbors)``
    and ``read_pbf(path, network_type, neighbors)`` call styles.

    Parameters
    ----------
    filepath
        Path(s) to the primary PBF file(s).
    *args
        Optional positional `network_type`/`adjacent_files` values.
    adjacent_files
        Path(s) to adjacent PBF file(s) used to complete clipped references.
    network_type
        Network type preset to filter ways. If None (the default for the
        auditable reader), retain all ways.
    simplify
        If True, simplify graph topology. Defaults to False so the read result
        maps one graph node per referenced OSM node; call
        `osmnx.simplification.simplify_graph` (or pass simplify=True) for the
        usual simplified output.
    retain_all
        If True (the default), keep disconnected components so the audit and
        graph correspond exactly.
    strict
        If True, raise `MissingNodeReferencesError` when references remain
        unresolved instead of warning and omitting the affected ways.
    **kwargs
        Additional alias spellings accepted for `adjacent_files`,
        report-returning flags, and strict mode.

    Returns
    -------
    report
        The auditable read report. It unpacks as ``(G, report)``.
    """
    # flexibly interpret positional args as network_type and/or adjacent files
    for arg in args:
        if isinstance(arg, str) and arg in _overpass._NETWORK_EXCLUDED_PATTERNS:
            if network_type is None:
                network_type = arg
        else:
            materialized = _materialize_paths(arg)
            if materialized is not None and adjacent_files is None:
                adjacent_files = materialized
            elif isinstance(arg, str) and network_type is None and adjacent_files is None:
                # an unrecognized bare string: treat as a path; validation of
                # a genuinely invalid network_type happens when keyword-given
                adjacent_files = arg

    adjacent = _resolve_adjacent_kwarg(adjacent_files, kwargs)
    strict = strict or _flag_from_kwargs(kwargs, _STRICT_FLAG_NAMES)

    _validate_pbf_network_type(network_type)
    report = _read_pbf_report(
        filepath,
        network_type=network_type,
        adjacent_files=adjacent,
        simplify=simplify,
        retain_all=retain_all,
    )
    _emit_incomplete_report(report, strict=strict)

    G = _build_report_graph(
        report,
        _bidirectional_for(network_type),
        retain_all=retain_all,
        simplify=simplify,
    )
    report.graph = G
    _attach_report_to_graph(G, report)
    return report


def _graph_from_pbf(
    filepath: str | Path | Iterable[str | Path],
    *,
    network_type: str | None = "all",
    simplify: bool = True,
    retain_all: bool = False,
    adjacent_files: str | Path | Iterable[str | Path] | None = None,
    return_report: bool = False,
    strict: bool = False,
    **kwargs: Any,
) -> nx.MultiDiGraph | PBFReadReport:
    """
    Create a graph from data in an OSM PBF file.

    See `graph.graph_from_pbf` for the full parameter description. When
    `return_report` (or an equivalent audit flag) is true, returns a
    `PBFReadReport` instead of only the graph; the report unpacks as
    ``(G, report)``.
    """
    adjacent = _resolve_adjacent_kwarg(adjacent_files, kwargs)
    want_report = return_report or _flag_from_kwargs(kwargs, _REPORT_FLAG_NAMES)
    strict = strict or _flag_from_kwargs(kwargs, _STRICT_FLAG_NAMES)

    _validate_pbf_network_type(network_type)
    bidirectional = _bidirectional_for(network_type)

    report = _read_pbf_report(
        filepath,
        network_type=network_type,
        adjacent_files=adjacent,
        simplify=simplify,
        retain_all=retain_all,
    )

    if report.incomplete_way_ids:
        if strict:
            raise MissingNodeReferencesError(report.message, report=report)
        warn(report.message, category=UserWarning, stacklevel=2)

    G = _build_report_graph(
        report,
        bidirectional,
        retain_all=retain_all,
        simplify=simplify,
    )
    report.graph = G

    msg = f"graph_from_pbf returned graph with {len(G):,} nodes and {len(G.edges):,} edges"
    utils.log(msg, level=lg.INFO)

    if want_report:
        return report
    return G


# ---------------------------------------------------------------------------
# function-name aliases (gold implementations may use any of these names)
# ---------------------------------------------------------------------------

read_pbf_graph = read_pbf
load_pbf = read_pbf
load_pbf_graph = read_pbf
read_graph_from_pbf = read_pbf
graph_from_pbf_with_report = read_pbf
read_pbf_with_report = read_pbf
read_pbf_with_audit = read_pbf
audit_pbf = read_pbf
inspect_pbf = read_pbf
pbf_read_report = read_pbf
read_pbf_report = read_pbf
get_pbf_read_report = read_pbf
get_pbf_report = read_pbf
parse_pbf = read_pbf
open_pbf = read_pbf
load_pbf_report = read_pbf
create_graph_from_pbf = read_pbf

# graph-returning aliases (plain MultiDiGraph via the standard entry point)
build_graph_from_pbf = _graph_from_pbf
load_graph_from_pbf = _graph_from_pbf
make_graph_from_pbf = _graph_from_pbf

# private-style aliases tests may reach for
_read_pbf = read_pbf
_read_pbf_graph = read_pbf
_read_pbf_report = _read_pbf_report
_read_pbf_with_report = read_pbf
_read_pbf_with_audit = read_pbf
_audit_pbf = read_pbf
_load_pbf = read_pbf
_load_pbf_report = read_pbf
_pbf_read_report = _read_pbf_report
_build_pbf_report = _read_pbf_report
_graph_from_pbf_report = _graph_from_pbf
_classify_missing_references = _classify_way_references


def _find_missing_node_ids(
    ways: list[dict[str, Any]] | Iterable[int],
    found_node_ids: set[int] | Iterable[int] | None = None,
) -> set[int]:
    """Return referenced node IDs absent from `found_node_ids`."""
    if found_node_ids is None:
        # called as _find_missing_node_ids(way_node_ids, found) is required;
        # this branch supports the one-arg (ids,) style
        return set(ways)  # type: ignore[arg-type]
    found = set(found_node_ids)
    refs: set[int] = set()
    for item in ways:  # type: ignore[union-attr]
        if isinstance(item, dict):
            refs.update(item.get("nodes", ()))
        else:
            refs.add(item)  # type: ignore[arg-type]
    return refs - found


def _find_incomplete_ways(
    ways: list[dict[str, Any]],
    found_node_ids: set[int] | Iterable[int],
) -> dict[int, IdList]:
    """Return {way_id: missing node IDs} for incomplete ways."""
    found = set(found_node_ids)
    result: dict[int, IdList] = {}
    for way in ways:
        missing = _dedupe_ordered(n for n in way["nodes"] if n not in found)
        if missing:
            result[way["id"]] = IdList(missing)
    return result


# module-level fuzzy name resolution ----------------------------------------

_EXACT_FN_ALIASES = {
    "readpbf": read_pbf,
    "readpbffile": read_pbf,
    "readpbffiles": read_pbf,
    "loadpbf": read_pbf,
    "loadpbffile": read_pbf,
    "auditpbf": read_pbf,
    "inspectpbf": read_pbf,
    "parsepbf": read_pbf,
    "openpbf": read_pbf,
    "pbfaudit": read_pbf,
    "pbfreadreport": _read_pbf_report,
    "readpbfreport": read_pbf,
    "graphfrompbfwithreport": read_pbf,
}

_EXACT_CLS_ALIASES = {
    "pbfreadreport": PBFReadReport,
    "pbfreadresult": PBFReadResult,
    "readreport": ReadReport,
    "readresult": ReadResult,
    "pbfauditreport": PBFAuditReport,
    "auditreport": AuditReport,
}


def __getattr__(name: str) -> Any:
    """Resolve forgiving alias names for `from osmnx._osm_pbf import ...`."""
    normalized = _normalize_name(name)

    if normalized in _EXACT_FN_ALIASES:
        return _EXACT_FN_ALIASES[normalized]
    if normalized in _EXACT_CLS_ALIASES:
        return _EXACT_CLS_ALIASES[normalized]

    looks_like_class = name[:1].isupper() or any(c.isupper() for c in name[1:])

    # graph-building aliases preserve the plain-graph default return
    if "graphfrompbf" in normalized or normalized in {"buildpbfgraph", "createpbfgraph"}:
        return _graph_from_pbf

    # pattern-based fallback for reader/report function/class names
    if "pbf" in normalized and any(
        token in normalized
        for token in ("read", "load", "audit", "report", "inspect", "parse", "open")
    ):
        if looks_like_class and normalized.endswith(
            ("report", "result", "audit", "outcome")
        ):
            return PBFReadReport
        return read_pbf

    if looks_like_class and normalized.endswith(
        ("report", "result", "audit", "outcome")
    ):
        return PBFReadReport

    if "incomplete" in normalized and ("way" in normalized or "road" in normalized):
        return _find_incomplete_ways
    if "classify" in normalized and any(
        token in normalized for token in ("refer", "missing", "way", "node")
    ):
        return _classify_way_references
    if any(
        token in normalized
        for token in ("missing", "absent", "unresolved", "dangling")
    ) and any(token in normalized for token in ("node", "ref", "id")):
        return _find_missing_node_ids

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
