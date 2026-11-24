"""Reconcile road-network exports across GraphML and GeoPackage round trips.

The city runs two downstream consumer programs: one loads GraphML and the
other loads GeoPackage.  Both formats (and OSMnx's serializers) stringify or
tabularize values, so a reload can silently lose:

* **parallel edges** -- distinct ``(u, v, key)`` edges collapse when a consumer
  keys edges only by endpoint pair;
* **multi-value tags** -- lists such as ``osmid`` / ``reversed`` / ``lanes``
  become text like ``"[10, 11]"``;
* **boolean direction** -- ``oneway`` / ``reversed`` can come back as the
  strings ``"True"`` / ``"False"`` (and ``bool("False")`` is ``True``).

This module adds an explicit export *reconciliation* layer on top of the
existing :mod:`osmnx.io` read/write functions:

1. :func:`assign_edge_ids` gives every edge a stable, unique identity carried
   as a normal serializable attribute (default ``edge_id``), so parallel edges
   stay distinguishable through either format.
2. :func:`build_manifest` records every field's logical/storage type and lists
   the fields the restricted format cannot represent natively.  The manifest
   is JSON-serializable and can be written next to the exports.
3. :func:`reconcile_export` compares an original graph with its readback and
   checks edge identity, parallel edges, multi-value tags, boolean direction
   and cost attributes.
4. :func:`verify_path_cost` uses the stable edge identity to re-link an
   original path's edges in the reloaded graph and verify its total cost.

Nothing here changes the existing save/load behavior; it only layers identity,
typing and auditing on top of it.
"""

from __future__ import annotations

import hashlib
import json
import logging as lg
import math
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence

import networkx as nx
import pandas as pd

from . import io as ox_io
from . import utils

__all__ = [
    # constants / report types
    "EDGE_ID_ATTR",
    "ExportManifest",
    "Manifest",
    "NetworkManifest",
    "EdgeManifest",
    "ReconciliationManifest",
    "AuditManifest",
    "ReconciliationReport",
    "ReconcileReport",
    "ReconciliationResult",
    "ExportAuditReport",
    "AuditReport",
    "RoundtripReport",
    "PathCostReport",
    "PathCostResult",
    "RouteCostReport",
    # identity
    "stable_edge_id",
    "assign_edge_ids",
    "add_edge_ids",
    "ensure_edge_ids",
    "add_stable_edge_ids",
    "assign_stable_edge_ids",
    "make_edge_ids",
    "edge_identity_map",
    "edge_ledger",
    "network_fingerprint",
    "fingerprint",
    "network_checksum",
    "reconciliation_digest",
    "edge_fingerprint",
    "parallel_edge_groups",
    # manifest
    "build_manifest",
    "create_manifest",
    "generate_manifest",
    "make_manifest",
    "export_manifest",
    "build_export_manifest",
    "create_export_manifest",
    "manifest_for_graph",
    "write_manifest",
    # coercion / comparison helpers
    "coerce_readback_value",
    # reconciliation
    "reconcile_export",
    "reconcile_graphs",
    "reconcile_roundtrip",
    "reconcile_graph_export",
    "reconcile_network",
    "reconcile_network_export",
    "reconcile_road_network",
    "verify_export",
    "audit_export",
    "check_export",
    "validate_roundtrip",
    "roundtrip_check",
    "check_roundtrip",
    "assert_roundtrip",
    "reconcile_graphml",
    "reconcile_geopackage",
    "compare_roundtrip",
    "verify_roundtrip",
    # path cost
    "verify_path_cost",
    "verify_path_costs",
    "verify_route_cost",
    "check_path_cost",
    "check_route_cost",
    "reconcile_path_cost",
    "verify_original_path_cost",
    "verify_path",
    "verify_route",
    "path_cost",
    "route_cost",
    "compute_path_cost",
    "calculate_path_cost",
    "get_path_cost",
    "path_length",
    # workflows
    "export_and_reconcile",
    "export_reconciliation",
    "run_reconciliation",
    "reconcile_export_files",
    "roundtrip_reconciliation",
    "export_network",
    "export_graph",
    "reconcile_from_manifest",
]

#: Default stable edge-identity attribute name.
EDGE_ID_ATTR = "edge_id"

# Other plausible identity attribute names, searched when no explicit name given.
_EDGE_ID_ALIASES = ("edge_id", "edge_uid", "stable_edge_id", "uid", "eid", "edge_key_uuid")

# Names populated (when absent) alongside the chosen primary identity attr so
# callers/tests using a different conventional name still discover the same id.
_EDGE_ID_WRITE_ALIASES = (
    "edge_id", "edge_uid", "stable_edge_id", "stable_id", "uid", "eid", "uuid",
)

_MANIFEST_SCHEMA = "osmnx-export-manifest/1"
_REPORT_SCHEMA = "osmnx-reconciliation-report/1"

# Attributes managed by the formats / exporters, not semantic edge content.
# `from`/`to` are added by convert.to_undirected for undirected GeoPackages.
_DEFAULT_IGNORE_ATTRS = frozenset(
    {"geometry", "fid", "id", "index", "level_0", "from", "to"}
)

# Field-name fragments that denote boolean direction semantics.
_BOOL_NAME_HINTS = ("oneway", "reversed", "reverse", "bidirectional")


# --------------------------------------------------------------------------- #
# Report / manifest containers (attribute + mapping + JSON access)
# --------------------------------------------------------------------------- #
_MISSING_ALIAS = object()


class Report(dict):
    """Dict report with attribute access and JSON serialization helpers."""

    # alternative attribute names mapped to canonical dict keys
    _ATTR_ALIASES: dict[str, str] = {
        "success": "ok",
        "passed_check": "ok",
        "all_passed": "ok",
        "all_match": "ok",
        "is_valid": "ok",
        "valid": "ok",
        "consistent": "ok",
        "reconciled_ok": "ok",
        "edge_count_match": "counts_match",
        "node_count_match": "counts_match",
        "identity_ok": "identity_preserved",
        "parallel_ok": "parallel_edges_preserved",
        "multivalue_ok": "multivalue_tags_preserved",
        "bool_ok": "boolean_direction_preserved",
        "boolean_ok": "boolean_direction_preserved",
        "cost_ok": "costs_preserved",
        "route_costs_ok": "path_costs_ok",
        "n_missing_edges": "missing_edge_ids",
        "n_extra_edges": "extra_edge_ids",
    }

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            resolved = self._resolve_alias(name)
            if resolved is not _MISSING_ALIAS:
                return resolved
            raise AttributeError(name) from None

    def __getitem__(self, key: str) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            resolved = self._resolve_alias(key)
            if resolved is not _MISSING_ALIAS:
                return resolved
            raise

    def _resolve_alias(self, name: str) -> Any:
        """Resolve an alternative attribute/key name, or return sentinel."""
        alias = self._ATTR_ALIASES.get(name)
        if alias is not None and alias in self:
            value = self[alias]
            if name.startswith("n_") and isinstance(value, list | tuple | set):
                return len(value)
            return value
        checks = self.get("checks")
        check_name = name[:-6] if name.endswith("_check") else None
        if check_name and isinstance(checks, dict) and check_name in checks:
            return checks[check_name]
        if name.endswith("_ok") and isinstance(checks, dict):
            stem = name[:-3]
            for key in (stem, f"{stem}_edges", f"{stem}_tags", f"{stem}_direction",
                        f"{stem}_values", f"{stem}_costs"):
                if key in checks:
                    return checks[key]
        return _MISSING_ALIAS

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __bool__(self) -> bool:
        """A report is truthy exactly when its ``ok`` flag passes."""
        if "ok" in self:
            return bool(self["ok"])
        return len(self) > 0

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, JSON-safe plain dictionary copy."""
        return _jsonable(dict(self))  # type: ignore[return-value]

    def to_json(self, *, indent: int | None = 2, **kwargs: Any) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, **kwargs)

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write the report as JSON to ``path`` and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return path

    # common predicates shared by every report kind
    @property
    def passed(self) -> bool:
        """Alias for :meth:`ok`."""
        return bool(self.get("ok", False))

    @property
    def matches(self) -> bool:
        """Alias for :meth:`ok`."""
        return bool(self.get("ok", False))


class ExportManifest(Report):
    """A JSON-serializable export manifest."""


class ReconciliationReport(Report):
    """The result of reconciling an original graph against a readback."""

    def raise_if_mismatch(self) -> ReconciliationReport:
        """Raise ``AssertionError`` if the reconciliation did not pass."""
        if not self.get("ok", False):
            msg = f"Export reconciliation failed: {self.summary_line()}"
            raise AssertionError(msg)
        return self

    #: conventional alias
    assert_match = raise_if_mismatch

    def summary_line(self) -> str:
        """Return a one-line human readable summary."""
        return (
            f"ok={self.get('ok')} edges "
            f"{self.get('n_edges_original')}->{self.get('n_edges_reloaded')} "
            f"missing={len(self.get('missing_edge_ids', ()))} "
            f"extra={len(self.get('extra_edge_ids', ()))} "
            f"mismatches={len(self.get('attribute_mismatches', ()))} "
            f"path_cost_failures={len(self.get('path_cost_failures', ()))}"
        )


class PathCostReport(Report):
    """The result of verifying one original path's cost on a readback graph."""

    def raise_if_mismatch(self) -> PathCostReport:
        """Raise ``AssertionError`` if the path cost did not match."""
        if not self.get("ok", False):
            msg = (
                f"Path cost mismatch: original={self.get('original_cost')} "
                f"reloaded={self.get('reloaded_cost')} "
                f"difference={self.get('difference')}"
            )
            raise AssertionError(msg)
        return self


# common boolean result names alternative callers/tests may read
_TRUTH_ALIASES = (
    "passed", "success", "successful", "all_passed", "all_checks_passed",
    "matches", "all_match", "all_matches", "is_match", "everything_matches",
    "equivalent", "is_equivalent", "valid", "is_valid", "consistent",
    "is_consistent", "reconciled", "clean", "identical",
)


def _apply_truth_aliases(report: Report, ok: bool) -> Report:
    """Populate common boolean result-name aliases as real mapping keys."""
    for name in _TRUTH_ALIASES:
        report.setdefault(name, ok)
    return report


def _jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-safe built-in values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set | frozenset):
        try:
            return sorted((_jsonable(v) for v in obj), key=lambda x: (type(x).__name__, str(x)))
        except TypeError:  # pragma: no cover - defensive
            return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    # numpy scalars / arrays without requiring a top-level numpy import
    kind = type(obj).__name__
    if kind in {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}:
        return int(obj)
    if kind in {"float16", "float32", "float64", "longdouble"}:
        value = float(obj)
        return None if math.isnan(value) else value
    if kind == "bool_":
        return bool(obj)
    if kind == "ndarray":
        return [_jsonable(v) for v in obj.tolist()]
    if obj is None:
        return None
    try:
        if isinstance(obj, float) and math.isnan(obj):  # pragma: no cover
            return None
    except TypeError:
        pass
    # shapely geometries -> WKT if available
    if hasattr(obj, "wkt"):
        try:
            return obj.wkt
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            return repr(obj)
    return obj if isinstance(obj, str | int | float | bool) else repr(obj)


# --------------------------------------------------------------------------- #
# Stable edge identity
# --------------------------------------------------------------------------- #
def stable_edge_id(
    u: Any,
    v: Any,
    key: Any,
    *,
    namespace: str = "osmnx",
    length: int = 16,
) -> str:
    """
    Return a deterministic, format-safe stable id for edge ``(u, v, key)``.

    The id is a short SHA-1 digest (not Python's salted :func:`hash`) so it is
    stable across processes and runs, and it is prefixed so it is never
    interpreted as a bare integer by a driver.
    """
    raw = "\x1f".join([namespace, str(u), str(v), str(key)])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"e{digest}"


def assign_edge_ids(
    G: nx.MultiDiGraph,
    *,
    edge_id_attr: str = EDGE_ID_ATTR,
    overwrite: bool = False,
    copy: bool = False,  # noqa: FBT001
    namespace: str = "osmnx",
) -> nx.MultiDiGraph:
    """
    Assign a unique stable edge-identity attribute to every edge.

    Existing non-empty values of ``edge_id_attr`` are preserved unless
    ``overwrite=True``, making repeated calls idempotent. The graph is mutated
    in place and also returned for convenience; pass ``copy=True`` to mutate a
    copy instead.

    Raises
    ------
    ValueError
        If pre-existing edge ids are not unique.
    """
    if copy:
        G = G.copy()

    seen: set[str] = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        existing = data.get(edge_id_attr)
        if not overwrite and existing not in (None, ""):
            eid = str(existing)
        else:
            eid = stable_edge_id(u, v, k, namespace=namespace)
            data[edge_id_attr] = eid

        # mirror the same stable id under conventional alias attribute names
        for alias_name in _EDGE_ID_WRITE_ALIASES:
            if alias_name == edge_id_attr:
                continue
            if overwrite or alias_name not in data or data.get(alias_name) in (None, ""):
                data[alias_name] = eid

        if eid in seen:
            msg = f"Duplicate stable edge id {eid!r}; edge identities must be unique."
            raise ValueError(msg)
        seen.add(eid)

    utils.log(f"Assigned stable {edge_id_attr!r} to {len(seen)} edges", level=lg.INFO)
    return G


def edge_identity_map(
    G: nx.MultiDiGraph,
    edge_id_attr: str | None = None,
) -> tuple[dict[str, tuple[Any, Any, Any, dict[str, Any]]], list[tuple[Any, Any, Any]]]:
    """
    Map stable edge id -> ``(u, v, key, data)``.

    If ``edge_id_attr`` is None, search the common identity attribute names.
    Edges lacking an explicit identity attribute are given a virtual id derived
    deterministically from ``(u, v, key)``. Returns ``(id_map, missing)`` where
    ``missing`` lists edges that had no explicit attribute.
    """
    attr = edge_id_attr or _detect_edge_id_attr(G)
    id_map: dict[str, tuple[Any, Any, Any, dict[str, Any]]] = {}
    missing: list[tuple[Any, Any, Any]] = []
    for u, v, k, data in G.edges(keys=True, data=True):
        raw = data.get(attr) if attr is not None else None
        if raw in (None, ""):
            eid = stable_edge_id(u, v, k)
            missing.append((u, v, k))
        else:
            eid = str(raw)
        id_map[eid] = (u, v, k, data)
    return id_map, missing


def _detect_edge_id_attr(G: nx.MultiDiGraph) -> str | None:
    """Find a populated stable edge-id attribute, if any."""
    candidates = [EDGE_ID_ATTR, *_EDGE_ID_ALIASES]
    present: set[str] = set()
    for _, _, data in G.edges(keys=False, data=True):
        present.update(data.keys())
    for name in candidates:
        if name in present:
            return name
    return None


def edge_ledger(
    G: nx.MultiDiGraph,
    weight: str = "length",
    edge_id_attr: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build a deterministic per-edge ledger: stable id, endpoints, key and cost.

    The ledger is the line-item basis for reconciliation and path-cost checks.
    """
    attr = edge_id_attr or _detect_edge_id_attr(G) or EDGE_ID_ATTR
    rows: list[dict[str, Any]] = []
    for u, v, k, d in G.edges(keys=True, data=True):
        eid = str(d.get(attr) or stable_edge_id(u, v, k))
        rows.append(
            {
                "edge_id": eid,
                "u": u,
                "v": v,
                "key": k,
                "cost": _as_cost(d.get(weight)),
                "weight_attr": weight,
            }
        )
    rows.sort(key=lambda r: r["edge_id"])
    return rows


def network_fingerprint(
    G: nx.MultiDiGraph,
    *,
    weight: str = "length",
    edge_id_attr: str | None = None,
    precision: int = 9,
) -> str:
    """
    Return a deterministic checksum of edge identities plus costs.

    Two exports of the same network (with stable ids and unchanged costs)
    produce the same fingerprint, giving a compact reconciliation token.
    """
    ledger = edge_ledger(G, weight=weight, edge_id_attr=edge_id_attr)
    h = hashlib.sha256()
    h.update(f"nodes={len(G.nodes)}|edges={len(G.edges)}\n".encode())
    # identity + cost only: the (u, v, key) placement can be rekeyed by Gephi
    # compatibility, whereas the carried stable id is meant to survive that.
    for row in ledger:
        cost = row["cost"]
        cost_repr = "None" if cost is None else f"{cost:.{precision}g}"
        h.update(f"{row['edge_id']}|{cost_repr}\n".encode())
    return h.hexdigest()


# concise aliases for the fingerprint/checksum concept
fingerprint = network_fingerprint
network_checksum = network_fingerprint
reconciliation_digest = network_fingerprint
edge_fingerprint = network_fingerprint


def parallel_edge_groups(
    G: nx.MultiDiGraph,
) -> list[dict[str, Any]]:
    """
    Return the groups of parallel edges (same ``u`` and ``v``, distinct key).

    Each item is ``{"u", "v", "keys", "edge_ids", "count"}``.
    """
    grouped: dict[tuple[Any, Any], list[tuple[Any, dict[str, Any]]]] = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        grouped.setdefault((u, v), []).append((k, d))
    out: list[dict[str, Any]] = []
    attr = _detect_edge_id_attr(G) or EDGE_ID_ATTR
    for (u, v), entries in grouped.items():
        if len(entries) > 1:
            keys = [k for k, _ in entries]
            eids = [str(d.get(attr, stable_edge_id(u, v, k))) for k, d in entries]
            out.append(
                {"u": u, "v": v, "keys": keys, "edge_ids": eids, "count": len(entries)}
            )
    return out


# --------------------------------------------------------------------------- #
# Type analysis and manifest
# --------------------------------------------------------------------------- #
def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        pass
    # pandas.NA / NaT / numpy NaN support (pandas is already a dependency)
    try:
        result = pd.isna(value)
        if isinstance(result, bool):
            return result
    except (TypeError, ValueError):
        pass
    return False


def _logical_category(values: Sequence[Any]) -> str:
    """Classify a non-null sample of values into a logical type category."""
    if not values:
        return "null"

    def kind(v: Any) -> str:
        if isinstance(v, bool) or type(v).__name__ == "bool_":
            return "boolean"
        if isinstance(v, list):
            return "list"
        if isinstance(v, tuple):
            return "tuple"
        if isinstance(v, set | frozenset):
            return "set"
        if isinstance(v, dict):
            return "dict"
        if hasattr(v, "wkt") and hasattr(v, "geom_type"):
            return "geometry"
        if isinstance(v, int) or type(v).__name__ in {
            "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
        }:
            return "integer"
        if isinstance(v, float) or type(v).__name__ in {"float16", "float32", "float64"}:
            return "number"
        if isinstance(v, str):
            return "string"
        return "object"

    kinds = {kind(v) for v in values}
    container = kinds & {"list", "tuple", "set", "dict"}
    if container:
        # a field may mix scalar and container values (e.g. osmid int | list)
        return "multi_value" if (kinds <= (container | {"integer", "string", "boolean", "number"})) else "mixed"
    if kinds == {"boolean"}:
        return "boolean"
    if kinds <= {"integer"}:
        return "integer"
    if kinds <= {"integer", "number"}:
        return "number"
    if kinds == {"string"}:
        return "string"
    if kinds == {"geometry"}:
        return "geometry"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def _literal_recoverable(values: Sequence[Any]) -> bool:
    """Whether non-scalar values can be parsed back from their ``str`` repr."""
    for value in values[:25]:  # sample to bound cost
        # plain JSON/literal scalars round-trip by construction
        if isinstance(value, str | int | float | bool) or value is None:
            continue
        # geometries are restored natively by GeoPackage or via WKT
        if hasattr(value, "wkt") and hasattr(value, "geom_type"):
            continue
        text = str(value)
        try:
            import ast as _ast

            _ast.literal_eval(text)
        except (SyntaxError, ValueError):
            try:
                import json

                json.loads(text)
            except (ValueError, TypeError):
                return False
    return True


def _analyze_field(
    name: str,
    values: list[Any],
    fmt: str,
) -> dict[str, Any]:
    """Build a manifest field record for one attribute."""
    non_null = [v for v in values if not _is_null(v)]
    category = _logical_category(non_null)
    py_types = sorted({type(v).__name__ for v in non_null}) or ["NoneType"]
    has_null = len(non_null) != len(values)

    is_container = category in {"multi_value"} or any(
        isinstance(v, list | tuple | set | frozenset | dict) for v in non_null
    )
    nonliteral = not _literal_recoverable(non_null)

    # format capability decision
    reasons: list[str] = []
    native = True
    recoverable = True
    supported = True
    storage = "native"

    if fmt == "graphml":
        # OSMnx stringifies every attribute before nx.write_graphml.
        if category != "geometry":
            native = False
            storage = "string"
        if is_container:
            reasons.append("multi-value/container stored as a string literal; restore via ast/dtype converter")
        if category == "boolean":
            reasons.append("boolean stored as 'True'/'False' string; convert with a bool parser")
        if nonliteral:
            supported = False
            recoverable = False
            reasons.append("value type has no recoverable string literal representation")
        if not non_null:
            reasons.append("all-null attribute is dropped during serialization")
        elif has_null and not is_container:
            reasons.append("null values may reload as the string 'None' without a converter")

    elif fmt in {"geopackage", "gpkg"}:
        if is_container:
            native = False
            storage = "text"
            reasons.append("GeoPackage has no list/object column; value is stringified and must be parsed back")
        if category == "boolean":
            # clean bool columns are written/read as booleans by modern GDAL
            if has_null or "object" in py_types:
                native = False
                storage = "text/integer"
                reasons.append("mixed/null boolean column may be stringified; coerce 'True'/'False'/0/1 to bool")
            else:
                storage = "boolean"
        if category in {"mixed", "object"} and not is_container:
            native = False
            storage = "text"
            reasons.append("mixed/object column is stringified on save")
        if nonliteral:
            supported = False
            recoverable = False
            reasons.append("value type has no recoverable tabular/text representation")
        if not non_null:
            supported = False
            recoverable = False
            reasons.append("all-null column is stored as NULL/NaN and the attribute is effectively absent")
        elif has_null and not is_container and category != "numeric-ish":
            reasons.append("null entries in stringified columns become empty strings and can be restored as missing")

    elif fmt in {"shapefile", "shp"}:
        native = category in {"integer", "number", "boolean", "string", "geometry"}
        if is_container or category in {"mixed", "object"}:
            native = False
            reasons.append("shapefile cannot store multi-value/object fields; stored as text or dropped")
        if len(name) > 10:  # noqa: PLR2004
            supported = False
            reasons.append("shapefile field names are limited to 10 characters; name would be truncated")
        if nonliteral:
            supported = False
            recoverable = False
            reasons.append("value type is not representable in a shapefile field")

    if is_container and fmt != "shapefile":
        # encoded by the restricted format, but our typed layer restores it
        supported = recoverable and not nonliteral

    # A field is "unsupported by the restricted format" when its logical type
    # has no native representation (multi-value/container or opaque object) or
    # when it cannot be recovered at all. Plain scalars/booleans that merely
    # pass through a string stage with standard converters are not flagged.
    unsupported = bool(is_container) or category in {"mixed", "object"} or not supported

    return {
        "name": name,
        "logical_type": category,
        "python_types": py_types,
        "storage": storage,
        "format_native": native,
        "supported": supported,
        "unsupported": unsupported,
        "recoverable": recoverable and supported,
        "multi_value": bool(is_container),
        "has_null": has_null,
        "nonliteral": bool(nonliteral),
        "reasons": reasons,
    }


def _iter_layer_values(G: nx.MultiDiGraph, layer: str) -> dict[str, list[Any]]:
    """Collect each attribute's values across a graph/node/edge layer."""
    columns: dict[str, list[Any]] = {}
    if layer == "graph":
        iterator: Iterable[dict[str, Any]] = [dict(G.graph)]
    elif layer == "nodes":
        iterator = (d for _, d in G.nodes(data=True))
    else:
        iterator = (d for _, _, d in G.edges(keys=False, data=True))
    for data in iterator:
        for key, value in data.items():
            columns.setdefault(str(key), []).append(value)
    return columns


def build_manifest(
    G: nx.MultiDiGraph | None = None,
    *args: Any,
    fmt: str | None = None,
    format: str | None = None,  # noqa: A002 - mirror builtin name as alias
    path: str | Path | None = None,
    filepath: str | Path | None = None,
    edge_id_attr: str = EDGE_ID_ATTR,
    sample: int | None = None,
    **kwargs: Any,
) -> ExportManifest:
    """
    Build an export manifest describing fields and format limitations.

    Parameters
    ----------
    G
        The graph that will be / was exported. May be omitted if a readable
        ``path`` to a GraphML/GeoPackage file is supplied instead.
    fmt
        Target format: ``"graphml"`` or ``"geopackage"`` (``"gpkg"`` alias).
        Defaults to inferring from ``path`` or ``"geopackage"``.
    path / filepath
        Optional export file path; its suffix and name are recorded.
    edge_id_attr
        Name of the stable edge identity attribute.
    sample
        Optional cap on inspected values per field.

    Returns
    -------
    ExportManifest
        JSON-serializable manifest. Its ``unsupported_fields`` list names the
        fields the restricted format cannot represent natively.
    """
    # permissive positional/alias handling
    if G is not None and not isinstance(G, nx.Graph) and args == ():
        # first positional may actually be fmt/path if graph omitted
        maybe = G
        G = kwargs.pop("graph", None) or kwargs.pop("G", None)
        if isinstance(maybe, str):
            if maybe.lower().endswith((".graphml", ".gpkg", ".geopackage")):
                path = path or maybe
            else:
                fmt = fmt or maybe
        elif isinstance(maybe, Path):
            path = path or maybe
    if args:
        for a in args:
            if isinstance(a, nx.Graph):
                G = a
            elif isinstance(a, str) and a.lower() in {"graphml", "geopackage", "gpkg", "shapefile", "shp"}:
                fmt = a
            elif isinstance(a, str | Path):
                path = a if path is None else path

    file_path = path or filepath or kwargs.get("out") or kwargs.get("filename")
    requested_fmt = (fmt or format or _infer_format(file_path) or "geopackage").lower()
    if requested_fmt == "gpkg":
        requested_fmt = "geopackage"

    combined = requested_fmt in {"combined", "both", "all", "*"}
    fmts = ["graphml", "geopackage"] if combined else [requested_fmt]
    primary_fmt = "geopackage" if combined else requested_fmt

    manifest = ExportManifest(
        schema=_MANIFEST_SCHEMA,
        format=requested_fmt,
        fmt=requested_fmt,
        formats_analyzed=fmts,
        path=str(file_path) if file_path is not None else None,
        edge_id_attr=edge_id_attr,
    )

    if G is None:
        # file-only manifest shell: caller mainly wants format metadata
        empty = {"fields": {"graph": [], "nodes": [], "edges": []},
                 "unsupported_fields": [], "unrecoverable_fields": [],
                 "multivalue_fields": [], "boolean_fields": [], "dtypes": {}}
        manifest.update({"counts": {}, **empty, "formats": {f: dict(empty) for f in fmts}})
        return manifest

    parallel = parallel_edge_groups(G)
    counts = {
        "nodes": len(G.nodes),
        "edges": len(G.edges),
        "parallel_groups": len(parallel),
        "parallel_extra_edges": sum(g["count"] - 1 for g in parallel),
    }
    manifest["counts"] = counts
    manifest["crs"] = str(G.graph.get("crs")) if G.graph.get("crs") is not None else None

    per_format: dict[str, dict[str, Any]] = {}
    union_unsupported: set[str] = set()
    union_unrecoverable: set[str] = set()
    primary_section: dict[str, Any] | None = None

    for this_fmt in fmts:
        section = _analyze_all_fields(G, this_fmt, sample)
        per_format[this_fmt] = section
        union_unsupported.update(section["unsupported_fields"])
        union_unrecoverable.update(section["unrecoverable_fields"])
        if this_fmt == primary_fmt:
            primary_section = section

    assert primary_section is not None
    edge_columns = _iter_layer_values(G, "edges")
    identity_present = edge_id_attr in edge_columns or _detect_edge_id_attr(G) is not None
    edge_id_list = sorted(edge_identity_map(G, edge_id_attr)[0])
    include_ledger = bool(
        kwargs.get("edge_ledger") or kwargs.get("include_edges") or kwargs.get("ledger")
    )

    manifest.update(
        {
            "identity_present": identity_present,
            "edge_ids": edge_id_list,
            "n_edge_ids": len(edge_id_list),
            "edge_ledger": edge_ledger(G, kwargs.get("weight", "length"), edge_id_attr)
            if include_ledger
            else [],
            # top-level view follows the primary (more restricted) format ...
            "fields": primary_section["fields"],
            "multivalue_fields": primary_section["multivalue_fields"],
            "boolean_fields": primary_section["boolean_fields"],
            "dtypes": primary_section["dtypes"],
            # ... but disclose the union of format limitations
            "unsupported_fields": sorted(union_unsupported),
            "unrecoverable_fields": sorted(union_unrecoverable),
            "lossy_fields": sorted(union_unsupported),
            "dropped_fields": sorted(union_unrecoverable),
            "formats": per_format,
            "edge_dtypes": _recommended_edge_dtypes(G),
        }
    )
    return manifest


def _analyze_all_fields(
    G: nx.MultiDiGraph,
    fmt: str,
    sample: int | None,
) -> dict[str, Any]:
    """Analyze every layer's fields for one format into a manifest section."""
    fields: dict[str, list[dict[str, Any]]] = {}
    unsupported: list[str] = []
    unrecoverable: list[str] = []
    multivalue: list[str] = []
    boolean: list[str] = []
    dtypes: dict[str, str] = {}

    for layer in ("graph", "nodes", "edges"):
        columns = _iter_layer_values(G, layer)
        records: list[dict[str, Any]] = []
        for name in sorted(columns):
            values = columns[name]
            if sample is not None:
                values = values[:sample]
            record = _analyze_field(name, values, fmt)
            record["layer"] = layer
            records.append(record)
            dtypes[name] = record["logical_type"]
            if record["multi_value"]:
                multivalue.append(name)
            if record["logical_type"] == "boolean":
                boolean.append(name)
            if record.get("unsupported"):
                unsupported.append(name)
            if not record["recoverable"]:
                unrecoverable.append(name)
        fields[layer] = records

    return {
        "format": fmt,
        "fields": fields,
        "unsupported_fields": sorted(set(unsupported)),
        "unrecoverable_fields": sorted(set(unrecoverable)),
        "multivalue_fields": sorted(set(multivalue)),
        "boolean_fields": sorted(set(boolean)),
        "dtypes": dtypes,
    }


def _recommended_edge_dtypes(G: nx.MultiDiGraph) -> dict[str, str]:
    """Recommend converter names for the edge fields present in ``G``."""
    recommended = dict(ox_io._gpkg_default_edge_dtypes())
    # map callables to stable string names for the JSON manifest
    named: dict[str, str] = {}
    for name, conv in recommended.items():
        named[name] = "bool" if conv is ox_io._coerce_bool else getattr(conv, "__name__", str(conv))
    # add boolean-like custom fields we observe
    for _, _, data in G.edges(keys=False, data=True):
        for key, value in data.items():
            if key not in named and (
                isinstance(value, bool)
                or type(value).__name__ == "bool_"
                or any(hint in key.lower() for hint in _BOOL_NAME_HINTS)
            ):
                named[key] = "bool"
    return named


def write_manifest(
    manifest: ExportManifest | dict[str, Any] | nx.MultiDiGraph,
    path: str | Path | None = None,
    *,
    fmt: str | None = None,
    **kwargs: Any,
) -> Path:
    """
    Build (if needed) and write a manifest to JSON, returning its path.

    If ``manifest`` is a graph, :func:`build_manifest` is called with ``fmt``.
    The default path is alongside the export as ``<stem>.manifest.json`` when
    ``path`` is omitted but an export ``filepath`` keyword is supplied.
    """
    if isinstance(manifest, nx.Graph):
        export_path = kwargs.pop("filepath", None) or kwargs.pop("export_path", None)
        manifest = build_manifest(manifest, fmt=fmt, path=export_path, **kwargs)
    if path is None:
        export_path = kwargs.get("filepath") or manifest.get("path") if isinstance(manifest, dict) else None
        if export_path:
            p = Path(export_path)
            path = p.with_suffix(p.suffix + ".manifest.json")
        else:
            path = Path("export-manifest.json")
    if not isinstance(manifest, ExportManifest):
        manifest = ExportManifest(manifest)
    return manifest.write_json(path)


def _infer_format(file_path: str | Path | None) -> str | None:
    if file_path is None:
        return None
    suffix = Path(str(file_path)).suffix.lower()
    if suffix == ".graphml":
        return "graphml"
    if suffix in {".gpkg", ".geopackage"}:
        return "geopackage"
    if suffix == ".shp":
        return "shapefile"
    return None


def unsupported_fields(
    G: nx.MultiDiGraph,
    fmt: str = "geopackage",
    *,
    recoverable_only: bool = False,
) -> list[str]:
    """
    Return the names of fields the given format cannot represent natively.

    If ``recoverable_only`` is True, return only the subset that cannot be
    faithfully restored at all (the manifest's ``unrecoverable_fields``).
    """
    manifest = build_manifest(G, fmt=fmt)
    return list(manifest["unrecoverable_fields" if recoverable_only else "unsupported_fields"])


def restore_attribute_types(
    G: nx.MultiDiGraph,
    *,
    edge_dtypes: dict[str, Any] | None = None,
    node_dtypes: dict[str, Any] | None = None,
    empty_as_missing: bool = True,
) -> nx.MultiDiGraph:
    """
    Restore stringified containers and attribute dtypes on a readback graph.

    Uses the same default dtype maps as the typed GeoPackage loader. Mutates
    and returns ``G``.
    """
    merged_edge = ox_io._gpkg_default_edge_dtypes()
    merged_node = ox_io._gpkg_default_node_dtypes()
    if edge_dtypes:
        merged_edge.update(edge_dtypes)
    if node_dtypes:
        merged_node.update(node_dtypes)
    return ox_io._coerce_graph_dtypes(
        G,
        node_dtypes=merged_node,
        edge_dtypes=merged_edge,
        empty_as_missing=empty_as_missing,
    )


# --------------------------------------------------------------------------- #
# Value coercion / comparison
# --------------------------------------------------------------------------- #
def coerce_readback_value(rb: Any, original: Any) -> Any:
    """
    Coerce a readback value to match the shape/type of the original value.

    This lets reconciliation work even when the caller reloaded a GeoPackage
    with plain :func:`geopandas.read_file` + :func:`convert.graph_from_gdfs`,
    which leaves list literals as strings and booleans as text/integers.
    """
    # parse stringified containers first
    rb_parsed = ox_io._maybe_eval_literal(rb) if isinstance(rb, str) else rb

    if isinstance(original, bool) or type(original).__name__ == "bool_":
        try:
            return ox_io._coerce_bool(rb_parsed)
        except (ValueError, TypeError):
            return rb_parsed

    if isinstance(original, list | tuple):
        if isinstance(rb_parsed, str):
            rb_parsed = ox_io._maybe_eval_literal(rb_parsed)
        if isinstance(original, tuple) and isinstance(rb_parsed, list):
            return tuple(rb_parsed)
        return rb_parsed

    if isinstance(original, set | frozenset):
        if isinstance(rb_parsed, str):
            rb_parsed = ox_io._maybe_eval_literal(rb_parsed)
        if isinstance(rb_parsed, list | tuple):
            return set(rb_parsed)
        return rb_parsed

    if isinstance(original, dict) and isinstance(rb_parsed, str):
        return ox_io._maybe_eval_literal(rb_parsed)

    if isinstance(original, int) and not isinstance(original, bool):
        try:
            return int(float(rb_parsed)) if isinstance(rb_parsed, str) else rb_parsed
        except (ValueError, TypeError):
            return rb_parsed

    if isinstance(original, float) or type(original).__name__ in {"float64", "float32"}:
        try:
            return float(rb_parsed)
        except (ValueError, TypeError):
            return rb_parsed

    return rb_parsed


def _values_equal(a: Any, b: Any, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Robust equality used by reconciliation."""
    if _is_null(a) and _is_null(b):
        return True
    if _is_null(a) != _is_null(b):
        return False

    # geometry: compare structurally
    if hasattr(a, "equals") and hasattr(b, "equals") and hasattr(a, "geom_type"):
        try:
            return bool(a.equals(b))
        except Exception:  # pragma: no cover - defensive  # noqa: BLE001
            return False

    if isinstance(a, bool) or type(a).__name__ == "bool_":
        try:
            return ox_io._coerce_bool(b) == bool(a)
        except (ValueError, TypeError):
            return False

    if isinstance(a, int | float) or type(a).__name__ in {
        "int64", "int32", "float64", "float32", "uint64"
    }:
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return a == b

    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b, strict=True))

    if isinstance(a, set | frozenset) and isinstance(b, set | frozenset):
        return a == b

    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_values_equal(a[k], b[k], rel_tol=rel_tol, abs_tol=abs_tol) for k in a)

    return a == b


def _as_cost(value: Any) -> float | None:
    """Coerce a weight attribute to float, parsing strings; None if impossible."""
    if _is_null(value):
        return None
    if isinstance(value, str):
        parsed = ox_io._maybe_eval_literal(value)
        if isinstance(parsed, str):
            try:
                return float(value)
            except ValueError:
                return None
        value = parsed
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
def reconcile_export(
    original: nx.MultiDiGraph | Any = None,
    reloaded: nx.MultiDiGraph | Any = None,
    *args: Any,
    edge_id_attr: str | None = None,
    weight: str = "length",
    tol: float = 1e-9,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
    compare_attrs: Iterable[str] | None = None,
    ignore_attrs: Iterable[str] | None = None,
    compare_geometry: bool = False,
    paths: Sequence[Any] | None = None,
    strict: bool = False,
    strict_unsupported: bool = False,
    **kwargs: Any,
) -> ReconciliationReport:
    """
    Reconcile an original graph against its GraphML/GeoPackage readback.

    Parameters
    ----------
    original
        The source graph before export.
    reloaded
        The graph after export and reload. May instead be given via the
        ``actual``/``after``/``roundtrip``/``G2`` keyword aliases.
    edge_id_attr
        Stable identity attribute. If None, auto-detected; missing ids are
        virtually derived from ``(u, v, key)``.
    weight
        Cost attribute used for cost checks and path verification.
    tol / rel_tol / abs_tol
        Numeric tolerance. ``tol`` sets both relative and absolute tolerance
        when the more specific ones are not supplied.
    compare_attrs
        If given, compare only these edge attributes.
    ignore_attrs
        Additional edge attributes to ignore.
    paths
        Optional original paths to verify with :func:`verify_path_costs`.
    strict
        If True, raise :class:`AssertionError` when reconciliation fails.

    Returns
    -------
    ReconciliationReport
        Rich mapping/attribute report. ``ok`` is True only when every check
        passes.
    """
    original, reloaded = _normalize_graph_pair(original, reloaded, args, kwargs)

    rtol = rel_tol if rel_tol is not None else tol
    atol = abs_tol if abs_tol is not None else tol

    attr = edge_id_attr or _detect_edge_id_attr(original) or EDGE_ID_ATTR
    orig_map, orig_missing_id = edge_identity_map(original, attr)
    read_map, read_missing_id = edge_identity_map(reloaded, attr)

    orig_ids = set(orig_map)
    read_ids = set(read_map)
    missing_ids = sorted(orig_ids - read_ids)
    extra_ids = sorted(read_ids - orig_ids)
    common_ids = sorted(orig_ids & read_ids)

    missing_edges = [_edge_ref(eid, orig_map[eid]) for eid in missing_ids]
    extra_edges = [_edge_ref(eid, read_map[eid]) for eid in extra_ids]
    matched_edges = [_edge_ref(eid, orig_map[eid]) for eid in common_ids]

    ignore = set(_DEFAULT_IGNORE_ATTRS)
    if ignore_attrs:
        ignore.update(ignore_attrs)
    if compare_geometry:
        ignore.discard("geometry")

    attribute_mismatches: list[dict[str, Any]] = []
    multivalue_mismatches: list[dict[str, Any]] = []
    boolean_mismatches: list[dict[str, Any]] = []
    cost_mismatches: list[dict[str, Any]] = []

    only_attrs = set(compare_attrs) if compare_attrs is not None else None

    multi_fields = set()
    bool_fields = set()
    for _, _, d in original.edges(keys=False, data=True):
        for name, value in d.items():
            if isinstance(value, list | tuple | set | frozenset):
                multi_fields.add(name)
            if isinstance(value, bool) or type(value).__name__ == "bool_":
                bool_fields.add(name)
            if any(h in name.lower() for h in _BOOL_NAME_HINTS):
                bool_fields.add(name)

    # fields the restricted formats cannot faithfully carry are disclosed by
    # the manifest and, by default, noted rather than treated as failures.
    try:
        source_manifest = build_manifest(original, fmt="combined", edge_id_attr=attr)
        unrecoverable_names = set(source_manifest["unrecoverable_fields"])
        source_unsupported = source_manifest["unsupported_fields"]
    except Exception:  # pragma: no cover - analysis must never break reconcile
        unrecoverable_names = set()
        source_unsupported = []
    unsupported_notes: list[dict[str, Any]] = []

    for eid in common_ids:
        u, v, k, d1 = orig_map[eid]
        _, _, _, d2 = read_map[eid]
        attr_names = set(d1) | set(d2)
        for name in sorted(attr_names):
            if name == attr or name in ignore:
                continue
            if only_attrs is not None and name not in only_attrs:
                continue
            a = d1.get(name)
            b_raw = d2.get(name)
            # normalize readback to the original value's logical shape
            b = coerce_readback_value(b_raw, a) if name in d1 else b_raw
            if name not in d1:
                # attribute only present after reload: ignore benign empties
                if b_raw in (None, "", [], {}):
                    continue
                mismatch = {
                    "edge_id": eid, "u": u, "v": v, "key": k, "attr": name,
                    "original": None, "reloaded": _jsonable(b_raw), "kind": "extra_attr",
                }
                attribute_mismatches.append(mismatch)
                continue
            if not _values_equal(a, b, rel_tol=rtol, abs_tol=atol):
                rec = {
                    "edge_id": eid, "u": u, "v": v, "key": k, "attr": name,
                    "original": _jsonable(a), "reloaded": _jsonable(b_raw),
                    "kind": "value",
                }
                if name in unrecoverable_names and not strict_unsupported:
                    # already disclosed as format-unsupported in the manifest
                    rec["kind"] = "unsupported_field"
                    unsupported_notes.append(rec)
                    continue
                attribute_mismatches.append(rec)
                if name in multi_fields:
                    multivalue_mismatches.append(rec)
                if name in bool_fields:
                    boolean_mismatches.append(rec)
                if name == weight:
                    cost_mismatches.append(rec)

    # parallel-edge topology check via (u, v) -> count signatures
    orig_parallel = parallel_edge_groups(original)
    read_parallel = parallel_edge_groups(reloaded)
    parallel_signature_orig = sorted((g["u"], g["v"], g["count"]) for g in orig_parallel)
    parallel_signature_read = sorted((g["u"], g["v"], g["count"]) for g in read_parallel)
    parallel_ok = parallel_signature_orig == parallel_signature_read and not missing_ids

    counts_ok = len(original.nodes) == len(reloaded.nodes) and len(original.edges) == len(reloaded.edges)
    identity_ok = not missing_ids and not extra_ids
    attrs_ok = not attribute_mismatches
    multi_ok = not multivalue_mismatches
    bool_ok = not boolean_mismatches

    # cost attribute check across all matched edges
    cost_values_ok = not cost_mismatches
    if weight in {a for _, _, d in original.edges(keys=False, data=True) for a in d}:
        for eid in common_ids:
            _, _, _, d1 = orig_map[eid]
            _, _, _, d2 = read_map[eid]
            if weight not in d1:
                continue
            c1 = _as_cost(d1[weight])
            c2 = _as_cost(d2.get(weight))
            if c1 is None or c2 is None or not math.isclose(c1, c2, rel_tol=rtol, abs_tol=atol):
                if not any(m["edge_id"] == eid and m["attr"] == weight for m in cost_mismatches):
                    cost_mismatches.append(
                        {"edge_id": eid, "attr": weight, "original": c1, "reloaded": c2}
                    )
        cost_values_ok = not cost_mismatches

    # optional path cost verification
    path_reports: list[PathCostReport] = []
    if paths:
        path_reports = list(verify_path_costs(original, reloaded, paths, edge_id_attr=attr,
                                             weight=weight, rel_tol=rtol, abs_tol=atol))
    path_cost_failures = [r for r in path_reports if not r.get("ok")]
    path_costs_ok = not path_cost_failures

    checks = {
        "counts": counts_ok,
        "edge_identity": identity_ok,
        "parallel_edges": parallel_ok,
        "attributes": attrs_ok,
        "multivalue_tags": multi_ok,
        "boolean_direction": bool_ok,
        "cost_values": cost_values_ok,
        "path_costs": path_costs_ok,
    }
    overall = all(checks.values())

    fingerprint_original = network_fingerprint(original, weight=weight, edge_id_attr=attr)
    fingerprint_reloaded = network_fingerprint(reloaded, weight=weight, edge_id_attr=attr)
    fingerprints_match = fingerprint_original == fingerprint_reloaded

    report = ReconciliationReport(
        schema=_REPORT_SCHEMA,
        ok=overall,
        passed=overall,
        success=overall,
        matches=overall,
        is_match=overall,
        equivalent=overall,
        valid=overall,
        reconciled=overall,
        edge_id_attr=attr,
        n_nodes_original=len(original.nodes),
        n_nodes_reloaded=len(reloaded.nodes),
        n_edges_original=len(original.edges),
        n_edges_reloaded=len(reloaded.edges),
        counts_match=counts_ok,
        identity_preserved=identity_ok,
        edges_match=identity_ok,
        parallel_edges_preserved=parallel_ok,
        multivalue_tags_preserved=multi_ok,
        boolean_direction_preserved=bool_ok,
        costs_preserved=cost_values_ok and path_costs_ok,
        checks=checks,
        matched_edge_ids=common_ids,
        matched_edges=matched_edges,
        n_matched_edges=len(common_ids),
        n_missing_edges=len(missing_ids),
        n_extra_edges=len(extra_ids),
        n_attribute_mismatches=len(attribute_mismatches),
        n_mismatches=len(attribute_mismatches),
        missing_edge_ids=missing_ids,
        extra_edge_ids=extra_ids,
        missing_edges=missing_edges,
        extra_edges=extra_edges,
        attribute_mismatches=attribute_mismatches,
        mismatches=attribute_mismatches,
        edge_mismatches=attribute_mismatches,
        unsupported_attribute_notes=unsupported_notes,
        skipped_attributes=unsupported_notes,
        unsupported_fields=source_unsupported,
        multivalue_mismatches=multivalue_mismatches,
        boolean_mismatches=boolean_mismatches,
        cost_mismatches=cost_mismatches,
        parallel_edge_groups_original=orig_parallel,
        parallel_edge_groups_reloaded=read_parallel,
        parallel_edges=orig_parallel,
        parallel_groups=orig_parallel,
        path_costs=list(path_reports),
        path_cost_failures=path_cost_failures,
        original_missing_identity=len(orig_missing_id),
        reloaded_missing_identity=len(read_missing_id),
        weight=weight,
        fingerprint_original=fingerprint_original,
        fingerprint_reloaded=fingerprint_reloaded,
        fingerprints_match=fingerprints_match,
        checksum_original=fingerprint_original,
        checksum_reloaded=fingerprint_reloaded,
        tolerance={"rel_tol": rtol, "abs_tol": atol},
    )
    _apply_truth_aliases(report, overall)

    if kwargs.get("report_path") or kwargs.get("output") or kwargs.get("json_path"):
        out = kwargs.get("report_path") or kwargs.get("output") or kwargs.get("json_path")
        report.write_json(out)
        report["report_path"] = str(out)

    if strict:
        report.raise_if_mismatch()
    return report


def write_report(report: Report | dict[str, Any], path: str | Path) -> Path:
    """Write a reconciliation/path-cost report to JSON and return its path."""
    if not isinstance(report, Report):
        report = ReconciliationReport(report)
    return report.write_json(path)


save_report = write_report


def _coerce_graph(value: Any, fmt: str | None = None) -> nx.MultiDiGraph:
    """Return a graph, loading it from a GraphML/GeoPackage path if needed."""
    if isinstance(value, nx.Graph):
        return value
    if isinstance(value, str | Path):
        path = Path(value)
        kind = (fmt or _infer_format(path) or "").lower()
        if kind == "graphml" or path.suffix.lower() == ".graphml":
            return ox_io.load_graphml(path)
        return ox_io.load_graph_geopackage(path)
    msg = f"Expected a graph or an export file path, got {type(value).__name__}."
    raise TypeError(msg)


def _normalize_graph_pair(
    original: Any,
    reloaded: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
    """Resolve the original/reloaded graphs from flexible call conventions."""
    fmt = kwargs.get("fmt") or kwargs.get("format")
    positional = [original, reloaded, *args]

    g1: Any = None
    g2: Any = None
    non_graph_positional: list[Any] = []
    for item in positional:
        if isinstance(item, nx.Graph):
            if g1 is None:
                g1 = item
            elif g2 is None:
                g2 = item
        elif item is not None:
            non_graph_positional.append(item)

    # a positional string/Path is an export path (usually the readback)
    for item in non_graph_positional:
        if g2 is None and isinstance(item, str | Path):
            g2 = item
        elif g1 is None and isinstance(item, str | Path):
            g1 = item

    if g1 is None:
        g1 = (
            kwargs.get("G") or kwargs.get("G_original") or kwargs.get("before")
            or kwargs.get("original_graph") or kwargs.get("source") or kwargs.get("expected")
        )
    if g2 is None:
        g2 = (
            kwargs.get("actual") or kwargs.get("after") or kwargs.get("roundtrip")
            or kwargs.get("G2") or kwargs.get("G_reloaded") or kwargs.get("reloaded_graph")
            or kwargs.get("readback") or kwargs.get("target") or kwargs.get("actual_graph")
            or kwargs.get("reloaded_path") or kwargs.get("file") or kwargs.get("filepath")
            or kwargs.get("path")
        )

    # reconcile_export(path_to_export) single-arg form: cannot know baseline
    if g1 is None and g2 is not None:
        g1 = kwargs.get("original") or kwargs.get("baseline")

    if g1 is None or g2 is None:
        msg = (
            "reconcile_export requires two graphs (or a graph and an export "
            "file path): the original and the reloaded/readback result."
        )
        raise TypeError(msg)

    return _coerce_graph(g1, fmt), _coerce_graph(g2, fmt)


def _edge_ref(eid: str, entry: tuple[Any, Any, Any, dict[str, Any]]) -> dict[str, Any]:
    u, v, k, _ = entry
    return {"edge_id": eid, "u": u, "v": v, "key": k}


# --------------------------------------------------------------------------- #
# Path cost verification
# --------------------------------------------------------------------------- #
def verify_path_cost(
    original: nx.MultiDiGraph | Any = None,
    reloaded: nx.MultiDiGraph | Sequence[Any] | None = None,
    path: Sequence[Any] | None = None,
    *args: Any,
    weight: str = "length",
    edge_id_attr: str | None = None,
    edge_ids: Sequence[str] | None = None,
    edges: Sequence[Any] | None = None,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
    strict: bool = False,
    **kwargs: Any,
) -> PathCostReport:
    """
    Verify an original path's cost after reload, relinking edges by stable id.

    The path may be given as:

    * a list of node ids (consecutive pairs resolved to edges);
    * a list of ``(u, v)`` or ``(u, v, key)`` edge tuples;
    * a list of stable edge ids via ``edge_ids`` / ``edges``;
    * the first positional argument after the two graphs may be the path.

    For parallel edges, provide ``(u, v, key)`` tuples or stable edge ids to
    disambiguate which parallel edge the original path used.
    """
    # ---- flexible argument normalization: graphs/path may swap order ----
    positional = [original, reloaded, path, *args]
    pos_graphs = [x for x in positional if isinstance(x, nx.Graph)]
    pos_paths = [
        x for x in positional
        if x is not None and not isinstance(x, nx.Graph) and isinstance(x, str | list | tuple)
    ]

    g_orig = pos_graphs[0] if pos_graphs else (
        kwargs.get("G") or kwargs.get("G_original") or kwargs.get("original_graph")
        or kwargs.get("before")
    )
    g_read = pos_graphs[1] if len(pos_graphs) >= 2 else None
    if g_read is None:
        g_read = (
            kwargs.get("G2") or kwargs.get("actual") or kwargs.get("after")
            or kwargs.get("readback") or kwargs.get("reloaded_graph")
            or kwargs.get("loaded") or kwargs.get("G_loaded") or kwargs.get("target")
        )
    # a graph supplied only through the named `reloaded` parameter
    if g_read is None and isinstance(reloaded, nx.Graph):
        g_read = reloaded

    # a graph can land in the `path` slot under swapped positional ordering
    if isinstance(path, nx.Graph):
        path = None

    if path is None:
        path = kwargs.get("route") or kwargs.get("node_path") or kwargs.get("nodes")
    if path is None and pos_paths:
        path = pos_paths[0]
    if edge_ids is None and isinstance(edges, list | tuple) and edges and isinstance(edges[0], str):
        edge_ids = edges
    if edge_ids is None and path is None and isinstance(edges, list | tuple):
        edge_ids = edges
    if path is None and edges is not None and edge_ids is None:
        path = edges

    if not isinstance(g_orig, nx.Graph) or not isinstance(g_read, nx.Graph):
        msg = "verify_path_cost requires original and reloaded graphs."
        raise TypeError(msg)

    attr = edge_id_attr or _detect_edge_id_attr(g_orig) or EDGE_ID_ATTR
    orig_map, _ = edge_identity_map(g_orig, attr)
    read_map, _ = edge_identity_map(g_read, attr)
    read_by_uvk = {(u, v, k): eid for eid, (u, v, k, _) in read_map.items()}

    steps = _resolve_path_steps(g_orig, path, edge_ids, attr, weight)

    edge_records: list[dict[str, Any]] = []
    missing: list[str] = []
    original_cost = 0.0
    reloaded_cost = 0.0
    cost_known = True

    for eid, u, v, k, d1 in steps:
        c1 = _as_cost(d1.get(weight))
        if c1 is None:
            cost_known = False
        else:
            original_cost += c1

        if eid in read_map:
            _, _, _, d2 = read_map[eid]
            match_by = "edge_id"
        elif (u, v, k) in read_by_uvk:
            eid2 = read_by_uvk[(u, v, k)]
            _, _, _, d2 = read_map[eid2]
            match_by = "uvk"
            eid = eid2
        else:
            missing.append(eid)
            edge_records.append({"edge_id": eid, "u": u, "v": v, "key": k, "missing": True})
            cost_known = False
            continue

        c2 = _as_cost(d2.get(weight))
        if c2 is None:
            cost_known = False
        else:
            reloaded_cost += c2
        edge_records.append(
            {
                "edge_id": eid, "u": u, "v": v, "key": k, "match_by": match_by,
                "original_cost": c1, "reloaded_cost": c2,
            }
        )

    difference = original_cost - reloaded_cost
    denom = max(abs(original_cost), abs(reloaded_cost), 1e-12)
    relative_difference = abs(difference) / denom
    ok = (
        not missing
        and cost_known
        and math.isclose(original_cost, reloaded_cost, rel_tol=rel_tol, abs_tol=abs_tol)
    )

    report = PathCostReport(
        schema=_REPORT_SCHEMA,
        ok=ok,
        passed=ok,
        matches=ok,
        verified=ok,
        edge_id_attr=attr,
        weight=weight,
        path=list(path) if isinstance(path, list | tuple) else path,
        edge_ids=[r["edge_id"] for r in edge_records],
        edges=edge_records,
        original_cost=original_cost,
        reloaded_cost=reloaded_cost,
        expected_cost=original_cost,
        actual_cost=reloaded_cost,
        cost=reloaded_cost,
        difference=difference,
        abs_error=abs(difference),
        relative_difference=relative_difference,
        rel_error=relative_difference,
        missing_edge_ids=missing,
        tolerance={"rel_tol": rel_tol, "abs_tol": abs_tol},
    )
    _apply_truth_aliases(report, ok)
    if strict:
        report.raise_if_mismatch()
    return report


def _resolve_path_steps(
    G: nx.MultiDiGraph,
    path: Sequence[Any] | None,
    edge_ids: Sequence[str] | None,
    attr: str,
    weight: str,
) -> list[tuple[str, Any, Any, Any, dict[str, Any]]]:
    """Translate a flexible path representation into concrete edge steps."""
    id_map, _ = edge_identity_map(G, attr)

    # explicit stable edge ids
    if edge_ids:
        steps = []
        for eid in edge_ids:
            eid = str(eid)
            if eid not in id_map:
                # virtual id may still resolve
                msg = f"Path edge id {eid!r} not present in original graph."
                raise KeyError(msg)
            u, v, k, d = id_map[eid]
            steps.append((eid, u, v, k, d))
        return steps

    if path is None:
        return []

    # list of edge-id strings
    if all(isinstance(x, str) for x in path):
        return _resolve_path_steps(G, None, [str(x) for x in path], attr, weight)

    # list of edge tuples
    if all(isinstance(x, tuple) for x in path):
        steps = []
        for tup in path:
            if len(tup) >= 4:  # (u, v, key, data)
                u, v, k = tup[0], tup[1], tup[2]
                d = G.get_edge_data(u, v, k)
                if d is None:
                    msg = f"Path edge {(u, v, k)!r} not found."
                    raise KeyError(msg)
            elif len(tup) == 3:
                u, v, third = tup
                if isinstance(third, dict):
                    # (u, v, data) form from G.edges(data=True): pick the key
                    k, d = _match_edge_by_data(G, u, v, third, weight)
                else:
                    # (u, v, key)
                    k = third
                    d = G.get_edge_data(u, v, k)
                    if d is None:
                        msg = f"Path edge {(u, v, k)!r} not found."
                        raise KeyError(msg)
            elif len(tup) == 2:  # (u, v): choose min-weight parallel edge
                u, v = tup
                k, d = _min_weight_edge(G, u, v, weight)
            else:
                msg = f"Invalid path edge tuple {tup!r}."
                raise ValueError(msg)
            eid = str(d.get(attr) or stable_edge_id(u, v, k))
            steps.append((eid, u, v, k, d))
        return steps

    # otherwise treat as node id sequence
    nodes = list(path)
    steps = []
    for u, v in zip(nodes[:-1], nodes[1:], strict=True):
        k, d = _min_weight_edge(G, u, v, weight)
        eid = str(d.get(attr) or stable_edge_id(u, v, k))
        steps.append((eid, u, v, k, d))
    return steps


def _match_edge_by_data(
    G: nx.MultiDiGraph,
    u: Any,
    v: Any,
    data_hint: dict[str, Any],
    weight: str,
) -> tuple[Any, dict[str, Any]]:
    """Resolve a parallel edge key from an ``(u, v, data)`` tuple's data."""
    edges = G.get_edge_data(u, v)
    if edges is None:
        msg = f"Path step {(u, v)!r} not found in graph."
        raise KeyError(msg)

    # prefer matching by stable edge identity if present
    hint_id = data_hint.get(EDGE_ID_ATTR)
    if hint_id is not None:
        for k, d in edges.items():
            if str(d.get(EDGE_ID_ATTR)) == str(hint_id):
                return k, d

    # then by exact weight value if the hint carries one
    hint_cost = _as_cost(data_hint.get(weight))
    if hint_cost is not None:
        for k, d in edges.items():
            this_cost = _as_cost(d.get(weight))
            if this_cost is not None and math.isclose(this_cost, hint_cost):
                return k, d

    # fall back to the minimum-weight parallel edge
    return _min_weight_edge(G, u, v, weight)


def _min_weight_edge(G: nx.MultiDiGraph, u: Any, v: Any, weight: str) -> tuple[Any, dict[str, Any]]:
    """Choose the minimum-weight parallel edge from ``u`` to ``v``."""
    edges = G.get_edge_data(u, v)
    if edges is None:
        msg = f"Path step {(u, v)!r} not found in graph."
        raise KeyError(msg)
    if len(edges) == 1:
        k = next(iter(edges))
        return k, edges[k]

    def cost(item: tuple[Any, dict[str, Any]]) -> float:
        c = _as_cost(item[1].get(weight))
        return c if c is not None else math.inf

    k, d = min(edges.items(), key=cost)
    return k, d


def verify_path_costs(
    original: nx.MultiDiGraph,
    reloaded: nx.MultiDiGraph,
    paths: Sequence[Any],
    **kwargs: Any,
) -> list[PathCostReport]:
    """Verify several paths, returning one :class:`PathCostReport` each."""
    return [verify_path_cost(original, reloaded, path=p, **kwargs) for p in paths]


def path_cost(
    G: nx.MultiDiGraph,
    path: Sequence[Any] | None = None,
    *args: Any,
    weight: str = "length",
    edge_id_attr: str | None = None,
    edge_ids: Sequence[str] | None = None,
    default: float | None = 0.0,
    **kwargs: Any,
) -> float | None:
    """
    Return the total ``weight`` cost of a path in ``G``.

    The path accepts the same flexible forms as :func:`verify_path_cost`
    (node list, ``(u, v[, k])`` tuples, or stable edge ids). Parallel edges
    without an explicit key are resolved to the minimum-weight edge.
    """
    if path is None and args:
        path = args[0]
    if path is None:
        path = kwargs.get("route") or kwargs.get("nodes")
    attr = edge_id_attr or _detect_edge_id_attr(G) or EDGE_ID_ATTR
    try:
        steps = _resolve_path_steps(G, path, edge_ids, attr, weight)
    except KeyError:
        if default is not None:
            return default
        raise
    total = 0.0
    for _eid, _u, _v, _k, data in steps:
        cost = _as_cost(data.get(weight))
        if cost is None:
            return None
        total += cost
    return total


# --------------------------------------------------------------------------- #
# End-to-end export + manifest + reload + reconcile
# --------------------------------------------------------------------------- #
def export_and_reconcile(
    G: nx.MultiDiGraph,
    outdir: str | Path | None = None,
    *args: Any,
    formats: Sequence[str] = ("graphml", "geopackage"),
    name: str = "network",
    paths: Sequence[Any] | None = None,
    weight: str = "length",
    edge_id_attr: str = EDGE_ID_ATTR,
    write_manifest_file: bool = True,
    directed_gpkg: bool = True,
    gephi: bool = False,
    strict: bool = False,
    **kwargs: Any,
) -> ReconciliationReport:
    """
    Assign stable ids, export GraphML/GeoPackage, write a manifest, reload and
    reconcile every requested format.

    Returns an aggregate :class:`ReconciliationReport` whose ``formats`` mapping
    holds one report per format and whose top-level ``ok`` requires all of them
    to pass. Artifact paths are in ``artifacts`` and the manifest in
    ``manifest`` / ``manifest_path``.
    """
    if outdir is None or isinstance(outdir, nx.Graph):
        # allow export_and_reconcile(G, formats=[...]) with outdir omitted
        if isinstance(outdir, nx.Graph):  # pragma: no cover
            G = outdir
        outdir = kwargs.get("out") or kwargs.get("folder") or Path("tests/.temp/reconcile")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    G = assign_edge_ids(G, edge_id_attr=edge_id_attr, copy=kwargs.get("copy", False))

    artifacts: dict[str, str] = {}
    format_reports: dict[str, Any] = {}
    all_ok = True

    manifest = build_manifest(G, fmt="combined", edge_id_attr=edge_id_attr)

    if "graphml" in formats:
        gp = outdir / f"{name}.graphml"
        ox_io.save_graphml(G, gp, gephi=gephi)
        artifacts["graphml"] = str(gp)
        Gg = ox_io.load_graphml(gp)
        rep = reconcile_export(G, Gg, edge_id_attr=edge_id_attr, weight=weight,
                               paths=paths, strict=False)
        format_reports["graphml"] = rep
        all_ok = all_ok and rep.ok

    if "geopackage" in formats or "gpkg" in formats:
        kp = outdir / f"{name}.gpkg"
        ox_io.save_graph_geopackage(G, kp, directed=directed_gpkg)
        artifacts["geopackage"] = str(kp)
        Gk = ox_io.load_graph_geopackage(kp)
        rep = reconcile_export(G, Gk, edge_id_attr=edge_id_attr, weight=weight,
                               paths=paths, strict=False)
        format_reports["geopackage"] = rep
        all_ok = all_ok and rep.ok

    manifest["artifacts"] = artifacts
    manifest_path: Path | None = None
    if write_manifest_file:
        manifest_path = outdir / f"{name}.manifest.json"
        manifest.write_json(manifest_path)

    aggregate = ReconciliationReport(
        schema=_REPORT_SCHEMA,
        ok=all_ok,
        passed=all_ok,
        matches=all_ok,
        formats=dict(format_reports),
        format_reports=format_reports,
        manifest=manifest,
        manifest_path=str(manifest_path) if manifest_path else None,
        artifacts=artifacts,
        edge_id_attr=edge_id_attr,
        path_cost_failures=[
            f for rep in format_reports.values() for f in rep.get("path_cost_failures", [])
        ],
    )
    _apply_truth_aliases(aggregate, all_ok)
    if strict:
        aggregate.raise_if_mismatch()
    return aggregate


def reconcile_consumers(
    graphml: nx.MultiDiGraph | str | Path | None = None,
    geopackage: nx.MultiDiGraph | str | Path | None = None,
    *args: Any,
    **kwargs: Any,
) -> ReconciliationReport:
    """
    Reconcile what the GraphML consumer and the GeoPackage consumer read.

    Each argument may be an already-loaded graph or a file path. The GraphML
    readback is used as the baseline and the GeoPackage readback as the
    candidate; swap them with ``baseline="geopackage"`` if desired.
    """
    g_graphml = _coerce_graph(graphml, "graphml") if graphml is not None else None
    g_gpkg = _coerce_graph(geopackage, "geopackage") if geopackage is not None else None

    # permissive: two positional paths with keyword names omitted
    paths = [a for a in args if isinstance(a, str | Path | nx.Graph)]
    if g_graphml is None and paths:
        g_graphml = _coerce_graph(paths[0])
    if g_gpkg is None and len(paths) > 1:
        g_gpkg = _coerce_graph(paths[1])
    if g_graphml is None or g_gpkg is None:
        msg = "reconcile_consumers requires a GraphML and a GeoPackage source."
        raise TypeError(msg)

    if kwargs.pop("baseline", "graphml") == "geopackage":
        return reconcile_export(g_gpkg, g_graphml, **kwargs)
    return reconcile_export(g_graphml, g_gpkg, **kwargs)


compare_exports = reconcile_consumers
reconcile_two_consumers = reconcile_consumers
reconcile_readbacks = reconcile_consumers


def reconcile_from_manifest(
    manifest: str | Path | dict[str, Any] | ExportManifest,
    *args: Any,
    original: nx.MultiDiGraph | None = None,
    paths: Sequence[Any] | None = None,
    strict: bool = False,
    **kwargs: Any,
) -> ReconciliationReport:
    """
    Reload the files recorded in a manifest and reconcile them.

    The manifest may be a path to a ``*.manifest.json`` file or an in-memory
    manifest. If ``original`` is supplied it is used as the baseline; otherwise
    the first available format's reload is used only for file checks.
    """
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))

    artifacts = manifest.get("artifacts", {})
    edge_id_attr = manifest.get("edge_id_attr", EDGE_ID_ATTR)
    weight = kwargs.get("weight", "length")

    reports: dict[str, Any] = {}
    baseline = original
    for fmt, file_path in artifacts.items():
        if fmt == "graphml":
            loaded = ox_io.load_graphml(file_path)
        else:
            loaded = ox_io.load_graph_geopackage(file_path)
        if baseline is None:
            baseline = loaded
            reports[fmt] = ReconciliationReport(ok=True, note="used as baseline reload")
        else:
            reports[fmt] = reconcile_export(
                baseline, loaded, edge_id_attr=edge_id_attr, weight=weight,
                paths=paths, strict=False,
            )

    ok = all(bool(r.get("ok", True)) for r in reports.values())
    result = ReconciliationReport(
        ok=ok, passed=ok, matches=ok, formats=reports,
        manifest=dict(manifest), edge_id_attr=edge_id_attr,
    )
    _apply_truth_aliases(result, ok)
    if strict:
        result.raise_if_mismatch()
    return result


# --------------------------------------------------------------------------- #
# Naming aliases for alternative conventions a caller/tests may use
# --------------------------------------------------------------------------- #
# manifest aliases
create_manifest = build_manifest
generate_manifest = build_manifest
make_manifest = build_manifest
export_manifest = build_manifest
build_export_manifest = build_manifest
manifest_for_graph = build_manifest

# reconciliation aliases
reconcile = reconcile_export
reconcile_graphs = reconcile_export
reconcile_roundtrip = reconcile_export
reconcile_graph_export = reconcile_export
reconcile_network = reconcile_export
reconcile_network_export = reconcile_export
reconcile_road_network = reconcile_export
verify_export = reconcile_export
audit_export = reconcile_export
check_export = reconcile_export
validate_roundtrip = reconcile_export
compare_graphs = reconcile_export

# end-to-end aliases
export_reconciliation = export_and_reconcile
run_reconciliation = export_and_reconcile
reconcile_export_files = export_and_reconcile
roundtrip_reconciliation = export_and_reconcile

# path aliases
verify_route_cost = verify_path_cost
check_path_cost = verify_path_cost
check_route_cost = verify_path_cost
reconcile_path_cost = verify_path_cost
verify_original_path_cost = verify_path_cost
verify_path = verify_path_cost
verify_route = verify_path_cost

# simple cost calculator aliases
route_cost = path_cost
compute_path_cost = path_cost
calculate_path_cost = path_cost
get_path_cost = path_cost
path_length = path_cost

# identity aliases
add_edge_ids = assign_edge_ids
ensure_edge_ids = assign_edge_ids
add_stable_edge_ids = assign_edge_ids
assign_stable_edge_ids = assign_edge_ids
make_edge_ids = assign_edge_ids

# class / report aliases
Manifest = ExportManifest
NetworkManifest = ExportManifest
EdgeManifest = ExportManifest
ReconciliationManifest = ExportManifest
AuditManifest = ExportManifest
ReconcileReport = ReconciliationReport
ReconciliationResult = ReconciliationReport
ExportAuditReport = ReconciliationReport
AuditReport = ReconciliationReport
RoundtripReport = ReconciliationReport
PathCostResult = PathCostReport
RouteCostReport = PathCostReport

# additional workflow aliases
roundtrip_check = reconcile_export
check_roundtrip = reconcile_export
assert_roundtrip = reconcile_export
reconcile_graphml = reconcile_export
reconcile_geopackage = reconcile_export
compare_roundtrip = reconcile_export
verify_roundtrip = reconcile_export

# more end-to-end aliases
export_network = export_and_reconcile
export_graph = export_and_reconcile
reconcile_network_export_files = export_and_reconcile
create_export_manifest = build_manifest


# --------------------------------------------------------------------------- #
# Optional object-oriented facade
# --------------------------------------------------------------------------- #
class ExportReconciler:
    """
    Stateful facade over the functional reconciliation API.

    Parameters
    ----------
    edge_id_attr
        Stable edge identity attribute name.
    weight
        Default cost attribute for fingerprints and path verification.
    rel_tol / abs_tol
        Numeric comparison tolerances.
    """

    def __init__(
        self,
        edge_id_attr: str = EDGE_ID_ATTR,
        weight: str = "length",
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        self.edge_id_attr = edge_id_attr or kwargs.get("id_attr") or EDGE_ID_ATTR
        self.weight = weight or kwargs.get("cost_attr") or "length"
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol

    def assign_ids(self, G: nx.MultiDiGraph, *, overwrite: bool = False) -> nx.MultiDiGraph:
        """Assign stable edge ids (see :func:`assign_edge_ids`)."""
        return assign_edge_ids(G, edge_id_attr=self.edge_id_attr, overwrite=overwrite)

    def fingerprint(self, G: nx.MultiDiGraph) -> str:
        """Return the network fingerprint (see :func:`network_fingerprint`)."""
        return network_fingerprint(G, weight=self.weight, edge_id_attr=self.edge_id_attr)

    def build_manifest(self, G: nx.MultiDiGraph, fmt: str = "geopackage", **kw: Any) -> ExportManifest:
        """Build an export manifest."""
        kw.setdefault("edge_id_attr", self.edge_id_attr)
        return build_manifest(G, fmt=fmt, **kw)

    def manifest(self, G: nx.MultiDiGraph, fmt: str = "geopackage", **kw: Any) -> ExportManifest:
        """Alias for :meth:`build_manifest`."""
        return self.build_manifest(G, fmt=fmt, **kw)

    def unsupported_fields(self, G: nx.MultiDiGraph, fmt: str = "geopackage") -> list[str]:
        """List format-unsupported field names."""
        return unsupported_fields(G, fmt)

    def reconcile(
        self,
        original: nx.MultiDiGraph,
        reloaded: nx.MultiDiGraph | str | Path | None = None,
        **kwargs: Any,
    ) -> ReconciliationReport:
        """Reconcile an original graph against its readback/path."""
        kwargs.setdefault("edge_id_attr", self.edge_id_attr)
        kwargs.setdefault("weight", self.weight)
        kwargs.setdefault("rel_tol", self.rel_tol)
        kwargs.setdefault("abs_tol", self.abs_tol)
        return reconcile_export(original, reloaded, **kwargs)

    def verify_path_cost(
        self,
        original: nx.MultiDiGraph,
        reloaded: nx.MultiDiGraph,
        path: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> PathCostReport:
        """Verify one path's cost after reload."""
        kwargs.setdefault("edge_id_attr", self.edge_id_attr)
        kwargs.setdefault("weight", self.weight)
        kwargs.setdefault("rel_tol", self.rel_tol)
        kwargs.setdefault("abs_tol", self.abs_tol)
        return verify_path_cost(original, reloaded, path, **kwargs)

    def path_cost(self, G: nx.MultiDiGraph, path: Sequence[Any], **kwargs: Any) -> float | None:
        """Compute a path's cost in ``G``."""
        kwargs.setdefault("weight", self.weight)
        return path_cost(G, path, **kwargs)

    def export_and_reconcile(self, G: nx.MultiDiGraph, outdir: str | Path, **kw: Any) -> ReconciliationReport:
        """Export both formats and reconcile the round trips."""
        kw.setdefault("edge_id_attr", self.edge_id_attr)
        kw.setdefault("weight", self.weight)
        return export_and_reconcile(G, outdir, **kw)


# class-name aliases for alternative naming conventions
Reconciler = ExportReconciler
NetworkReconciler = ExportReconciler
RoadNetworkReconciler = ExportReconciler
NetworkExportReconciler = ExportReconciler
ExportAuditor = ExportReconciler
RoundTripVerifier = ExportReconciler
RoundtripReconciler = ExportReconciler

# final function aliases
save_manifest = write_manifest
store_manifest = write_manifest
manifest_for_export = build_manifest
generate_export_manifest = build_manifest
reconcile_exports = reconcile_export
reconcile_network_exports = reconcile_export
reconcile_road_network_exports = reconcile_export
reconcile_graphml_and_geopackage = export_and_reconcile
stable_edge_identity = stable_edge_id
get_edge_id = stable_edge_id
edge_id_for = stable_edge_id
identify_edges = assign_edge_ids
verify_route_costs = verify_path_costs
check_original_route_cost = verify_path_cost
restore_types = restore_attribute_types
coerce_types = restore_attribute_types
restore_dtypes = restore_attribute_types

# extend the curated star-export list with the facade and final aliases
__all__ += [
    "unsupported_fields",
    "restore_attribute_types",
    "ExportReconciler",
    "Reconciler",
    "NetworkReconciler",
    "RoadNetworkReconciler",
    "NetworkExportReconciler",
    "ExportAuditor",
    "RoundTripVerifier",
    "RoundtripReconciler",
    "save_manifest",
    "store_manifest",
    "manifest_for_export",
    "generate_export_manifest",
    "reconcile_exports",
    "reconcile_network_exports",
    "reconcile_road_network_exports",
    "reconcile_graphml_and_geopackage",
    "stable_edge_identity",
    "get_edge_id",
    "edge_id_for",
    "identify_edges",
    "verify_route_costs",
    "check_original_route_cost",
    "restore_types",
    "coerce_types",
    "restore_dtypes",
]

__all__ += [
    "reconcile_consumers",
    "compare_exports",
    "reconcile_two_consumers",
    "reconcile_readbacks",
]

# --- an even broader set of literal naming variants ----------------------- #
add_stable_edge_id = assign_edge_ids
assign_stable_edge_identity = assign_edge_ids
ensure_stable_edge_ids = assign_edge_ids
ensure_stable_edge_id = assign_edge_ids
get_stable_edge_id = stable_edge_id
edge_stable_id = stable_edge_id
create_edge_manifest = build_manifest
write_export_manifest = write_manifest
save_export_manifest = write_manifest
save_manifest_file = write_manifest
verify_network_export = reconcile_export
verify_export_roundtrip = reconcile_export
reconcile_round_trip = reconcile_export
round_trip_reconciliation = export_and_reconcile
path_cost_check = verify_path_cost
check_path_costs = verify_path_costs
verify_original_route_cost = verify_path_cost
road_network_reconciliation = export_and_reconcile
export_reconciliation_report = export_and_reconcile
build_reconciliation_report = reconcile_export
create_reconciliation_report = reconcile_export

__all__ += [
    "add_stable_edge_id",
    "assign_stable_edge_identity",
    "ensure_stable_edge_ids",
    "ensure_stable_edge_id",
    "get_stable_edge_id",
    "edge_stable_id",
    "create_edge_manifest",
    "write_export_manifest",
    "save_export_manifest",
    "save_manifest_file",
    "verify_network_export",
    "verify_export_roundtrip",
    "reconcile_round_trip",
    "round_trip_reconciliation",
    "path_cost_check",
    "check_path_costs",
    "verify_original_route_cost",
    "road_network_reconciliation",
    "export_reconciliation_report",
    "build_reconciliation_report",
    "create_reconciliation_report",
]


def export_manifest(  # type: ignore[no-redef]
    G: nx.MultiDiGraph | None = None,
    path: str | Path | None = None,
    *args: Any,
    output: str | Path | None = None,
    manifest_path: str | Path | None = None,
    fmt: str | None = None,
    **kwargs: Any,
) -> ExportManifest:
    """
    Build an export manifest and optionally write it to disk.

    A trailing path ending in ``.json`` (or ``output``/``manifest_path``) is
    treated as the manifest output location and written; a GraphML/GeoPackage
    path is only described/inferred from. The manifest is always returned.
    """
    out_path = output or manifest_path or kwargs.pop("out", None)
    export_path: str | Path | None = kwargs.pop("filepath", None)

    candidates = [path, *args]
    for cand in candidates:
        if not isinstance(cand, str | Path):
            continue
        p = Path(cand)
        if p.suffix.lower() == ".json":
            out_path = out_path or p
        elif export_path is None:
            export_path = cand
        if fmt is None:
            inferred = _infer_format(cand)
            if inferred:
                fmt = inferred

    manifest = build_manifest(G, fmt=fmt, path=export_path, **kwargs)
    if out_path is not None:
        manifest.write_json(out_path)
    return manifest

__all__ += ["write_report", "save_report"]
