"""Define custom errors and exceptions."""


class CacheOnlyInterruptError(InterruptedError):
    """Exception for `settings.cache_only_mode=True` interruption."""


class GraphSimplificationError(ValueError):
    """Exception for a problem with graph simplification."""


class ValidationError(ValueError):
    """Exception for failed graph or GeoDataFrame validation."""


class InsufficientResponseError(ValueError):
    """Exception for empty or too few results in server response."""


class SourceConflictError(ValueError):
    """
    Exception for an unresolvable conflict between duplicate source elements.

    Raised when the same OSM element is returned by multiple source parcels
    with tied version facts but differing geometry, so no observation can be
    selected without guessing.
    """


class ResponseStatusCodeError(ValueError):
    """Exception for an unhandled server response status code."""
