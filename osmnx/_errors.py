"""Define custom errors and exceptions."""


class CacheOnlyInterruptError(InterruptedError):
    """Exception for `settings.cache_only_mode=True` interruption."""


class GraphSimplificationError(ValueError):
    """Exception for a problem with graph simplification."""


class ValidationError(ValueError):
    """Exception for failed graph or GeoDataFrame validation."""


class InsufficientResponseError(ValueError):
    """Exception for empty or too few results in server response."""


class IncompleteCoverageError(IOError):
    """Exception when one or more planned area shards could not be retrieved.

    Raised after every allowed retry is exhausted for at least one shard of a
    subdivided query. The successfully retrieved shards are preserved in the
    local HTTP cache, so re-running the same query resumes from the failed
    shards rather than re-downloading the completed ones.
    """


class ResponseStatusCodeError(ValueError):
    """Exception for an unhandled server response status code."""
