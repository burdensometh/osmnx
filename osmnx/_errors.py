"""Define custom errors and exceptions."""


class CacheOnlyInterruptError(InterruptedError):
    """Exception for `settings.cache_only_mode=True` interruption."""


class GraphSimplificationError(ValueError):
    """Exception for a problem with graph simplification."""


class ValidationError(ValueError):
    """Exception for failed graph or GeoDataFrame validation."""


class InsufficientResponseError(ValueError):
    """Exception for empty or too few results in server response."""


class ResponseStatusCodeError(ValueError):
    """Exception for an unhandled server response status code."""


class MissingNodeReferencesError(ValueError):
    """Exception for way node references that cannot be found in any PBF source."""

    def __init__(self, message: str, report: object | None = None) -> None:
        """
        Create the exception, optionally carrying the audit report.

        Parameters
        ----------
        message
            Error description.
        report
            The `PBFReadReport` describing the missing references, if available.
        """
        super().__init__(message)
        self.report = report
