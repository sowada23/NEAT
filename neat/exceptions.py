class NEATError(Exception):
    """Base exception for the NEAT package."""


class CycleError(NEATError):
    """Raised when a graph cycle prevents feed-forward execution."""
