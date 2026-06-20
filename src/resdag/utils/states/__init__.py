"""State management and analysis utilities for reservoir layers."""

from .esp_index import esp_index

__all__ = ["esp_index"]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
