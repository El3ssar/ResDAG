"""Private delegation used by the new cell-backed sequence layers."""

from typing import Any

from .base_reservoir import BaseReservoirLayer


class CellReservoirLayer(BaseReservoirLayer):
    """Expose cell configuration while retaining BaseReservoirLayer state handling."""

    _live_attributes: frozenset[str] = frozenset()
    _fixed_attributes: frozenset[str] = frozenset()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            if "cell" in modules:
                try:
                    return getattr(modules["cell"], name)
                except AttributeError:
                    pass
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        modules = self.__dict__.get("_modules", {})
        if "cell" in modules:
            if name in self._fixed_attributes:
                raise AttributeError(
                    f"{name} is fixed at construction; create a new layer to change it"
                )
            if name in self._live_attributes:
                setattr(modules["cell"], name, value)
                return
        super().__setattr__(name, value)
