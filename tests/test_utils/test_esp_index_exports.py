"""Public-API export tests for ``esp_index`` (issue #180).

The ESP index is the library's signature stability diagnostic, so it must be
reachable from the short, discoverable import paths — not only the deep
``resdag.utils.states.esp_index`` path. These tests pin each acceptance
criterion from issue #180:

* ``from resdag.utils import esp_index`` works;
* the ``states`` subpackage is importable as ``resdag.utils.states``;
* ``esp_index`` appears in the relevant ``__all__`` lists;
* the symbol is the *same* object regardless of import path.
"""

import resdag
import resdag.utils
import resdag.utils.states
from resdag.utils.states.esp_index import esp_index as canonical_esp_index


def test_top_level_esp_index_import() -> None:
    """``from resdag import esp_index`` resolves to the canonical function."""
    from resdag import esp_index

    assert esp_index is canonical_esp_index


def test_utils_esp_index_import() -> None:
    """``from resdag.utils import esp_index`` resolves to the canonical function."""
    from resdag.utils import esp_index

    assert esp_index is canonical_esp_index


def test_states_subpackage_is_importable() -> None:
    """The ``states`` subpackage is reachable as ``resdag.utils.states``."""
    assert resdag.utils.states.esp_index is canonical_esp_index


def test_esp_index_in_top_level_all() -> None:
    """``esp_index`` is advertised in the top-level public API."""
    assert "esp_index" in resdag.__all__
    assert "esp_index" in dir(resdag)


def test_esp_index_in_utils_all() -> None:
    """``esp_index`` and ``states`` are advertised in ``resdag.utils.__all__``."""
    assert "esp_index" in resdag.utils.__all__
    assert "states" in resdag.utils.__all__
    assert "esp_index" in dir(resdag.utils)
    assert "states" in dir(resdag.utils)


def test_all_import_paths_agree() -> None:
    """Every documented import path yields the identical function object."""
    assert resdag.esp_index is resdag.utils.esp_index is resdag.utils.states.esp_index
