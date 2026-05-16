"""Tests for the input/feedback initializer registry."""

import pytest

from resdag.init.input_feedback import (
    get_input_feedback,
    show_input_initializers,
)


class TestInputFeedbackRegistry:
    """Tests for input/feedback initializer registry."""

    def test_show_initializers_returns_list(self):
        """``show_input_initializers()`` returns the sorted list of names."""
        names = show_input_initializers()

        assert isinstance(names, list)
        assert len(names) > 0
        assert names == sorted(names)
        # A couple of names we know are registered.
        assert "random" in names
        assert "chebyshev" in names

    def test_show_initializers_with_name_returns_none(self):
        """When a name is given, only details are printed and ``None`` returned."""
        result = show_input_initializers("chebyshev")
        assert result is None

    def test_show_initializers_unknown_raises(self):
        """Unknown names raise a ``ValueError`` with the available list."""
        with pytest.raises(ValueError, match="Unknown initializer"):
            show_input_initializers("definitely_not_an_initializer")

    def test_get_input_feedback_known_name(self):
        """``get_input_feedback`` returns an initializer for a known name."""
        init = get_input_feedback("chebyshev")
        assert init is not None
        assert hasattr(init, "initialize")
