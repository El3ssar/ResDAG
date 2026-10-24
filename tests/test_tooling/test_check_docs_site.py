"""Built-site validation catches raw HTML media and copied-template mistakes."""

import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "tools/check_docs_site.py"
spec = importlib.util.spec_from_file_location("_check_docs_site", SOURCE)
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)


def _write(root, relative, text=""):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_resolves_deployed_prefix_directory_links_anchors_and_video(tmp_path):
    _write(tmp_path, "index.html", '<a href="/ResDAG/workflows/visualize/#movie">Movie</a>')
    _write(tmp_path, "assets/movie.mp4")
    _write(tmp_path, "assets/poster.png")
    _write(
        tmp_path,
        "workflows/visualize/index.html",
        '<h2 id="movie">Movie</h2><video poster="../../assets/poster.png">'
        '<source src="../../assets/movie.mp4"></video><a href="../../">Home</a>',
    )
    assert checker.check_site(tmp_path, base_path="/ResDAG/") == []


def test_checks_raw_html_media_paths_including_wrong_directory_depth(tmp_path):
    _write(tmp_path, "assets/poster.png")
    _write(
        tmp_path,
        "workflows/visualize/index.html",
        '<video poster="../assets/poster.png"><source src="../../assets/missing.mp4"></video>',
    )
    problems = checker.check_site(tmp_path, base_path="/ResDAG/")
    assert len(problems) == 2
    assert all("workflows/visualize/index.html:1:" in problem for problem in problems)
    assert any("poster=" in problem for problem in problems)
    assert any("src=" in problem for problem in problems)


def test_copied_template_is_not_silently_exempt_from_link_validation(tmp_path):
    _write(tmp_path, "index.html", "Home")
    _write(tmp_path, "start/index.html", "Start")
    _write(tmp_path, "overrides/home.html", '<a href="start/">Start</a>')
    problems = checker.check_site(tmp_path)
    assert len(problems) == 1
    assert problems[0].startswith("overrides/home.html:1:")


def test_missing_html_anchor_and_outside_base_are_reported(tmp_path):
    _write(tmp_path, "index.html", '<a href="#absent">Missing</a><img src="/assets/poster.png">')
    _write(tmp_path, "assets/poster.png")
    problems = checker.check_site(tmp_path, base_path="/ResDAG/")
    assert len(problems) == 2
    assert any("missing HTML anchor" in problem for problem in problems)
    assert any("outside deployed base" in problem for problem in problems)


def test_external_data_urls_encoded_anchors_and_text_fragments(tmp_path):
    _write(
        tmp_path,
        "index.html",
        '<h2 id="a.b">API</h2><a href="#a%2Eb">Here</a>'
        '<a href="#:~:text=API">Text</a><a href="https://example.invalid/unavailable">External</a>'
        '<img src="data:image/png;base64,abc"><script src="//cdn.invalid/script.js"></script>',
    )
    assert checker.check_site(tmp_path) == []


def test_empty_output_and_escaping_encoded_path_are_rejected(tmp_path):
    assert checker.check_site(tmp_path)[0].startswith("No built HTML")
    _write(tmp_path, "index.html", '<a href="%2e%2e/private.html">Outside</a>')
    assert "escapes" in checker.check_site(tmp_path)[0]
    with pytest.raises(ValueError, match="URL path"):
        checker.check_site(tmp_path, base_path="https://example.invalid/")


def test_unrendered_docstring_roles_and_directives_are_rejected_in_prose(tmp_path):
    _write(
        tmp_path,
        "index.html",
        "<p>Return a :class:<code>torch.Tensor</code>.</p>\n"
        "<p>.. deprecated:: 0.7.0</p>\n<p>.. math::</p>",
    )
    problems = checker.check_site(tmp_path)
    assert len(problems) == 3
    assert all("unrendered reStructuredText" in problem for problem in problems)


def test_examples_source_and_script_literals_are_not_mistaken_for_rendering_failures(tmp_path):
    _write(
        tmp_path,
        "index.html",
        "<pre><code>:class:`torch.Tensor`\n.. math::</code></pre>"
        '<code>:func:`example`</code><script>const literal = ":math:";</script>'
        "<style>/* :class: */</style><p>Ordinary Markdown prose.</p>",
    )
    assert checker.check_site(tmp_path) == []


def test_duplicate_ids_from_repeated_module_reference_directives_are_rejected(tmp_path):
    _write(tmp_path, "index.html", '<h2 id="resdag.data">Data</h2><h2 id="resdag.data">Data</h2>')
    assert checker.check_site(tmp_path) == [
        "index.html: duplicate HTML anchor 'resdag.data' (2 copies)"
    ]
    _write(tmp_path, "index.html", '<a id="same" name="same">One anchor</a>')
    assert checker.check_site(tmp_path) == []
