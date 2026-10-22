"""MkDocs build hooks that make the docs modular.

Three mechanisms, zero extra dependencies:

1. Registry pages — one page per registered topology and input/feedback
   initializer is *generated at build time* from the live registries
   (docstring summary, parameter table, gallery figure). Registering a new
   component in the library is all it takes for its page to appear.

2. Nav autogen — the nav sections listed in AUTOSECTIONS pick up every
   page in their directory automatically; adding a layer or architecture
   page is "drop the file in the folder".

3. Card autogen — `<!-- nb-cards: <docs-relative-dir> -->` on any index
   page expands into a card grid built from the target pages' frontmatter
   (title + description), including generated pages.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

# Executable recipes are metadata, not guesses from a common reservoir size.
# These components have size constraints or required builder arguments.
CATALOG_EXAMPLES = {
    "topologies": {
        "kleinberg_small_world": {"reservoir_size": 484},
        "multi_cycle": {"reservoir_size": 498},
        "spectral_cascade": {"reservoir_size": 496},
    },
    "initializers": {
        "chain_of_neurons_input": {"reservoir_size": 501},
        "dendrocycle_input": {"parameters": {"c": 0.5}},
        "informed_init": {"parameters": {"model_in_size": 1}},
        "ring_window": {"parameters": {"c": 0.5, "window": 5}},
    },
}

# Sections whose nav children are discovered, not listed.
AUTOSECTIONS = {
    "Layers": "build/layers",
    "Readouts": "build/readouts",
    "Architectures": "build/architectures",
    "Topologies": "build/initialization/topologies",
    "Initializers": "build/initialization/initializers",
}


# ---------------------------------------------------------------------------
# Registry page generation
# ---------------------------------------------------------------------------


def _registries():
    from resdag.init.input_feedback.registry import _INPUT_FEEDBACK_REGISTRY
    from resdag.init.topology.registry import _TOPOLOGY_REGISTRY

    return _TOPOLOGY_REGISTRY, _INPUT_FEEDBACK_REGISTRY


def _doc_summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    lines = [ln.strip() for ln in doc.splitlines()]
    out = []
    for ln in lines:
        if not ln and out:
            break
        if ln:
            out.append(ln)
    return " ".join(out) or "No description yet."


def _param_table(fn, defaults=None, skip=("n", "rows", "cols", "self", "seed")) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ""
    rows = []
    for pname, p in sig.parameters.items():
        if pname in skip or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        value = (defaults or {}).get(pname, p.default)
        default = "—" if value is inspect.Parameter.empty else f"`{value!r}`"
        rows.append(f"| `{pname}` | {default} |")
    if not rows:
        return ""
    return "| parameter | default |\n| --- | --- |\n" + "\n".join(rows)


def _catalog_example(name, category, *, radius_arg=""):
    recipe = CATALOG_EXAMPLES[category].get(name, {})
    size = recipe.get("reservoir_size", 500)
    parameters = recipe.get("parameters")
    spec = repr((name, parameters)) if parameters else repr(name)
    option = "recurrent_initializer" if category == "topologies" else "feedback_initializer"
    return f"ESNLayer({size}, feedback_size=3, {option}={spec}{radius_arg}, seed=7)"


def _topology_page(name: str, entry) -> str:
    # Forward-compatible unpack: the registry entry is
    # (builder, defaults, wrapper, prescaled, ...). Bind the leading fields we
    # use and ignore any trailing ones so future additions don't break the hook.
    builder, defaults, wrapper, *_ = entry
    prescaled = bool(entry[3]) if len(entry) > 3 else False
    kind = "graph" if wrapper.__name__ == "GraphTopology" else "matrix"
    summary = _doc_summary(builder)
    table = _param_table(builder, defaults)
    radius_arg = (
        ", spectral_radius=None"
        if prescaled or name in {"zeros", "delay_line"}
        else ", spectral_radius=0.9"
    )
    example = _catalog_example(name, "topologies", radius_arg=radius_arg)
    scaling = (
        "This builder preserves its native spectrum/scale by default. "
        "Use its parameters to control that scale, or explicitly configure "
        "`get_recurrent_initializer(..., rescale=True)` to allow an outer radius rescale."
        if prescaled
        else "A layer `spectral_radius` rescales this matrix after construction. "
        "Use `get_recurrent_initializer(..., rescale=False)` to keep the native scale."
    )
    if name in {"zeros", "delay_line"}:
        scaling = (
            "This builder has a zero eigenvalue radius. A nonzero outer radius "
            "cannot be obtained by multiplying its weights; control the "
            "connection magnitude through the builder's own parameters."
        )
    fig = f"../../../assets/figures/topologies/{name}.png"
    fig_block = (
        f"<figure markdown>\n![{name}]({fig})\n"
        f"<figcaption>Connectivity and weight matrix on a small reservoir.</figcaption>\n</figure>\n\n"
        if (DOCS / "assets/figures/topologies" / f"{name}.png").exists()
        else ""
    )
    return f"""---
description: "{summary[:140]}"
---

<span class="nb-kicker">Build · Topology ({kind})</span>

# `{name}`

{summary}

{fig_block}```python
from resdag.layers import ESNLayer

reservoir = {example}
```

{table}

Override parameters with `recurrent_initializer=("{name}", {{...}})` or
pre-configure with `get_recurrent_initializer("{name}", ...)`.
The existing `topology=` spelling and `get_topology()` also work.

{scaling}

## See also

- [Initialization](../index.md) — every way to specify structure
- [Reference](../../../reference/init.md) — the registry API
"""


def _initializer_page(name: str, entry) -> str:
    obj, defaults = entry
    target = obj.__init__ if inspect.isclass(obj) else obj
    summary = _doc_summary(obj)
    table = _param_table(target, defaults)
    example = _catalog_example(name, "initializers")
    fig = f"../../../assets/figures/initializers/{name}.png"
    fig_block = (
        f"<figure markdown>\n![{name}]({fig})\n"
        f"<figcaption>An 80×8 weight matrix drawn by this initializer.</figcaption>\n</figure>\n\n"
        if (DOCS / "assets/figures/initializers" / f"{name}.png").exists()
        else ""
    )
    return f"""---
description: "{summary[:140]}"
---

<span class="nb-kicker">Build · Initializer</span>

# `{name}`

{summary}

{fig_block}```python
from resdag.layers import ESNLayer

reservoir = {example}
```

{table}

Works for `feedback_initializer=` and `input_initializer=` alike; override
parameters with `("{name}", {{...}})` or `get_input_feedback("{name}", ...)`.

## See also

- [Initialization](../index.md) — every way to specify structure
- [Reference](../../../reference/init.md) — the registry API
"""


_GENERATED: dict[str, str] = {}


def _populate_generated():
    topo, init = _registries()
    _GENERATED.clear()
    for name in sorted(topo):
        uri = f"build/initialization/topologies/{name}.md"
        _GENERATED[uri] = _topology_page(name, topo[name])
    for name in sorted(init):
        uri = f"build/initialization/initializers/{name}.md"
        _GENERATED[uri] = _initializer_page(name, init[name])


def on_files(files, config):
    from mkdocs.structure.files import File

    for uri, content in _GENERATED.items():
        files.append(File.generated(config, uri, content=content))
    return files


# ---------------------------------------------------------------------------
# Nav autogen
# ---------------------------------------------------------------------------


def _walk_nav(items):
    for item in items:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, list):
                    if key in AUTOSECTIONS:
                        yield key, value
                    yield from _walk_nav(value)


def _discover(directory: str) -> list[str]:
    found = set()
    base = DOCS / directory
    if base.exists():
        found |= {f"{directory}/{p.name}" for p in base.glob("*.md")}
    found |= {uri for uri in _GENERATED if Path(uri).parent.as_posix() == directory}
    index = f"{directory}/index.md"
    rest = sorted(u for u in found if u != index)
    return ([index] if index in found else []) + rest


def on_config(config):
    # on_config runs before on_files — generate registry pages here so the
    # nav autogen below can see them.
    _populate_generated()
    if not config.nav:
        return config
    for key, children in _walk_nav(config.nav):
        directory = AUTOSECTIONS[key]
        # keep any explicit entries, append discovered ones not yet listed
        explicit = set()
        for c in children:
            explicit.add(c if isinstance(c, str) else next(iter(c.values())))
        for uri in _discover(directory):
            if uri not in explicit:
                children.append(uri)
    return config


# ---------------------------------------------------------------------------
# Card autogen
# ---------------------------------------------------------------------------

_CARD_RE = re.compile(r"<!--\s*nb-cards:\s*(\S+)\s*-->")


def _page_meta(uri: str) -> tuple[str, str]:
    """(title, description) for a docs-relative page uri."""
    if uri in _GENERATED:
        text = _GENERATED[uri]
    else:
        path = DOCS / uri
        if not path.exists():
            return Path(uri).stem, ""
        text = path.read_text(encoding="utf-8")
    title = Path(uri).stem.replace("-", " ").replace("_", " ")
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        title = m.group(1).strip().strip("`")
    desc = ""
    m = re.search(r'^description:\s*["\']?(.+?)["\']?\s*$', text, re.M)
    if m:
        desc = m.group(1).strip()
    return title, desc


def on_page_markdown(markdown, page, config, files):
    if "{{ resdag_version }}" in markdown:
        import resdag

        markdown = markdown.replace("{{ resdag_version }}", resdag.__version__)

    def expand(match):
        directory = match.group(1).rstrip("/")
        page_dir = Path(page.file.src_uri).parent
        cards = []
        for uri in _discover(directory):
            if uri.endswith("/index.md"):
                continue
            title, desc = _page_meta(uri)
            rel = (
                Path(uri).relative_to(page_dir)
                if Path(uri).is_relative_to(page_dir)
                else Path("..") / uri
            )  # fallback; index pages live above their children
            cards.append(f"- **[`{title}`]({rel.as_posix()})**\n\n    ---\n\n    {desc}\n")
        if not cards:
            return ""
        return '<div class="grid cards" markdown>\n\n' + "\n".join(cards) + "\n</div>"

    return _CARD_RE.sub(expand, markdown)
