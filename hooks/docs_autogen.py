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

from mkdocs.structure.files import File

DOCS = Path(__file__).resolve().parent.parent / "docs"

# Sections whose nav children are discovered, not listed.
AUTOSECTIONS = {
    "Layers": "build/layers",
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


def _param_table(fn, skip=("n", "rows", "cols", "self", "seed")) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ""
    rows = []
    for pname, p in sig.parameters.items():
        if pname in skip or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = "—" if p.default is inspect.Parameter.empty else f"`{p.default!r}`"
        rows.append(f"| `{pname}` | {default} |")
    if not rows:
        return ""
    return "| parameter | default |\n| --- | --- |\n" + "\n".join(rows)


def _topology_page(name: str, entry) -> str:
    builder, defaults, wrapper = entry
    kind = "graph" if wrapper.__name__ == "GraphTopology" else "matrix"
    summary = _doc_summary(builder)
    table = _param_table(builder)
    fig = f"../../../assets/figures/topologies/{name}.png"
    fig_block = (
        f"<figure markdown>\n![{name}]({fig})\n"
        f"<figcaption>Connectivity and weight matrix on a small reservoir.</figcaption>\n</figure>\n\n"
        if (DOCS / "assets/figures/topologies" / f"{name}.png").exists() else ""
    )
    return f"""---
description: "{summary[:140]}"
---

<span class="nb-kicker">Build · Topology ({kind})</span>

# `{name}`

{summary}

{fig_block}```python
from resdag.layers import ESNLayer

reservoir = ESNLayer(500, feedback_size=3, topology="{name}", spectral_radius=0.9)
```

{table}

Override any parameter with the tuple spec — `topology=("{name}", {{...}})` —
or pre-configure with `get_topology("{name}", ...)`. Scaling to a target
spectral radius happens after construction, whatever the source.

## See also

- [Initialization](../index.md) — every way to specify structure
- [Reference](../../../reference/init.md) — the registry API
"""


def _initializer_page(name: str, entry) -> str:
    obj, defaults = entry
    target = obj.__init__ if inspect.isclass(obj) else obj
    summary = _doc_summary(obj)
    table = _param_table(target)
    fig = f"../../../assets/figures/initializers/{name}.png"
    fig_block = (
        f"<figure markdown>\n![{name}]({fig})\n"
        f"<figcaption>An 80×8 weight matrix drawn by this initializer.</figcaption>\n</figure>\n\n"
        if (DOCS / "assets/figures/initializers" / f"{name}.png").exists() else ""
    )
    return f"""---
description: "{summary[:140]}"
---

<span class="nb-kicker">Build · Initializer</span>

# `{name}`

{summary}

{fig_block}```python
from resdag.layers import ESNLayer

reservoir = ESNLayer(500, feedback_size=3, feedback_initializer="{name}")
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
    def expand(match):
        directory = match.group(1).rstrip("/")
        page_dir = Path(page.file.src_uri).parent
        cards = []
        for uri in _discover(directory):
            if uri.endswith("/index.md"):
                continue
            title, desc = _page_meta(uri)
            rel = Path(uri).relative_to(page_dir) if Path(uri).is_relative_to(page_dir) \
                else Path("..") / uri  # fallback; index pages live above their children
            cards.append(f"- **[`{title}`]({rel.as_posix()})**\n\n    ---\n\n    {desc}\n")
        if not cards:
            return ""
        return '<div class="grid cards" markdown>\n\n' + "\n".join(cards) + "\n</div>"

    return _CARD_RE.sub(expand, markdown)
