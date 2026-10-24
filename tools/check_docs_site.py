"""Check local links, HTML anchors and media references in a built MkDocs site.

MkDocs validates Markdown links, but raw HTML video/poster/source references and
copied template files also need checking. This command reads local output only;
it never contacts external websites or executes page scripts.

    python tools/check_docs_site.py site --base-path /ResDAG/
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


@dataclass(frozen=True)
class _Reference:
    line: int
    attribute: str
    value: str


class _Page(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.references: list[_Reference] = []
        self.anchors: set[str] = set()
        self.anchor_counts: Counter[str] = Counter()
        self.prose: list[tuple[int, str]] = []
        self._ignored: list[str] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in {"pre", "code", "script", "style"}:
            self._ignored.append(tag)
        anchors = {values.get("id"), values.get("name") if tag == "a" else None} - {None, ""}
        self.anchors.update(anchors)
        self.anchor_counts.update(anchors)
        for attribute in ("href", "src", "poster"):
            if values.get(attribute):
                self.references.append(_Reference(self.getpos()[0], attribute, values[attribute]))

    def handle_endtag(self, tag: str) -> None:
        if tag in self._ignored:
            self._ignored.remove(tag)

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.prose.append((self.getpos()[0], data))


def _base_path(value: str) -> str:
    if not value.startswith("/") or "?" in value or "#" in value or ":" in value:
        raise ValueError("base_path must be a URL path such as /ResDAG/")
    return value.rstrip("/") + "/"


def check_site(site: Path, *, base_path: str = "/") -> list[str]:
    """Return broken local references with their source HTML paths and lines."""
    root = site.resolve()
    base = _base_path(base_path)
    pages = {
        path.resolve(): _Page(path.read_text(encoding="utf-8")) for path in root.rglob("*.html")
    }
    if not root.is_dir() or not pages:
        return [f"No built HTML pages found in {root}"]
    problems: list[str] = []
    for source, page in sorted(pages.items()):
        relative = source.relative_to(root).as_posix()
        for anchor, count in page.anchor_counts.items():
            if count > 1:
                problems.append(f"{relative}: duplicate HTML anchor {anchor!r} ({count} copies)")
        for line, prose in page.prose:
            if re.search(
                r":(?:(?:py|std):)?(?:attr|class|const|data|doi|exc|func|math|meth|mod|obj|pep|ref|term|type):"
                r"|(?m:^\s*\.\.\s+[a-z][\w-]*::)",
                prose,
            ):
                problems.append(f"{relative}:{line}: unrendered reStructuredText in prose")
        origin = f"https://docs.invalid{base}{relative}"
        for reference in page.references:
            parsed = urlsplit(reference.value)
            if parsed.scheme or parsed.netloc:
                continue
            resolved = urlsplit(urljoin(origin, reference.value))
            prefix = f"{relative}:{reference.line}: {reference.attribute}={reference.value!r}"
            decoded = unquote(resolved.path)
            if not decoded.startswith(base):
                problems.append(f"{prefix} is outside deployed base path {base}")
                continue
            target = (root / decoded.removeprefix(base)).resolve()
            if not target.is_relative_to(root):
                problems.append(f"{prefix} escapes the built site directory")
                continue
            if target.is_dir() or resolved.path.endswith("/"):
                target /= "index.html"
            if not target.is_file():
                problems.append(f"{prefix} points to a missing local target")
                continue
            anchor = unquote(resolved.fragment).partition(":~:text=")[0]
            if anchor and target in pages and anchor not in pages[target].anchors:
                problems.append(f"{prefix} points to a missing HTML anchor")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, nargs="?", default=Path("site"))
    parser.add_argument("--base-path", default="/", help="path from site_url; for example /ResDAG/")
    args = parser.parse_args()
    try:
        problems = check_site(args.site, base_path=args.base_path)
    except ValueError as error:
        parser.error(str(error))
    for problem in problems:
        print(problem)
    if problems:
        print(f"Found {len(problems)} broken local documentation references.")
        return 1
    print(f"Checked built HTML links, anchors and media in {args.site}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
