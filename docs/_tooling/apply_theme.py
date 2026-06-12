"""Render docs/_tooling/theme.json into the site stylesheet.

theme.json is the single source of truth for the documentation's colors
and fonts — the website CSS and every figure script read from it.

Usage (from the repo root):

    uv run python docs/_tooling/apply_theme.py             # CSS tokens only
    uv run python docs/_tooling/apply_theme.py --figures   # + regenerate all figures
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
THEME = json.loads((Path(__file__).parent / "theme.json").read_text())
CSS = ROOT / "docs" / "css" / "notebook.css"

START = "/* ===== TOKENS — generated from docs/_tooling/theme.json; edit there ===== */"
END = "/* ===== /TOKENS ===== */"


def scheme_block(selector: str, t: dict, syn: dict) -> str:
    # Material uses --md-primary-bg-color as the text color ON the header bar;
    # the header is dark in both schemes, so this must always be light.
    on_header = "#f2f4f2"
    return f"""{selector} {{
  --nb-paper: {t["bg"]};
  --nb-surface: {t["surface"]};
  --nb-ink: {t["ink"]};
  --nb-ink-soft: {t["ink_soft"]};
  --nb-muted: {t["muted"]};
  --nb-rule: {t["rule"]};
  --nb-rule-strong: {t["rule_strong"]};
  --nb-accent: {t["accent"]};
  --nb-accent-strong: {t["accent_strong"]};
  --nb-accent-wash: {t["accent_wash"]};
  --nb-code-bg: {t["code_bg"]};
  --nb-data-true: {t["data_true"]};

  --md-default-bg-color: {t["bg"]};
  --md-default-fg-color: {t["ink"]};
  --md-default-fg-color--light: {t["ink_soft"]};
  --md-default-fg-color--lighter: {t["muted"]};
  --md-typeset-color: {t["ink"]};
  --md-typeset-a-color: {t["accent"]};
  --md-accent-fg-color: {t["accent"]};
  --md-primary-fg-color: {t["header_bg"]};
  --md-primary-bg-color: {on_header};
  --md-footer-bg-color: {t["footer_bg"]};
  --md-footer-bg-color--dark: {t["footer_bg"]};

  --md-code-bg-color: {syn["bg"]};
  --md-code-fg-color: {syn["fg"]};
  --md-code-hl-keyword-color: {syn["keyword"]};
  --md-code-hl-string-color: {syn["string"]};
  --md-code-hl-number-color: {syn["number"]};
  --md-code-hl-function-color: {syn["function"]};
  --md-code-hl-constant-color: {syn["constant"]};
  --md-code-hl-name-color: {syn["name"]};
  --md-code-hl-operator-color: {syn["operator"]};
  --md-code-hl-punctuation-color: {syn["punctuation"]};
  --md-code-hl-comment-color: {syn["comment"]};
  --md-code-hl-special-color: {syn["special"]};
  --md-code-hl-variable-color: {syn["variable"]};
  --md-code-hl-generic-color: {syn["fg"]};
  --md-code-hl-color: {t["accent_wash"]};
}}"""


def main() -> None:
    f = THEME["fonts"]
    tokens = "\n\n".join([
        START,
        f""":root {{
  --nb-display: "{f["display"]}", sans-serif;
  --nb-body: "{f["body"]}", -apple-system, sans-serif;
  --nb-mono: "{f["mono"]}", ui-monospace, monospace;
}}""",
        scheme_block('[data-md-color-scheme="default"]', THEME["light"], THEME["syntax"]["light"]),
        scheme_block('[data-md-color-scheme="slate"]', THEME["dark"], THEME["syntax"]["dark"]),
        END,
    ])

    css = CSS.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if pattern.search(css):
        css = pattern.sub(tokens, css)
    else:
        sys.exit("token markers not found in notebook.css")
    CSS.write_text(css)
    print(f"tokens rendered into {CSS.relative_to(ROOT)}")

    if "--figures" in sys.argv:
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "figures" / "make_all.py")],
            check=True,
        )


if __name__ == "__main__":
    main()
