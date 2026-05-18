"""Thin wrapper: runs every figure generator in scripts/docs_figures/.

Run::

    .venv/bin/python scripts/build_docs_assets.py
    # or
    .venv/bin/python -m scripts.docs_figures
    # or one figure family at a time, e.g.
    .venv/bin/python -m scripts.docs_figures.sine

Each individual figure lives in its own module under scripts/docs_figures/
so you can edit (and re-run) one without touching the rest.
"""

from scripts.docs_figures.__main__ import main


if __name__ == "__main__":
    main()
