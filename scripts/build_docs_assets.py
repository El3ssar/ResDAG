"""Run the canonical documentation figure pipeline.

    python scripts/build_docs_assets.py
    python scripts/build_docs_assets.py --animations

The source generators and shared style live in docs/_tooling/figures/.
Forecast figures replay recorded data; this command never trains a model.
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    pipeline = Path(__file__).resolve().parents[1] / "docs/_tooling/figures/make_all.py"
    subprocess.run([sys.executable, str(pipeline), *sys.argv[1:]], check=True)
