"""Regenerate documentation figures; forecasts come from recorded observations.

Use --animations to also render the recorded forecast movies through TSDynamics.
No script in this entrypoint integrates systems or runs hyperparameter searches.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "forecast.py",
    "phase.py",
    "topologies.py",
    "initializers.py",
    "architectures.py",
    "hpo.py",
    "readme.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--animations", action="store_true", help="render forecast movies; requires FFmpeg"
    )
    args = parser.parse_args()
    scripts = [*SCRIPTS, "hero_forecasts.py"] if args.animations else SCRIPTS
    for script in scripts:
        print(f"[{script}]", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)


if __name__ == "__main__":
    main()
