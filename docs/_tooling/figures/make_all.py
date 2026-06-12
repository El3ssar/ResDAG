"""Regenerate every documentation figure. Run from anywhere."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["forecast.py", "phase.py", "topologies.py", "initializers.py",
           "architectures.py", "hpo.py", "readme.py"]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"[{script}]")
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
    print("note: hero_forecasts.py retrains the landing-page models; run it "
          "separately when the library's forecasting behavior changes.")
