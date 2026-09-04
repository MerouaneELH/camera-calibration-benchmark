"""Run every calibration model adapter using the shared benchmark workflow.

This is the convenient command-line entry point for model inference. The
individual adapters remain independently runnable, while ``benchmark`` keeps
their repeated image iteration and artifact-writing behavior in one place.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SCRIPTS = (
    ROOT / "models" / "run_anycalib.py",
    ROOT / "models" / "run_geocalib.py",
    ROOT / "models" / "run_perspective.py",
)


def main() -> None:
    """Run each model adapter with the active Python interpreter."""
    for script in MODEL_SCRIPTS:
        print(f"\n=== Running {script.stem} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
