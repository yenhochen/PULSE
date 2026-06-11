"""Entry point for synthetic analysis dataset generation."""

import subprocess
from pathlib import Path

BUILD_SCRIPT = Path(__file__).resolve().parent / "analysis" / "build.sh"

if __name__ == "__main__":
    subprocess.run(["sh", str(BUILD_SCRIPT)], check=True)
