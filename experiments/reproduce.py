"""Reproduce the report's fixed baseline comparisons on validation data."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from baselines.comparison import main

if __name__ == "__main__":
    main()
