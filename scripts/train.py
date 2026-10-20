"""Train the selected model or run its validation ablation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from mp2.training import main

if __name__ == "__main__":
    main()
