"""Command entry point; implementation lives in src/traffic_prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_prediction.inspect_data import main


if __name__ == "__main__":
    main()
