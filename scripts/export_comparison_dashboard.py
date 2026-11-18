"""Build the standalone LSTM/PatchSTG dashboard from frozen/cached forecasts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_prediction.comparison_dashboard import main


if __name__ == "__main__":
    main()
