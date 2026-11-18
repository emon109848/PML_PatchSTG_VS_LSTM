"""Refresh comparison documents/figures from saved scores; execute no models."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_prediction.finalize_context_lstm import build_comparison
from traffic_prediction.train_lstm import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "runs/final_comparison_v1")
    args = parser.parse_args()
    report = args.results_dir / "report.json"
    original_hash = sha256(report)
    result = json.loads(report.read_text())
    selection = json.loads((args.results_dir / "selection.json").read_text())
    if result["status"] != "complete" or not result["checkpoint_unchanged"]:
        raise ValueError("A completed frozen LSTM evaluation is required.")
    build_comparison(args.results_dir, selection, result, ROOT, ROOT)
    if sha256(report) != original_hash:
        raise RuntimeError("Measured LSTM evaluation report changed.")
    provenance = {"mode": "report/figure refresh from existing completed scores; no training or inference",
                  "lstm_test_report_sha256": original_hash,
                  "patchstg_test_report_sha256": sha256(ROOT / "reports/patchstg_sd_seed2023_test_v1.json"),
                  "report_builder_sha256": sha256(ROOT / "src/traffic_prediction/finalize_context_lstm.py")}
    (args.results_dir / "report_build.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Comparison report and figures refreshed from saved scores. No model executed.")
    print(args.results_dir / "PROJECT_REPORT.md")


if __name__ == "__main__":
    main()
