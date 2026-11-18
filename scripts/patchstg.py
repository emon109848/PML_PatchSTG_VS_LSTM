"""Convenient entry point for integrated PatchSTG tooling; no work without a command."""

import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    commands = {"check": "check_patchstg.py", "gpu-check": "check_patchstg_gpu.py",
                "train": "train_patchstg.py", "evaluate": "evaluate_patchstg.py",
                "export": "export_patchstg_replay.py"}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    source_path = str(root / "src")
    sys.path.insert(0, source_path)
    os.environ["PYTHONPATH"] = source_path + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    script = root / "scripts" / commands[args.command]
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
