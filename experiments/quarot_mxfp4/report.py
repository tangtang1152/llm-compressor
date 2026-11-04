"""Render figures locally from completed E1.2 summaries; no checkpoint needed."""

import argparse
import json
from pathlib import Path

from .diagnose import plots


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data["status"] != "complete":
        raise ValueError("report requires a completed diagnostic run")
    args.output.mkdir(parents=True, exist_ok=False)
    plots(
        args.output,
        data["matrices"],
        data["gamma"],
        data["comparisons"],
        data["data_kind"],
    )


if __name__ == "__main__":
    main()
