"""Join completed norm/OV reports without altering historical E2 artifacts."""

import argparse
import csv
import json
from pathlib import Path

from ..next_stage.common import csv_rows


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("norm", "ov", "norm-ab", "ov-ab", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = [
        json.loads((root / "results.json").read_text(encoding="utf-8"))
        for root in (args.norm, args.ov)
    ]
    if any(results[0][k] != results[1][k] for k in ("token_ids", "seed", "provenance")):
        raise ValueError("Norm/OV data identity, seed or runtime differs")
    gates = [
        json.loads((root / "parity.json").read_text())
        for root in (args.norm_ab, args.ov_ab)
    ]
    if (
        not all(g["passed"] for g in gates)
        or gates[0]["provenance"] != gates[1]["provenance"]
    ):
        raise ValueError("A/B provenance differs; do not combine")
    for filename, roots in [
        ("p1_search_ab.csv", (args.norm_ab, args.ov_ab)),
        ("p1_candidate_parity.csv", (args.norm_ab, args.ov_ab)),
        ("mxfp8_e2_completed.csv", (args.norm, args.ov)),
    ]:
        rows = []
        for root in roots:
            with (root / filename).open(newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
        csv_rows(args.output / filename, rows)
        if filename != "p1_candidate_parity.csv":
            columns = list(rows[0])
            (args.output / filename.replace(".csv", ".md")).write_text(
                "# Recipe MXFP8 E2\n\n| "
                + " | ".join(columns)
                + " |\n| "
                + " | ".join(["---"] * len(columns))
                + " |\n"
                + "\n".join(
                    "| " + " | ".join(str(r[k]) for k in columns) + " |" for r in rows
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
