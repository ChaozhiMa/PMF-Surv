from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from restot_surv.calibration import cox_probability_metrics, load_center_patient_ids
from restot_surv.config import load_config


def read_predictions(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        records = list(reader)
    results = {}
    for endpoint in ("os", "pfs"):
        for kind in ("time", "event", "risk"):
            field = f"{endpoint}_{kind}"
            if field not in fields and kind == "event":
                field = f"{endpoint}_status"
            if field not in fields:
                raise ValueError(f"{path} lacks column {field}.")
            results[f"{endpoint}_{kind}"] = np.asarray(
                [float(row[field]) for row in records], dtype=np.float64,
            )
    id_field = next((name for name in ("patient_id", "ID") if name in fields), None)
    ids = [str(row[id_field]).strip() for row in records] if id_field else []
    return results, ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Cox-recalibrated survival probabilities from frozen risk scores.",
    )
    parser.add_argument("--internal-validation", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Read external center membership from this config.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    validation, _ = read_predictions(args.internal_validation)
    external, ids = read_predictions(args.external)
    center_ids = {}
    if args.config:
        cfg = load_config(args.config)
        specs = cfg["data"].get("external_centers", {})
        if not specs:
            specs = cfg.get("evaluation", {}).get("external_centers", {})
        center_ids = load_center_patient_ids(
            specs, str(cfg["data"]["patient_id_col"]), PROJECT_ROOT,
        )
    scores = cox_probability_metrics(
        validation, external, ids, center_ids, bootstrap=args.bootstrap,
    )
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
