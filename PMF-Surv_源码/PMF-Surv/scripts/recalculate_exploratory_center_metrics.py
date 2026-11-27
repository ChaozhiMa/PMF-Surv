from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    "/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig/code"
)
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_external_centers import analyze_center, json_safe, normalize_id


RUN_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "rapot_os_pfs_mtlr_iterative_asym_hard60_508_127_204_seed2026_"
    "retrain6_rerun1"
)
ANALYSIS_ROOT = RUN_ROOT / "external_center_analysis"
EXPLORATORY_ROOT = ANALYSIS_ROOT / "exploratory_synthetic_reassignment_20"
PREDICTIONS = ANALYSIS_ROOT / "external_predictions_with_center.csv"
REFERENCE_RESULTS = ANALYSIS_ROOT / "center_metrics.json"
CENTER_TABLES = (
    (
        "anhui_medical_university",
        "安医大",
        EXPLORATORY_ROOT / "exploratory_anhui_after_move_126.csv",
        126,
    ),
    (
        "cancer_hospital",
        "肿瘤医院",
        EXPLORATORY_ROOT / "exploratory_cancer_after_move_47.csv",
        47,
    ),
    (
        "yijishan_hospital",
        "弋矶山",
        ANALYSIS_ROOT / "center_tables" / "yijishan_hospital_31.csv",
        31,
    ),
)




def align_table_and_predictions(
    table_path: Path,
    predictions: pd.DataFrame,
    expected_count: int,
) -> pd.DataFrame:
    table = pd.read_csv(table_path, dtype={"ID": str})
    table["ID"] = table["ID"].map(normalize_id)
    if len(table) != expected_count or table["ID"].duplicated().any():
        raise RuntimeError(f"Unexpected patients in {table_path}")
    selected = predictions[predictions["ID"].isin(set(table["ID"]))].copy()
    if set(selected["ID"]) != set(table["ID"]):
        raise RuntimeError(f"Prediction IDs do not match {table_path}")
    selected = selected.set_index("ID").loc[table["ID"]].reset_index()
    for endpoint in ("OS", "PFS"):
        if not np.allclose(
            table[f"{endpoint}_time"].to_numpy(dtype=np.float64),
            selected[f"{endpoint}_time"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=5.0e-4,
        ):
            raise RuntimeError(f"{endpoint} time mismatch in {table_path}")
        table_event = table[f"{endpoint}_status"].to_numpy(dtype=float) > 0.5
        predicted_event = (
            selected[f"{endpoint}_status"].to_numpy(dtype=float) > 0.5
        )
        if not np.array_equal(table_event, predicted_event):
            raise RuntimeError(f"{endpoint} event mismatch in {table_path}")
        table[f"{endpoint}_risk"] = selected[f"{endpoint}_risk"].to_numpy()
    return table


def main() -> None:
    required = [PREDICTIONS, REFERENCE_RESULTS]
    required.extend(spec[2] for spec in CENTER_TABLES)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(PREDICTIONS, dtype={"ID": str})
    predictions["ID"] = predictions["ID"].map(normalize_id)
    if len(predictions) != 204 or predictions["ID"].duplicated().any():
        raise RuntimeError("Frozen predictions must contain 204 unique patients.")
    reference = json.loads(REFERENCE_RESULTS.read_text(encoding="utf-8"))
    cutoffs = {
        endpoint: float(reference["risk_grouping"]["cutoffs"][endpoint])
        for endpoint in ("OS", "PFS")
    }

    metric_rows: list[Dict[str, Any]] = []
    auc_rows: list[Dict[str, Any]] = []
    all_ids: list[str] = []
    table_manifest: Dict[str, Any] = {}
    for center_key, center_name, table_path, expected_count in CENTER_TABLES:
        aligned = align_table_and_predictions(
            table_path,
            predictions,
            expected_count,
        )
        all_ids.extend(aligned["ID"].tolist())
        metrics, center_auc_rows = analyze_center(
            center_key,
            center_name,
            aligned,
            cutoffs,
        )
        metric_rows.append(metrics)
        auc_rows.extend(center_auc_rows)
        table_manifest[center_key] = {
            "center": center_name,
            "path": str(table_path),
            "patients": expected_count,
        }
    if (
        len(all_ids) != 204
        or len(set(all_ids)) != 204
        or set(all_ids) != set(predictions["ID"])
    ):
        raise RuntimeError("The three center tables do not partition all 204 patients.")

    metrics_path = EXPLORATORY_ROOT / "all_metrics_recalculated.csv"
    auc_path = EXPLORATORY_ROOT / "all_time_dependent_auc_recalculated.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(auc_rows).to_csv(auc_path, index=False, encoding="utf-8-sig")
    result = {
        "analysis_status": "exploratory synthetic reassignment only; not a validation result",
        "prediction_source": {
            "path": str(PREDICTIONS),
        },
        "risk_grouping": {
            "rule": "endpoint-specific training-set median risk, fixed for all centers",
            "cutoffs": cutoffs,
        },
        "center_tables": table_manifest,
        "methods": reference["methods"],
        "center_metrics": metric_rows,
        "files": {
            "metrics": str(metrics_path),
            "time_dependent_auc": str(auc_path),
        },
    }
    result_path = EXPLORATORY_ROOT / "all_metrics_recalculated.json"
    result_path.write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(pd.DataFrame(metric_rows).to_string(index=False), flush=True)
    print(f"Results: {result_path}", flush=True)


if __name__ == "__main__":
    main()
