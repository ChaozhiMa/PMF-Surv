from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(
    "/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig/code"
)
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_external_centers import analyze_center, json_safe, normalize_id


BASELINE_ROOT = PROJECT_ROOT.parent / "comparison_baselines_20260901"
MAIN_ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / (
    "rapot_os_pfs_mtlr_iterative_asym_hard60_508_127_204_seed2026_"
    "retrain6_rerun1"
) / "external_center_analysis"
EXPLORATORY_SOURCE_ROOT = (
    MAIN_ANALYSIS_ROOT / "exploratory_synthetic_reassignment_20"
)
DEFAULT_OUTPUT = BASELINE_ROOT / "exploratory_centers_126_47_31"
CENTER_TABLES = (
    (
        "anhui_medical_university",
        "安医大探索表",
        EXPLORATORY_SOURCE_ROOT / "exploratory_anhui_after_move_126.csv",
        126,
    ),
    (
        "cancer_hospital",
        "肿瘤医院探索表",
        EXPLORATORY_SOURCE_ROOT / "exploratory_cancer_after_move_47.csv",
        47,
    ),
    (
        "yijishan_hospital",
        "弋矶山原表",
        MAIN_ANALYSIS_ROOT / "center_tables" / "yijishan_hospital_31.csv",
        31,
    ),
)




def load_baseline_module() -> Any:
    path = SCRIPT_ROOT / "run_clinical_mtlr_rsf_baselines.py"
    spec = importlib.util.spec_from_file_location("clinical_baselines", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_feature_data(
    baseline: Any,
) -> tuple[dict[str, Any], Any, pd.DataFrame, np.ndarray]:
    cfg = baseline.load_yaml(baseline.DEFAULT_CONFIG)
    checkpoint = torch.load(
        BASELINE_ROOT / "clinical_mtlr" / "best_internal_os.pth",
        map_location="cpu",
    )
    preprocessor = baseline.ClinicalPreprocessor.from_state_dict(
        checkpoint["preprocessor"]
    )
    train_rows = baseline.read_rows(cfg["data"]["train_csv"])
    train_frame = baseline.frame_from_rows(train_rows)
    train_x = baseline.clinical_matrix(train_rows, preprocessor)
    if len(train_frame) != 508 or train_x.shape != (508, 17):
        raise RuntimeError(
            f"Unexpected training data shape: frame={len(train_frame)}, features={train_x.shape}"
        )
    return cfg, checkpoint, train_frame, train_x


def mtlr_train_risks(
    baseline: Any,
    cfg: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    train_frame: pd.DataFrame,
    train_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state = checkpoint["model_state"]
    hidden_weight = state["os_head.net.1.weight"]
    output_weight = state["os_head.net.4.weight"]
    input_dim = int(hidden_weight.shape[1])
    hidden_dim = int(hidden_weight.shape[0])
    num_bins = int(output_weight.shape[0])
    model = baseline.ClinicalMTLR(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_bins=num_bins,
        dropout=0.0,
    )
    baseline.initialize_mtlr_heads(
        model,
        train_frame,
        num_bins,
        cfg,
        torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["model_state"])
    outputs = baseline.predict_mtlr(model, train_x, torch.device("cpu"))
    return outputs["os_risk"].reshape(-1), outputs["pfs_risk"].reshape(-1)


def rsf_train_risks(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    os_model = joblib.load(BASELINE_ROOT / "rsf" / "os_model.joblib")
    pfs_model = joblib.load(BASELINE_ROOT / "rsf" / "pfs_model.joblib")
    return os_model.predict(train_x), pfs_model.predict(train_x)


def validate_source_tables() -> dict[str, Any]:
    all_ids: list[str] = []
    manifest: dict[str, Any] = {}
    for center_key, center_name, path, expected_count in CENTER_TABLES:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"ID": str})
        frame["ID"] = frame["ID"].map(normalize_id)
        if len(frame) != expected_count or frame["ID"].duplicated().any():
            raise RuntimeError(f"Unexpected source table contents: {path}")
        all_ids.extend(frame["ID"].tolist())
        manifest[center_key] = {
            "center": center_name,
            "path": str(path),
            "patients": expected_count,
        }
    if len(all_ids) != 204 or len(set(all_ids)) != 204:
        raise RuntimeError("The three source tables must contain 204 distinct patients in total.")
    return manifest


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
        missing = sorted(set(table["ID"]) - set(selected["ID"]))
        raise RuntimeError(f"Prediction IDs do not match {table_path}: {missing[:5]}")
    selected = selected.set_index("ID").loc[table["ID"]].reset_index()
    for endpoint in ("OS", "PFS"):
        table_time = table[f"{endpoint}_time"].to_numpy(dtype=np.float64)
        prediction_time = selected[f"{endpoint}_time"].to_numpy(dtype=np.float64)
        if not np.allclose(table_time, prediction_time, rtol=0.0, atol=5.0e-4):
            raise RuntimeError(f"{endpoint} time mismatch in {table_path}")
        table_event = table[f"{endpoint}_status"].to_numpy(dtype=float) > 0.5
        prediction_event = (
            selected[f"{endpoint}_status"].to_numpy(dtype=float) > 0.5
        )
        if not np.array_equal(table_event, prediction_event):
            raise RuntimeError(f"{endpoint} event mismatch in {table_path}")
        table[f"{endpoint}_risk"] = selected[f"{endpoint}_risk"].to_numpy(
            dtype=np.float64
        )
    return table


def evaluate_model(
    model_name: str,
    train_risks: tuple[np.ndarray, np.ndarray],
    predictions_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    predictions = pd.read_csv(predictions_path, dtype={"ID": str})
    predictions["ID"] = predictions["ID"].map(normalize_id)
    if len(predictions) != 204 or predictions["ID"].duplicated().any():
        raise RuntimeError(f"Expected 204 unique external predictions: {predictions_path}")
    cutoffs = {
        "OS": float(np.median(np.asarray(train_risks[0], dtype=np.float64))),
        "PFS": float(np.median(np.asarray(train_risks[1], dtype=np.float64))),
    }
    metric_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    evaluated_ids: list[str] = []
    for center_key, center_name, table_path, expected_count in CENTER_TABLES:
        aligned = align_table_and_predictions(
            table_path,
            predictions,
            expected_count,
        )
        evaluated_ids.extend(aligned["ID"].tolist())
        metrics, center_auc = analyze_center(
            center_key,
            center_name,
            aligned,
            cutoffs,
        )
        metrics = {"model": model_name, **metrics}
        center_auc = [{"model": model_name, **row} for row in center_auc]
        metric_rows.append(metrics)
        auc_rows.extend(center_auc)
    if (
        len(evaluated_ids) != 204
        or len(set(evaluated_ids)) != 204
        or set(evaluated_ids) != set(predictions["ID"])
    ):
        raise RuntimeError(f"The source tables do not partition {model_name} predictions.")
    return metric_rows, auc_rows, cutoffs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate clinical MTLR and RSF on the 126/47/31 exploratory center tables."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = validate_source_tables()
    baseline = load_baseline_module()
    cfg, checkpoint, train_frame, train_x = load_feature_data(baseline)
    risks = {
        "clinical_mtlr": mtlr_train_risks(
            baseline,
            cfg,
            checkpoint,
            train_frame,
            train_x,
        ),
        "rsf": rsf_train_risks(train_x),
    }
    prediction_paths = {
        "clinical_mtlr": BASELINE_ROOT
        / "clinical_mtlr"
        / "external_validation_predictions.csv",
        "rsf": BASELINE_ROOT / "rsf" / "external_validation_predictions.csv",
    }

    all_metrics: list[dict[str, Any]] = []
    all_auc: list[dict[str, Any]] = []
    cutoffs: dict[str, dict[str, float]] = {}
    for model_name in ("clinical_mtlr", "rsf"):
        metric_rows, auc_rows, model_cutoffs = evaluate_model(
            model_name,
            risks[model_name],
            prediction_paths[model_name],
        )
        all_metrics.extend(metric_rows)
        all_auc.extend(auc_rows)
        cutoffs[model_name] = model_cutoffs
        pd.DataFrame(metric_rows).to_csv(
            args.output_dir / f"{model_name}_center_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(auc_rows).to_csv(
            args.output_dir / f"{model_name}_time_dependent_auc.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics_path = args.output_dir / "center_metrics_all_models.csv"
    auc_path = args.output_dir / "time_dependent_auc_all_models.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(all_auc).to_csv(auc_path, index=False, encoding="utf-8-sig")
    result = {
        "analysis_status": "exploratory center reassignment only; models and weights are frozen",
        "models": ["clinical_mtlr", "rsf"],
        "risk_grouping": {
            "rule": "model-specific endpoint training-set median risk, fixed for all three centers",
            "training_patients": 508,
            "cutoffs": cutoffs,
        },
        "source_tables": manifest,
        "prediction_sources": {
            name: {
                "path": str(path),
            }
            for name, path in prediction_paths.items()
        },
        "metrics": all_metrics,
        "time_dependent_auc": all_auc,
        "output_files": {
            "all_metrics": str(metrics_path),
            "all_time_dependent_auc": str(auc_path),
        },
    }
    result_path = args.output_dir / "analysis.json"
    result_path.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    display_columns = [
        "model",
        "center",
        "patients",
        "os_cindex",
        "os_hr_high_vs_low",
        "os_auc_12m",
        "os_auc_24m",
        "os_auc_36m",
        "pfs_cindex",
        "pfs_hr_high_vs_low",
        "pfs_auc_6m",
        "pfs_auc_12m",
        "pfs_auc_18m",
        "pfs_auc_24m",
        "pfs_auc_36m",
    ]
    print(pd.DataFrame(all_metrics)[display_columns].to_string(index=False), flush=True)
    print(f"Results: {result_path}", flush=True)


if __name__ == "__main__":
    main()
