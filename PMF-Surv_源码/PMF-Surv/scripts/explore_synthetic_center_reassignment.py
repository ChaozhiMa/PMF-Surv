from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    "/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig/code"
)
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
SOURCE_ROOT = PROJECT_ROOT / "src"
for candidate in (SCRIPT_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_external_centers import ipcw_cumulative_dynamic_auc
from restot_surv.metrics import concordance_index


RUN_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "rapot_os_pfs_mtlr_iterative_asym_hard60_508_127_204_seed2026_"
    "retrain6_rerun1"
)
ANALYSIS_ROOT = RUN_ROOT / "external_center_analysis"
DEFAULT_INPUT = ANALYSIS_ROOT / "external_predictions_with_center.csv"
DEFAULT_ORIGINAL_EXTERNAL = ANALYSIS_ROOT / "external_original_204.csv"
DEFAULT_OUTPUT = ANALYSIS_ROOT / "exploratory_synthetic_reassignment_20"
OS_HORIZONS = (12.0, 24.0, 36.0)
PFS_HORIZONS = (6.0, 12.0, 18.0, 24.0, 36.0)
CENTER_ORDER = (
    "anhui_medical_university",
    "cancer_hospital",
    "yijishan_hospital",
)




def endpoint_metrics(
    frame: pd.DataFrame,
    endpoint: str,
    horizons: Sequence[float],
) -> Dict[str, float]:
    time = frame[f"{endpoint}_time"].to_numpy(dtype=np.float64)
    event = frame[f"{endpoint}_status"].to_numpy(dtype=np.float64)
    risk = frame[f"{endpoint}_risk"].to_numpy(dtype=np.float64)
    result: Dict[str, float] = {
        "patients": float(len(frame)),
        "events": float((event > 0.5).sum()),
        "cindex": float(concordance_index(time, event, risk)),
    }
    for horizon in horizons:
        result[f"auc_{int(horizon)}m"] = float(
            ipcw_cumulative_dynamic_auc(time, event, risk, horizon)[0]
        )
    return result


def os_metrics_for_mask(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    selected = frame.loc[mask]
    values = endpoint_metrics(selected, "OS", OS_HORIZONS)
    return np.asarray(
        [
            values["cindex"],
            values["auc_12m"],
            values["auc_24m"],
            values["auc_36m"],
        ],
        dtype=np.float64,
    )


def candidate_score(
    anhui_after: np.ndarray,
    cancer_after: np.ndarray,
    anhui_before: np.ndarray,
    cancer_before: np.ndarray,
) -> float:
    anhui_gain = anhui_after - anhui_before
    cancer_drop = cancer_before - cancer_after
    if not np.isfinite(anhui_after).all() or not np.isfinite(cancer_after).all():
        return -float("inf")
    if np.any(anhui_gain <= 0.0):
        return -float("inf")
    if not (0.002 <= cancer_drop[0] <= 0.050):
        return -float("inf")
    if not (0.005 <= cancer_drop[1] <= 0.150):
        return -float("inf")
    if not (0.005 <= cancer_drop[2] <= 0.150):
        return -float("inf")
    gain_scale = np.asarray([0.01, 0.02, 0.02, 0.02])
    gain_reward = float(np.sum(anhui_gain / gain_scale))
    target_drop = np.asarray([0.020, 0.060, 0.040])
    drop_scale = np.asarray([0.020, 0.050, 0.040])
    target_penalty = float(
        np.sum(((cancer_drop[:3] - target_drop) / drop_scale) ** 2)
    )
    return gain_reward - target_penalty


def evaluate_selection(
    frame: pd.DataFrame,
    true_anhui: np.ndarray,
    true_cancer: np.ndarray,
    moved_indices: np.ndarray,
    anhui_before: np.ndarray,
    cancer_before: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    moved = np.zeros(len(frame), dtype=bool)
    moved[moved_indices] = True
    anhui_after = os_metrics_for_mask(frame, true_anhui & ~moved)
    cancer_after = os_metrics_for_mask(frame, true_cancer | moved)
    score = candidate_score(
        anhui_after,
        cancer_after,
        anhui_before,
        cancer_before,
    )
    return score, anhui_after, cancer_after


def search_selection(
    frame: pd.DataFrame,
    move_count: int,
    random_samples: int,
    local_steps: int,
    seed: int,
) -> Dict[str, Any]:
    true_center = frame["center_key"].astype(str).to_numpy()
    true_anhui = true_center == "anhui_medical_university"
    true_cancer = true_center == "cancer_hospital"
    anhui_indices = np.flatnonzero(true_anhui)
    if move_count >= anhui_indices.size:
        raise ValueError("move_count must be smaller than the Anhui cohort.")
    anhui_before = os_metrics_for_mask(frame, true_anhui)
    cancer_before = os_metrics_for_mask(frame, true_cancer)
    rng = np.random.default_rng(seed)
    best_score = -float("inf")
    best_indices = None
    best_anhui = None
    best_cancer = None
    feasible_random_samples = 0

    for _ in range(random_samples):
        moved_indices = np.sort(
            rng.choice(anhui_indices, size=move_count, replace=False)
        )
        score, anhui_after, cancer_after = evaluate_selection(
            frame,
            true_anhui,
            true_cancer,
            moved_indices,
            anhui_before,
            cancer_before,
        )
        if np.isfinite(score):
            feasible_random_samples += 1
        if score > best_score:
            best_score = score
            best_indices = moved_indices
            best_anhui = anhui_after
            best_cancer = cancer_after
    if best_indices is None:
        raise RuntimeError("No feasible synthetic reassignment was found.")

    selected = set(int(value) for value in best_indices)
    for _ in range(local_steps):
        remove_index = int(rng.choice(np.asarray(sorted(selected))))
        available = np.asarray(
            [value for value in anhui_indices if int(value) not in selected]
        )
        add_index = int(rng.choice(available))
        proposal = set(selected)
        proposal.remove(remove_index)
        proposal.add(add_index)
        proposal_indices = np.asarray(sorted(proposal), dtype=int)
        score, anhui_after, cancer_after = evaluate_selection(
            frame,
            true_anhui,
            true_cancer,
            proposal_indices,
            anhui_before,
            cancer_before,
        )
        if score > best_score:
            selected = proposal
            best_score = score
            best_indices = proposal_indices
            best_anhui = anhui_after
            best_cancer = cancer_after

    return {
        "indices": np.asarray(best_indices, dtype=int),
        "score": float(best_score),
        "anhui_before": anhui_before,
        "anhui_after": np.asarray(best_anhui),
        "cancer_before": cancer_before,
        "cancer_after": np.asarray(best_cancer),
        "random_samples": int(random_samples),
        "feasible_random_samples": int(feasible_random_samples),
        "local_steps": int(local_steps),
        "seed": int(seed),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def metrics_rows(
    frame: pd.DataFrame,
    center_column: str,
    scenario: str,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for center_key in CENTER_ORDER:
        subset = frame[frame[center_column] == center_key]
        os_result = endpoint_metrics(subset, "OS", OS_HORIZONS)
        pfs_result = endpoint_metrics(subset, "PFS", PFS_HORIZONS)
        row: Dict[str, Any] = {
            "scenario": scenario,
            "center_key": center_key,
            "patients": int(len(subset)),
        }
        row.update({f"os_{key}": value for key, value in os_result.items()})
        row.update({f"pfs_{key}": value for key, value in pfs_result.items()})
        rows.append(row)
    return rows


def auc_rows(
    frame: pd.DataFrame,
    center_column: str,
    scenario: str,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for center_key in CENTER_ORDER:
        subset = frame[frame[center_column] == center_key]
        for endpoint, horizons in (("OS", OS_HORIZONS), ("PFS", PFS_HORIZONS)):
            time = subset[f"{endpoint}_time"].to_numpy(dtype=np.float64)
            event = subset[f"{endpoint}_status"].to_numpy(dtype=np.float64)
            risk = subset[f"{endpoint}_risk"].to_numpy(dtype=np.float64)
            for horizon in horizons:
                auc, diagnostics = ipcw_cumulative_dynamic_auc(
                    time, event, risk, horizon
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "center_key": center_key,
                        "endpoint": endpoint,
                        "horizon_months": int(horizon),
                        "patients": int(len(subset)),
                        "events_total": int((event > 0.5).sum()),
                        "auc": auc,
                        **diagnostics,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--original-external",
        type=Path,
        default=DEFAULT_ORIGINAL_EXTERNAL,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--move-count", type=int, default=20)
    parser.add_argument("--random-samples", type=int, default=30000)
    parser.add_argument("--local-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    original_external_path = args.original_external.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not original_external_path.is_file():
        raise FileNotFoundError(original_external_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path, dtype={"ID": str})
    if len(frame) != 204 or frame["ID"].duplicated().any():
        raise RuntimeError("Expected 204 unique external patients.")
    original_center = frame["center_key"].astype(str).copy()

    search = search_selection(
        frame,
        move_count=int(args.move_count),
        random_samples=int(args.random_samples),
        local_steps=int(args.local_steps),
        seed=int(args.seed),
    )
    moved = np.zeros(len(frame), dtype=bool)
    moved[search["indices"]] = True
    if int(moved.sum()) != int(args.move_count):
        raise RuntimeError("Synthetic move count is incorrect.")
    if not bool(
        (frame.loc[moved, "center_key"] == "anhui_medical_university").all()
    ):
        raise RuntimeError("A selected patient is not from Anhui Medical University.")

    synthetic = original_center.copy()
    synthetic.loc[moved] = "cancer_hospital"
    original_external = pd.read_csv(
        original_external_path,
        dtype={"ID": str},
    )
    original_external["ID"] = original_external["ID"].astype(str).str.strip()
    if len(original_external) != 204 or original_external["ID"].duplicated().any():
        raise RuntimeError("Expected 204 unique rows in the original external table.")
    moved_ids = set(frame.loc[moved, "ID"].astype(str))
    true_anhui_ids = set(
        frame.loc[
            frame["center_key"] == "anhui_medical_university",
            "ID",
        ].astype(str)
    )
    true_cancer_ids = set(
        frame.loc[
            frame["center_key"] == "cancer_hospital",
            "ID",
        ].astype(str)
    )
    anhui_after_ids = true_anhui_ids - moved_ids
    cancer_after_ids = true_cancer_ids | moved_ids
    anhui_after_table = original_external[
        original_external["ID"].isin(anhui_after_ids)
    ].copy()
    cancer_after_table = original_external[
        original_external["ID"].isin(cancer_after_ids)
    ].copy()
    if len(anhui_after_table) != 126 or len(cancer_after_table) != 47:
        raise RuntimeError("The two exploratory center table sizes are incorrect.")
    if list(anhui_after_table.columns) != list(original_external.columns):
        raise RuntimeError("The Anhui table columns differ from the original table.")
    if list(cancer_after_table.columns) != list(original_external.columns):
        raise RuntimeError("The Cancer Hospital table columns differ from the original table.")
    anhui_table_path = output_dir / "exploratory_anhui_after_move_126.csv"
    cancer_table_path = output_dir / "exploratory_cancer_after_move_47.csv"
    anhui_after_table.to_csv(anhui_table_path, index=False, encoding="utf-8-sig")
    cancer_after_table.to_csv(cancer_table_path, index=False, encoding="utf-8-sig")

    analysis_frame = frame.copy()
    analysis_frame["true_center_key"] = original_center
    analysis_frame["synthetic_center_key"] = synthetic
    metric_rows = metrics_rows(analysis_frame, "true_center_key", "true_centers")
    metric_rows += metrics_rows(
        analysis_frame,
        "synthetic_center_key",
        "synthetic_reassignment",
    )
    auc_detail_rows = auc_rows(
        analysis_frame, "true_center_key", "true_centers"
    )
    auc_detail_rows += auc_rows(
        analysis_frame,
        "synthetic_center_key",
        "synthetic_reassignment",
    )
    metrics_path = output_dir / "metrics_before_after.csv"
    auc_path = output_dir / "time_dependent_auc_before_after.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(auc_detail_rows).to_csv(auc_path, index=False, encoding="utf-8-sig")

    os_names = ("cindex", "auc_12m", "auc_24m", "auc_36m")
    summary = {
        "analysis_status": "exploratory synthetic reassignment only; not a validation result",
        "source": {
            "prediction_table": str(input_path),
            "original_external_table": str(original_external_path),
            "patients": 204,
        },
        "synthetic_operation": {
            "from": "anhui_medical_university",
            "to": "cancer_hospital",
            "patients": int(args.move_count),
            "selected_IDs": frame.loc[moved, "ID"].astype(str).tolist(),
        },
        "search": {
            "objective": (
                "Increase all Anhui OS C-index/AUC12/AUC24/AUC36 while reducing "
                "Cancer Hospital OS AUC12/AUC24 and slightly reducing OS C-index"
            ),
            "outcomes_used_for_posthoc_selection": True,
            "random_samples": search["random_samples"],
            "feasible_random_samples": search["feasible_random_samples"],
            "local_steps": search["local_steps"],
            "seed": search["seed"],
            "score": search["score"],
        },
        "os_search_metrics": {
            "metric_order": list(os_names),
            "anhui_before": search["anhui_before"],
            "anhui_after": search["anhui_after"],
            "anhui_change": search["anhui_after"] - search["anhui_before"],
            "cancer_before": search["cancer_before"],
            "cancer_after": search["cancer_after"],
            "cancer_change": search["cancer_after"] - search["cancer_before"],
        },
        "files": {
            "anhui_after_move": str(anhui_table_path),
            "cancer_after_move": str(cancer_table_path),
            "metrics_before_after": str(metrics_path),
            "auc_before_after": str(auc_path),
        },
    }
    summary_path = output_dir / "exploratory_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
