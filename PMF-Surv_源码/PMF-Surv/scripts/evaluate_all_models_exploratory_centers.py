from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2


PROJECT_ROOT = Path(
    "/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig/code"
)
BASELINE_ROOT = PROJECT_ROOT.parent / "comparison_baselines_20260901"
MAIN_RUN_ROOT = PROJECT_ROOT / "outputs" / (
    "rapot_os_pfs_mtlr_iterative_asym_hard60_508_127_204_seed2026_"
    "retrain6_rerun1"
)
MAIN_ANALYSIS_ROOT = MAIN_RUN_ROOT / "external_center_analysis"
EXPLORATORY_SOURCE_ROOT = (
    MAIN_ANALYSIS_ROOT / "exploratory_synthetic_reassignment_20"
)
DEFAULT_OUTPUT = BASELINE_ROOT / "exploratory_centers_126_47_31"
HORIZONS_MONTHS = (12.0, 24.0, 36.0)
DEFAULT_BOOTSTRAP_REPLICATES = 1000
DEFAULT_BOOTSTRAP_SEED = 2026


@dataclass(frozen=True)
class CenterSpec:
    key: str
    name: str
    table_path: Path
    expected_patients: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    prediction_path: Path
    train_prediction_path: Path | None = None
    cutoff_json_path: Path | None = None
    cutoff_json_model_key: str | None = None


CENTERS = (
    CenterSpec(
        "anhui_medical_university",
        "安医大探索表",
        EXPLORATORY_SOURCE_ROOT / "exploratory_anhui_after_move_126.csv",
        126,
    ),
    CenterSpec(
        "cancer_hospital",
        "肿瘤医院探索表",
        EXPLORATORY_SOURCE_ROOT / "exploratory_cancer_after_move_47.csv",
        47,
    ),
    CenterSpec(
        "yijishan_hospital",
        "弋矶山原表",
        MAIN_ANALYSIS_ROOT / "center_tables" / "yijishan_hospital_31.csv",
        31,
    ),
)


MODELS = (
    ModelSpec(
        "clinical_mtlr",
        "Clinical MTLR",
        BASELINE_ROOT / "clinical_mtlr" / "external_validation_predictions.csv",
        cutoff_json_path=DEFAULT_OUTPUT / "analysis.json",
        cutoff_json_model_key="clinical_mtlr",
    ),
    ModelSpec(
        "rsf",
        "RSF",
        BASELINE_ROOT / "rsf" / "external_validation_predictions.csv",
        cutoff_json_path=DEFAULT_OUTPUT / "analysis.json",
        cutoff_json_model_key="rsf",
    ),
    ModelSpec(
        "resnet18_3phase_mtlr",
        "3D ResNet-18 + MTLR",
        BASELINE_ROOT
        / "resnet18_3phase_mtlr"
        / "external_validation_predictions.csv",
        train_prediction_path=BASELINE_ROOT
        / "resnet18_3phase_mtlr"
        / "train_predictions.csv",
    ),
    ModelSpec(
        "daft_mtlr",
        "DAFT + MTLR",
        BASELINE_ROOT / "daft_mtlr" / "external_validation_predictions.csv",
        train_prediction_path=BASELINE_ROOT / "daft_mtlr" / "train_predictions.csv",
    ),
    ModelSpec(
        "cmc_surv_mtlr",
        "CMC-Surv + MTLR",
        BASELINE_ROOT / "cmc_surv_mtlr" / "external_validation_predictions.csv",
        train_prediction_path=BASELINE_ROOT
        / "cmc_surv_mtlr"
        / "train_predictions.csv",
    ),
    ModelSpec(
        "lite_prosenet",
        "Lite-ProSENet",
        BASELINE_ROOT / "lite_prosenet" / "external_validation_predictions.csv",
        train_prediction_path=BASELINE_ROOT
        / "lite_prosenet"
        / "train_predictions.csv",
    ),
    ModelSpec(
        "proposed_model",
        "Proposed model (Ours)",
        MAIN_ANALYSIS_ROOT / "external_predictions_with_center.csv",
        cutoff_json_path=MAIN_ANALYSIS_ROOT / "center_metrics.json",
    ),
)


def normalize_id(value: Any) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"([0-9]+)\.0+", text)
    return match.group(1) if match else text




def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def display_value(value: Any) -> Any:
    if value is None:
        return "/"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "/"
    return value


def concordance_index(
    times: Sequence[float], events: Sequence[float], risks: Sequence[float]
) -> float:
    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=np.float64) > 0.5
    risk = np.asarray(risks, dtype=np.float64)
    comparable = 0
    concordant = 0.0
    for index in np.flatnonzero(event):
        mask = time > time[index]
        comparable += int(mask.sum())
        differences = risk[index] - risk[mask]
        concordant += float((differences > 0.0).sum())
        concordant += 0.5 * float((differences == 0.0).sum())
    return float(concordant / comparable) if comparable else float("nan")


def fit_censoring_km(
    times: Sequence[float], events: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(times, dtype=np.float64).reshape(-1)
    event = np.asarray(events, dtype=np.float64).reshape(-1) > 0.5
    unique_times = np.unique(time)
    survival = 1.0
    values: list[float] = []
    for point in unique_times:
        at_risk = int(np.sum(time >= point))
        censored = int(np.sum((time == point) & ~event))
        if censored:
            survival *= 1.0 - censored / at_risk
        values.append(survival)
    return unique_times, np.asarray(values, dtype=np.float64)


def censoring_survival_at(
    query: Sequence[float],
    censoring_times: Sequence[float],
    censoring_survival: Sequence[float],
) -> np.ndarray:
    points = np.asarray(query, dtype=np.float64)
    times = np.asarray(censoring_times, dtype=np.float64)
    survival = np.asarray(censoring_survival, dtype=np.float64)
    indices = np.searchsorted(times, points, side="left") - 1
    result = np.ones_like(points)
    valid = indices >= 0
    result[valid] = survival[indices[valid]]
    return result


def ipcw_cumulative_dynamic_auc(
    times: Sequence[float],
    events: Sequence[float],
    risks: Sequence[float],
    horizon: float,
) -> tuple[float, dict[str, int]]:
    time = np.asarray(times, dtype=np.float64).reshape(-1)
    event = np.asarray(events, dtype=np.float64).reshape(-1) > 0.5
    risk = np.asarray(risks, dtype=np.float64).reshape(-1)
    cases = (time <= horizon) & event
    controls = time > horizon
    diagnostics = {
        "cases": int(cases.sum()),
        "controls": int(controls.sum()),
        "censored_before_or_at_horizon": int(np.sum((time <= horizon) & ~event)),
    }
    if cases.sum() == 0 or controls.sum() == 0:
        return float("nan"), diagnostics
    censor_times, censor_survival = fit_censoring_km(time, event)
    case_g = censoring_survival_at(time[cases], censor_times, censor_survival)
    if np.any(case_g <= 1.0e-6):
        return float("nan"), diagnostics
    weights = 1.0 / case_g
    differences = risk[cases, None] - risk[controls][None, :]
    concordance = (differences > 0.0).astype(np.float64)
    concordance += 0.5 * (differences == 0.0)
    numerator = float(np.sum(weights[:, None] * concordance))
    denominator = float(np.sum(weights) * controls.sum())
    return numerator / denominator, diagnostics


def cox_binary_hr(
    times: Sequence[float],
    events: Sequence[float],
    high_group: Sequence[bool],
) -> tuple[float, float, float]:
    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=np.float64) > 0.5
    high = np.asarray(high_group, dtype=np.float64)
    if high.min() == high.max() or event.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    event_times = np.unique(time[event])
    risk_sets = time[None, :] >= event_times[:, None]
    death_sets = (time[None, :] == event_times[:, None]) & event[None, :]
    n_total = risk_sets.sum(axis=1).astype(np.float64)
    n_high = risk_sets @ high
    n_low = n_total - n_high
    deaths_total = death_sets.sum(axis=1).astype(np.float64)
    deaths_high = death_sets @ high

    def score_information(beta: float) -> tuple[float, float]:
        exp_beta = float(np.exp(np.clip(beta, -30.0, 30.0)))
        denominator = n_low + n_high * exp_beta
        if np.any(denominator <= 0.0):
            return float("nan"), float("nan")
        probability_high = n_high * exp_beta / denominator
        score = np.sum(deaths_high - deaths_total * probability_high)
        information = np.sum(
            deaths_total * probability_high * (1.0 - probability_high)
        )
        return float(score), float(information)

    lower_score = score_information(-30.0)[0]
    upper_score = score_information(30.0)[0]
    if not (
        np.isfinite(lower_score)
        and np.isfinite(upper_score)
        and lower_score > 0.0
        and upper_score < 0.0
    ):
        return float("nan"), float("nan"), float("nan")
    beta = brentq(lambda value: score_information(value)[0], -30.0, 30.0)
    information = score_information(beta)[1]
    if information <= 0.0:
        return float("nan"), float("nan"), float("nan")
    standard_error = 1.0 / np.sqrt(information)
    return (
        float(np.exp(beta)),
        float(np.exp(beta - 1.9599639845 * standard_error)),
        float(np.exp(beta + 1.9599639845 * standard_error)),
    )


def percentile_interval(
    estimates: Sequence[float],
    point_estimate: float,
    minimum_valid: int,
) -> tuple[float, float, int]:
    values = np.asarray(estimates, dtype=np.float64)
    values = values[np.isfinite(values)]
    valid = int(values.size)
    if not np.isfinite(point_estimate) or valid < minimum_valid:
        return float("nan"), float("nan"), valid
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper), valid


def bootstrap_intervals(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    high_group: np.ndarray,
    point_estimates: Mapping[str, float],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    n = int(len(times))
    samples: dict[str, list[float]] = {
        "cindex": [],
        "hr": [],
        **{f"auc_{int(horizon)}m": [] for horizon in HORIZONS_MONTHS},
    }
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        boot_time = times[indices]
        boot_event = events[indices]
        boot_risk = risks[indices]
        boot_high = high_group[indices]
        samples["cindex"].append(
            concordance_index(boot_time, boot_event, boot_risk)
        )
        samples["hr"].append(
            cox_binary_hr(boot_time, boot_event, boot_high)[0]
        )
        for horizon in HORIZONS_MONTHS:
            auc, _ = ipcw_cumulative_dynamic_auc(
                boot_time, boot_event, boot_risk, horizon
            )
            samples[f"auc_{int(horizon)}m"].append(auc)

    minimum_valid = max(200, int(np.ceil(replicates * 0.5)))
    intervals: dict[str, dict[str, float | int]] = {}
    for metric, estimates in samples.items():
        lower, upper, valid = percentile_interval(
            estimates,
            float(point_estimates[metric]),
            minimum_valid,
        )
        intervals[metric] = {
            "lower": lower,
            "upper": upper,
            "valid_replicates": valid,
        }
    return intervals


def logrank_p_value(
    times: Sequence[float],
    events: Sequence[float],
    high_group: Sequence[bool],
) -> float:
    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=np.float64) > 0.5
    high = np.asarray(high_group, dtype=bool)
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for point in np.unique(time[event]):
        risk_set = time >= point
        n = float(risk_set.sum())
        n_high = float((risk_set & high).sum())
        deaths = (time == point) & event
        deaths_total = float(deaths.sum())
        observed += float((deaths & high).sum())
        expected += deaths_total * n_high / n
        if n > 1.0:
            variance += (
                deaths_total
                * n_high
                * (n - n_high)
                * (n - deaths_total)
                / (n * n * (n - 1.0))
            )
    if variance <= 0.0:
        return float("nan")
    return float(chi2.sf((observed - expected) ** 2 / variance, 1))


def load_center_table(center: CenterSpec) -> pd.DataFrame:
    frame = pd.read_csv(center.table_path, dtype={"ID": str})
    frame["ID"] = frame["ID"].map(normalize_id)
    if len(frame) != center.expected_patients:
        raise RuntimeError(
            f"{center.name} expected {center.expected_patients} patients, got {len(frame)}"
        )
    if frame["ID"].duplicated().any():
        raise RuntimeError(f"Duplicated patient ID inside {center.name}")
    required = {
        "ID",
        "OS_time",
        "OS_status",
        "PFS_time",
        "PFS_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {center.table_path}: {sorted(missing)}")
    return frame


def load_predictions(model: ModelSpec) -> pd.DataFrame:
    frame = pd.read_csv(model.prediction_path, dtype={"ID": str})
    frame["ID"] = frame["ID"].map(normalize_id)
    if frame["ID"].duplicated().any():
        raise RuntimeError(f"Duplicated patient ID in {model.prediction_path}")
    required = {
        "ID",
        "OS_time",
        "OS_status",
        "OS_risk",
        "PFS_time",
        "PFS_status",
        "PFS_risk",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {model.prediction_path}: {sorted(missing)}")
    return frame


def load_cutoffs(model: ModelSpec) -> tuple[dict[str, float], str]:
    if model.train_prediction_path is not None:
        train = pd.read_csv(model.train_prediction_path)
        cutoffs = {
            endpoint: float(np.median(train[f"{endpoint}_risk"].to_numpy(float)))
            for endpoint in ("OS", "PFS")
        }
        return cutoffs, f"training prediction median: {model.train_prediction_path}"
    if model.cutoff_json_path is None:
        raise RuntimeError(f"No cutoff source for {model.key}")
    payload = json.loads(model.cutoff_json_path.read_text(encoding="utf-8-sig"))
    if model.cutoff_json_model_key is None:
        values = payload["risk_grouping"]["cutoffs"]
    else:
        values = payload["risk_grouping"]["cutoffs"][model.cutoff_json_model_key]
    cutoffs = {endpoint: float(values[endpoint]) for endpoint in ("OS", "PFS")}
    return cutoffs, f"stored training-set median: {model.cutoff_json_path}"


def verify_proposed_checkpoint() -> dict[str, Any]:
    reference_path = MAIN_ANALYSIS_ROOT / "center_metrics.json"
    payload = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    checkpoint = payload["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    if checkpoint_path.name != "best_os.pth":
        raise RuntimeError(
            f"Proposed model must use best_os.pth, got {checkpoint_path.name}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    prediction_path = Path(payload["files"]["predictions_with_center"])
    expected_prediction_path = MAIN_ANALYSIS_ROOT / "external_predictions_with_center.csv"
    if prediction_path.resolve() != expected_prediction_path.resolve():
        raise RuntimeError(
            "Proposed-model prediction file is not the internal-OS-selected export"
        )
    return {
        "path": str(checkpoint_path),
        "prediction_path": str(prediction_path),
    }


def align_center_predictions(
    center: CenterSpec,
    center_frame: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    selected = predictions[predictions["ID"].isin(set(center_frame["ID"]))].copy()
    if set(selected["ID"]) != set(center_frame["ID"]):
        missing = sorted(set(center_frame["ID"]) - set(selected["ID"]))
        raise RuntimeError(
            f"{center.name} lacks predictions for {len(missing)} patients: {missing[:5]}"
        )
    selected = selected.set_index("ID").loc[center_frame["ID"]].reset_index()
    aligned = center_frame.copy()
    for endpoint in ("OS", "PFS"):
        observed_time = aligned[f"{endpoint}_time"].to_numpy(float)
        predicted_time = selected[f"{endpoint}_time"].to_numpy(float)
        if not np.allclose(observed_time, predicted_time, rtol=0.0, atol=5.0e-4):
            raise RuntimeError(f"{center.name} {endpoint} time mismatch")
        observed_event = aligned[f"{endpoint}_status"].to_numpy(float) > 0.5
        predicted_event = selected[f"{endpoint}_status"].to_numpy(float) > 0.5
        if not np.array_equal(observed_event, predicted_event):
            raise RuntimeError(f"{center.name} {endpoint} event mismatch")
        aligned[f"{endpoint}_risk"] = selected[f"{endpoint}_risk"].to_numpy(float)
    return aligned


def evaluate_endpoint(
    model: ModelSpec,
    center: CenterSpec,
    frame: pd.DataFrame,
    endpoint: str,
    cutoff: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    time = frame[f"{endpoint}_time"].to_numpy(float)
    event = frame[f"{endpoint}_status"].to_numpy(float)
    risk = frame[f"{endpoint}_risk"].to_numpy(float)
    high = risk >= cutoff
    cindex = concordance_index(time, event, risk)
    hr = cox_binary_hr(time, event, high)[0]
    row: dict[str, Any] = {
        "model_key": model.key,
        "model": model.name,
        "center_key": center.key,
        "center": center.name,
        "patients": int(len(frame)),
        "endpoint": endpoint,
        "events": int((event > 0.5).sum()),
        "cindex": cindex,
        "training_median_cutoff": cutoff,
        "low_risk_n": int((~high).sum()),
        "high_risk_n": int(high.sum()),
        "hr_high_vs_low": hr,
        "logrank_p": logrank_p_value(time, event, high),
    }
    auc_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS_MONTHS:
        auc, diagnostics = ipcw_cumulative_dynamic_auc(time, event, risk, horizon)
        row[f"auc_{int(horizon)}m"] = auc
        auc_rows.append(
            {
                "model_key": model.key,
                "model": model.name,
                "center_key": center.key,
                "center": center.name,
                "patients": int(len(frame)),
                "endpoint": endpoint,
                "events": int((event > 0.5).sum()),
                "horizon_months": int(horizon),
                "auc": auc,
                **diagnostics,
            }
        )
    point_estimates = {
        "cindex": cindex,
        "hr": hr,
        **{
            f"auc_{int(horizon)}m": float(row[f"auc_{int(horizon)}m"])
            for horizon in HORIZONS_MONTHS
        },
    }
    intervals = bootstrap_intervals(
        time,
        event,
        risk,
        high,
        point_estimates,
        bootstrap_replicates,
        bootstrap_seed,
    )
    row.update(
        {
            "cindex_ci_lower": intervals["cindex"]["lower"],
            "cindex_ci_upper": intervals["cindex"]["upper"],
            "cindex_bootstrap_valid": intervals["cindex"]["valid_replicates"],
            "hr_ci_lower": intervals["hr"]["lower"],
            "hr_ci_upper": intervals["hr"]["upper"],
            "hr_bootstrap_valid": intervals["hr"]["valid_replicates"],
        }
    )
    for horizon in HORIZONS_MONTHS:
        metric = f"auc_{int(horizon)}m"
        row[f"{metric}_ci_lower"] = intervals[metric]["lower"]
        row[f"{metric}_ci_upper"] = intervals[metric]["upper"]
        row[f"{metric}_bootstrap_valid"] = intervals[metric]["valid_replicates"]
    for auc_row in auc_rows:
        metric = f"auc_{int(auc_row['horizon_months'])}m"
        auc_row["auc_ci_lower"] = intervals[metric]["lower"]
        auc_row["auc_ci_upper"] = intervals[metric]["upper"]
        auc_row["bootstrap_valid"] = intervals[metric]["valid_replicates"]
    return row, auc_rows


def formatted_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    output = raw_results.copy()
    output["C-index (95% CI)"] = output.apply(
        lambda row: "/"
        if not (
            np.isfinite(float(row["cindex"]))
            and np.isfinite(float(row["cindex_ci_lower"]))
            and np.isfinite(float(row["cindex_ci_upper"]))
        )
        else (
            f"{row['cindex']:.3f} "
            f"({row['cindex_ci_lower']:.3f}-{row['cindex_ci_upper']:.3f})"
        ),
        axis=1,
    )
    output["HR (95% CI)"] = output.apply(
        lambda row: "/"
        if not (
            np.isfinite(float(row["hr_high_vs_low"]))
            and np.isfinite(float(row["hr_ci_lower"]))
            and np.isfinite(float(row["hr_ci_upper"]))
        )
        else (
            f"{row['hr_high_vs_low']:.3f} "
            f"({row['hr_ci_lower']:.3f}-{row['hr_ci_upper']:.3f})"
        ),
        axis=1,
    )
    for horizon in HORIZONS_MONTHS:
        months = int(horizon)
        output[f"AUC {months // 12}-year (95% CI)"] = output.apply(
            lambda row, metric=f"auc_{months}m": "/"
            if not (
                np.isfinite(float(row[metric]))
                and np.isfinite(float(row[f"{metric}_ci_lower"]))
                and np.isfinite(float(row[f"{metric}_ci_upper"]))
            )
            else (
                f"{row[metric]:.3f} "
                f"({row[f'{metric}_ci_lower']:.3f}-"
                f"{row[f'{metric}_ci_upper']:.3f})"
            ),
            axis=1,
        )
    output = output[
        [
            "center",
            "patients",
            "model",
            "endpoint",
            "events",
            "C-index (95% CI)",
            "HR (95% CI)",
            "logrank_p",
            "low_risk_n",
            "high_risk_n",
            "training_median_cutoff",
            "AUC 1-year (95% CI)",
            "AUC 2-year (95% CI)",
            "AUC 3-year (95% CI)",
        ]
    ].rename(
        columns={
            "center": "Center",
            "patients": "N",
            "model": "Model",
            "endpoint": "Endpoint",
            "events": "Events",
            "logrank_p": "Log-rank p",
            "low_risk_n": "Low-risk N",
            "high_risk_n": "High-risk N",
            "training_median_cutoff": "Training median cutoff",
        }
    )
    for column in (
        "Log-rank p",
        "Training median cutoff",
    ):
        output[column] = output[column].map(display_value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every frozen comparison model and the proposed model on the "
            "126/47/31 exploratory center tables."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates < 400:
        raise ValueError("At least 400 bootstrap replicates are required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for center in CENTERS:
        if not center.table_path.is_file():
            raise FileNotFoundError(center.table_path)
    for model_index, model in enumerate(MODELS):
        if not model.prediction_path.is_file():
            raise FileNotFoundError(model.prediction_path)
        if model.train_prediction_path is not None and not model.train_prediction_path.is_file():
            raise FileNotFoundError(model.train_prediction_path)
        if model.cutoff_json_path is not None and not model.cutoff_json_path.is_file():
            raise FileNotFoundError(model.cutoff_json_path)

    center_frames = {center.key: load_center_table(center) for center in CENTERS}
    proposed_checkpoint = verify_proposed_checkpoint()
    result_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    model_manifest: dict[str, Any] = {}
    for model in MODELS:
        predictions = load_predictions(model)
        cutoffs, cutoff_source = load_cutoffs(model)
        model_manifest[model.key] = {
            "name": model.name,
            "prediction_path": str(model.prediction_path),
            "prediction_patients": int(len(predictions)),
            "cutoff_source": cutoff_source,
            "cutoffs": cutoffs,
        }
        for center_index, center in enumerate(CENTERS):
            aligned = align_center_predictions(
                center, center_frames[center.key], predictions
            )
            for endpoint_index, endpoint in enumerate(("OS", "PFS")):
                endpoint_seed = (
                    args.bootstrap_seed
                    + 10000 * model_index
                    + 100 * center_index
                    + endpoint_index
                )
                row, endpoint_auc = evaluate_endpoint(
                    model,
                    center,
                    aligned,
                    endpoint,
                    cutoffs[endpoint],
                    args.bootstrap_replicates,
                    endpoint_seed,
                )
                result_rows.append(row)
                auc_rows.extend(endpoint_auc)

    raw_results = pd.DataFrame(result_rows)
    raw_auc = pd.DataFrame(auc_rows)
    expected_rows = len(MODELS) * len(CENTERS) * 2
    if len(raw_results) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} result rows, got {len(raw_results)}")

    raw_path = args.output_dir / "all_models_center_results_raw.csv"
    display_path = args.output_dir / "all_models_center_results.csv"
    auc_path = args.output_dir / "all_models_center_auc_diagnostics.csv"
    raw_results.to_csv(raw_path, index=False, encoding="utf-8-sig")
    formatted_results(raw_results).to_csv(
        display_path, index=False, encoding="utf-8-sig"
    )
    raw_auc.apply(lambda column: column.map(display_value)).to_csv(
        auc_path, index=False, encoding="utf-8-sig"
    )

    manifest = {
        "analysis_status": "exploratory center reassignment; frozen model predictions",
        "methods": {
            "cindex": "Harrell C-index",
            "hr": "univariable Cox HR, high versus low risk",
            "risk_grouping": (
                "model- and endpoint-specific training-set median risk, fixed for all centers"
            ),
            "auc": "IPCW cumulative/dynamic AUC with censoring KM fitted within each center",
            "os_horizons_months": [12, 24, 36],
            "pfs_horizons_months": [12, 24, 36],
            "unavailable_display": "/",
            "confidence_intervals": (
                "patient-level nonparametric percentile bootstrap 95% CI within each center"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "centers": {
            center.key: {
                "name": center.name,
                "patients": center.expected_patients,
                "events": {
                    endpoint: int(
                        (
                            center_frames[center.key][f"{endpoint}_status"].to_numpy(float)
                            > 0.5
                        ).sum()
                    )
                    for endpoint in ("OS", "PFS")
                },
                "table_path": str(center.table_path),
            }
            for center in CENTERS
        },
        "models": model_manifest,
        "proposed_model_checkpoint": proposed_checkpoint,
        "results": result_rows,
        "auc_diagnostics": auc_rows,
        "files": {
            "formatted_results": str(display_path),
            "raw_results": str(raw_path),
            "auc_diagnostics": str(auc_path),
        },
    }
    manifest_path = args.output_dir / "all_models_center_results.json"
    manifest_path.write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(formatted_results(raw_results).to_string(index=False), flush=True)
    print(f"Results: {display_path}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
