from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


PROJECT_ROOT = Path(
    "/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig/code"
)
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from restot_surv.config import paper_config_path
from restot_surv.data import ClinicalPreprocessor, read_rows
from restot_surv.losses import pairwise_rank_loss
from restot_surv.metrics import concordance_index
from restot_surv.models.survival import (
    MTLRHead,
    endpoint_consistency,
    fit_censoring_distribution,
    fit_ipcw_weight_cap,
    fit_time_boundaries,
    ipcw_event_weights,
)


DEFAULT_CONFIG = paper_config_path(PROJECT_ROOT)
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "comparison_baselines_20260901"
OS_HORIZONS = (12.0, 24.0, 36.0)
PFS_HORIZONS = (6.0, 12.0, 18.0, 24.0, 36.0)


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle))


def frame_from_rows(rows: Sequence[Mapping[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["ID"] = frame["ID"].astype(str)
    return frame


def clinical_matrix(
    rows: Sequence[Mapping[str, str]],
    preprocessor: ClinicalPreprocessor,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for row in rows:
        continuous, categorical, composite = preprocessor.transform(row)
        values.append(
            np.concatenate(
                [
                    continuous.astype(np.float32),
                    categorical.astype(np.float32),
                    composite.astype(np.float32),
                ]
            )
        )
    return np.stack(values).astype(np.float32)


def endpoint_arrays(
    frame: pd.DataFrame,
    endpoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    upper = endpoint.upper()
    return (
        frame[f"{upper}_time"].to_numpy(dtype=np.float32),
        frame[f"{upper}_status"].to_numpy(dtype=np.float32),
    )


def validate_splits(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> None:
    expected = {"train": 508, "val": 127, "test": 204}
    frames = {"train": train_frame, "val": val_frame, "test": test_frame}
    for name, frame in frames.items():
        if len(frame) != expected[name]:
            raise RuntimeError(f"Expected {expected[name]} {name} cases, found {len(frame)}.")
        if frame["ID"].duplicated().any():
            raise RuntimeError(f"Duplicate IDs in {name} split.")
        if frame[list(feature_names)].isna().any().any():
            raise RuntimeError(f"Missing clinical values in {name} split.")


def fit_censoring_km(
    times: Sequence[float], events: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(times, dtype=np.float64).reshape(-1)
    event = np.asarray(events, dtype=np.float64).reshape(-1) > 0.5
    unique_times = np.unique(time)
    survival = 1.0
    values = []
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


def metric_rows(
    model_name: str,
    split: str,
    endpoint: str,
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "model": model_name,
            "split": split,
            "endpoint": endpoint.upper(),
            "metric": "c_index",
            "horizon_months": np.nan,
            "value": concordance_index(times, events, risks),
            "cases": int((events > 0.5).sum()),
            "controls": np.nan,
        }
    ]
    horizons = OS_HORIZONS if endpoint.lower() == "os" else PFS_HORIZONS
    for horizon in horizons:
        auc, diagnostics = ipcw_cumulative_dynamic_auc(times, events, risks, horizon)
        rows.append(
            {
                "model": model_name,
                "split": split,
                "endpoint": endpoint.upper(),
                "metric": "time_dependent_auc",
                "horizon_months": horizon,
                "value": auc,
                "cases": diagnostics["cases"],
                "controls": diagnostics["controls"],
            }
        )
    return rows


class ClinicalMTLR(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.os_head = MTLRHead(input_dim, hidden_dim, num_bins, dropout)
        self.pfs_head = MTLRHead(input_dim, hidden_dim, num_bins, dropout)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        os_logits = self.os_head(features)
        pfs_logits = self.pfs_head(features)
        os_survival = self.os_head.survival(os_logits)
        pfs_survival = self.pfs_head.survival(pfs_logits)
        return {
            "os_logits": os_logits,
            "pfs_logits": pfs_logits,
            "os_survival": os_survival,
            "pfs_survival": pfs_survival,
            "os_risk": self.os_head.risk(os_logits),
            "pfs_risk": self.pfs_head.risk(pfs_logits),
        }


def initialize_mtlr_heads(
    model: ClinicalMTLR,
    train_frame: pd.DataFrame,
    num_bins: int,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> None:
    ipcw_cfg = cfg["loss"].get("rank_ipcw", {})
    for endpoint, head in (("os", model.os_head), ("pfs", model.pfs_head)):
        times, events = endpoint_arrays(train_frame, endpoint)
        boundaries = fit_time_boundaries(times, events, num_bins)
        censor_times, censor_survival = fit_censoring_distribution(times, events)
        head.set_boundaries(torch.as_tensor(boundaries, device=device))
        head.set_censoring_distribution(
            torch.as_tensor(censor_times, device=device),
            torch.as_tensor(censor_survival, device=device),
        )
        if bool(ipcw_cfg.get("enabled", False)) and (
            endpoint == "os" or bool(ipcw_cfg.get("apply_to_pfs", False))
        ):
            cap = fit_ipcw_weight_cap(
                times,
                events,
                head.censoring_times.detach().cpu(),
                head.censoring_survival.detach().cpu(),
                survival_floor=float(ipcw_cfg.get("survival_floor", 0.1)),
                quantile=float(ipcw_cfg.get("cap_quantile", 0.99)),
            )
            head.set_rank_ipcw_cap(cap)


def tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def predict_mtlr(
    model: ClinicalMTLR,
    features: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad():
        outputs = model(tensor(features, device))
    return {
        name: value.detach().float().cpu().numpy()
        for name, value in outputs.items()
    }


def train_one_mtlr_candidate(
    cfg: Mapping[str, Any],
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    train_x: np.ndarray,
    val_x: np.ndarray,
    candidate: Mapping[str, Any],
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    set_seed(seed)
    num_bins = int(cfg["model"]["num_time_bins"])
    model = ClinicalMTLR(
        input_dim=train_x.shape[1],
        hidden_dim=int(candidate["hidden_dim"]),
        num_bins=num_bins,
        dropout=float(candidate["dropout"]),
    ).to(device)
    initialize_mtlr_heads(model, train_frame, num_bins, cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
    )
    train_features = tensor(train_x, device)
    train_values: dict[str, torch.Tensor] = {}
    for endpoint in ("os", "pfs"):
        times, events = endpoint_arrays(train_frame, endpoint)
        train_values[f"{endpoint}_time"] = tensor(times, device)
        train_values[f"{endpoint}_event"] = tensor(events, device)

    loss_cfg = cfg["loss"]
    ipcw_cfg = loss_cfg.get("rank_ipcw", {})
    os_event_weights = None
    if bool(ipcw_cfg.get("enabled", False)):
        os_event_weights = ipcw_event_weights(
            train_values["os_time"],
            train_values["os_event"],
            model.os_head.censoring_times,
            model.os_head.censoring_survival,
            survival_floor=float(ipcw_cfg.get("survival_floor", 0.1)),
            weight_cap=model.os_head.rank_ipcw_cap,
        )

    val_os_time, val_os_event = endpoint_arrays(val_frame, "os")
    val_pfs_time, val_pfs_event = endpoint_arrays(val_frame, "pfs")
    best_score = -float("inf")
    best_pfs = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    final_loss = float("nan")
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(train_features)
        os_nll = model.os_head.loss(
            outputs["os_logits"],
            train_values["os_time"],
            train_values["os_event"],
        )
        pfs_nll = model.pfs_head.loss(
            outputs["pfs_logits"],
            train_values["pfs_time"],
            train_values["pfs_event"],
        )
        rank_scale = min(epoch / max(int(loss_cfg.get("rank_warmup_epochs", 5)), 1), 1.0)
        os_rank = pairwise_rank_loss(
            outputs["os_risk"],
            train_values["os_time"],
            train_values["os_event"],
            float(loss_cfg["rank_temperature"]),
            os_event_weights,
        )
        pfs_rank = pairwise_rank_loss(
            outputs["pfs_risk"],
            train_values["pfs_time"],
            train_values["pfs_event"],
            float(loss_cfg["rank_temperature"]),
        )
        consistency = endpoint_consistency(
            outputs["os_survival"].detach(),
            outputs["pfs_survival"],
            model.os_head.boundaries,
            model.pfs_head.boundaries,
        )
        os_loss = os_nll + rank_scale * float(loss_cfg["os_rank_weight"]) * os_rank
        pfs_loss = pfs_nll + rank_scale * float(loss_cfg["pfs_rank_weight"]) * pfs_rank
        loss = (
            float(loss_cfg["os_weight"]) * os_loss
            + float(loss_cfg["pfs_weight"]) * pfs_loss
            + float(loss_cfg["endpoint_consistency_weight"]) * consistency
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite clinical MTLR loss at epoch {epoch}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())

        val_outputs = predict_mtlr(model, val_x, device)
        val_os = concordance_index(val_os_time, val_os_event, val_outputs["os_risk"])
        val_pfs = concordance_index(val_pfs_time, val_pfs_event, val_outputs["pfs_risk"])
        improved = np.isfinite(val_os) and val_os > best_score
        if improved:
            best_score = float(val_os)
            best_pfs = float(val_pfs)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch >= 40 and no_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Clinical MTLR did not produce a valid checkpoint.")
    result = dict(candidate)
    result.update(
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "val_os_c_index": best_score,
            "val_pfs_c_index": best_pfs,
            "final_train_loss": final_loss,
        }
    )
    return result, best_state


def prediction_frame(
    frame: pd.DataFrame,
    os_risk: np.ndarray,
    pfs_risk: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": frame["ID"].astype(str).to_numpy(),
            "OS_time": frame["OS_time"].to_numpy(dtype=np.float64),
            "OS_status": frame["OS_status"].to_numpy(dtype=np.float64),
            "OS_risk": np.asarray(os_risk, dtype=np.float64).reshape(-1),
            "PFS_time": frame["PFS_time"].to_numpy(dtype=np.float64),
            "PFS_status": frame["PFS_status"].to_numpy(dtype=np.float64),
            "PFS_risk": np.asarray(pfs_risk, dtype=np.float64).reshape(-1),
        }
    )


def evaluate_prediction_frame(
    model_name: str,
    split: str,
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for endpoint in ("os", "pfs"):
        upper = endpoint.upper()
        rows.extend(
            metric_rows(
                model_name,
                split,
                endpoint,
                predictions[f"{upper}_time"].to_numpy(dtype=np.float64),
                predictions[f"{upper}_status"].to_numpy(dtype=np.float64),
                predictions[f"{upper}_risk"].to_numpy(dtype=np.float64),
            )
        )
    return rows


def run_clinical_mtlr(
    cfg: Mapping[str, Any],
    output_root: Path,
    preprocessor: ClinicalPreprocessor,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    output_dir = output_root / "clinical_mtlr"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        {
            "hidden_dim": hidden_dim,
            "dropout": 0.1,
            "learning_rate": learning_rate,
            "weight_decay": 1.0e-3,
        }
        for hidden_dim in (32, 64, 128)
        for learning_rate in (1.0e-3, 3.0e-3)
    ]
    grid_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for index, candidate in enumerate(candidates, start=1):
        print(f"[clinical_mtlr] candidate {index}/{len(candidates)}: {candidate}", flush=True)
        result, state = train_one_mtlr_candidate(
            cfg,
            train_frame,
            val_frame,
            train_x,
            val_x,
            candidate,
            seed,
            device,
            max_epochs=400,
            patience=50,
        )
        grid_results.append(result)
        print(
            "[clinical_mtlr] "
            f"best_epoch={result['best_epoch']} "
            f"val_os={result['val_os_c_index']:.6f} "
            f"val_pfs={result['val_pfs_c_index']:.6f}",
            flush=True,
        )
        val_score = float(result["val_os_c_index"])
        if np.isfinite(val_score) and (
            best_result is None or val_score > float(best_result["val_os_c_index"])
        ):
            best_result = result
            best_state = state

    if best_result is None or best_state is None:
        raise RuntimeError("No clinical MTLR candidate was selected.")
    pd.DataFrame(grid_results).to_csv(output_dir / "internal_grid_results.csv", index=False)
    selected = ClinicalMTLR(
        input_dim=train_x.shape[1],
        hidden_dim=int(best_result["hidden_dim"]),
        num_bins=int(cfg["model"]["num_time_bins"]),
        dropout=float(best_result["dropout"]),
    ).to(device)
    initialize_mtlr_heads(
        selected,
        train_frame,
        int(cfg["model"]["num_time_bins"]),
        cfg,
        device,
    )
    selected.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "preprocessor": preprocessor.state_dict(),
        },
        output_dir / "best_internal_os.pth",
    )
    with (output_dir / "selected_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(best_result, handle, ensure_ascii=False, indent=2)

    rows: list[dict[str, Any]] = []
    for split, frame, features in (
        ("internal_validation", val_frame, val_x),
        ("external_validation", test_frame, test_x),
    ):
        outputs = predict_mtlr(selected, features, device)
        predictions = prediction_frame(
            frame,
            outputs["os_risk"],
            outputs["pfs_risk"],
        )
        predictions.to_csv(output_dir / f"{split}_predictions.csv", index=False)
        rows.extend(evaluate_prediction_frame("clinical_mtlr", split, predictions))
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    print(f"[clinical_mtlr] selected: {best_result}", flush=True)
    return rows


def survival_target(frame: pd.DataFrame, endpoint: str) -> np.ndarray:
    times, events = endpoint_arrays(frame, endpoint)
    return Surv.from_arrays(event=events > 0.5, time=times.astype(np.float64))


def run_rsf(
    output_root: Path,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    output_dir = output_root / "rsf"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_os = survival_target(train_frame, "os")
    val_os_time, val_os_event = endpoint_arrays(val_frame, "os")
    grid: list[dict[str, Any]] = []
    for min_samples_leaf in (3, 5, 10, 15):
        for max_features in ("sqrt", 0.5, 1.0):
            grid.append(
                {
                    "n_estimators": 500,
                    "min_samples_leaf": min_samples_leaf,
                    "min_samples_split": max(2, min_samples_leaf * 2),
                    "max_features": max_features,
                    "max_depth": None,
                }
            )

    grid_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_os_model: RandomSurvivalForest | None = None
    for index, params in enumerate(grid, start=1):
        print(f"[rsf] candidate {index}/{len(grid)}: {params}", flush=True)
        model = RandomSurvivalForest(
            **params,
            random_state=seed,
            n_jobs=8,
            low_memory=True,
        )
        model.fit(train_x, train_os)
        val_risk = model.predict(val_x)
        val_score = concordance_index(val_os_time, val_os_event, val_risk)
        result = dict(params)
        result["val_os_c_index"] = float(val_score)
        grid_results.append(result)
        print(f"[rsf] val_os={val_score:.6f}", flush=True)
        if np.isfinite(val_score) and (
            best_result is None or val_score > float(best_result["val_os_c_index"])
        ):
            best_result = result
            best_os_model = model

    if best_result is None or best_os_model is None:
        raise RuntimeError("No RSF candidate was selected.")
    pd.DataFrame(grid_results).to_csv(output_dir / "internal_grid_results.csv", index=False)
    selected_params = {
        name: best_result[name]
        for name in (
            "n_estimators",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "max_depth",
        )
    }
    pfs_model = RandomSurvivalForest(
        **selected_params,
        random_state=seed,
        n_jobs=8,
        low_memory=True,
    )
    pfs_model.fit(train_x, survival_target(train_frame, "pfs"))
    joblib.dump(best_os_model, output_dir / "os_model.joblib")
    joblib.dump(pfs_model, output_dir / "pfs_model.joblib")
    with (output_dir / "selected_hyperparameters.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **best_result,
                "selection_rule": "maximum internal-validation OS C-index",
                "pfs_rule": "fit with the OS-selected shared RSF hyperparameters",
                "feature_names": list(feature_names),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    rows: list[dict[str, Any]] = []
    for split, frame, features in (
        ("internal_validation", val_frame, val_x),
        ("external_validation", test_frame, test_x),
    ):
        predictions = prediction_frame(
            frame,
            best_os_model.predict(features),
            pfs_model.predict(features),
        )
        predictions.to_csv(output_dir / f"{split}_predictions.csv", index=False)
        rows.extend(evaluate_prediction_frame("rsf", split, predictions))
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    print(f"[rsf] selected: {best_result}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pure-clinical MTLR and RSF comparison baselines."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", choices=("all", "mtlr", "rsf"), default="all")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["experiment"].get("seed", 2026))
    set_seed(seed)
    data_cfg = cfg["data"]
    train_rows = read_rows(data_cfg["train_csv"])
    val_rows = read_rows(data_cfg["val_csv"])
    test_rows = read_rows(data_cfg["test_csv"])
    train_frame = frame_from_rows(train_rows)
    val_frame = frame_from_rows(val_rows)
    test_frame = frame_from_rows(test_rows)
    feature_names = (
        list(data_cfg["continuous_vars"])
        + list(data_cfg["categorical_vars"])
        + list(data_cfg["composite_vars"])
    )
    validate_splits(train_frame, val_frame, test_frame, feature_names)
    if len(feature_names) != 17:
        raise RuntimeError(f"Expected 17 clinical variables, found {len(feature_names)}.")
    preprocessor = ClinicalPreprocessor.fit(
        train_rows,
        data_cfg["continuous_vars"],
        data_cfg["categorical_vars"],
        data_cfg["composite_vars"],
    )
    train_x = clinical_matrix(train_rows, preprocessor)
    val_x = clinical_matrix(val_rows, preprocessor)
    test_x = clinical_matrix(test_rows, preprocessor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "experiment_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": str(args.config),
                "seed": seed,
                "split_sizes": {"train": 508, "internal_validation": 127, "external_validation": 204},
                "feature_names": feature_names,
                "selection": "internal validation only; external validation evaluated after selection",
                "external_csv": str(data_cfg["test_csv"]),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    all_rows: list[dict[str, Any]] = []
    if args.only in ("all", "mtlr"):
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        all_rows.extend(
            run_clinical_mtlr(
                cfg,
                args.output_dir,
                preprocessor,
                train_frame,
                val_frame,
                test_frame,
                train_x,
                val_x,
                test_x,
                seed,
                device,
            )
        )
    if args.only in ("all", "rsf"):
        all_rows.extend(
            run_rsf(
                args.output_dir,
                train_frame,
                val_frame,
                test_frame,
                train_x,
                val_x,
                test_x,
                feature_names,
                seed,
            )
        )

    metrics_path = args.output_dir / "comparison_metrics.csv"
    if metrics_path.exists() and args.only != "all":
        previous = pd.read_csv(metrics_path)
        replaced = {row["model"] for row in all_rows}
        previous = previous[~previous["model"].isin(replaced)]
        all_rows = previous.to_dict("records") + all_rows
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(metrics_path, index=False)
    cindex = metrics[metrics["metric"] == "c_index"].copy()
    print("\nFinal C-index summary", flush=True)
    print(
        cindex[["model", "split", "endpoint", "value"]].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
