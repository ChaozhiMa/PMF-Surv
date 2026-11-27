from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "Simple")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(
    "/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig/code"
)
SOURCE_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_clinical_mtlr_rsf_baselines as baseline_metrics
from restot_surv.config import load_config, paper_config_path
from restot_surv.data import (
    JointEventBatchSampler,
    SynchronizedBatchAugmentation,
    build_datasets,
)
from restot_surv.metrics import concordance_index
from restot_surv.models.lite_prosenet import LiteProSENet
from restot_surv.utils import seed_worker


DEFAULT_CONFIG = paper_config_path(PROJECT_ROOT)
DEFAULT_OUTPUT = (
    PROJECT_ROOT.parent
    / "comparison_baselines_20260901"
    / "lite_prosenet"
)


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_epoch_seed(seed: int, rank: int, epoch: int) -> None:
    value = int(seed) + int(rank) + int(epoch) * 100_003
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({key: payload[key] for key in ("model", "model_state", "preprocessor") if key in payload}, temporary)
    temporary.replace(path)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def move_tensor(
    batch: Mapping[str, Any],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    return batch[name].to(device=device, non_blocking=True)


def endpoint_max_times(
    records: Sequence[Any],
    data_cfg: Mapping[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for endpoint in ("os", "pfs"):
        column = data_cfg[f"{endpoint}_time_col"]
        maximum = max(float(record.row[column]) for record in records)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise RuntimeError(f"Invalid maximum {endpoint.upper()} time: {maximum}.")
        result[endpoint] = maximum
    return result


def forward_model(
    model: nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    augment: SynchronizedBatchAugmentation | None = None,
) -> dict[str, torch.Tensor]:
    image = move_tensor(batch, "image", device)
    if augment is not None:
        image = augment(image)
    return model(
        image,
        move_tensor(batch, "clinical_continuous", device),
        move_tensor(batch, "clinical_categorical", device),
        move_tensor(batch, "clinical_composite", device),
    )


def native_endpoint_loss(
    outputs: Mapping[str, torch.Tensor],
    endpoint: str,
    target: torch.Tensor,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = outputs[f"{endpoint}_prediction"]
    branch_one = outputs[f"{endpoint}_branch_one"]
    branch_two = outputs[f"{endpoint}_branch_two"]
    mse = F.mse_loss(prediction, target)
    smooth_l1 = F.smooth_l1_loss(prediction, target)
    branch_one_loss = F.mse_loss(branch_one, target)
    branch_two_loss = F.smooth_l1_loss(branch_two, target)
    total = mse + smooth_l1 + alpha * branch_one_loss + beta * branch_two_loss
    return total, {
        "mse": mse,
        "smooth_l1": smooth_l1,
        "branch_one": branch_one_loss,
        "branch_two": branch_two_loss,
    }


def compute_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    max_times: Mapping[str, float],
    device: torch.device,
    alpha: float,
    beta: float,
    pfs_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    os_target = move_tensor(batch, "os_time", device).float() / float(max_times["os"])
    pfs_target = move_tensor(batch, "pfs_time", device).float() / float(
        max_times["pfs"]
    )
    os_loss, os_components = native_endpoint_loss(
        outputs,
        "os",
        os_target,
        alpha,
        beta,
    )
    pfs_loss, pfs_components = native_endpoint_loss(
        outputs,
        "pfs",
        pfs_target,
        alpha,
        beta,
    )
    total = os_loss + pfs_weight * pfs_loss
    return total, {
        "os_loss": float(os_loss.detach().cpu()),
        "pfs_loss": float(pfs_loss.detach().cpu()),
        "os_mse": float(os_components["mse"].detach().cpu()),
        "os_smooth_l1": float(os_components["smooth_l1"].detach().cpu()),
        "pfs_mse": float(pfs_components["mse"].detach().cpu()),
        "pfs_smooth_l1": float(pfs_components["smooth_l1"].detach().cpu()),
    }


def make_eval_loader(
    dataset: Any,
    cfg: Mapping[str, Any],
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.no_grad()
def collect_predictions(
    model: LiteProSENet,
    loader: DataLoader,
    device: torch.device,
    max_times: Mapping[str, float],
) -> pd.DataFrame:
    model.eval()
    ids: list[str] = []
    arrays: dict[str, list[np.ndarray]] = {
        "os_time": [],
        "os_event": [],
        "os_risk": [],
        "os_prediction": [],
        "pfs_time": [],
        "pfs_event": [],
        "pfs_risk": [],
        "pfs_prediction": [],
    }
    for batch in loader:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = forward_model(model, batch, device)
        patient_ids = batch["patient_id"]
        if isinstance(patient_ids, str):
            patient_ids = [patient_ids]
        ids.extend(str(value) for value in patient_ids)
        for endpoint in ("os", "pfs"):
            arrays[f"{endpoint}_time"].append(
                batch[f"{endpoint}_time"].detach().float().cpu().numpy().reshape(-1)
            )
            arrays[f"{endpoint}_event"].append(
                batch[f"{endpoint}_event"].detach().float().cpu().numpy().reshape(-1)
            )
            normalized_prediction = (
                outputs[f"{endpoint}_prediction"]
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            arrays[f"{endpoint}_prediction"].append(
                normalized_prediction * float(max_times[endpoint])
            )
            arrays[f"{endpoint}_risk"].append(-normalized_prediction)
    return pd.DataFrame(
        {
            "ID": ids,
            "OS_time": np.concatenate(arrays["os_time"]),
            "OS_status": np.concatenate(arrays["os_event"]),
            "OS_predicted_time": np.concatenate(arrays["os_prediction"]),
            "OS_risk": np.concatenate(arrays["os_risk"]),
            "PFS_time": np.concatenate(arrays["pfs_time"]),
            "PFS_status": np.concatenate(arrays["pfs_event"]),
            "PFS_predicted_time": np.concatenate(arrays["pfs_prediction"]),
            "PFS_risk": np.concatenate(arrays["pfs_risk"]),
        }
    )


def cindex_pair(predictions: pd.DataFrame) -> tuple[float, float]:
    return (
        concordance_index(
            predictions["OS_time"],
            predictions["OS_status"],
            predictions["OS_risk"],
        ),
        concordance_index(
            predictions["PFS_time"],
            predictions["PFS_status"],
            predictions["PFS_risk"],
        ),
    )


def learning_rates(
    epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> tuple[float, float]:
    encoder_peak = 1.0e-4
    head_peak = 3.0e-4
    minimum_ratio = 0.01
    if epoch <= warmup_epochs:
        factor = epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        factor = minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    return encoder_peak * factor, head_peak * factor


def update_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    encoder_lr: float,
    head_lr: float,
) -> None:
    optimizer.param_groups[0]["lr"] = encoder_lr
    optimizer.param_groups[1]["lr"] = head_lr
    optimizer.param_groups[2]["lr"] = head_lr


def make_checkpoint(
    model: LiteProSENet,
    datasets: Any,
) -> dict[str, Any]:
    return {
        "model_state": cpu_state_dict(model),
        "preprocessor": datasets.preprocessor.state_dict(),
    }



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the adapted Lite-ProSENet comparison baseline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--raw-frame-weight", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--pfs-weight", type=float, default=0.3)
    args = parser.parse_args()

    rank, local_rank, world_size = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("Lite-ProSENet training requires CUDA.")
    if args.batch_size_per_gpu < 2:
        raise ValueError(
            "Lite-ProSENet uses BatchNorm in its native heads; "
            "batch-size-per-gpu must be at least 2."
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cfg = load_config(args.config)
    seed = int(cfg["experiment"].get("seed", 2026))
    set_seed(seed, rank)
    datasets = build_datasets(cfg)
    expected = {"train": 508, "val": 127, "test": 204}
    for name, count in expected.items():
        if len(getattr(datasets, name)) != count:
            raise RuntimeError(f"Expected {count} {name} cases.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_times = endpoint_max_times(datasets.train.records, cfg["data"])

    raw_model = LiteProSENet(
        image_shape=cfg["data"]["image_shape"],
        num_continuous=len(cfg["data"]["continuous_vars"]),
        categorical_cardinalities=datasets.preprocessor.categorical_cardinalities(),
        composite_cardinalities=datasets.preprocessor.composite_cardinalities(),
        base_channels=args.base_channels,
        raw_frame_weight=args.raw_frame_weight,
    ).to(device)
    if raw_model.clinical_tower.variable_count != 17:
        raise RuntimeError(
            "This adapted baseline expects the configured 17 clinical variables."
        )
    model: nn.Module = raw_model
    if world_size > 1:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

    train_sampler = JointEventBatchSampler(
        datasets.train,
        batch_size=args.batch_size_per_gpu,
        seed=seed,
        rank=rank,
        world_size=world_size,
        max_oversample=float(cfg["training"].get("max_stratum_oversample", 2.0)),
    )
    train_loader = DataLoader(
        datasets.train,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + rank),
    )
    augmentation = SynchronizedBatchAugmentation(cfg)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(raw_model.image_tower.parameters())
                + list(raw_model.clinical_tower.parameters()),
                "lr": 1.0e-4,
            },
            {"params": raw_model.os_head.parameters(), "lr": 3.0e-4},
            {"params": raw_model.pfs_head.parameters(), "lr": 3.0e-4},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    val_loader = (
        make_eval_loader(datasets.val, cfg, args.eval_batch_size, seed + 10_000)
        if rank == 0
        else None
    )
    history: list[dict[str, Any]] = []
    best_os = -float("inf")
    best_pfs = -float("inf")
    best_epoch = 0
    no_improvement = 0
    start_time = time.time()

    if rank == 0:
        contract = {
            "model": "Lite-ProSENet with native survival-time heads",
            "source": "Wu et al., Multimodal learning for non-small cell lung cancer prognosis",
            "input": "three-phase CT as channels [AP, PVP, DP]",
            "clinical_variables": list(cfg["data"]["continuous_vars"])
            + list(cfg["data"]["categorical_vars"])
            + list(cfg["data"]["composite_vars"]),
            "uses_mtlr": False,
            "uses_qred": False,
            "uses_reliability": False,
            "uses_cross_phase_interaction": False,
            "frame_difference_axis": "axial depth",
            "pfs_updates_encoder": False,
            "pfs_training_scope": "PFS native head only; fused features are detached",
            "native_training_target": "observed survival time normalized by training-set maximum",
            "native_loss_uses_censoring_indicator": False,
            "split_sizes": expected,
            "seed": seed,
            "world_size": world_size,
            "visible_gpu_count": torch.cuda.device_count(),
            "selection": "internal validation OS C-index; PFS C-index tie-break",
            "parameter_count": int(sum(p.numel() for p in raw_model.parameters())),
            "hyperparameters": {
                "raw_frame_weight": args.raw_frame_weight,
                "difference_frame_weight_total": 1.0 - args.raw_frame_weight,
                "alpha": args.alpha,
                "beta": args.beta,
                "pfs_weight": args.pfs_weight,
                "base_channels": args.base_channels,
            },
        }
        (args.output_dir / "experiment_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(contract, ensure_ascii=False), flush=True)

    for epoch in range(1, args.epochs + 1):
        set_epoch_seed(seed, rank, epoch)
        train_sampler.set_epoch(epoch)
        encoder_lr, head_lr = learning_rates(epoch, args.epochs, args.warmup_epochs)
        update_optimizer_lr(optimizer, encoder_lr, head_lr)
        model.train()
        totals = {
            "loss": 0.0,
            "os_loss": 0.0,
            "pfs_loss": 0.0,
            "os_mse": 0.0,
            "os_smooth_l1": 0.0,
            "pfs_mse": 0.0,
            "pfs_smooth_l1": 0.0,
        }
        steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = forward_model(model, batch, device, augmentation)
                loss, components = compute_loss(
                    outputs,
                    batch,
                    max_times,
                    device,
                    args.alpha,
                    args.beta,
                    args.pfs_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {steps + 1}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach().cpu())
            for name, value in components.items():
                totals[name] += value
            steps += 1

        if world_size > 1:
            dist.barrier()
        stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
        if rank == 0:
            assert val_loader is not None
            val_predictions = collect_predictions(
                raw_model,
                val_loader,
                device,
                max_times,
            )
            val_os, val_pfs = cindex_pair(val_predictions)
            improved = math.isfinite(val_os) and val_os > best_os
            if improved:
                best_os = float(val_os)
                best_pfs = float(val_pfs)
                best_epoch = epoch
                no_improvement = 0
                checkpoint = make_checkpoint(raw_model, datasets)

                atomic_torch_save(
                    checkpoint,
                    args.output_dir / "best_internal_os.pth",
                )
            else:
                no_improvement += 1
            record = {
                "epoch": epoch,
                **{
                    name: value / max(steps, 1)
                    for name, value in totals.items()
                },
                "internal_os_cindex": val_os,
                "internal_pfs_cindex": val_pfs,
                "best_epoch": best_epoch,
                "best_internal_os_cindex": best_os,
                "best_internal_pfs_cindex": best_pfs,
                "encoder_lr": encoder_lr,
                "head_lr": head_lr,
                "elapsed_minutes": (time.time() - start_time) / 60.0,
            }
            history.append(record)
            pd.DataFrame(history).to_csv(
                args.output_dir / "training_history.csv",
                index=False,
            )
            (args.output_dir / "latest_epoch_metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "internal_os_cindex": float(val_os),
                        "internal_pfs_cindex": float(val_pfs),
                        "best_epoch_by_internal_os": best_epoch,
                        "best_internal_os_cindex": best_os,
                        "best_internal_pfs_cindex": best_pfs,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"epoch={epoch:03d} loss={record['loss']:.6f} "
                f"internal_os={val_os:.6f} internal_pfs={val_pfs:.6f} "
                f"best_epoch={best_epoch:03d} best_os={best_os:.6f} "
                f"elapsed_min={record['elapsed_minutes']:.1f}",
                flush=True,
            )
            if epoch >= 40 and no_improvement >= args.patience:
                stop_flag.fill_(1)
        if world_size > 1:
            dist.broadcast(stop_flag, src=0)
        if bool(stop_flag.item()):
            break

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        checkpoint_path = args.output_dir / "best_internal_os.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        raw_model.load_state_dict(checkpoint["model_state"])
        metric_rows: list[dict[str, Any]] = []
        for split, dataset in (
            ("train", datasets.train),
            ("internal_validation", datasets.val),
            ("external_validation", datasets.test),
        ):
            split_seed = {
                "train": seed + 30_000,
                "internal_validation": seed + 40_000,
                "external_validation": seed + 50_000,
            }[split]
            loader = make_eval_loader(dataset, cfg, args.eval_batch_size, split_seed)
            predictions = collect_predictions(
                raw_model,
                loader,
                device,
                max_times,
            )
            predictions.to_csv(
                args.output_dir / f"{split}_predictions.csv",
                index=False,
            )
            metric_rows.extend(
                baseline_metrics.evaluate_prediction_frame(
                    "lite_prosenet",
                    split,
                    predictions,
                )
            )
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        cindex = metrics[metrics["metric"] == "c_index"]
        print("Final C-index summary", flush=True)
        print(
            cindex[["model", "split", "endpoint", "value"]].to_string(index=False),
            flush=True,
        )
        summary = {
            "best_epoch": int(best_epoch),
            "best_internal_os_cindex": float(best_os),
            "best_internal_pfs_cindex": float(best_pfs),
            "epochs_run": int(epoch),
            "stopped_early": bool(epoch < args.epochs),
            "metrics_file": str(args.output_dir / "metrics.csv"),
        }
        (args.output_dir / "result_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
