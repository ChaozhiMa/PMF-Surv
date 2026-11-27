from __future__ import annotations

import argparse
import copy
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
from monai.networks.nets import resnet18
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
from restot_surv.losses import (
    distributed_pairwise_rank_loss,
    pairwise_rank_loss,
)
from restot_surv.metrics import concordance_index
from restot_surv.models.survival import (
    MTLRHead,
    endpoint_consistency,
    fit_censoring_distribution,
    fit_ipcw_weight_cap,
    fit_time_boundaries,
    ipcw_event_weights,
)
from restot_surv.utils import seed_worker


DEFAULT_CONFIG = paper_config_path(PROJECT_ROOT)
DEFAULT_OUTPUT = (
    PROJECT_ROOT.parent
    / "comparison_baselines_20260901"
    / "resnet18_3phase_mtlr"
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


class ResNet18ThreePhaseMTLR(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = resnet18(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=3,
            conv1_t_size=7,
            conv1_t_stride=2,
            no_max_pool=False,
            feed_forward=False,
        )
        self.os_head = MTLRHead(512, hidden_dim, num_bins, dropout)
        self.pfs_head = MTLRHead(512, hidden_dim, num_bins, dropout)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(image)
        os_logits = self.os_head(embedding)
        pfs_logits = self.pfs_head(embedding.detach())
        os_survival = self.os_head.survival(os_logits)
        pfs_survival = self.pfs_head.survival(pfs_logits)
        return {
            "embedding": embedding,
            "os_logits": os_logits,
            "pfs_logits": pfs_logits,
            "os_survival": os_survival,
            "pfs_survival": pfs_survival,
            "os_risk": self.os_head.risk(os_logits),
            "pfs_risk": self.pfs_head.risk(pfs_logits),
        }


def record_endpoint_arrays(
    records: Sequence[Any],
    data_cfg: Mapping[str, Any],
    endpoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = endpoint.lower()
    time_col = data_cfg[f"{prefix}_time_col"]
    event_col = data_cfg[f"{prefix}_event_col"]
    return (
        np.asarray([float(record.row[time_col]) for record in records], dtype=np.float32),
        np.asarray([float(record.row[event_col]) for record in records], dtype=np.float32),
    )


def initialize_heads(
    model: ResNet18ThreePhaseMTLR,
    train_records: Sequence[Any],
    cfg: Mapping[str, Any],
    device: torch.device,
) -> None:
    num_bins = int(cfg["model"]["num_time_bins"])
    ipcw_cfg = cfg["loss"].get("rank_ipcw", {})
    for endpoint, head in (("os", model.os_head), ("pfs", model.pfs_head)):
        times, events = record_endpoint_arrays(train_records, cfg["data"], endpoint)
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


def move_tensor(batch: Mapping[str, Any], name: str, device: torch.device) -> torch.Tensor:
    return batch[name].to(device=device, non_blocking=True)


def compute_loss(
    model: ResNet18ThreePhaseMTLR,
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    cfg: Mapping[str, Any],
    device: torch.device,
    epoch: int,
    distributed: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = cfg["loss"]
    os_time = move_tensor(batch, "os_time", device)
    os_event = move_tensor(batch, "os_event", device)
    pfs_time = move_tensor(batch, "pfs_time", device)
    pfs_event = move_tensor(batch, "pfs_event", device)
    os_nll = model.os_head.loss(outputs["os_logits"], os_time, os_event)
    pfs_nll = model.pfs_head.loss(outputs["pfs_logits"], pfs_time, pfs_event)
    ipcw_cfg = loss_cfg.get("rank_ipcw", {})
    os_event_weights = None
    if bool(ipcw_cfg.get("enabled", False)):
        os_event_weights = ipcw_event_weights(
            os_time,
            os_event,
            model.os_head.censoring_times,
            model.os_head.censoring_survival,
            survival_floor=float(ipcw_cfg.get("survival_floor", 0.1)),
            weight_cap=model.os_head.rank_ipcw_cap,
        )
    rank_loss = distributed_pairwise_rank_loss if distributed else pairwise_rank_loss
    os_rank = rank_loss(
        outputs["os_risk"],
        os_time,
        os_event,
        float(loss_cfg["rank_temperature"]),
        os_event_weights,
    )
    pfs_rank = rank_loss(
        outputs["pfs_risk"],
        pfs_time,
        pfs_event,
        float(loss_cfg["rank_temperature"]),
    )
    rank_scale = min(
        float(epoch) / max(int(loss_cfg.get("rank_warmup_epochs", 5)), 1),
        1.0,
    )
    consistency = endpoint_consistency(
        outputs["os_survival"].detach(),
        outputs["pfs_survival"],
        model.os_head.boundaries,
        model.pfs_head.boundaries,
    )
    os_loss = os_nll + rank_scale * float(loss_cfg["os_rank_weight"]) * os_rank
    pfs_loss = pfs_nll + rank_scale * float(loss_cfg["pfs_rank_weight"]) * pfs_rank
    total = (
        float(loss_cfg["os_weight"]) * os_loss
        + float(loss_cfg["pfs_weight"]) * pfs_loss
        + float(loss_cfg["endpoint_consistency_weight"]) * consistency
    )
    components = {
        "os_nll": float(os_nll.detach().cpu()),
        "pfs_nll": float(pfs_nll.detach().cpu()),
        "os_rank": float(os_rank.detach().cpu()),
        "pfs_rank": float(pfs_rank.detach().cpu()),
        "endpoint_consistency": float(consistency.detach().cpu()),
    }
    return total, components


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
    model: ResNet18ThreePhaseMTLR,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    ids: list[str] = []
    arrays: dict[str, list[np.ndarray]] = {
        "os_time": [],
        "os_event": [],
        "os_risk": [],
        "pfs_time": [],
        "pfs_event": [],
        "pfs_risk": [],
    }
    for batch in loader:
        image = move_tensor(batch, "image", device)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(image)
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
            arrays[f"{endpoint}_risk"].append(
                outputs[f"{endpoint}_risk"].detach().float().cpu().numpy().reshape(-1)
            )
    return pd.DataFrame(
        {
            "ID": ids,
            "OS_time": np.concatenate(arrays["os_time"]),
            "OS_status": np.concatenate(arrays["os_event"]),
            "OS_risk": np.concatenate(arrays["os_risk"]),
            "PFS_time": np.concatenate(arrays["pfs_time"]),
            "PFS_status": np.concatenate(arrays["pfs_event"]),
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


def learning_rates(epoch: int, epochs: int, warmup_epochs: int) -> tuple[float, float]:
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


def make_checkpoint(
    model: ResNet18ThreePhaseMTLR,
    datasets: Any,
) -> dict[str, Any]:
    return {
        "model_state": cpu_state_dict(model),
        "preprocessor": datasets.preprocessor.state_dict(),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a three-phase 3D ResNet-18 + MTLR baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()

    rank, local_rank, world_size = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("This 3D baseline requires CUDA.")
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

    raw_model = ResNet18ThreePhaseMTLR(
        hidden_dim=int(cfg["model"]["survival_hidden_dim"]),
        num_bins=int(cfg["model"]["num_time_bins"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    initialize_heads(raw_model, datasets.train.records, cfg, device)
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
            {"params": raw_model.encoder.parameters(), "lr": 1.0e-4},
            {
                "params": list(raw_model.os_head.parameters())
                + list(raw_model.pfs_head.parameters()),
                "lr": 3.0e-4,
            },
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
            "model": "3D ResNet-18 + dual MTLR",
            "input": "three-phase CT as channels [AP, PVP, DP]",
            "uses_clinical": False,
            "uses_qred": False,
            "uses_reliability": False,
            "uses_cross_phase_interaction": False,
            "pfs_updates_encoder": False,
            "pfs_training_scope": "PFS MTLR head only; encoder features are detached",
            "split_sizes": expected,
            "seed": seed,
            "world_size": world_size,
            "visible_gpu_count": torch.cuda.device_count(),
            "selection": "internal validation OS C-index only; external observed every epoch but never used for selection",
            "parameter_count": int(sum(p.numel() for p in raw_model.parameters())),
            "determinism": {
                "torch_deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "deterministic_warn_only": True,
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
                "nccl_algo": os.environ.get("NCCL_ALGO"),
                "nccl_proto": os.environ.get("NCCL_PROTO"),
                "worker_seeded": True,
                "epoch_seeded": True,
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
        encoder_lr, head_lr = learning_rates(
            epoch,
            args.epochs,
            args.warmup_epochs,
        )
        update_optimizer_lr(optimizer, encoder_lr, head_lr)
        model.train()
        totals = {
            "loss": 0.0,
            "os_nll": 0.0,
            "pfs_nll": 0.0,
            "os_rank": 0.0,
            "pfs_rank": 0.0,
            "endpoint_consistency": 0.0,
        }
        steps = 0
        for batch in train_loader:
            image = move_tensor(batch, "image", device)
            image = augmentation(image)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(image)
                loss, components = compute_loss(
                    raw_model,
                    outputs,
                    batch,
                    cfg,
                    device,
                    epoch,
                    distributed=world_size > 1,
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
            val_predictions = collect_predictions(raw_model, val_loader, device)
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
                "train_loss": totals["loss"] / max(steps, 1),
                "os_nll": totals["os_nll"] / max(steps, 1),
                "pfs_nll": totals["pfs_nll"] / max(steps, 1),
                "os_rank": totals["os_rank"] / max(steps, 1),
                "pfs_rank": totals["pfs_rank"] / max(steps, 1),
                "endpoint_consistency": totals["endpoint_consistency"] / max(steps, 1),
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
            pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
            (args.output_dir / "latest_epoch_metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "internal_os_cindex": float(val_os),
                        "internal_pfs_cindex": float(val_pfs),
                        "best_epoch_by_internal_os": best_epoch,
                        "best_internal_os_cindex": best_os,
                        "best_internal_pfs_cindex": best_pfs,
                        "external_used_for_selection": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"epoch={epoch:03d} loss={record['train_loss']:.6f} "
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
            loader = make_eval_loader(
                dataset,
                cfg,
                args.eval_batch_size,
                split_seed,
            )
            predictions = collect_predictions(raw_model, loader, device)
            predictions.to_csv(
                args.output_dir / f"{split}_predictions.csv",
                index=False,
            )
            metric_rows.extend(
                baseline_metrics.evaluate_prediction_frame(
                    "resnet18_3phase_mtlr",
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
