from __future__ import annotations

import csv
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from .config import save_resolved_config
from .data import (
    DatasetBundle,
    JointEventBatchSampler,
    SynchronizedBatchAugmentation,
)
from .losses import Objective
from .calibration import cox_probability_metrics, load_center_patient_ids
from .metrics import concordance_index, integrated_brier_score
from .models.model import PMFSurv
from .models.survival import (
    fit_censoring_distribution,
    fit_ipcw_weight_cap,
    fit_time_boundaries,
)
from .utils import (
    atomic_torch_save,
    git_commit,
    seed_everything,
    seed_worker,
    write_json,
)


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _warmup_cuda_libraries(device: torch.device) -> None:
    """Initialize convolution and matrix-product backward handles early."""
    for dtype in (torch.float32, torch.float16):
        sample = torch.ones(
            (1, 2, 16, 16, 16), device=device, dtype=dtype, requires_grad=True
        )
        kernel = torch.ones(
            (4, 2, 3, 3, 3), device=device, dtype=dtype, requires_grad=True
        )
        output = torch.nn.functional.conv3d(sample, kernel, padding=1)
        output.float().mean().backward()
        del sample, kernel, output
        matrix = torch.ones(
            (256, 256), device=device, dtype=dtype, requires_grad=True
        )
        product = matrix @ matrix.t()
        product.float().mean().backward()
        del matrix, product
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def load_external_center_ids(
    center_paths: Mapping[str, str],
    patient_id_col: str,
    project_root: str | Path,
) -> dict[str, set[str]]:
    """Read center membership once; keep patient identifiers as strings."""
    centers = {}
    for name, source in center_paths.items():
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = Path(project_root) / path
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            centers[str(name)] = {
                str(row[patient_id_col]).strip() for row in csv.DictReader(handle)
            }
    return centers


def external_center_metrics(
    patient_ids: Sequence[str],
    predictions: Mapping[str, np.ndarray],
    center_ids: Mapping[str, set[str]],
) -> dict[str, float]:
    """Slice the same evaluated predictions by ID, independent of rank order."""
    identifiers = np.asarray([str(value).strip() for value in patient_ids])
    metrics = {}
    for center, members in center_ids.items():
        selected = np.isin(identifiers, list(members))
        metrics[f"{center}_n"] = float(selected.sum())
        for endpoint in ("os", "pfs"):
            metrics[f"{center}_{endpoint}_cindex"] = concordance_index(
                predictions[f"{endpoint}_time"][selected],
                predictions[f"{endpoint}_event"][selected],
                predictions[f"{endpoint}_risk"][selected],
            )
    return metrics


def optimizer_group_lr(
    cfg: Mapping[str, Any],
    group_name: str,
    epoch: int,
    step_in_epoch: int,
    steps_per_epoch: int,
) -> float:
    training = cfg["training"]
    schedule = cfg["schedule"]
    structure_end = int(schedule["structure_end_epoch"])
    transition_end = int(schedule["transition_end_epoch"])
    total_epochs = int(training["epochs"])
    steps_per_epoch = max(int(steps_per_epoch), 1)
    step_in_epoch = min(max(int(step_in_epoch), 1), steps_per_epoch)
    boundary_lr = float(training["stage_boundary_lr"])
    peak_lrs = {
        "encoder": float(training["encoder_lr"]),
        "fusion": float(training["fusion_lr"]),
        "survival_heads": float(training["head_lr"]),
        "auxiliary": float(training["auxiliary_lr"]),
        "token_reliability": float(
            training.get("token_reliability_lr", training["encoder_lr"])
        ),
    }
    peak = peak_lrs[group_name]

    if structure_end == 0 and transition_end == 0:
        current = (epoch - 1) * steps_per_epoch + step_in_epoch
        warmup = min(int(training["stage1_warmup_steps"]), total_epochs * steps_per_epoch)
        if warmup > 1 and current <= warmup:
            progress = (current - 1) / (warmup - 1)
            return boundary_lr + progress * (peak - boundary_lr)
        return peak

    if epoch <= structure_end:
        current = (epoch - 1) * steps_per_epoch + step_in_epoch
        total = structure_end * steps_per_epoch
        warmup = min(int(training["stage1_warmup_steps"]), total)
        if warmup > 0 and current <= warmup:
            progress = current / warmup
            return boundary_lr + progress * (peak - boundary_lr)
        progress = (current - warmup) / max(total - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return boundary_lr + cosine * (peak - boundary_lr)

    if epoch <= transition_end or group_name in {"encoder", "auxiliary"}:
        return boundary_lr

    current = (epoch - transition_end - 1) * steps_per_epoch + step_in_epoch
    total = max((total_epochs - transition_end) * steps_per_epoch, 1)
    warmup = min(
        int(training["stage2_head_warmup_epochs"]) * steps_per_epoch,
        total,
    )
    if current <= warmup:
        progress = current / max(warmup, 1)
        return boundary_lr + progress * (peak - boundary_lr)
    progress = (current - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return boundary_lr + cosine * (peak - boundary_lr)


def next_no_improve_count(
    epoch: int,
    transition_end_epoch: int,
    current: int,
    improved: bool,
) -> int:
    if int(epoch) <= int(transition_end_epoch):
        return 0
    return 0 if improved else int(current) + 1


def optimizer_steps_per_epoch(micro_steps_per_epoch: int, accumulation: int) -> int:
    accumulation = max(int(accumulation), 1)
    return max(1, math.ceil(int(micro_steps_per_epoch) / accumulation))


def optimizer_step_for_micro_step(micro_step: int, accumulation: int) -> int:
    accumulation = max(int(accumulation), 1)
    return max(1, math.ceil(int(micro_step) / accumulation))


def accumulation_divisor_for_micro_step(
    micro_step: int,
    total_micro_steps: int,
    accumulation: int,
) -> int:
    accumulation = max(int(accumulation), 1)
    micro_step = int(micro_step)
    total_micro_steps = int(total_micro_steps)
    remainder = total_micro_steps % accumulation
    if remainder and micro_step > total_micro_steps - remainder:
        return remainder
    return accumulation


def should_compute_auxiliary(
    epoch: int,
    transition_end_epoch: int,
) -> bool:
    if int(transition_end_epoch) <= 0:
        return True
    return int(epoch) <= int(transition_end_epoch)


def should_compute_dynamic_contrastive(
    epoch: int,
    step_in_epoch: int,
    max_epoch: int,
    every_steps: int,
) -> bool:
    interval = int(every_steps)
    if interval <= 0:
        raise ValueError("every_steps must be positive.")
    return (
        int(epoch) <= int(max_epoch)
        and (int(step_in_epoch) - 1) % interval == 0
    )


def should_compute_token_reliability_calibration(
    step_in_epoch: int,
    every_steps: int,
) -> bool:
    interval = int(every_steps)
    if interval <= 0:
        raise ValueError("every_steps must be positive.")
    return (int(step_in_epoch) - 1) % interval == 0


def token_reliability_gate_progress(
    epoch: int,
    step_in_epoch: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> float:
    if int(warmup_epochs) <= 0:
        return 1.0
    steps = max(int(steps_per_epoch), 1)
    completed = (max(int(epoch), 1) - 1) * steps + min(
        max(int(step_in_epoch), 1),
        steps,
    )
    return min(max(completed / (int(warmup_epochs) * steps), 0.0), 1.0)


def prepare_output_directory(
    output_dir: str | Path,
    resume: str | Path | None,
    write_artifacts: bool,
) -> None:
    output = Path(output_dir)
    if not write_artifacts:
        return
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError(
            f"Refusing to reuse non-empty output directory without resume: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)


class Trainer:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        datasets: DatasetBundle,
        model: PMFSurv,
        project_root: str | Path,
        write_artifacts: bool = True,
    ) -> None:
        self.cfg = cfg
        self.datasets = datasets
        self.project_root = Path(project_root)
        self.training_cfg = cfg["training"]
        self.external_center_ids = load_external_center_ids(
            cfg["data"].get("external_centers", {}),
            str(cfg["data"]["patient_id_col"]),
            self.project_root,
        )
        self.write_artifacts = bool(write_artifacts)
        self.rank, self.local_rank, self.world_size = _distributed_context()
        self.is_main = self.rank == 0
        if not torch.cuda.is_available():
            raise RuntimeError("ReST-OT training requires CUDA.")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        if self.training_cfg.get("cuda_library_warmup", False):
            _warmup_cuda_libraries(self.device)
        seed_everything(
            int(cfg["experiment"]["seed"]),
            self.rank,
            deterministic=bool(self.training_cfg["deterministic"]),
        )
        self.raw_model = model.to(self.device)
        self._set_time_boundaries()
        self.model = (
            DistributedDataParallel(
                self.raw_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            if self.world_size > 1
            else self.raw_model
        )
        self.objective = Objective(cfg)
        self.train_augmentation = SynchronizedBatchAugmentation(cfg)
        self.output_dir = Path(cfg["experiment"]["output_dir"]).expanduser()
        if not self.output_dir.is_absolute():
            self.output_dir = (self.project_root / self.output_dir).resolve()
        output_error = [None]
        if self.is_main:
            try:
                prepare_output_directory(
                    self.output_dir,
                    self.training_cfg.get("resume"),
                    self.write_artifacts,
                )
            except Exception as error:
                output_error[0] = str(error)
        if self.world_size > 1:
            dist.broadcast_object_list(output_error, src=0)
        if output_error[0] is not None:
            raise RuntimeError(output_error[0])
        self.optimizer = self._build_optimizer()
        self.train_steps_per_epoch = len(self._train_loader())
        schedule = cfg["schedule"]
        self.lr_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=4,
                threshold=0.0, threshold_mode="abs",
                min_lr=float(self.training_cfg["stage_boundary_lr"]), eps=0.0,
            )
            if schedule["structure_end_epoch"] == schedule["transition_end_epoch"] == 0
            else None
        )
        precision = str(self.training_cfg["mixed_precision"]).lower()
        self.autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        self.use_autocast = precision in {"fp16", "bf16"}
        self.scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")
        self.start_epoch = 1
        self.best_os = -float("inf")
        self.best_os_epoch = 0
        self.checkpoint_selection_metric = "os_cindex"
        self.best_selection_value = -float("inf")
        self.best_checkpoint_name = "best_os.pth"
        self.best_selection_label = "best_os"
        self.checkpoint_selection_split = "internal_validation"
        self.best_selection_epoch = 0
        self.extra_checkpoint_epochs = {
            int(value)
            for value in self.training_cfg.get(
                "extra_checkpoint_epochs",
                [],
            )
        }
        self.minimum_training_epochs = int(
            self.training_cfg.get("minimum_training_epochs", 0)
        )
        self.no_improve_epochs = 0
        resume = self.training_cfg.get("resume")
        if resume:
            self._resume(resume)
        elif self.is_main and self.write_artifacts:
            save_resolved_config(cfg, self.output_dir / "config.yaml")
            self._write_metadata()

    def _set_time_boundaries(self) -> None:
        rows = [record.row for record in self.datasets.train.records]
        data_cfg = self.cfg["data"]
        num_bins = int(self.cfg["model"]["num_time_bins"])
        os_boundaries = fit_time_boundaries(
            [float(row[data_cfg["os_time_col"]]) for row in rows],
            [float(row[data_cfg["os_event_col"]]) for row in rows],
            num_bins,
        )
        pfs_boundaries = fit_time_boundaries(
            [float(row[data_cfg["pfs_time_col"]]) for row in rows],
            [float(row[data_cfg["pfs_event_col"]]) for row in rows],
            num_bins,
        )
        self.raw_model.set_time_boundaries(
            torch.from_numpy(os_boundaries),
            torch.from_numpy(pfs_boundaries),
        )
        os_censoring = fit_censoring_distribution(
            [float(row[data_cfg["os_time_col"]]) for row in rows],
            [float(row[data_cfg["os_event_col"]]) for row in rows],
        )
        pfs_censoring = fit_censoring_distribution(
            [float(row[data_cfg["pfs_time_col"]]) for row in rows],
            [float(row[data_cfg["pfs_event_col"]]) for row in rows],
        )
        self.raw_model.os_head.set_censoring_distribution(
            torch.from_numpy(os_censoring[0]),
            torch.from_numpy(os_censoring[1]),
        )
        self.raw_model.pfs_head.set_censoring_distribution(
            torch.from_numpy(pfs_censoring[0]),
            torch.from_numpy(pfs_censoring[1]),
        )
        ipcw_cfg = self.cfg["loss"].get("rank_ipcw", {})
        if bool(ipcw_cfg.get("enabled", False)):
            survival_floor = float(ipcw_cfg.get("survival_floor", 0.1))
            cap_quantile = float(ipcw_cfg.get("cap_quantile", 0.99))
            os_cap = fit_ipcw_weight_cap(
                [float(row[data_cfg["os_time_col"]]) for row in rows],
                [float(row[data_cfg["os_event_col"]]) for row in rows],
                torch.from_numpy(os_censoring[0]),
                torch.from_numpy(os_censoring[1]),
                survival_floor,
                cap_quantile,
            )
            self.raw_model.os_head.set_rank_ipcw_cap(os_cap)
            if bool(ipcw_cfg.get("apply_to_pfs", False)):
                pfs_cap = fit_ipcw_weight_cap(
                    [float(row[data_cfg["pfs_time_col"]]) for row in rows],
                    [float(row[data_cfg["pfs_event_col"]]) for row in rows],
                    torch.from_numpy(pfs_censoring[0]),
                    torch.from_numpy(pfs_censoring[1]),
                    survival_floor,
                    cap_quantile,
                )
                self.raw_model.pfs_head.set_rank_ipcw_cap(pfs_cap)

    def _write_metadata(self) -> None:
        write_json(
            {
                "git_commit": git_commit(self.project_root),
                "preprocessor": self.datasets.preprocessor.state_dict(),
            },
            self.output_dir / "metadata.json",
        )

    def _parameter_groups(self) -> list[dict[str, Any]]:
        fusion_parameters = (
            list(self.raw_model.clinical_encoder.parameters())
            + list(self.raw_model.image_risk_pooler.parameters())
        )
        fusion_parameters += list(self.raw_model.interaction.parameters())
        if self.raw_model.qred is not None:
            fusion_parameters += list(self.raw_model.qred.parameters())
        if self.raw_model.token_quality_gate is not None:
            fusion_parameters += list(
                self.raw_model.token_quality_gate.parameters()
            )
        groups = [
            {
                "params": list(self.raw_model.encoder.parameters()),
                "lr": float(self.training_cfg["encoder_lr"]),
                "name": "encoder",
            },
            {
                "params": fusion_parameters,
                "lr": float(self.training_cfg["fusion_lr"]),
                "name": "fusion",
            },
            {
                "params": (
                    list(self.raw_model.os_head.parameters())
                    + list(self.raw_model.pfs_head.parameters())
                ),
                "lr": float(self.training_cfg["head_lr"]),
                "name": "survival_heads",
            },
            {
                "params": self.raw_model.auxiliary.parameters(),
                "lr": float(self.training_cfg["auxiliary_lr"]),
                "name": "auxiliary",
            },
        ]
        if self.raw_model.token_reliability_estimator is not None:
            groups.append(
                {
                    "params": self.raw_model.token_reliability_estimator.parameters(),
                    "lr": float(self.training_cfg["token_reliability_lr"]),
                    "name": "token_reliability",
                }
            )
        return groups

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self._parameter_groups(),
            weight_decay=float(self.training_cfg["weight_decay"]),
        )

    def _clip_gradients(self) -> None:
        max_norm = float(self.training_cfg["gradient_clip"])
        if self.raw_model.os_prediction_mode != "time_query_independent":
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), max_norm)
            return

        pfs_parameters = list(self.raw_model.pfs_head.parameters())
        pfs_parameter_ids = {id(parameter) for parameter in pfs_parameters}
        os_and_shared_parameters = [
            parameter
            for parameter in self.raw_model.parameters()
            if id(parameter) not in pfs_parameter_ids
        ]
        torch.nn.utils.clip_grad_norm_(os_and_shared_parameters, max_norm)
        torch.nn.utils.clip_grad_norm_(pfs_parameters, max_norm)

    def _adjust_lr(self, epoch: int, optimizer_step_in_epoch: int) -> None:
        accumulation = max(
            int(self.training_cfg["gradient_accumulation_steps"]),
            1,
        )
        steps_per_epoch = optimizer_steps_per_epoch(
            self.train_steps_per_epoch,
            accumulation,
        )
        if self.lr_scheduler is not None:
            step = (epoch - 1) * steps_per_epoch + optimizer_step_in_epoch
            warmup = min(
                int(self.training_cfg["stage1_warmup_steps"]),
                int(self.training_cfg["epochs"]) * steps_per_epoch,
            )
            if step > warmup:
                return
        for group in self.optimizer.param_groups:
            group_name = str(group["name"])
            group["lr"] = optimizer_group_lr(
                self.cfg,
                group_name,
                epoch,
                optimizer_step_in_epoch,
                steps_per_epoch,
            )

    def _step_validation_lr(
        self, val_metrics: Mapping[str, float], epoch: int,
    ) -> None:
        if self.lr_scheduler is None:
            return
        steps_per_epoch = optimizer_steps_per_epoch(
            self.train_steps_per_epoch,
            max(int(self.training_cfg["gradient_accumulation_steps"]), 1),
        )
        warmup = min(
            int(self.training_cfg["stage1_warmup_steps"]),
            int(self.training_cfg["epochs"]) * steps_per_epoch,
        )
        if epoch * steps_per_epoch < warmup:
            return
        score = torch.tensor(
            float(val_metrics["os_cindex"]) if self.is_main else float("nan"),
            device=self.device, dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.broadcast(score, src=0)
        self.lr_scheduler.step(float(score.item()))

    def _loader_generator(self, stream: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(
            int(self.cfg["experiment"]["seed"])
            + int(self.rank)
            + 1_000_003 * int(stream)
        )
        return generator

    def _train_loader(self) -> DataLoader:
        sampler = JointEventBatchSampler(
            self.datasets.train,
            batch_size=int(self.training_cfg["batch_size_per_gpu"]),
            seed=int(self.cfg["experiment"]["seed"]),
            rank=self.rank,
            world_size=self.world_size,
            max_oversample=float(self.training_cfg["max_stratum_oversample"]),
        )
        return DataLoader(
            self.datasets.train,
            batch_sampler=sampler,
            num_workers=int(self.cfg["data"]["num_workers"]),
            pin_memory=bool(self.cfg["data"]["pin_memory"]),
            persistent_workers=bool(self.cfg["data"]["persistent_workers"])
            and int(self.cfg["data"]["num_workers"]) > 0,
            worker_init_fn=seed_worker,
            generator=self._loader_generator(0),
        )

    def _eval_loader(self, split: str) -> DataLoader:
        dataset = getattr(self.datasets, split)
        if self.world_size > 1:
            dataset = Subset(
                dataset,
                list(range(self.rank, len(dataset), self.world_size)),
            )
        stream = {"val": 1, "test": 2}.get(split, 3)
        return DataLoader(
            dataset,
            batch_size=int(self.training_cfg["eval_batch_size_per_gpu"]),
            shuffle=False,
            num_workers=int(self.cfg["data"]["num_workers"]),
            pin_memory=bool(self.cfg["data"]["pin_memory"]),
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=self._loader_generator(stream),
        )

    def _distributed_concat_first_dim(
        self,
        local_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return local_tensor
        local_size = torch.tensor(
            [local_tensor.shape[0]],
            dtype=torch.long,
            device=self.device,
        )
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(gathered_sizes, local_size)
        sizes = [int(value.item()) for value in gathered_sizes]
        maximum_size = max(sizes)
        padded_shape = (maximum_size, *local_tensor.shape[1:])
        padded = local_tensor.new_zeros(padded_shape)
        padded[: local_tensor.shape[0]] = local_tensor
        gathered = [torch.zeros_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded.contiguous())
        return torch.cat(
            [tensor[:size] for tensor, size in zip(gathered, sizes)],
            dim=0,
        )

    def _autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.use_autocast
            else nullcontext()
        )

    def _run_train_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        accumulation = max(int(self.training_cfg["gradient_accumulation_steps"]), 1)
        totals: dict[str, float] = {}
        sample_count = 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(loader, start=1):
            optimizer_step_in_epoch = optimizer_step_for_micro_step(
                step,
                accumulation,
            )
            optimizer_steps = optimizer_steps_per_epoch(
                len(loader),
                accumulation,
            )
            accumulation_divisor = accumulation_divisor_for_micro_step(
                step,
                len(loader),
                accumulation,
            )
            batch = _move_batch(raw_batch, self.device)
            source_image = batch["image"]
            batch["image"] = self.train_augmentation(source_image)
            auxiliary_enabled = should_compute_auxiliary(
                epoch,
                int(self.cfg["schedule"]["transition_end_epoch"]),
            )
            dynamic_cfg = self.cfg["auxiliary"]
            contrastive_active = (
                auxiliary_enabled
                and bool(dynamic_cfg.get("dynamic_contrastive_enabled", True))
                and should_compute_dynamic_contrastive(
                    epoch,
                    step,
                    int(dynamic_cfg["dynamic_contrastive_until_epoch"]),
                    int(dynamic_cfg["dynamic_contrastive_every_steps"]),
                )
            )
            contrastive_indices = None
            contrastive_image = None
            if contrastive_active:
                contrastive_batch = min(
                    int(
                        dynamic_cfg[
                            "dynamic_contrastive_batch_size_per_gpu"
                        ]
                    ),
                    int(source_image.shape[0]),
                )
                contrastive_indices = torch.randperm(
                    source_image.shape[0],
                    device=source_image.device,
                )[:contrastive_batch]
                contrastive_image = self.train_augmentation(
                    source_image[contrastive_indices]
                )
            reliability_cfg = self.cfg["model"].get(
                "token_reliability", {}
            )
            reliability_calibration_cfg = reliability_cfg.get(
                "calibration", {}
            )
            reliability_calibration_active = bool(
                reliability_cfg.get("enabled", False)
            ) and should_compute_token_reliability_calibration(
                step,
                int(reliability_calibration_cfg.get("every_steps", 1)),
            )
            reliability_calibration_image = None
            if reliability_calibration_active:
                reliability_batch_size = min(
                    int(
                        reliability_calibration_cfg.get(
                            "batch_size_per_gpu", 1
                        )
                    ),
                    int(source_image.shape[0]),
                )
                reliability_indices = torch.randperm(
                    source_image.shape[0],
                    device=source_image.device,
                )[:reliability_batch_size]
                reliability_calibration_image = source_image[
                    reliability_indices
                ]
            reliability_gate_progress = token_reliability_gate_progress(
                epoch,
                step,
                len(loader),
                int(reliability_cfg.get("gate_warmup_epochs", 0)),
            )
            with self._autocast():
                outputs = self.model(
                    batch["image"],
                    batch["clinical_continuous"],
                    batch["clinical_categorical"],
                    batch["clinical_composite"],
                    compute_auxiliary=auxiliary_enabled,
                    contrastive_image=contrastive_image,
                    contrastive_indices=contrastive_indices,
                    token_reliability_gate_progress=(
                        reliability_gate_progress
                    ),
                    compute_token_reliability_calibration=(
                        reliability_calibration_active
                    ),
                    token_reliability_calibration_image=(
                        reliability_calibration_image
                    ),
                )
                loss, components, weights = self.objective(
                    self.raw_model,
                    outputs,
                    batch,
                    epoch,
                    training=True,
                    step_in_epoch=optimizer_step_in_epoch,
                    steps_per_epoch=optimizer_steps,
                )
                scaled_loss = loss / accumulation_divisor
            try:
                self.scaler.scale(scaled_loss).backward()
            except RuntimeError:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                print(
                    f"backward_failed rank={self.rank} epoch={epoch} step={step} "
                    f"allocated_MiB={torch.cuda.memory_allocated(self.device) / 2**20:.0f} "
                    f"reserved_MiB={torch.cuda.memory_reserved(self.device) / 2**20:.0f} "
                    f"free_MiB={free_bytes / 2**20:.0f} total_MiB={total_bytes / 2**20:.0f}",
                    flush=True,
                )
                raise
            if step % accumulation == 0 or step == len(loader):
                self.scaler.unscale_(self.optimizer)
                self._clip_gradients()
                self._adjust_lr(epoch, optimizer_step_in_epoch)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            batch_size = int(batch["image"].shape[0])
            sample_count += batch_size
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            log_every = int(self.training_cfg.get("log_every_steps", 0))
            if self.is_main and log_every > 0 and (step == 1 or step % log_every == 0):
                print(
                    f"train_epoch={epoch:03d} step={step}/{len(loader)} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"batch_per_gpu={batch_size} world_size={self.world_size} "
                    f"amp_scale={self.scaler.get_scale():.1f}",
                    flush=True,
                )
            totals["weight_survival"] = totals.get("weight_survival", 0.0) + weights.survival * batch_size
            totals["weight_reconstruction"] = (
                totals.get("weight_reconstruction", 0.0) + weights.reconstruction * batch_size
            )
            totals["weight_spatial"] = totals.get("weight_spatial", 0.0) + weights.spatial * batch_size
            totals["weight_dynamic"] = totals.get("weight_dynamic", 0.0) + weights.dynamic * batch_size
            totals["weight_post_structure_regularization"] = (
                totals.get("weight_post_structure_regularization", 0.0)
                + weights.post_structure_regularization * batch_size
            )
            totals["weight_ecomix_scale"] = (
                totals.get("weight_ecomix_scale", 0.0)
                + weights.ecomix_scale * batch_size
            )
            for group in self.optimizer.param_groups:
                lr_key = f"lr_{group['name']}"
                totals[lr_key] = (
                    totals.get(lr_key, 0.0)
                    + float(group["lr"]) * batch_size
                )
        tensor = torch.tensor(
            [sample_count] + [totals[key] for key in sorted(totals)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        count = max(float(tensor[0].item()), 1.0)
        return {
            key: float(tensor[index + 1].item()) / count
            for index, key in enumerate(sorted(totals))
        }

    def evaluate(self, split: str, epoch: int) -> dict[str, float]:
        if split != "test":
            return self._evaluate_split(split, epoch)[0]
        _, validation, _ = self._evaluate_split("val", epoch)
        scores, external, identifiers = self._evaluate_split("test", epoch)
        if self.is_main:
            specs = self.cfg["data"].get("external_centers", {})
            if not specs:
                specs = self.cfg.get("evaluation", {}).get("external_centers", {})
            center_ids = load_center_patient_ids(
                specs, str(self.cfg["data"]["patient_id_col"]), self.project_root,
            )
            scores.update(cox_probability_metrics(validation, external, identifiers, center_ids))
        return scores

    @torch.no_grad()
    def _evaluate_split(
        self, split: str, epoch: int,
    ) -> tuple[dict[str, float], dict[str, np.ndarray], list[str]]:
        self.raw_model.eval()
        totals: dict[str, float] = {}
        sample_count = 0
        report_centers = split == "test" and bool(self.external_center_ids)
        collect_identifiers = split == "test"
        patient_ids: list[str] = []
        arrays: dict[str, list[torch.Tensor]] = {
            "os_time": [],
            "os_event": [],
            "pfs_time": [],
            "pfs_event": [],
            "os_risk": [],
            "pfs_risk": [],
            "os_survival": [],
            "pfs_survival": [],
        }
        for raw_batch in self._eval_loader(split):
            batch = _move_batch(raw_batch, self.device)
            if collect_identifiers:
                patient_ids.extend(str(value) for value in batch["patient_id"])
            with self._autocast():
                outputs = self.raw_model(
                    batch["image"],
                    batch["clinical_continuous"],
                    batch["clinical_categorical"],
                    batch["clinical_composite"],
                    compute_auxiliary=False,
                    token_reliability_gate_progress=(
                        token_reliability_gate_progress(
                            epoch,
                            self.train_steps_per_epoch,
                            self.train_steps_per_epoch,
                            int(
                                self.cfg["model"]
                                .get("token_reliability", {})
                                .get("gate_warmup_epochs", 0)
                            ),
                        )
                    ),
                )
                _, components, _ = self.objective(
                    self.raw_model,
                    outputs,
                    batch,
                    epoch,
                    training=False,
                    step_in_epoch=self.train_steps_per_epoch,
                    steps_per_epoch=self.train_steps_per_epoch,
                )
            batch_size = int(batch["image"].shape[0])
            sample_count += batch_size
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            for key in arrays:
                source = outputs[key] if key in outputs else batch[key]
                arrays[key].append(source.detach().float())
        total_names = sorted(totals)
        total_tensor = torch.tensor(
            [sample_count] + [totals[key] for key in total_names],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        total_sample_count = max(float(total_tensor[0].item()), 1.0)
        metrics = {
            key: float(total_tensor[index + 1].item()) / total_sample_count
            for index, key in enumerate(total_names)
        }
        gathered = {
            key: self._distributed_concat_first_dim(
                torch.cat(values, dim=0)
            )
            for key, values in arrays.items()
        }
        if collect_identifiers and self.world_size > 1:
            rank_patient_ids = [None for _ in range(self.world_size)]
            dist.all_gather_object(rank_patient_ids, patient_ids)
            # Tensor gathering above also concatenates rank 0, rank 1, ... .
            patient_ids = [
                patient_id
                for rank_ids in rank_patient_ids
                for patient_id in rank_ids
            ]
        if not self.is_main:
            return {}, {}, []
        merged = {
            key: value.cpu().numpy()
            for key, value in gathered.items()
        }
        os_cindex = concordance_index(
            merged["os_time"],
            merged["os_event"],
            merged["os_risk"],
        )
        pfs_cindex = concordance_index(
            merged["pfs_time"],
            merged["pfs_event"],
            merged["pfs_risk"],
        )
        metrics.update(
            {
                "os_cindex": os_cindex,
                "pfs_cindex": pfs_cindex,
                "os_harrell_cindex": os_cindex,
                "pfs_harrell_cindex": pfs_cindex,
                "os_ibs": integrated_brier_score(
                    merged["os_time"],
                    merged["os_event"],
                    merged["os_survival"],
                    self.raw_model.os_head.boundaries.cpu().numpy(),
                    self.raw_model.os_head.censoring_times.cpu().numpy(),
                    self.raw_model.os_head.censoring_survival.cpu().numpy(),
                ),
                "pfs_ibs": integrated_brier_score(
                    merged["pfs_time"],
                    merged["pfs_event"],
                    merged["pfs_survival"],
                    self.raw_model.pfs_head.boundaries.cpu().numpy(),
                    self.raw_model.pfs_head.censoring_times.cpu().numpy(),
                    self.raw_model.pfs_head.censoring_survival.cpu().numpy(),
                ),
            }
        )
        if report_centers:
            metrics.update(
                external_center_metrics(
                    patient_ids, merged, self.external_center_ids
                )
            )
        return metrics, merged, patient_ids

    def _checkpoint(self) -> dict[str, Any]:
        """Return only inference state and the clinical preprocessing state."""
        return {
            "model": self.raw_model.state_dict(),
            "preprocessor": self.datasets.preprocessor.state_dict(),
        }

    def _prepare_model_for_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
    ) -> None:
        state = checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise RuntimeError("Checkpoint has no model state mapping.")

        def required(name: str) -> torch.Tensor:
            value = state.get(name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Checkpoint model state has no tensor: {name}")
            return value

        self.raw_model.set_time_boundaries(
            required("os_head.boundaries"),
            required("pfs_head.boundaries"),
        )
        self.raw_model.os_head.set_censoring_distribution(
            required("os_head.censoring_times"),
            required("os_head.censoring_survival"),
        )
        self.raw_model.pfs_head.set_censoring_distribution(
            required("pfs_head.censoring_times"),
            required("pfs_head.censoring_survival"),
        )
        if "os_head.rank_ipcw_cap" in state:
            self.raw_model.os_head.set_rank_ipcw_cap(
                required("os_head.rank_ipcw_cap")
            )
        if "pfs_head.rank_ipcw_cap" in state:
            self.raw_model.pfs_head.set_rank_ipcw_cap(
                required("pfs_head.rank_ipcw_cap")
            )

    def _resume(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("preprocessor") != self.datasets.preprocessor.state_dict():
            raise RuntimeError(
                "Checkpoint clinical preprocessor does not match the dataset."
            )
        self._prepare_model_for_checkpoint(checkpoint)
        self.raw_model.load_state_dict(checkpoint["model"], strict=True)
        # Model-only checkpoints are warm starts. Optimizer, AMP and random
        # state are intentionally initialized from the current configuration.
        self.start_epoch = 1
        self.best_os = -float("inf")
        self.best_os_epoch = 0
        self.best_selection_epoch = 0
        self.no_improve_epochs = 0

    def _write_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Mapping[str, float],
        val_metrics: Mapping[str, float],
    ) -> None:
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        path = self.output_dir / "metrics.csv"
        write_header = not path.exists() or path.stat().st_size == 0
        fieldnames = list(row)
        if not write_header:
            with path.open("r", encoding="utf-8", newline="") as handle:
                existing_fields = next(csv.reader(handle), [])
            if [field for field in existing_fields if not field.startswith("test_")] != fieldnames:
                raise RuntimeError(
                    "Existing metrics.csv schema does not match the current run. "
                    "Use a new experiment.output_dir or resume with the matching code."
                )
            fieldnames = existing_fields
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _load_model_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        return self.load_checkpoint_for_evaluation(checkpoint)

    def load_checkpoint_for_evaluation(
        self,
        checkpoint: Mapping[str, Any],
    ) -> int:
        self._prepare_model_for_checkpoint(checkpoint)
        self.raw_model.load_state_dict(checkpoint["model"], strict=True)
        return 1

    def _final_test(self) -> None:
        reports = {}
        path = self.output_dir / self.best_checkpoint_name
        if path.exists():
            self._load_model_checkpoint(path)
            selected_epoch = [self.best_selection_epoch if self.is_main else None]
            if self.world_size > 1:
                dist.broadcast_object_list(selected_epoch, src=0)
            epoch = int(selected_epoch[0] or 1)
            test_metrics = self.evaluate("test", epoch)
            if self.is_main:
                reports[self.best_selection_label] = {
                    "epoch": epoch,
                    "selection_metric": self.checkpoint_selection_metric,
                    "selection_split": self.checkpoint_selection_split,
                    "selection_value": self.best_selection_value,
                    "test": test_metrics,
                }
        if self.is_main:
            write_json(reports, self.output_dir / "test_results.json")

    def fit(self) -> None:
        loader = self._train_loader()
        if self.is_main:
            print(
                f"PMF-Surv start output={self.output_dir}",
                flush=True,
            )
        for epoch in range(self.start_epoch, int(self.training_cfg["epochs"]) + 1):
            started = time.time()
            train_metrics = self._run_train_epoch(epoch, loader)
            if self.world_size > 1:
                dist.barrier()
            val_metrics = self.evaluate("val", epoch)
            self._step_validation_lr(val_metrics, epoch)
            if self.world_size > 1:
                dist.barrier()
            if self.is_main:
                self._write_epoch_metrics(
                    epoch,
                    train_metrics,
                    val_metrics,
                )
                weights = self.objective.schedule.at(epoch)
                selection_value = float(val_metrics["os_cindex"])
                selection_improved = (
                    weights.survival > 0
                    and math.isfinite(selection_value)
                    and selection_value > self.best_selection_value
                )
                if selection_improved:
                    self.best_os = selection_value
                    self.best_os_epoch = epoch
                    self.best_selection_value = selection_value
                    self.best_selection_epoch = epoch
                monitoring_warmup_end = int(
                    self.cfg["schedule"]["transition_end_epoch"]
                )
                self.no_improve_epochs = next_no_improve_count(
                    epoch,
                    monitoring_warmup_end,
                    self.no_improve_epochs,
                    selection_improved,
                )
                checkpoint = self._checkpoint()
                if selection_improved:
                    atomic_torch_save(
                        checkpoint,
                        self.output_dir / self.best_checkpoint_name,
                    )
                    write_json(
                        {
                            "epoch": epoch,
                            "checkpoint": self.best_checkpoint_name,
                            "selection_metric": self.checkpoint_selection_metric,
                            "selection_split": self.checkpoint_selection_split,
                            "selection_value": selection_value,
                        },
                        self.output_dir / "best_checkpoint.json",
                    )
                if epoch in self.extra_checkpoint_epochs:
                    atomic_torch_save(
                        checkpoint,
                        self.output_dir / f"epoch_{epoch:03d}.pth",
                    )
                print(
                    f"epoch={epoch:03d} train={train_metrics['total_loss']:.4f} "
                    f"val={val_metrics['total_loss']:.4f} "
                    f"val_os={val_metrics['os_cindex']:.4f} val_pfs={val_metrics['pfs_cindex']:.4f} "
                    f"{self.best_selection_label}="
                    f"{self.best_selection_value:.4f}"
                    f"@{self.best_selection_epoch:03d} "
                    f"seconds={time.time() - started:.1f}",
                    flush=True,
                )
            stop_tensor = torch.tensor(
                [
                    int(
                        self.is_main
                        and self.no_improve_epochs >= int(self.training_cfg["patience"])
                        and epoch >= self.minimum_training_epochs
                        and epoch
                        > int(self.cfg["schedule"]["transition_end_epoch"])
                    )
                ],
                device=self.device,
            )
            if self.world_size > 1:
                dist.broadcast(stop_tensor, src=0)
            if bool(stop_tensor.item()):
                break
        if self.world_size > 1:
            dist.barrier()
        self._final_test()
        if self.world_size > 1:
            dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
