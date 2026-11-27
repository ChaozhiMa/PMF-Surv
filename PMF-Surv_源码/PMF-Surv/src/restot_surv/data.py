from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset


def _clean_category(value: object) -> str:
    text = str(value if value is not None else "").strip()
    return "__missing__" if text == "" or text.lower() in {"nan", "na", "none", "null"} else text


def _to_float(value: object) -> float:
    text = str(value if value is not None else "").strip()
    if text == "" or text.lower() in {"nan", "na", "none", "null"}:
        return float("nan")
    return float(text)


@dataclass(frozen=True)
class CaseRecord:
    patient_id: str
    image_path: Path
    row: Mapping[str, str]


@dataclass
class ClinicalPreprocessor:
    continuous_vars: tuple[str, ...]
    categorical_vars: tuple[str, ...]
    composite_vars: tuple[str, ...]
    continuous_mean: dict[str, float]
    continuous_std: dict[str, float]
    categorical_maps: dict[str, dict[str, int]]
    composite_maps: dict[str, dict[str, int]]

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, str]],
        continuous_vars: Sequence[str],
        categorical_vars: Sequence[str],
        composite_vars: Sequence[str],
    ) -> "ClinicalPreprocessor":
        continuous_mean: dict[str, float] = {}
        continuous_std: dict[str, float] = {}
        for name in continuous_vars:
            values = np.asarray([_to_float(row.get(name, "")) for row in rows], dtype=np.float32)
            finite = values[np.isfinite(values)]
            mean = float(finite.mean()) if finite.size else 0.0
            std = float(finite.std()) if finite.size else 1.0
            continuous_mean[str(name)] = mean
            continuous_std[str(name)] = max(std, 1e-6)

        def make_maps(names: Sequence[str]) -> dict[str, dict[str, int]]:
            result: dict[str, dict[str, int]] = {}
            for name in names:
                mapping = {"__missing__": 0}
                for value in sorted({_clean_category(row.get(name, "")) for row in rows}):
                    if value not in mapping:
                        mapping[value] = len(mapping)
                result[str(name)] = mapping
            return result

        return cls(
            continuous_vars=tuple(str(v) for v in continuous_vars),
            categorical_vars=tuple(str(v) for v in categorical_vars),
            composite_vars=tuple(str(v) for v in composite_vars),
            continuous_mean=continuous_mean,
            continuous_std=continuous_std,
            categorical_maps=make_maps(categorical_vars),
            composite_maps=make_maps(composite_vars),
        )

    def transform(self, row: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        continuous = []
        for name in self.continuous_vars:
            value = _to_float(row.get(name, ""))
            if not np.isfinite(value):
                value = self.continuous_mean[name]
            continuous.append((value - self.continuous_mean[name]) / self.continuous_std[name])

        def categories(names: Sequence[str], maps: Mapping[str, Mapping[str, int]]) -> np.ndarray:
            values = []
            for name in names:
                text = _clean_category(row.get(name, ""))
                values.append(int(maps[name].get(text, 0)))
            return np.asarray(values, dtype=np.int64)

        return (
            np.asarray(continuous, dtype=np.float32),
            categories(self.categorical_vars, self.categorical_maps),
            categories(self.composite_vars, self.composite_maps),
        )

    def categorical_cardinalities(self) -> list[int]:
        return [len(self.categorical_maps[name]) for name in self.categorical_vars]

    def composite_cardinalities(self) -> list[int]:
        return [len(self.composite_maps[name]) for name in self.composite_vars]

    def state_dict(self) -> dict[str, Any]:
        return {
            "continuous_vars": list(self.continuous_vars),
            "categorical_vars": list(self.categorical_vars),
            "composite_vars": list(self.composite_vars),
            "continuous_mean": self.continuous_mean,
            "continuous_std": self.continuous_std,
            "categorical_maps": self.categorical_maps,
            "composite_maps": self.composite_maps,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ClinicalPreprocessor":
        return cls(
            continuous_vars=tuple(state["continuous_vars"]),
            categorical_vars=tuple(state["categorical_vars"]),
            composite_vars=tuple(state["composite_vars"]),
            continuous_mean={str(k): float(v) for k, v in state["continuous_mean"].items()},
            continuous_std={str(k): float(v) for k, v in state["continuous_std"].items()},
            categorical_maps={
                str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in state["categorical_maps"].items()
            },
            composite_maps={
                str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in state["composite_maps"].items()
            },
        )


def read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return [dict(row) for row in reader]


def find_image(
    root: str | Path,
    patient_id: str,
    patterns: Sequence[str],
    extensions: Sequence[str],
) -> Path | None:
    directory = Path(root)
    search_directories = [directory]
    if directory.is_dir():
        search_directories.extend(
            sorted(path for path in directory.iterdir() if path.is_dir())
        )
    for pattern in patterns:
        stem = pattern.format(pid=patient_id)
        for extension in extensions:
            filename = f"{stem}{extension}"
            for search_directory in search_directories:
                candidate = search_directory / filename
                if candidate.is_file():
                    return candidate
    return None


def build_records(cfg: Mapping[str, Any], split: str) -> list[CaseRecord]:
    data_cfg = cfg["data"]
    rows = read_rows(data_cfg[f"{split}_csv"])
    records: list[CaseRecord] = []
    missing: list[str] = []
    seen_patient_ids: set[str] = set()
    for row in rows:
        patient_id = str(row.get(data_cfg["patient_id_col"], "")).strip()
        if not patient_id:
            raise ValueError(f"Missing patient id in {split} CSV.")
        if patient_id in seen_patient_ids:
            raise ValueError(f"Duplicate patient id {patient_id!r} in {split} CSV.")
        seen_patient_ids.add(patient_id)
        for endpoint in ("os", "pfs"):
            time = _to_float(row.get(data_cfg[f"{endpoint}_time_col"], ""))
            event = _to_float(row.get(data_cfg[f"{endpoint}_event_col"], ""))
            if not np.isfinite(time) or time < 0.0:
                raise ValueError(
                    f"{split} patient {patient_id}: invalid {endpoint.upper()} time {time}."
                )
            if event not in {0.0, 1.0}:
                raise ValueError(
                    f"{split} patient {patient_id}: invalid {endpoint.upper()} event {event}."
                )
        image = find_image(
            data_cfg[f"{split}_image_dir"],
            patient_id,
            data_cfg["filename_patterns"],
            data_cfg["image_exts"],
        )
        if image is None:
            missing.append(patient_id)
            continue
        records.append(CaseRecord(patient_id, image, row))
    if missing and bool(data_cfg.get("strict_images", True)):
        raise FileNotFoundError(f"{split}: {len(missing)} images missing, examples={missing[:10]}")
    return records


def _ensure_phase_first(array: np.ndarray, num_phases: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"Expected 4D multi-phase volume, got {array.shape}.")
    if array.shape[0] == num_phases:
        return array
    if array.shape[-1] == num_phases:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"Cannot identify {num_phases} phases in volume shape {array.shape}.")


def load_volume(path: str | Path, num_phases: int) -> np.ndarray:
    file_path = Path(path)
    name = file_path.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        array = np.asarray(nib.load(str(file_path)).get_fdata(), dtype=np.float32)
    elif name.endswith(".npy"):
        array = np.asarray(np.load(file_path), dtype=np.float32)
    elif name.endswith(".npz"):
        with np.load(file_path) as payload:
            key = next((key for key in ("image", "volume", "arr_0") if key in payload), next(iter(payload)))
            array = np.asarray(payload[key], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported image format: {file_path}")
    return _ensure_phase_first(array, num_phases)


def center_crop_or_pad(
    volume: np.ndarray,
    target_shape: Sequence[int],
) -> np.ndarray:
    target = tuple(int(v) for v in target_shape)
    output = np.zeros(target, dtype=np.float32)
    source_slices = []
    destination_slices = []
    for source_size, destination_size in zip(volume.shape, target):
        if source_size >= destination_size:
            start = (source_size - destination_size) // 2
            source_slices.append(slice(start, start + destination_size))
            destination_slices.append(slice(0, destination_size))
        else:
            start = (destination_size - source_size) // 2
            source_slices.append(slice(0, source_size))
            destination_slices.append(slice(start, start + source_size))
    output[tuple(destination_slices)] = volume[tuple(source_slices)]
    return output


def normalize_per_phase(volume: np.ndarray) -> np.ndarray:
    output = np.zeros_like(volume, dtype=np.float32)
    for phase in range(volume.shape[0]):
        values = volume[phase]
        finite = values[np.isfinite(values)]
        if finite.size == 0 or float(finite.std()) < 1e-6:
            continue
        mean = float(finite.mean())
        std = float(finite.std())
        clean_values = np.where(np.isfinite(values), values, mean)
        output[phase] = (clean_values - mean) / std
    return output


def normalize_joint(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0 or float(finite.std()) < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    mean = float(finite.mean())
    return (
        (np.where(np.isfinite(volume), volume, mean) - mean) / float(finite.std())
    ).astype(np.float32)


def normalize_volume(volume: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        finite = volume[np.isfinite(volume)]
        fill = float(finite.mean()) if finite.size else 0.0
        return np.ascontiguousarray(
            np.where(np.isfinite(volume), volume, fill),
            dtype=np.float32,
        )
    if mode == "zscore_joint":
        return normalize_joint(volume)
    if mode == "zscore_per_phase":
        return normalize_per_phase(volume)
    raise ValueError(f"Unsupported normalization mode: {mode}")


class SynchronizedBatchAugmentation:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        augmentation = cfg["augmentation"]
        self.enabled = bool(augmentation.get("enabled", True))
        self.flip_probability = float(augmentation.get("flip_probability", 0.5))
        self.noise_probability = float(augmentation.get("noise_probability", 0.15))
        self.noise_std = float(augmentation.get("noise_std", 0.02))
        self.intensity_probability = float(augmentation.get("intensity_probability", 0.2))
        self.intensity_scale = tuple(float(v) for v in augmentation.get("intensity_scale", (0.9, 1.1)))
        self.intensity_shift = tuple(float(v) for v in augmentation.get("intensity_shift", (-0.1, 0.1)))

    def __call__(
        self,
        volume: torch.Tensor,
    ) -> torch.Tensor:
        if volume.ndim != 5:
            raise ValueError(
                "Synchronized batch augmentation expects [B,P,D,H,W]."
            )
        if not self.enabled:
            return volume
        augmented = []
        for sample in volume:
            output = sample
            for axis in (1, 2, 3):
                if torch.rand((), device=sample.device) < self.flip_probability:
                    output = torch.flip(output, dims=(axis,))
            if torch.rand((), device=sample.device) < self.intensity_probability:
                scale = torch.empty(
                    (),
                    device=sample.device,
                    dtype=sample.dtype,
                ).uniform_(*self.intensity_scale)
                shift = torch.empty(
                    (),
                    device=sample.device,
                    dtype=sample.dtype,
                ).uniform_(*self.intensity_shift)
                output = output * scale + shift
            if torch.rand((), device=sample.device) < self.noise_probability:
                shared_noise = torch.randn(
                    (1, *sample.shape[-3:]),
                    device=sample.device,
                    dtype=sample.dtype,
                )
                output = output + shared_noise * self.noise_std
            augmented.append(output)
        return torch.stack(augmented, dim=0)


class SurvivalDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CaseRecord],
        preprocessor: ClinicalPreprocessor,
        cfg: Mapping[str, Any],
    ) -> None:
        self.records = list(records)
        self.preprocessor = preprocessor
        self.data_cfg = cfg["data"]
        self.num_phases = int(cfg["model"]["num_phases"])
        self.image_shape = tuple(int(v) for v in self.data_cfg["image_shape"])
        self.source_phase_order = tuple(
            str(value)
            for value in self.data_cfg.get(
                "source_phase_order",
                self.data_cfg["phase_order"],
            )
        )
        self.phase_order = tuple(
            str(value) for value in self.data_cfg["phase_order"]
        )
        self.phase_indices = tuple(
            self.source_phase_order.index(phase) for phase in self.phase_order
        )
        if len(self.phase_indices) != self.num_phases:
            raise ValueError(
                "The selected data phases must match model.num_phases."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        raw_image = load_volume(record.image_path, len(self.source_phase_order))
        raw_image = raw_image[list(self.phase_indices)]
        raw_image = center_crop_or_pad(raw_image, self.image_shape)
        image = normalize_volume(
            raw_image,
            str(self.data_cfg.get("normalization", "zscore_joint")),
        )
        continuous, categorical, composite = self.preprocessor.transform(record.row)
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "patient_id": record.patient_id,
            "image": torch.from_numpy(image),
            "clinical_continuous": torch.from_numpy(continuous),
            "clinical_categorical": torch.from_numpy(categorical),
            "clinical_composite": torch.from_numpy(composite),
            "os_time": torch.tensor(_to_float(record.row[self.data_cfg["os_time_col"]]), dtype=torch.float32),
            "os_event": torch.tensor(_to_float(record.row[self.data_cfg["os_event_col"]]), dtype=torch.float32),
            "pfs_time": torch.tensor(_to_float(record.row[self.data_cfg["pfs_time_col"]]), dtype=torch.float32),
            "pfs_event": torch.tensor(_to_float(record.row[self.data_cfg["pfs_event_col"]]), dtype=torch.float32),
        }


@dataclass
class DatasetBundle:
    train: SurvivalDataset
    val: SurvivalDataset
    test: SurvivalDataset
    preprocessor: ClinicalPreprocessor


def build_datasets(
    cfg: Mapping[str, Any],
    preprocessor_state: Mapping[str, Any] | None = None,
) -> DatasetBundle:
    train_records = build_records(cfg, "train")
    val_records = build_records(cfg, "val")
    test_records = build_records(cfg, "test")
    data_cfg = cfg["data"]
    preprocessor = (
        ClinicalPreprocessor.from_state_dict(preprocessor_state)
        if preprocessor_state is not None
        else ClinicalPreprocessor.fit(
            [record.row for record in train_records],
            data_cfg["continuous_vars"],
            data_cfg["categorical_vars"],
            data_cfg["composite_vars"],
        )
    )
    return DatasetBundle(
        train=SurvivalDataset(train_records, preprocessor, cfg),
        val=SurvivalDataset(val_records, preprocessor, cfg),
        test=SurvivalDataset(test_records, preprocessor, cfg),
        preprocessor=preprocessor,
    )


class JointEventBatchSampler(BatchSampler):
    """OS/PFS event coverage with no duplicate patient in a DDP global batch."""

    def __init__(
        self,
        dataset: SurvivalDataset,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        max_oversample: float = 2.0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_oversample = float(max_oversample)
        self.epoch = 0
        self.strata: dict[tuple[int, int], list[int]] = {}
        for index, record in enumerate(dataset.records):
            row = record.row
            key = (
                int(float(row[dataset.data_cfg["os_event_col"]]) > 0.5),
                int(float(row[dataset.data_cfg["pfs_event_col"]]) > 0.5),
            )
            self.strata.setdefault(key, []).append(index)
        self.nonempty = tuple(key for key, values in sorted(self.strata.items()) if values)
        if not self.nonempty:
            raise ValueError("No non-empty OS/PFS event strata.")
        if self.batch_size <= 0 or self.world_size <= 0:
            raise ValueError("batch_size and world_size must be positive.")
        if self.max_oversample < 1.0:
            raise ValueError("max_oversample must be at least 1.")
        global_batch = self.batch_size * self.world_size
        self.steps_per_rank = max(1, math.ceil(len(dataset) / global_batch))
        self.global_batch_size = self.batch_size * self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_rank

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1009)
        pools = {key: values.copy() for key, values in self.strata.items() if values}
        for values in pools.values():
            rng.shuffle(values)
        pointers = {key: 0 for key in pools}
        draws = {key: 0 for key in pools}
        caps = {
            key: max(len(values), int(math.ceil(len(values) * self.max_oversample)))
            for key, values in pools.items()
        }

        def take(key: tuple[int, int], forbidden_ids: set[str]) -> int | None:
            if draws[key] >= caps[key]:
                return None
            values = pools[key]
            for _ in range(len(values) + 1):
                if pointers[key] >= len(values):
                    rng.shuffle(values)
                    pointers[key] = 0
                candidate = values[pointers[key]]
                pointers[key] += 1
                patient_id = str(self.dataset.records[candidate].patient_id)
                if patient_id not in forbidden_ids:
                    draws[key] += 1
                    return candidate
            return None

        for batch_index in range(self.steps_per_rank):
            global_batch: list[int] = []
            forbidden_ids: set[str] = set()
            start = batch_index % len(self.nonempty)
            order = self.nonempty[start:] + self.nonempty[:start]
            for key in order:
                if len(global_batch) >= self.global_batch_size:
                    break
                candidate = take(key, forbidden_ids)
                if candidate is not None:
                    global_batch.append(candidate)
                    forbidden_ids.add(
                        str(self.dataset.records[candidate].patient_id)
                    )
            while len(global_batch) < self.global_batch_size:
                available = [
                    key
                    for key in self.nonempty
                    if draws[key] < caps[key]
                ]
                available.sort(
                    key=lambda key: (
                        draws[key] / len(pools[key]),
                        -len(pools[key]),
                    )
                )
                candidate = None
                for key in available:
                    candidate = take(key, forbidden_ids)
                    if candidate is not None:
                        break
                if candidate is None:
                    raise RuntimeError(
                        "Cannot construct a full unique-patient batch within "
                        "max_stratum_oversample. Increase the cap or reduce batch size."
                    )
                global_batch.append(candidate)
                forbidden_ids.add(str(self.dataset.records[candidate].patient_id))
            start_index = self.rank * self.batch_size
            end_index = start_index + self.batch_size
            yield global_batch[start_index:end_index]
