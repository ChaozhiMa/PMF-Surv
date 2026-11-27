from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(
    "/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig/code"
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from restot_surv.config import load_config, paper_config_path
from restot_surv.data import build_datasets
from restot_surv.models.lite_prosenet import LiteProSENet


def main() -> None:
    config_path = paper_config_path(PROJECT_ROOT)
    cfg = load_config(config_path)
    datasets = build_datasets(cfg)
    loader = DataLoader(datasets.train, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = torch.device("cuda:0")
    model = LiteProSENet(
        image_shape=cfg["data"]["image_shape"],
        num_continuous=len(cfg["data"]["continuous_vars"]),
        categorical_cardinalities=datasets.preprocessor.categorical_cardinalities(),
        composite_cardinalities=datasets.preprocessor.composite_cardinalities(),
    ).to(device)
    model.train()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(
            batch["image"].to(device),
            batch["clinical_continuous"].to(device),
            batch["clinical_categorical"].to(device),
            batch["clinical_composite"].to(device),
        )
        loss = outputs["os_prediction"].mean() + outputs["pfs_prediction"].mean()
    loss.backward()
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise RuntimeError("Non-finite Lite-ProSENet output on a real batch.")
    if not any(parameter.grad is not None for parameter in model.image_tower.parameters()):
        raise RuntimeError("OS loss did not update the ProSENet image tower.")
    print(
        {
            "split_sizes": {
                "train": len(datasets.train),
                "val": len(datasets.val),
                "test": len(datasets.test),
            },
            "image_shape": tuple(batch["image"].shape),
            "clinical_variable_count": model.clinical_tower.variable_count,
            "output_shapes": {name: tuple(value.shape) for name, value in outputs.items()},
            "loss": float(loss.detach().cpu()),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
    )


if __name__ == "__main__":
    main()
