from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from restot_surv.config import load_config, validate_config, paper_config_path
from restot_surv.data import build_datasets
from restot_surv.models import PMFSurv
from restot_surv.trainer import (
    Trainer,
    cleanup_distributed,
)
from restot_surv.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PMF-Surv.")
    parser.add_argument("--config", type=Path, default=paper_config_path(PROJECT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None and args.seed != int(cfg["experiment"]["seed"]):
        cfg["experiment"]["seed"] = args.seed
        if args.output_dir is None:
            output = Path(cfg["experiment"]["output_dir"])
            cfg["experiment"]["output_dir"] = str(
                output.with_name(f"{output.name}_seed{args.seed}")
            )
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.resume:
        cfg["training"]["resume"] = args.resume
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
        cfg["schedule"]["transition_end_epoch"] = min(
            int(cfg["schedule"]["transition_end_epoch"]),
            int(args.epochs),
        )
        cfg["schedule"]["structure_end_epoch"] = min(
            int(cfg["schedule"]["structure_end_epoch"]),
            max(0, int(cfg["schedule"]["transition_end_epoch"]) - 1),
        )
    validate_config(cfg)

    # Seed before dataset/model construction so every DDP rank starts from
    # the same deterministic initialization. Trainer then derives rank-local
    # streams for stochastic training operations.
    seed_everything(
        int(cfg["experiment"]["seed"]),
        rank=0,
        deterministic=bool(cfg["training"]["deterministic"]),
    )

    preprocessor_state = None
    if cfg["training"].get("resume"):
        resume_checkpoint = torch.load(
            cfg["training"]["resume"],
            map_location="cpu",
        )
        preprocessor_state = resume_checkpoint.get("preprocessor")
        if preprocessor_state is None:
            raise RuntimeError("Resume checkpoint has no clinical preprocessor.")
    datasets = build_datasets(cfg, preprocessor_state=preprocessor_state)
    model = PMFSurv(
        cfg,
        datasets.preprocessor.categorical_cardinalities(),
        datasets.preprocessor.composite_cardinalities(),
    )
    try:
        Trainer(cfg, datasets, model, PROJECT_ROOT).fit()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
