from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from restot_surv.config import load_config, paper_config_path
from restot_surv.data import build_datasets
from restot_surv.models import PMFSurv
from restot_surv.trainer import (
    Trainer,
    cleanup_distributed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PMF-Surv checkpoint.")
    parser.add_argument("--config", type=Path, default=paper_config_path(PROJECT_ROOT))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    preprocessor_state = checkpoint.get("preprocessor")
    if preprocessor_state is None:
        raise RuntimeError("Checkpoint has no clinical preprocessor.")
    datasets = build_datasets(cfg, preprocessor_state=preprocessor_state)
    model = PMFSurv(
        cfg,
        datasets.preprocessor.categorical_cardinalities(),
        datasets.preprocessor.composite_cardinalities(),
    )
    trainer = Trainer(
        cfg,
        datasets,
        model,
        PROJECT_ROOT,
        write_artifacts=False,
    )
    epoch = trainer.load_checkpoint_for_evaluation(checkpoint)
    try:
        metrics = trainer.evaluate(args.split, epoch)
        if trainer.is_main:
            print(json.dumps(metrics, indent=2, sort_keys=True))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

