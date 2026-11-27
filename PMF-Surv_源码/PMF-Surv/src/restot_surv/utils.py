from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def seed_everything(
    seed: int,
    rank: int = 0,
    deterministic: bool = False,
) -> None:
    actual = int(seed) + int(rank)
    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)
    torch.cuda.manual_seed_all(actual)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Propagate PyTorch's worker seed to Python and NumPy RNGs."""
    del worker_id
    worker_seed = int(torch.initial_seed()) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def git_commit(root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "uncommitted"


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, output)


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
