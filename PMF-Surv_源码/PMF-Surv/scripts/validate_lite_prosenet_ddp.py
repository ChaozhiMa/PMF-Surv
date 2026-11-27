from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(
    "/home/image020/mm_tqsurv_rapot_midband_quality_codeconfig/code"
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from restot_surv.config import load_config, paper_config_path
from restot_surv.data import build_datasets
from restot_surv.models.lite_prosenet import LiteProSENet


def main() -> None:
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config_path = paper_config_path(PROJECT_ROOT)
    cfg = load_config(config_path)
    datasets = build_datasets(cfg)
    subset = Subset(datasets.train, [2 * rank, 2 * rank + 1])
    batch = next(iter(DataLoader(subset, batch_size=2, num_workers=0)))
    raw_model = LiteProSENet(
        image_shape=cfg["data"]["image_shape"],
        num_continuous=len(cfg["data"]["continuous_vars"]),
        categorical_cardinalities=datasets.preprocessor.categorical_cardinalities(),
        composite_cardinalities=datasets.preprocessor.composite_cardinalities(),
    ).to(device)
    model = DistributedDataParallel(raw_model, device_ids=[local_rank])
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(
            batch["image"].to(device),
            batch["clinical_continuous"].to(device),
            batch["clinical_categorical"].to(device),
            batch["clinical_composite"].to(device),
        )
        loss = outputs["os_prediction"].mean() + outputs["pfs_prediction"].mean()
    loss.backward()
    finite = torch.tensor(
        float(torch.isfinite(loss).item()),
        device=device,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            {
                "world_size": dist.get_world_size(),
                "all_ranks_finite": bool(finite.item()),
                "peak_gpu_memory_gib_per_rank": torch.cuda.max_memory_allocated()
                / 1024**3,
            }
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
