from __future__ import annotations

import sys

from run_multimodal_mtlr_baselines import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--model", "cmc_surv"]
    main()
