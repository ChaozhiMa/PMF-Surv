from __future__ import annotations

from typing import Any, Mapping


def configured_clinical_token_count(cfg: Mapping[str, Any]) -> int:
    data_cfg = cfg["data"]
    return sum(
        len(data_cfg[key])
        for key in (
            "continuous_vars",
            "categorical_vars",
            "composite_vars",
        )
    )


def configured_image_risk_token_count(cfg: Mapping[str, Any]) -> int:
    return int(cfg["model"]["image_risk_tokens"])


def configured_matched_token_count(cfg: Mapping[str, Any]) -> int:
    semantic_transfer = bool(
        cfg["model"].get("ablation", {}).get("semantic_transfer", True)
    )
    return configured_clinical_token_count(cfg) if semantic_transfer else 0


def configured_evidence_token_count(cfg: Mapping[str, Any]) -> int:
    return (
        configured_clinical_token_count(cfg)
        + configured_matched_token_count(cfg)
        + configured_image_risk_token_count(cfg)
    )
