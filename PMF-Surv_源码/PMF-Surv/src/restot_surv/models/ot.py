from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .evidence import configured_image_risk_token_count


class GlobalImagingFeaturePooler(nn.Module):
    """Pool regional imaging features into the configured set of global imaging features."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        token_dim = int(model_cfg["token_dim"])
        self.num_queries = configured_image_risk_token_count(cfg)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_queries, token_dim)
        )
        self.query_norm = nn.LayerNorm(token_dim)
        self.key_norm = nn.LayerNorm(token_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.LayerNorm(token_dim),
        )
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        queries = self.query_norm(
            self.query_tokens.expand(image_tokens.shape[0], -1, -1)
        )
        keys = self.key_norm(image_tokens)
        scale = image_tokens.shape[-1] ** -0.5
        attention = torch.softmax(
            torch.bmm(queries, keys.transpose(1, 2)) * scale,
            dim=-1,
        )
        pooled = torch.bmm(attention.to(image_tokens.dtype), image_tokens)
        return self.output(pooled)


ImageRiskPooler = GlobalImagingFeaturePooler
