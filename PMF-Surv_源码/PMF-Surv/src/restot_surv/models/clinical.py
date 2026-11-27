from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .evidence import configured_clinical_token_count


class ClinicalInformationEncoder(nn.Module):
    """Each real clinical variable remains an explicit, identifiable token."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
    ) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        self.token_dim = int(model_cfg["token_dim"])
        embedding_dim = int(model_cfg["clinical_embedding_dim"])
        self.continuous_count = len(data_cfg["continuous_vars"])
        self.categorical_count = len(data_cfg["categorical_vars"])
        self.composite_count = len(data_cfg["composite_vars"])
        self.continuous_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, self.token_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.token_dim),
                )
                for _ in range(self.continuous_count)
            ]
        )
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(max(int(cardinality), 1), embedding_dim) for cardinality in categorical_cardinalities]
        )
        self.composite_embeddings = nn.ModuleList(
            [nn.Embedding(max(int(cardinality), 1), embedding_dim) for cardinality in composite_cardinalities]
        )
        self.category_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.token_dim),
            nn.GELU(),
            nn.LayerNorm(self.token_dim),
        )
        token_count = configured_clinical_token_count(cfg)
        self.variable_embedding = nn.Parameter(torch.zeros(1, token_count, self.token_dim))
        self.output_norm = nn.LayerNorm(self.token_dim)
        nn.init.normal_(self.variable_embedding, std=0.02)

    @property
    def token_count(self) -> int:
        return int(self.variable_embedding.shape[1])

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
    ) -> torch.Tensor:
        tokens = []
        for index, projection in enumerate(self.continuous_projections):
            tokens.append(projection(continuous[:, index : index + 1]).unsqueeze(1))
        for index, embedding in enumerate(self.categorical_embeddings):
            value = categorical[:, index].long().clamp(0, embedding.num_embeddings - 1)
            tokens.append(self.category_projection(embedding(value)).unsqueeze(1))
        for index, embedding in enumerate(self.composite_embeddings):
            value = composite[:, index].long().clamp(0, embedding.num_embeddings - 1)
            tokens.append(self.category_projection(embedding(value)).unsqueeze(1))
        output = torch.cat(tokens, dim=1)
        return self.output_norm(output + self.variable_embedding)


ClinicalTokenEncoder = ClinicalInformationEncoder
