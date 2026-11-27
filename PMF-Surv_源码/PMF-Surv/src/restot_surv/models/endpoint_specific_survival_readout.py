from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evidence import (
    configured_clinical_token_count,
    configured_image_risk_token_count,
)

from .survival import MTLRHead, PiecewiseExponentialHead, interval_geometry


class ContinuousTimeEmbedding(nn.Module):
    """Embed real interval centers and widths in a shared month-based space."""

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(
        self,
        boundaries: torch.Tensor,
        num_bins: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers, widths, horizon = interval_geometry(boundaries, num_bins)
        normalizer = horizon.clamp_min(1.0)
        log_normalizer = torch.log1p(normalizer).clamp_min(1.0)
        features = torch.stack(
            [
                centers / normalizer,
                widths / normalizer,
                torch.log1p(centers.clamp_min(0.0)) / log_normalizer,
                torch.log1p(widths.clamp_min(0.0)) / log_normalizer,
            ],
            dim=-1,
        )
        return self.net(features.float()), centers, widths


class TypeAwareEvidenceAttention(nn.Module):
    """Cross-attention with time/type bias and evidence-count normalization."""

    def __init__(
        self,
        token_dim: int,
        heads: int,
        dropout: float,
        num_time_bins: int,
        num_evidence_types: int = 3,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.heads = int(heads)
        self.head_dim = self.token_dim // self.heads
        self.num_time_bins = int(num_time_bins)
        self.num_evidence_types = int(num_evidence_types)
        if self.token_dim % self.heads:
            raise ValueError("token_dim must be divisible by attention heads.")
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.context_norm = nn.LayerNorm(self.token_dim)
        self.query_projection = nn.Linear(self.token_dim, self.token_dim)
        self.key_projection = nn.Linear(self.token_dim, self.token_dim)
        self.value_projection = nn.Linear(self.token_dim, self.token_dim)
        self.output_projection = nn.Linear(self.token_dim, self.token_dim)
        self.dropout = nn.Dropout(dropout)
        self.type_bias = nn.Parameter(
            torch.zeros(
                self.num_time_bins,
                self.heads,
                self.num_evidence_types,
            )
        )

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(
            batch,
            length,
            self.heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        evidence_types: torch.Tensor,
        evidence_type_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[1] != self.num_time_bins:
            raise ValueError("One learned query is required for every time bin.")
        if evidence_types.shape != (context.shape[1],):
            raise ValueError("Each evidence token requires one type identifier.")
        if evidence_type_counts.shape != (self.num_evidence_types,):
            raise ValueError("evidence_type_counts has the wrong shape.")

        projected_query = self._split(
            self.query_projection(self.query_norm(query))
        )
        normalized_context = self.context_norm(context)
        projected_key = self._split(self.key_projection(normalized_context))
        projected_value = self._split(self.value_projection(context))
        scores = torch.matmul(
            projected_query.float(),
            projected_key.float().transpose(-2, -1),
        ) / math.sqrt(self.head_dim)
        learned_bias = self.type_bias.permute(1, 0, 2)[
            :, :, evidence_types
        ]
        count_bias = -torch.log(
            evidence_type_counts.float().clamp_min(1.0)[evidence_types]
        )
        scores = scores + learned_bias[None] + count_bias[None, None, None]
        attention = torch.softmax(scores, dim=-1)
        values = torch.matmul(
            self.dropout(attention).to(projected_value.dtype),
            projected_value,
        )
        batch, _, query_count, _ = values.shape
        values = values.permute(0, 2, 1, 3).reshape(
            batch,
            query_count,
            self.token_dim,
        )
        return self.output_projection(values), attention.mean(dim=1)


class EndpointSpecificSurvivalReadoutCore(nn.Module):
    """Build endpoint-specific temporal features from the available evidence."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        endpoint: str,
        detach_evidence: bool,
    ) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.endpoint = str(endpoint)
        self.detach_evidence = bool(detach_evidence)
        if self.endpoint not in {"os", "pfs"}:
            raise ValueError("endpoint must be os or pfs.")
        if self.endpoint == "os" and self.detach_evidence:
            raise ValueError("The primary OS head must receive trainable evidence.")
        if self.endpoint == "pfs" and not self.detach_evidence:
            raise ValueError("The independent PFS head must detach shared evidence.")

        self.token_dim = int(model_cfg["token_dim"])
        self.num_bins = int(model_cfg["num_time_bins"])
        self.semantic_transfer_enabled = bool(
            model_cfg.get("ablation", {}).get("semantic_transfer", True)
        )
        self.clinical_token_count = configured_clinical_token_count(cfg)
        self.image_risk_token_count = configured_image_risk_token_count(cfg)
        dropout = float(model_cfg["dropout"])
        heads = int(model_cfg["time_query"]["attention_heads"])
        self.learned_time_queries = nn.Parameter(
            torch.zeros(1, self.num_bins, self.token_dim)
        )
        self.time_embedding = ContinuousTimeEmbedding(self.token_dim, dropout)
        evidence_type_counts = [float(self.clinical_token_count)]
        evidence_type_ids = [
            torch.zeros(self.clinical_token_count, dtype=torch.long)
        ]
        if self.semantic_transfer_enabled:
            evidence_type_counts.append(float(self.clinical_token_count))
            evidence_type_ids.append(
                torch.full(
                    (self.clinical_token_count,),
                    len(evidence_type_counts) - 1,
                    dtype=torch.long,
                )
            )
        evidence_type_counts.append(float(self.image_risk_token_count))
        evidence_type_ids.append(
            torch.full(
                (self.image_risk_token_count,),
                len(evidence_type_counts) - 1,
                dtype=torch.long,
            )
        )
        self.evidence_token_count = int(sum(evidence_type_counts))
        self.evidence_type_embedding = nn.Parameter(
            torch.zeros(1, len(evidence_type_counts), self.token_dim)
        )
        self.evidence_attention = TypeAwareEvidenceAttention(
            self.token_dim,
            heads,
            dropout,
            num_time_bins=self.num_bins,
            num_evidence_types=len(evidence_type_counts),
        )
        self.evidence_norm = nn.LayerNorm(self.token_dim)
        self.evidence_ffn = nn.Sequential(
            nn.Linear(self.token_dim, 4 * self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.token_dim, self.token_dim),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "evidence_type_ids",
            torch.cat(evidence_type_ids),
            persistent=False,
        )
        self.register_buffer(
            "evidence_type_counts",
            torch.tensor(evidence_type_counts),
            persistent=False,
        )
        nn.init.normal_(self.learned_time_queries, std=0.02)
        nn.init.normal_(self.evidence_type_embedding, std=0.02)

    def forward(
        self,
        clinical_tokens: torch.Tensor,
        matched_tokens: torch.Tensor,
        image_evidence_tokens: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = clinical_tokens.shape[0]
        if (
            matched_tokens.shape[0] != batch
            or image_evidence_tokens.shape[0] != batch
        ):
            raise ValueError("All time-query evidence must share a batch dimension.")
        expected_matched_tokens = (
            self.clinical_token_count if self.semantic_transfer_enabled else 0
        )
        if clinical_tokens.shape[1] != self.clinical_token_count:
            raise ValueError(
                "Clinical evidence count does not match the configured variables."
            )
        if matched_tokens.shape[1] != expected_matched_tokens:
            raise ValueError(
                "The aligned-image evidence count does not match the "
                "semantic-transfer setting."
            )
        if image_evidence_tokens.shape[1] != self.image_risk_token_count:
            raise ValueError(
                "Image-risk evidence count does not match the configuration."
            )
        if self.detach_evidence:
            clinical_tokens = clinical_tokens.detach()
            matched_tokens = matched_tokens.detach()
            image_evidence_tokens = image_evidence_tokens.detach()

        time_embedding, time_centers, time_widths = self.time_embedding(
            boundaries,
            self.num_bins,
        )
        time_queries = self.learned_time_queries.expand(batch, -1, -1)
        time_queries = time_queries + time_embedding[None].to(time_queries.dtype)
        evidence = torch.cat(
            [clinical_tokens, matched_tokens, image_evidence_tokens],
            dim=1,
        )
        if evidence.shape[1] != self.evidence_token_count:
            raise RuntimeError(
                "The survival evidence bank size does not match the active "
                "components."
            )
        evidence_types = self.evidence_type_ids
        evidence = evidence + self.evidence_type_embedding[:, evidence_types]
        evidence_context, evidence_attention = self.evidence_attention(
            time_queries,
            evidence,
            evidence_types,
            self.evidence_type_counts,
        )
        time_evidence = self.evidence_norm(time_queries + evidence_context)
        time_evidence = self.evidence_norm(
            time_evidence + self.evidence_ffn(time_evidence)
        )
        diagnostics = {
            f"{self.endpoint}_time_query_features": time_evidence,
            f"{self.endpoint}_time_centers": time_centers,
            f"{self.endpoint}_time_widths": time_widths,
            f"{self.endpoint}_to_evidence_attention": evidence_attention,
            f"{self.endpoint}_evidence_type_ids": evidence_types,
            f"{self.endpoint}_evidence_bank_size": torch.as_tensor(
                evidence.shape[1],
                device=evidence.device,
            ),
        }
        return time_evidence, diagnostics


class EndpointSpecificSurvivalReadout(MTLRHead):
    """MTLR head using endpoint-specific queries over available evidence."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        endpoint: str,
        detach_evidence: bool,
    ) -> None:
        model_cfg = cfg["model"]
        token_dim = int(model_cfg["token_dim"])
        hidden_dim = int(model_cfg["survival_hidden_dim"])
        num_bins = int(model_cfg["num_time_bins"])
        dropout = float(model_cfg["dropout"])
        super().__init__(token_dim, hidden_dim, num_bins, dropout)
        self.net = nn.Identity()
        self.endpoint = str(endpoint)
        self.core = EndpointSpecificSurvivalReadoutCore(
            cfg,
            endpoint=endpoint,
            detach_evidence=detach_evidence,
        )
        self.shared_logit_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        clinical_tokens: torch.Tensor,
        matched_tokens: torch.Tensor,
        image_evidence_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        time_evidence, diagnostics = self.core(
            clinical_tokens,
            matched_tokens,
            image_evidence_tokens,
            self.boundaries,
        )
        # Preserve the 2026-08-17 MTLR formulation while evaluating its final
        # projection in FP32. The current ImageRiskPooler path overflows this
        # backward pass under FP16 AMP, whereas the old image extractor did not.
        with torch.autocast(
            device_type=time_evidence.device.type,
            enabled=False,
        ):
            logits = self.shared_logit_projection(
                time_evidence.float()
            ).squeeze(-1)
        return logits, diagnostics


class EndpointSpecificPiecewiseExponentialReadout(
    PiecewiseExponentialHead
):
    """Endpoint-specific head predicting 12 piecewise-constant hazard rates."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        endpoint: str,
        detach_evidence: bool,
    ) -> None:
        model_cfg = cfg["model"]
        token_dim = int(model_cfg["token_dim"])
        hidden_dim = int(model_cfg["survival_hidden_dim"])
        num_bins = int(model_cfg["num_time_bins"])
        dropout = float(model_cfg["dropout"])
        super().__init__(num_bins)
        self.endpoint = str(endpoint)
        self.core = EndpointSpecificSurvivalReadoutCore(
            cfg,
            endpoint=endpoint,
            detach_evidence=detach_evidence,
        )
        self.shared_rate_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        survival_cfg = model_cfg.get("survival_head", {})
        initial_rate = float(survival_cfg.get("initial_hazard_rate", 0.01))
        initial_weight_std = float(
            survival_cfg.get("initial_rate_weight_std", 1.0e-3)
        )
        if initial_rate <= 0.0 or initial_weight_std <= 0.0:
            raise ValueError(
                "Initial hazard rate and output-weight std must be positive."
            )
        output_layer = self.shared_rate_projection[-1]
        nn.init.normal_(
            output_layer.weight,
            mean=0.0,
            std=initial_weight_std,
        )
        nn.init.constant_(
            output_layer.bias,
            math.log(math.expm1(initial_rate)),
        )

    def forward(
        self,
        clinical_tokens: torch.Tensor,
        matched_tokens: torch.Tensor,
        image_evidence_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        time_evidence, diagnostics = self.core(
            clinical_tokens,
            matched_tokens,
            image_evidence_tokens,
            self.boundaries,
        )
        # Execute the small-weight output projection in FP32. Casting only
        # after an autocast FP16 projection collapses patient-specific risks.
        with torch.autocast(
            device_type=time_evidence.device.type,
            enabled=False,
        ):
            rate_logits = self.shared_rate_projection(
                time_evidence.float()
            ).squeeze(-1)
            rates = F.softplus(rate_logits).clamp_min(1.0e-8)
        diagnostics[f"{self.endpoint}_interval_hazard_rates"] = rates
        return rates, diagnostics


class PooledEvidenceCore(nn.Module):
    """Collapse the available evidence into one patient token without time queries."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        endpoint: str,
        detach_evidence: bool,
    ) -> None:
        super().__init__()
        self.endpoint = str(endpoint)
        self.detach_evidence = bool(detach_evidence)
        self.semantic_transfer_enabled = bool(
            cfg["model"].get("ablation", {}).get("semantic_transfer", True)
        )
        self.clinical_token_count = configured_clinical_token_count(cfg)
        self.image_risk_token_count = configured_image_risk_token_count(cfg)
        if self.endpoint not in {"os", "pfs"}:
            raise ValueError("endpoint must be os or pfs.")
        if self.endpoint == "os" and self.detach_evidence:
            raise ValueError("The primary OS head must receive trainable evidence.")
        if self.endpoint == "pfs" and not self.detach_evidence:
            raise ValueError("The independent PFS head must detach shared evidence.")

    def forward(
        self,
        clinical_tokens: torch.Tensor,
        matched_tokens: torch.Tensor,
        image_evidence_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected_matched_tokens = (
            self.clinical_token_count if self.semantic_transfer_enabled else 0
        )
        if clinical_tokens.shape[1] != self.clinical_token_count:
            raise ValueError(
                "Clinical evidence count does not match the configured variables."
            )
        if matched_tokens.shape[1] != expected_matched_tokens:
            raise ValueError(
                "The aligned-image evidence count does not match the "
                "semantic-transfer setting."
            )
        if image_evidence_tokens.shape[1] != self.image_risk_token_count:
            raise ValueError(
                "Image-risk evidence count does not match the configuration."
            )
        if self.detach_evidence:
            clinical_tokens = clinical_tokens.detach()
            matched_tokens = matched_tokens.detach()
            image_evidence_tokens = image_evidence_tokens.detach()
        evidence = torch.cat(
            [clinical_tokens, matched_tokens, image_evidence_tokens],
            dim=1,
        )
        patient_token = evidence.mean(dim=1)
        diagnostics = {
            f"{self.endpoint}_pooled_patient_token": patient_token[:, None],
            f"{self.endpoint}_evidence_bank_size": torch.as_tensor(
                evidence.shape[1],
                device=evidence.device,
            ),
        }
        return patient_token, diagnostics


class PooledEvidenceMTLRHead(MTLRHead):
    """Ordinary MTLR over one pooled patient token when time queries are absent."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        endpoint: str,
        detach_evidence: bool,
    ) -> None:
        model_cfg = cfg["model"]
        super().__init__(
            int(model_cfg["token_dim"]),
            int(model_cfg["survival_hidden_dim"]),
            int(model_cfg["num_time_bins"]),
            float(model_cfg["dropout"]),
        )
        self.endpoint = str(endpoint)
        self.core = PooledEvidenceCore(
            cfg,
            endpoint=endpoint,
            detach_evidence=detach_evidence,
        )

    def forward(
        self,
        clinical_tokens: torch.Tensor,
        matched_tokens: torch.Tensor,
        image_evidence_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patient_token, diagnostics = self.core(
            clinical_tokens,
            matched_tokens,
            image_evidence_tokens,
        )
        with torch.autocast(
            device_type=patient_token.device.type,
            enabled=False,
        ):
            logits = self.net(patient_token.float())
        return logits, diagnostics


class PooledEvidencePiecewiseExponentialHead(PiecewiseExponentialHead):
    """Piecewise hazards from one pooled patient token without time queries."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        endpoint: str,
        detach_evidence: bool,
    ) -> None:
        model_cfg = cfg["model"]
        token_dim = int(model_cfg["token_dim"])
        hidden_dim = int(model_cfg["survival_hidden_dim"])
        num_bins = int(model_cfg["num_time_bins"])
        dropout = float(model_cfg["dropout"])
        super().__init__(num_bins)
        self.endpoint = str(endpoint)
        self.core = PooledEvidenceCore(
            cfg,
            endpoint=endpoint,
            detach_evidence=detach_evidence,
        )
        self.rate_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins),
        )
        survival_cfg = model_cfg.get("survival_head", {})
        initial_rate = float(survival_cfg.get("initial_hazard_rate", 0.01))
        initial_weight_std = float(
            survival_cfg.get("initial_rate_weight_std", 1.0e-3)
        )
        output_layer = self.rate_projection[-1]
        nn.init.normal_(output_layer.weight, std=initial_weight_std)
        nn.init.constant_(
            output_layer.bias,
            math.log(math.expm1(initial_rate)),
        )

    def forward(
        self,
        clinical_tokens: torch.Tensor,
        matched_tokens: torch.Tensor,
        image_evidence_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patient_token, diagnostics = self.core(
            clinical_tokens,
            matched_tokens,
            image_evidence_tokens,
        )
        with torch.autocast(
            device_type=patient_token.device.type,
            enabled=False,
        ):
            rates = F.softplus(
                self.rate_projection(patient_token.float())
            ).clamp_min(1.0e-8)
        diagnostics[f"{self.endpoint}_interval_hazard_rates"] = rates
        return rates, diagnostics


__all__ = [
    "EndpointSpecificSurvivalReadout",
    "EndpointSpecificPiecewiseExponentialReadout",
    "PooledEvidenceMTLRHead",
    "PooledEvidencePiecewiseExponentialHead",
]
