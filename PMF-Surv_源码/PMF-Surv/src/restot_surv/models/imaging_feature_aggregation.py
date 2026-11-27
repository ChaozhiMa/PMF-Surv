from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evidence import configured_clinical_token_count


class OneSidedUnbalancedOptimalTransport(nn.Module):
    """Single-scale clinical-image UOT with an exact image-side marginal.

    Every contextual image token contributes the same input mass and that mass
    is fully assigned across the clinical queries. The clinical/query marginal
    remains soft through a KL-style row penalty, so row masses may differ.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        clinical_query_count: int,
        image_token_count: int,
        epsilon: float,
        query_balance_weight: float,
        iterations: int,
        tolerance: float,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.clinical_query_count = int(clinical_query_count)
        self.image_token_count = int(image_token_count)
        self.epsilon = max(float(epsilon), 1.0e-6)
        self.query_balance_weight = float(query_balance_weight)
        self.iterations = max(int(iterations), 1)
        self.tolerance = max(float(tolerance), 1.0e-12)
        if self.query_balance_weight < 0.0:
            raise ValueError("query_balance_weight must be non-negative.")

    def semantic_cost(
        self,
        clinical_queries: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return C[j,i] = 1 - cos(Q[j], H[i]) at the native token scale."""

        if clinical_queries.ndim != 3 or image_tokens.ndim != 3:
            raise ValueError("Clinical queries and image tokens must be rank-3 tensors.")
        if clinical_queries.shape[0] != image_tokens.shape[0]:
            raise ValueError("Clinical queries and image tokens must share a batch size.")
        if clinical_queries.shape[1:] != (
            self.clinical_query_count,
            self.token_dim,
        ):
            raise ValueError(
                "Clinical semantic queries must have shape "
                f"[B,{self.clinical_query_count},{self.token_dim}]."
            )
        if image_tokens.shape[1:] != (self.image_token_count, self.token_dim):
            raise ValueError(
                "Contextual image tokens must have shape "
                f"[B,{self.image_token_count},{self.token_dim}]."
            )
        queries = F.normalize(clinical_queries.float(), dim=-1)
        images = F.normalize(image_tokens.float(), dim=-1)
        return 1.0 - torch.bmm(queries, images.transpose(1, 2))

    def _solve(
        self,
        cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_count, image_count = cost.shape
        kernel = torch.exp(-cost.float() / self.epsilon).clamp_min(1.0e-30)
        image_marginal = torch.full(
            (batch, image_count),
            1.0 / float(image_count),
            dtype=kernel.dtype,
            device=kernel.device,
        )
        query_reference = torch.full(
            (batch, query_count),
            1.0 / float(query_count),
            dtype=kernel.dtype,
            device=kernel.device,
        )
        exponent = self.query_balance_weight / (
            self.query_balance_weight + self.epsilon
        )
        row_scaling = torch.ones_like(query_reference)
        column_scaling = torch.ones_like(image_marginal)
        residual = kernel.new_full((batch,), float("inf"))
        iterations_used = 0
        for iteration in range(self.iterations):
            previous = row_scaling
            row_scaling = (
                query_reference
                / torch.bmm(kernel, column_scaling.unsqueeze(-1))
                .squeeze(-1)
                .clamp_min(1.0e-30)
            ).pow(exponent)
            column_scaling = image_marginal / torch.bmm(
                kernel.transpose(1, 2), row_scaling.unsqueeze(-1)
            ).squeeze(-1).clamp_min(1.0e-30)
            residual = (row_scaling - previous).abs().amax(dim=1)
            iterations_used = iteration + 1
            if residual.detach().amax().item() <= self.tolerance:
                break

        transport = (
            row_scaling.unsqueeze(-1)
            * kernel
            * column_scaling.unsqueeze(1)
        )
        # Enforce sum_j pi[j,i] = a[i] exactly after finite iterations.
        transport = transport * (
            image_marginal / transport.sum(dim=1).clamp_min(1.0e-30)
        ).unsqueeze(1)
        iteration_count = residual.new_full(
            (batch,),
            float(iterations_used),
        )
        return transport, residual, iteration_count

    def forward(
        self,
        clinical_queries: torch.Tensor,
        image_tokens: torch.Tensor,
        *,
        return_transport_fp32: bool = False,
    ) -> dict[str, torch.Tensor]:
        with torch.autocast(device_type=image_tokens.device.type, enabled=False):
            cost = self.semantic_cost(clinical_queries, image_tokens)
            transport, residual, iteration_count = self._solve(cost)
            row_mass = transport.sum(dim=-1)
            normalized_rows = transport / row_mass.unsqueeze(-1).clamp_min(1.0e-12)
            aligned = torch.bmm(normalized_rows, image_tokens.float())
            expected_image_mass = 1.0 / float(self.image_token_count)
            image_marginal_error = (
                transport.sum(dim=1) - expected_image_mass
            ).abs().amax(dim=1)
        output_dtype = torch.float32 if return_transport_fp32 else image_tokens.dtype
        return {
            "query_aligned_image_features": aligned.to(output_dtype),
            "transport_plan": transport.to(output_dtype),
            "query_mass": row_mass.to(output_dtype),
            "semantic_cost_mean": cost.mean(dim=(1, 2)).to(image_tokens.dtype),
            "image_marginal_error": image_marginal_error.to(image_tokens.dtype),
            "qred_sinkhorn_residual": residual.to(image_tokens.dtype),
            "qred_sinkhorn_iterations": iteration_count.to(image_tokens.dtype),
            "qred_converged": (residual <= self.tolerance).to(image_tokens.dtype),
        }


class BalancedClinicalImageOT(OneSidedUnbalancedOptimalTransport):
    """Entropic OT with both clinical and image marginals constrained."""

    def _solve(
        self,
        cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_count, image_count = cost.shape
        kernel = torch.exp(-cost.float() / self.epsilon).clamp_min(1.0e-30)
        query_marginal = torch.full(
            (batch, query_count),
            1.0 / float(query_count),
            dtype=kernel.dtype,
            device=kernel.device,
        )
        image_marginal = torch.full(
            (batch, image_count),
            1.0 / float(image_count),
            dtype=kernel.dtype,
            device=kernel.device,
        )
        row_scaling = torch.ones_like(query_marginal)
        column_scaling = torch.ones_like(image_marginal)
        residual = kernel.new_full((batch,), float("inf"))
        iterations_used = 0
        for iteration in range(self.iterations):
            row_scaling = query_marginal / torch.bmm(
                kernel,
                column_scaling.unsqueeze(-1),
            ).squeeze(-1).clamp_min(1.0e-30)
            column_scaling = image_marginal / torch.bmm(
                kernel.transpose(1, 2),
                row_scaling.unsqueeze(-1),
            ).squeeze(-1).clamp_min(1.0e-30)
            transport = (
                row_scaling.unsqueeze(-1)
                * kernel
                * column_scaling.unsqueeze(1)
            )
            row_error = (
                transport.sum(dim=-1) - query_marginal
            ).abs().amax(dim=1)
            column_error = (
                transport.sum(dim=1) - image_marginal
            ).abs().amax(dim=1)
            residual = torch.maximum(row_error, column_error)
            iterations_used = iteration + 1
            if residual.detach().amax().item() <= self.tolerance:
                break
        iteration_count = residual.new_full(
            (batch,),
            float(iterations_used),
        )
        return transport, residual, iteration_count


class ConventionalClinicalImageUOT(OneSidedUnbalancedOptimalTransport):
    """Conventional entropic UOT with KL relaxation on both marginals."""

    def __init__(self, *, mass_penalty: float, **kwargs: Any) -> None:
        super().__init__(query_balance_weight=mass_penalty, **kwargs)
        self.mass_penalty = float(mass_penalty)
        if self.mass_penalty <= 0.0:
            raise ValueError("mass_penalty must be positive.")

    def _solve(
        self,
        cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, query_count, image_count = cost.shape
        kernel = torch.exp(-cost.float() / self.epsilon).clamp_min(1.0e-30)
        query_reference = torch.full(
            (batch, query_count),
            1.0 / float(query_count),
            dtype=kernel.dtype,
            device=kernel.device,
        )
        image_reference = torch.full(
            (batch, image_count),
            1.0 / float(image_count),
            dtype=kernel.dtype,
            device=kernel.device,
        )
        exponent = self.mass_penalty / (self.mass_penalty + self.epsilon)
        row_scaling = torch.ones_like(query_reference)
        column_scaling = torch.ones_like(image_reference)
        residual = kernel.new_full((batch,), float("inf"))
        iterations_used = 0
        for iteration in range(self.iterations):
            previous_row = row_scaling
            previous_column = column_scaling
            row_scaling = (
                query_reference
                / torch.bmm(kernel, column_scaling.unsqueeze(-1))
                .squeeze(-1)
                .clamp_min(1.0e-30)
            ).pow(exponent)
            column_scaling = (
                image_reference
                / torch.bmm(
                    kernel.transpose(1, 2),
                    row_scaling.unsqueeze(-1),
                )
                .squeeze(-1)
                .clamp_min(1.0e-30)
            ).pow(exponent)
            row_change = (row_scaling - previous_row).abs().amax(dim=1)
            column_change = (
                column_scaling - previous_column
            ).abs().amax(dim=1)
            residual = torch.maximum(row_change, column_change)
            iterations_used = iteration + 1
            if residual.detach().amax().item() <= self.tolerance:
                break
        transport = (
            row_scaling.unsqueeze(-1)
            * kernel
            * column_scaling.unsqueeze(1)
        )
        iteration_count = residual.new_full(
            (batch,),
            float(iterations_used),
        )
        return transport, residual, iteration_count


class StandardClinicalImageCrossAttention(nn.Module):
    """Ordinary cross-modal attention used instead of semantic OT."""

    def __init__(
        self,
        *,
        token_dim: int,
        clinical_query_count: int,
        image_token_count: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.clinical_query_count = int(clinical_query_count)
        self.image_token_count = int(image_token_count)
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.image_norm = nn.LayerNorm(self.token_dim)
        self.attention = nn.MultiheadAttention(
            self.token_dim,
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        self.output_ffn = nn.Sequential(
            nn.Linear(self.token_dim, 4 * self.token_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.token_dim, self.token_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        clinical_queries: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if clinical_queries.shape[1:] != (
            self.clinical_query_count,
            self.token_dim,
        ):
            raise ValueError(
                "Clinical semantic queries have the wrong shape."
            )
        if image_tokens.shape[1:] != (
            self.image_token_count,
            self.token_dim,
        ):
            raise ValueError("Contextual image tokens have the wrong shape.")
        normalized_images = self.image_norm(image_tokens)
        attended, attention = self.attention(
            self.query_norm(clinical_queries),
            normalized_images,
            normalized_images,
            need_weights=True,
            average_attn_weights=True,
        )
        aligned = self.output_norm(attended)
        aligned = self.output_norm(aligned + self.output_ffn(aligned))
        transport = attention.float() / float(self.clinical_query_count)
        query_mass = transport.sum(dim=-1)
        expected_image_mass = 1.0 / float(self.image_token_count)
        image_marginal_error = (
            transport.sum(dim=1) - expected_image_mass
        ).abs().amax(dim=1)
        batch = image_tokens.shape[0]
        zero = image_tokens.new_zeros((batch,))
        return {
            "query_aligned_image_features": aligned,
            "transport_plan": transport.to(image_tokens.dtype),
            "query_mass": query_mass.to(image_tokens.dtype),
            "semantic_cost_mean": (1.0 - attention).mean(dim=(1, 2)).to(
                image_tokens.dtype
            ),
            "image_marginal_error": image_marginal_error.to(
                image_tokens.dtype
            ),
            "qred_sinkhorn_residual": zero,
            "qred_sinkhorn_iterations": zero,
            "qred_converged": torch.ones_like(zero),
        }


class ClinicalInformationConditionedImagingFeatureAggregation(nn.Module):
    """Match configured clinical tokens to all contextual image tokens."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        ot_cfg = model_cfg["ot"]
        self.num_phases = int(model_cfg["num_phases"])
        self.patch_grid = tuple(int(value) for value in model_cfg["patch_grid"])
        self.token_dim = int(model_cfg["token_dim"])
        self.clinical_query_count = configured_clinical_token_count(cfg)
        self.image_token_count = self.num_phases
        for value in self.patch_grid:
            self.image_token_count *= value
        ablation_cfg = model_cfg.get("ablation", {})
        self.semantic_fusion = str(
            ablation_cfg.get("semantic_fusion", "proposed_ot")
        ).lower()
        self.assignment_mode = str(ot_cfg.get("assignment_mode", "hard_top1")).lower()
        self.hard_top1_straight_through = bool(
            ot_cfg.get("hard_top1_straight_through", True)
        )
        if self.assignment_mode not in {"soft", "hard_top1"}:
            raise ValueError("ot.assignment_mode must be soft or hard_top1.")
        common_transport_kwargs = {
            "token_dim": self.token_dim,
            "clinical_query_count": self.clinical_query_count,
            "image_token_count": self.image_token_count,
            "epsilon": float(ot_cfg["epsilon"]),
            "iterations": int(ot_cfg["iterations"]),
            "tolerance": float(ot_cfg["tolerance"]),
        }
        if self.semantic_fusion == "proposed_ot":
            self.transport = OneSidedUnbalancedOptimalTransport(
                query_balance_weight=float(
                    ot_cfg.get("query_balance_weight", 0.5)
                ),
                **common_transport_kwargs,
            )
        elif self.semantic_fusion == "standard_cross_attention":
            self.transport = StandardClinicalImageCrossAttention(
                token_dim=self.token_dim,
                clinical_query_count=self.clinical_query_count,
                image_token_count=self.image_token_count,
                heads=int(model_cfg["interaction_heads"]),
                dropout=float(model_cfg["dropout"]),
            )
        elif self.semantic_fusion == "balanced_ot":
            self.transport = BalancedClinicalImageOT(
                query_balance_weight=float(
                    ot_cfg.get("query_balance_weight", 0.5)
                ),
                **common_transport_kwargs,
            )
        elif self.semantic_fusion == "conventional_uot":
            self.transport = ConventionalClinicalImageUOT(
                mass_penalty=float(
                    ot_cfg.get(
                        "mass_penalty",
                        ot_cfg.get("query_balance_weight", 0.5),
                    )
                ),
                **common_transport_kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported semantic fusion mode: {self.semantic_fusion}."
            )

    @staticmethod
    def scale_aligned_evidence(
        matched_tokens: torch.Tensor,
        query_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Scale each conditioned imaging feature by its relative query mass."""

        if query_mass.shape != matched_tokens.shape[:2]:
            raise ValueError(
                "query_mass must align with the conditioned imaging features."
            )
        relative_mass = query_mass / query_mass.mean(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        return matched_tokens * relative_mass.clamp(0.0, 2.0).unsqueeze(-1)

    def forward(
        self,
        clinical_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if self.semantic_fusion == "proposed_ot" and self.assignment_mode == "hard_top1":
            # Decide ownership before any FP16 cast can round close scores into ties.
            result = self.transport(
                clinical_tokens, image_tokens, return_transport_fp32=True
            )
            result = self._hard_top1_readout(result, image_tokens)
        else:
            result = self.transport(clinical_tokens, image_tokens)
        diagnostics = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "query_aligned_image_features",
                "transport_plan",
                "query_mass",
            }
        }
        return (
            result["query_aligned_image_features"],
            result["transport_plan"],
            result["query_mass"],
            diagnostics,
        )

    def _hard_top1_readout(
        self,
        soft: dict[str, torch.Tensor],
        image_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Unique image ownership after solving the original soft transport.

        Argmax compares clinical queries within each original image column,
        before row normalization. Empty queries remain zero; no quota is added.
        The solver diagnostics still describe the preceding soft OT solve.
        """
        soft_plan = soft["transport_plan"]
        winner = soft_plan.detach().argmax(dim=1, keepdim=True)
        image_mass = 1.0 / float(image_tokens.shape[1])
        # FP32 keeps the ownership masses accurate under mixed precision.
        hard_plan = torch.zeros_like(soft_plan, dtype=torch.float32)
        hard_plan.scatter_(1, winner, image_mass)
        hard_mass = hard_plan.sum(dim=-1)
        denominator = torch.where(
            hard_mass > 0.0, hard_mass, torch.ones_like(hard_mass)
        )
        with torch.autocast(device_type=image_tokens.device.type, enabled=False):
            hard_aligned = torch.bmm(
                hard_plan / denominator.unsqueeze(-1), image_tokens.float()
            )
            hard_scaled = self.scale_aligned_evidence(hard_aligned, hard_mass)
        soft_scaled = self.scale_aligned_evidence(
            soft["query_aligned_image_features"], soft["query_mass"]
        )
        use_straight_through = (
            self.training and self.hard_top1_straight_through
            and torch.is_grad_enabled()
        )
        if use_straight_through:
            # Apply ST AFTER both normalization and mass scaling. Applying ST
            # separately to an empty row's feature and mass would lose its
            # surrogate gradient (both forward factors would be zero).
            scaled = hard_scaled.detach() + (soft_scaled - soft_scaled.detach())
        else:
            scaled = hard_scaled
        result = dict(soft)
        result.update({
            "query_aligned_image_features": hard_aligned.to(image_tokens.dtype),
            "transport_plan": hard_plan,
            "query_mass": hard_mass,
            # Consumed by build_evidence, never exposed as a logging diagnostic.
            "_scaled_matched_tokens": scaled.to(image_tokens.dtype),
            "image_marginal_error": (
                hard_plan.sum(dim=1) - image_mass
            ).abs().amax(dim=1),
            "qred_hard_top1_active_queries": (hard_mass > 0.0).sum(dim=1).float(),
            "qred_hard_top1_empty_queries": (hard_mass == 0.0).sum(dim=1).float(),
        })
        return result


__all__ = [
    "BalancedClinicalImageOT",
    "ConventionalClinicalImageUOT",
    "OneSidedUnbalancedOptimalTransport",
    "ClinicalInformationConditionedImagingFeatureAggregation",
    "StandardClinicalImageCrossAttention",
]
