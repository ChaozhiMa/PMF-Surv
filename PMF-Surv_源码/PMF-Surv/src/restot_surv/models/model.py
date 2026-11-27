from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .auxiliary import AuxiliaryObjectives
from .clinical import ClinicalInformationEncoder
from .cross_phase_fusion import CrossPhaseFusion, build_image_encoder
from .ot import GlobalImagingFeaturePooler
from .imaging_feature_aggregation import ClinicalInformationConditionedImagingFeatureAggregation
from .reliability import (
    MeanAnchoredRankModulation,
    LocalTokenDegradation,
    RegionalFeatureStabilityScorer,
    midband_quality_weights,
    token_reliability_calibration_loss,
    token_stability_target,
)
from .endpoint_specific_survival_readout import (
    EndpointSpecificSurvivalReadout,
    EndpointSpecificPiecewiseExponentialReadout,
    PooledEvidenceMTLRHead,
    PooledEvidencePiecewiseExponentialHead,
)


class PMFSurv(nn.Module):
    """PMF-Surv with optional cross-phase fusion, imaging feature aggregation, and survival readout stages."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
    ) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        ablation_cfg = model_cfg.get("ablation", {})
        self.semantic_transfer_enabled = bool(
            ablation_cfg.get("semantic_transfer", True)
        )
        self.time_autonomous_query_enabled = bool(
            ablation_cfg.get("time_autonomous_query", True)
        )
        self.encoder = build_image_encoder(cfg)
        self.interaction = CrossPhaseFusion(cfg)

        reliability_cfg = model_cfg.get("token_reliability", {})
        self.token_reliability_enabled = bool(
            reliability_cfg.get("enabled", False)
        )
        quality_band_cfg = reliability_cfg.get("quality_band", {})
        self.token_quality_band_enabled = bool(
            quality_band_cfg.get("enabled", False)
        )
        self.token_quality_band_mode = str(
            quality_band_cfg.get("mode", "fixed_hard")
        ).lower()
        self.token_quality_lower_threshold = float(
            quality_band_cfg.get("lower_threshold", 0.35)
        )
        self.token_quality_upper_threshold = float(
            quality_band_cfg.get("upper_threshold", 0.80)
        )
        self.token_quality_high_weight = float(
            quality_band_cfg.get("high_quality_weight", 1.0)
        )
        self.token_quality_low_weight = float(
            quality_band_cfg.get("low_quality_weight", 0.02)
        )
        self.token_reliability_estimator: RegionalFeatureStabilityScorer | None = (
            RegionalFeatureStabilityScorer(cfg)
            if self.token_reliability_enabled
            else None
        )
        self.token_reliability_degradation: LocalTokenDegradation | None = (
            LocalTokenDegradation(cfg)
            if self.token_reliability_enabled
            else None
        )
        self.token_quality_gate: MeanAnchoredRankModulation | None = (
            MeanAnchoredRankModulation(cfg)
            if (
                self.token_reliability_enabled
                and self.token_quality_band_enabled
                and self.token_quality_band_mode
                == MeanAnchoredRankModulation.MODE
            )
            else None
        )
        calibration_cfg = reliability_cfg.get("calibration", {})
        self.token_reliability_target_temperature = float(
            calibration_cfg.get("target_temperature", 0.15)
        )
        self.token_reliability_corruption_penalty = float(
            calibration_cfg.get("corruption_penalty", 0.75)
        )
        self.token_reliability_ranking_margin = float(
            calibration_cfg.get("ranking_margin", 0.15)
        )
        self.token_reliability_ranking_weight = float(
            calibration_cfg.get("ranking_weight", 0.5)
        )

        self.clinical_encoder = ClinicalInformationEncoder(
            cfg,
            categorical_cardinalities,
            composite_cardinalities,
        )
        self.qred: ClinicalInformationConditionedImagingFeatureAggregation | None = (
            ClinicalInformationConditionedImagingFeatureAggregation(cfg)
            if self.semantic_transfer_enabled
            else None
        )
        if (
            self.qred is not None
            and self.qred.clinical_query_count
            != self.clinical_encoder.token_count
        ):
            raise ValueError(
                "The imaging feature aggregation query count must match the configured clinical-information features."
            )
        self.image_risk_pooler = GlobalImagingFeaturePooler(cfg)
        self.expected_evidence_bank_size = (
            self.clinical_encoder.token_count
            + (
                self.qred.clinical_query_count
                if self.qred is not None
                else 0
            )
            + self.image_risk_pooler.num_queries
        )
        self.auxiliary = AuxiliaryObjectives(cfg)

        os_prediction_cfg = model_cfg["os_prediction"]
        self.os_prediction_mode = str(os_prediction_cfg["mode"])
        if self.os_prediction_mode != "time_query_independent":
            raise ValueError(
                "PMF-Surv supports only "
                "model.os_prediction.mode=time_query_independent."
            )
        survival_head_cfg = model_cfg.get("survival_head", {"type": "mtlr"})
        self.survival_head_type = str(
            survival_head_cfg.get("type", "mtlr")
        ).lower()
        if self.survival_head_type == "piecewise_exponential":
            head_class = (
                EndpointSpecificPiecewiseExponentialReadout
                if self.time_autonomous_query_enabled
                else PooledEvidencePiecewiseExponentialHead
            )
        elif self.survival_head_type == "mtlr":
            head_class = (
                EndpointSpecificSurvivalReadout
                if self.time_autonomous_query_enabled
                else PooledEvidenceMTLRHead
            )
        else:
            raise ValueError(
                "model.survival_head.type must be mtlr or "
                "piecewise_exponential."
            )
        self.os_head = head_class(
            cfg,
            endpoint="os",
            detach_evidence=False,
        )
        self.pfs_head = head_class(
            cfg,
            endpoint="pfs",
            detach_evidence=True,
        )

    def set_time_boundaries(
        self,
        os_boundaries: torch.Tensor,
        pfs_boundaries: torch.Tensor,
    ) -> None:
        self.os_head.set_boundaries(os_boundaries)
        self.pfs_head.set_boundaries(pfs_boundaries)

    def predict_from_evidence(
        self,
        clinical_tokens: torch.Tensor,
        scaled_matched_tokens: torch.Tensor,
        image_risk_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pfs_parameters, pfs_diagnostics = self.pfs_head(
            clinical_tokens=clinical_tokens,
            matched_tokens=scaled_matched_tokens,
            image_evidence_tokens=image_risk_tokens,
        )
        os_parameters, os_diagnostics = self.os_head(
            clinical_tokens=clinical_tokens,
            matched_tokens=scaled_matched_tokens,
            image_evidence_tokens=image_risk_tokens,
        )
        pfs_survival = self.pfs_head.survival(pfs_parameters)
        os_survival = self.os_head.survival(os_parameters)
        predictions = {
            "os_head_output": os_parameters,
            "pfs_head_output": pfs_parameters,
            "os_survival": os_survival,
            "pfs_survival": pfs_survival,
            "os_risk": self.os_head.risk(os_parameters),
            "pfs_risk": self.pfs_head.risk(pfs_parameters),
            **os_diagnostics,
            **pfs_diagnostics,
        }
        if self.survival_head_type == "piecewise_exponential":
            predictions.update(
                {
                    "os_hazard_rates": os_parameters,
                    "pfs_hazard_rates": pfs_parameters,
                }
            )
        else:
            predictions.update(
                {
                    "os_logits": os_parameters,
                    "pfs_logits": pfs_parameters,
                }
            )
        return predictions

    def encode_image(
        self,
        image: torch.Tensor,
        *,
        token_reliability_gate_progress: float = 1.0,
        return_token_reliability: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            dict[str, torch.Tensor],
        ]
    ):
        base_tokens = self.encoder(image.float())
        if not self.token_reliability_enabled:
            contextual_tokens = self.interaction(base_tokens)
            if return_token_reliability:
                return base_tokens, contextual_tokens, None, None, {}
            return base_tokens, contextual_tokens

        if self.token_reliability_estimator is None:
            raise RuntimeError("Token reliability estimator is not initialized.")
        # Reliability calibration is independent of survival gradients. The
        # quality gate below has no trainable parameters and returns a hard mask.
        with torch.no_grad(), torch.autocast(
            device_type=base_tokens.device.type,
            enabled=False,
        ):
            predicted_reliability = self.token_reliability_estimator(
                base_tokens.detach().float(),
            )
        progress = min(max(float(token_reliability_gate_progress), 0.0), 1.0)
        gate_diagnostics: dict[str, torch.Tensor] = {}
        mapped_reliability = predicted_reliability
        if self.token_quality_gate is not None:
            mapped_reliability, gate_diagnostics = self.token_quality_gate(
                predicted_reliability,
            )
        elif self.token_quality_band_enabled:
            mapped_reliability, hard_membership = midband_quality_weights(
                predicted_reliability,
                lower_threshold=self.token_quality_lower_threshold,
                upper_threshold=self.token_quality_upper_threshold,
                high_quality_weight=self.token_quality_high_weight,
                low_quality_weight=self.token_quality_low_weight,
            )
            hard_membership_fraction = hard_membership.float().mean()
            gate_diagnostics = {
                "token_quality_lower_threshold": (
                    predicted_reliability.new_tensor(
                        self.token_quality_lower_threshold
                    )
                ),
                "token_quality_upper_threshold": (
                    predicted_reliability.new_tensor(
                        self.token_quality_upper_threshold
                    )
                ),
                "token_quality_hard_membership_fraction": (
                    hard_membership_fraction
                ),
            }
        if self.token_quality_band_enabled:
            # Hard-gate modes are binary from the first step. In particular,
            # there is no warmup interpolation between 1.0 and 0.02.
            effective_reliability = mapped_reliability
        else:
            effective_reliability = 1.0 + progress * (
                mapped_reliability - 1.0
            )
        contextual_tokens = self.interaction(
            base_tokens,
            reliability=effective_reliability,
        )
        if return_token_reliability:
            return (
                base_tokens,
                contextual_tokens,
                predicted_reliability,
                effective_reliability,
                gate_diagnostics,
            )
        return base_tokens, contextual_tokens

    def calibrate_token_reliability(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            not self.token_reliability_enabled
            or self.token_reliability_estimator is None
            or self.token_reliability_degradation is None
        ):
            zero = image.float().sum() * 0.0
            return {
                "token_reliability_calibration_loss": zero,
                "token_reliability_calibration_active": zero,
            }

        cuda_devices = []
        if image.is_cuda:
            cuda_devices.append(
                torch.cuda.current_device()
                if image.device.index is None
                else image.device.index
            )
        with torch.random.fork_rng(devices=cuda_devices):
            degraded_image, corruption_strength, kind_indices = (
                self.token_reliability_degradation(image.float())
            )
            calibration_batch = int(image.shape[0])
            paired_fused = self.encoder.forward_calibration(
                torch.cat([image.float(), degraded_image], dim=0)
            )
        clean_fused, degraded_fused = paired_fused.split(calibration_batch, dim=0)
        target, representation_drift = token_stability_target(
            clean_fused,
            degraded_fused,
            corruption_strength,
            temperature=self.token_reliability_target_temperature,
            corruption_penalty=self.token_reliability_corruption_penalty,
            minimum_probability=(
                self.token_reliability_estimator.minimum_probability
            ),
        )
        with torch.autocast(
            device_type=degraded_fused.device.type,
            enabled=False,
        ):
            prediction = self.token_reliability_estimator(
                degraded_fused.detach().float(),
            )
        calibration = token_reliability_calibration_loss(
            prediction,
            target,
            corruption_strength,
            ranking_margin=self.token_reliability_ranking_margin,
            ranking_weight=self.token_reliability_ranking_weight,
        )
        calibration.update(
            {
                "token_reliability_calibration_active": prediction.new_ones(()),
                "token_reliability_representation_drift_mean": (
                    representation_drift.mean()
                ),
                "token_reliability_corruption_fraction": (
                    (corruption_strength > 0).float().mean()
                ),
                "token_reliability_degradation_kind_mean": (
                    kind_indices.float().mean()
                ),
            }
        )
        return calibration

    def build_evidence(
        self,
        image_tokens: torch.Tensor,
        clinical_tokens: torch.Tensor,
        detach_image_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        evidence_image_tokens = (
            image_tokens.detach() if detach_image_tokens else image_tokens
        )
        if self.qred is not None:
            matched, transport, query_mass, qred_diagnostics = self.qred(
                clinical_tokens,
                evidence_image_tokens,
            )
            scaled_matched_tokens = qred_diagnostics.pop(
                "_scaled_matched_tokens", None
            )
            if scaled_matched_tokens is None:
                scaled_matched_tokens = self.qred.scale_aligned_evidence(
                    matched,
                    query_mass,
                )
        else:
            batch_size, image_token_count, token_dim = (
                evidence_image_tokens.shape
            )
            matched = evidence_image_tokens.new_empty(
                (batch_size, 0, token_dim)
            )
            scaled_matched_tokens = matched
            transport = evidence_image_tokens.new_empty(
                (batch_size, 0, image_token_count)
            )
            query_mass = evidence_image_tokens.new_empty((batch_size, 0))
            qred_diagnostics = {}
        image_risk_tokens = self.image_risk_pooler(evidence_image_tokens)
        evidence_bank_size = (
            clinical_tokens.shape[1]
            + scaled_matched_tokens.shape[1]
            + image_risk_tokens.shape[1]
        )
        if evidence_bank_size != self.expected_evidence_bank_size:
            raise RuntimeError(
                "The survival evidence bank size does not match the active "
                "semantic-transfer setting."
            )
        return {
            "matched_tokens": matched,
            "scaled_matched_tokens": scaled_matched_tokens,
            "transport_plan": transport,
            "query_mass": query_mass,
            "image_risk_tokens": image_risk_tokens,
            "evidence_bank_size": torch.as_tensor(
                evidence_bank_size,
                device=image_tokens.device,
            ),
            **qred_diagnostics,
        }

    def mix_clinical_evidence(
        self,
        image_tokens: torch.Tensor,
        clinical_tokens: torch.Tensor,
        permutation: torch.Tensor,
        lam: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if lam.ndim != 1 or lam.shape[0] != clinical_tokens.shape[0]:
            raise ValueError("lam must contain one value per patient.")
        clinical_weight = lam.to(clinical_tokens.dtype).view(-1, 1, 1)
        mixed_clinical = (
            clinical_weight * clinical_tokens
            + (1.0 - clinical_weight) * clinical_tokens[permutation]
        )
        outputs = self.build_evidence(
            image_tokens,
            mixed_clinical,
            detach_image_tokens=True,
        )
        outputs["mixed_clinical_tokens"] = mixed_clinical
        return outputs

    def forward(
        self,
        image: torch.Tensor,
        clinical_continuous: torch.Tensor,
        clinical_categorical: torch.Tensor,
        clinical_composite: torch.Tensor,
        compute_auxiliary: bool = True,
        contrastive_image: torch.Tensor | None = None,
        contrastive_indices: torch.Tensor | None = None,
        token_reliability_gate_progress: float = 1.0,
        compute_token_reliability_calibration: bool = False,
        token_reliability_calibration_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        (
            base_tokens,
            contextual_tokens,
            predicted_token_reliability,
            effective_token_reliability,
            gate_diagnostics,
        ) = self.encode_image(
            image,
            token_reliability_gate_progress=token_reliability_gate_progress,
            return_token_reliability=True,
        )
        image_tokens = contextual_tokens.flatten(1, 2)
        clinical_tokens = self.clinical_encoder(
            clinical_continuous.float(),
            clinical_categorical.long(),
            clinical_composite.long(),
        )
        evidence_outputs = self.build_evidence(image_tokens, clinical_tokens)
        outputs = {
            "base_tokens": base_tokens,
            "image_tokens": image_tokens,
            "clinical_tokens": clinical_tokens,
            **evidence_outputs,
            **self.predict_from_evidence(
                clinical_tokens,
                evidence_outputs["scaled_matched_tokens"],
                evidence_outputs["image_risk_tokens"],
            ),
            **gate_diagnostics,
        }
        if predicted_token_reliability is not None:
            outputs.update(
                {
                    "predicted_token_reliability": predicted_token_reliability,
                    "effective_token_reliability": effective_token_reliability,
                    "token_reliability_mean": predicted_token_reliability.mean(),
                    "token_reliability_gate_progress": (
                        predicted_token_reliability.new_tensor(
                            min(
                                max(
                                    float(token_reliability_gate_progress),
                                    0.0,
                                ),
                                1.0,
                            )
                        )
                    ),
                }
            )
        if compute_auxiliary:
            dynamic_contrastive_tokens = None
            if contrastive_image is not None:
                if contrastive_indices is None:
                    if contrastive_image.shape[0] != image.shape[0]:
                        raise ValueError(
                            "contrastive_indices are required for a "
                            "contrastive subset."
                        )
                    contrastive_indices = torch.arange(
                        image.shape[0],
                        device=image.device,
                    )
                _, dynamic_contrastive_tokens = self.encode_image(
                    contrastive_image
                )
            outputs.update(
                self.auxiliary(
                    base_tokens,
                    contextual_tokens,
                    image,
                    dynamic_contrastive_tokens,
                    contrastive_indices,
                )
            )
        if self.token_reliability_enabled:
            calibration_image = (
                image
                if token_reliability_calibration_image is None
                else token_reliability_calibration_image
            )
            if compute_token_reliability_calibration:
                calibration = self.calibrate_token_reliability(
                    calibration_image
                )
            else:
                zero = image.float().sum() * 0.0
                calibration = {
                    "token_reliability_calibration_loss": zero,
                    "token_reliability_calibration_active": zero,
                }
            outputs.update(calibration)
        return outputs


ReSTOTSurvivalModel = PMFSurv
