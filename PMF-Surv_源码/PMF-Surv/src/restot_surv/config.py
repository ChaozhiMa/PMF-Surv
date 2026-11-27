from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _normalize_pre_qred_module_names(cfg: dict[str, Any]) -> None:
    """Convert historical configuration names before validation and use."""

    model = cfg.get("model")
    if not isinstance(model, dict):
        return
    matching = model.get("matching")
    if not isinstance(matching, dict):
        return
    if str(matching.get("type", "")).lower() == "rapot":
        matching["type"] = "qred"
    matching.pop("rapot_variant", None)


def paper_config_path(project_root: str | Path) -> Path:
    """Resolve the locally bundled or shared PMF-Surv training profile."""
    project = Path(project_root).resolve()
    local = project / "config" / "hard_top1.yaml"
    if local.is_file():
        return local
    for parent in (project, *project.parents):
        shared = parent / "restot_surv_hard_top1" / "config" / "hard_top1.yaml"
        if shared.is_file():
            return shared
    return local


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping.")
    cfg = copy.deepcopy(dict(raw))
    _normalize_pre_qred_module_names(cfg)
    cfg["_config_path"] = str(config_path)
    validate_config(cfg)
    return cfg


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {context}.{key}")
    return mapping[key]


def validate_config(cfg: Mapping[str, Any]) -> None:
    for section in ("experiment", "data", "model", "auxiliary", "loss", "schedule", "training"):
        _require(cfg, section, "config")
    data = cfg["data"]
    model = cfg["model"]
    training = cfg["training"]
    image_shape = tuple(int(v) for v in _require(data, "image_shape", "data"))
    patch_grid = tuple(int(v) for v in _require(model, "patch_grid", "model"))
    patch_size = tuple(int(v) for v in _require(model, "patch_size", "model"))
    if len(image_shape) != 4 or image_shape[0] != int(model["num_phases"]):
        raise ValueError("data.image_shape must be [num_phases, depth, height, width].")
    phase_order = tuple(str(value) for value in data["phase_order"])
    source_phase_order = tuple(
        str(value)
        for value in data.get("source_phase_order", phase_order)
    )
    if (
        not source_phase_order
        or len(source_phase_order) != len(set(source_phase_order))
    ):
        raise ValueError(
            "data.source_phase_order must contain unique phase names."
        )
    if (
        not phase_order
        or len(phase_order) != len(set(phase_order))
        or any(phase not in source_phase_order for phase in phase_order)
    ):
        raise ValueError(
            "data.phase_order must be a non-empty unique subset of "
            "data.source_phase_order."
        )
    if len(phase_order) != int(model["num_phases"]):
        raise ValueError(
            "The number of selected data phases must equal model.num_phases."
        )
    expected = tuple(patch_grid[i] * patch_size[i] for i in range(3))
    if image_shape[1:] != expected:
        raise ValueError(
            f"Image spatial shape {image_shape[1:]} must equal patch_grid * patch_size = {expected}."
        )
    backbone = model.get("backbone", {})
    if not isinstance(backbone, Mapping):
        raise ValueError("model.backbone must be a mapping.")
    backbone_type = str(backbone.get("type", "swin")).lower().replace("-", "_")
    supported_backbones = {
        "swin",
        "swin3d",
        "3d_swin",
        "resnet50",
        "resnet_50",
        "3d_resnet50",
        "densenet121",
        "densenet_121",
        "3d_densenet121",
        "convnext_tiny",
        "convnexttiny",
        "3d_convnext_tiny",
        "vit_small",
        "vitsmall",
        "3d_vit_small",
        "efficientnet_b1",
        "efficientnetb1",
        "3d_efficientnet_b1",
    }
    if backbone_type not in supported_backbones:
        raise ValueError(
            "Unsupported model.backbone.type; expected one of "
            f"{sorted(supported_backbones)}."
        )
    if backbone_type in {"swin", "swin3d", "3d_swin"}:
        swin = _require(model, "swin", "model")
        swin_patch_size = tuple(
            int(v) for v in _require(swin, "patch_size", "model.swin")
        )
        window_size = tuple(
            int(v) for v in _require(swin, "window_size", "model.swin")
        )
        depths = tuple(int(v) for v in _require(swin, "depths", "model.swin"))
        num_heads = tuple(
            int(v) for v in _require(swin, "num_heads", "model.swin")
        )
        if len(swin_patch_size) != 3 or len(window_size) != 3:
            raise ValueError(
                "model.swin.patch_size and window_size must contain three dimensions."
            )
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError(
                "model.swin.depths and num_heads must describe four Swin stages."
            )
        if any(
            value <= 0
            for value in (*swin_patch_size, *window_size, *depths, *num_heads)
        ):
            raise ValueError("All Swin spatial, depth, and head settings must be positive.")
        hierarchy_scales = []
        for reconstruction_size, swin_size in zip(patch_size, swin_patch_size):
            if reconstruction_size % swin_size:
                raise ValueError(
                    "model.patch_size must be divisible by model.swin.patch_size."
                )
            scale = reconstruction_size // swin_size
            if scale not in {1, 2, 4, 8, 16}:
                raise ValueError(
                    "The reconstruction grid must align with one of the five Swin hierarchy levels."
                )
            hierarchy_scales.append(scale)
        if len(set(hierarchy_scales)) != 1:
            raise ValueError(
                "The Swin output hierarchy level must be the same on all spatial axes."
            )
        feature_size = int(_require(swin, "feature_size", "model.swin"))
        for stage, heads in enumerate(num_heads):
            if feature_size * (2**stage) % heads != 0:
                raise ValueError(
                    f"Swin stage {stage} width must be divisible by its attention head count."
                )
    else:
        token_grid = tuple(
            int(v) for v in backbone.get("token_grid", patch_grid)
        )
        if token_grid != patch_grid:
            raise ValueError(
                "Non-Swin backbones must use backbone.token_grid equal to model.patch_grid."
            )
        if len(token_grid) != 3 or any(value <= 0 for value in token_grid):
            raise ValueError("model.backbone.token_grid must contain three positive dimensions.")
    token_dim = int(model["token_dim"])
    if token_dim % int(model["interaction_heads"]) != 0:
        raise ValueError("model.token_dim must be divisible by model.interaction_heads.")
    tolerance_radius = int(
        _require(
            model,
            "interaction_tolerance_radius",
            "model",
        )
    )
    if tolerance_radius < 0 or tolerance_radius > 2:
        raise ValueError(
            "model.interaction_tolerance_radius must be between 0 and 2."
        )
    token_reliability = model.get("token_reliability", {})
    if not isinstance(token_reliability, Mapping):
        raise ValueError("model.token_reliability must be a mapping.")
    token_reliability_enabled = token_reliability.get("enabled", False)
    if not isinstance(token_reliability_enabled, bool):
        raise ValueError("model.token_reliability.enabled must be a boolean.")
    if token_reliability_enabled:
        if int(token_reliability.get("hidden_dim", token_dim)) <= 0:
            raise ValueError(
                "model.token_reliability.hidden_dim must be positive."
            )
        initial_probability = float(
            token_reliability.get("initial_probability", 0.9)
        )
        minimum_probability = float(
            token_reliability.get("minimum_probability", 0.02)
        )
        if not 0.0 < initial_probability < 1.0:
            raise ValueError(
                "model.token_reliability.initial_probability must be in (0, 1)."
            )
        if not 0.0 < minimum_probability < initial_probability:
            raise ValueError(
                "model.token_reliability.minimum_probability must be positive "
                "and below initial_probability."
            )
        if float(token_reliability.get("initial_weight_std", 1.0e-3)) <= 0.0:
            raise ValueError(
                "model.token_reliability.initial_weight_std must be positive."
            )
        if float(
            token_reliability.get("attention_log_bias_strength", 1.0)
        ) < 0.0:
            raise ValueError(
                "model.token_reliability.attention_log_bias_strength cannot "
                "be negative."
            )
        if int(token_reliability.get("gate_warmup_epochs", 0)) < 0:
            raise ValueError(
                "model.token_reliability.gate_warmup_epochs cannot be negative."
            )
        quality_band = token_reliability.get("quality_band", {})
        if not isinstance(quality_band, Mapping):
            raise ValueError(
                "model.token_reliability.quality_band must be a mapping."
            )
        quality_band_enabled = quality_band.get("enabled", False)
        if not isinstance(quality_band_enabled, bool):
            raise ValueError(
                "model.token_reliability.quality_band.enabled must be a boolean."
            )
        high_quality_weight = float(
            quality_band.get("high_quality_weight", 1.0)
        )
        low_quality_weight = float(
            quality_band.get("low_quality_weight", 0.02)
        )
        if not 0.0 < low_quality_weight < high_quality_weight <= 1.0:
            raise ValueError(
                "model.token_reliability.quality_band weights must satisfy "
                "0 < low_quality_weight < high_quality_weight <= 1."
            )
        if quality_band_enabled:
            mode = str(quality_band.get("mode", "fixed_hard")).lower()
            supported_modes = {
                "fixed_hard",
                "iterative_asymmetric_hard",
            }
            if mode not in supported_modes:
                raise ValueError(
                    "model.token_reliability.quality_band.mode must be one of "
                    f"{sorted(supported_modes)}."
                )
            if "learnable" in quality_band:
                raise ValueError(
                    "quality_band.learnable is obsolete; choose an explicit "
                    "hard-gate mode instead."
                )
            if mode == "fixed_hard":
                lower_threshold = float(
                    quality_band.get("lower_threshold", 0.35)
                )
                upper_threshold = float(
                    quality_band.get("upper_threshold", 0.80)
                )
                if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
                    raise ValueError(
                        "Fixed quality-band thresholds must satisfy "
                        "0 <= lower_threshold < upper_threshold <= 1."
                    )
            else:
                if str(
                    quality_band.get("grouping", "patient_phase")
                ).lower() != "patient_phase":
                    raise ValueError(
                        "iterative_asymmetric_hard requires "
                        "quality_band.grouping=patient_phase."
                    )
                target_fraction = float(
                    quality_band.get("target_fraction", 0.60)
                )
                if not 0.0 < target_fraction < 1.0:
                    raise ValueError(
                        "quality_band.target_fraction must lie in (0, 1)."
                    )
                if int(token_reliability.get("gate_warmup_epochs", 0)) != 0:
                    raise ValueError(
                        "iterative_asymmetric_hard requires gate_warmup_epochs=0 "
                        "so effective weights remain strictly binary."
                    )
        calibration = token_reliability.get("calibration", {})
        if not isinstance(calibration, Mapping):
            raise ValueError(
                "model.token_reliability.calibration must be a mapping."
            )
        if int(calibration.get("every_steps", 1)) <= 0:
            raise ValueError(
                "model.token_reliability.calibration.every_steps must be positive."
            )
        calibration_batch_size = int(
            calibration.get("batch_size_per_gpu", 1)
        )
        if not 1 <= calibration_batch_size <= int(
            training["batch_size_per_gpu"]
        ):
            raise ValueError(
                "model.token_reliability.calibration.batch_size_per_gpu must "
                "be between 1 and training.batch_size_per_gpu."
            )
        if float(calibration.get("loss_weight", 0.0)) < 0.0:
            raise ValueError(
                "model.token_reliability.calibration.loss_weight cannot be negative."
            )
        supported_degradations = {
            "noise",
            "blur",
            "stripes",
            "slice_loss",
            "intensity",
        }
        degradation_kinds = tuple(
            str(value).lower()
            for value in calibration.get(
                "degradation_kinds", supported_degradations
            )
        )
        if not degradation_kinds or not set(degradation_kinds).issubset(
            supported_degradations
        ):
            raise ValueError(
                "model.token_reliability.calibration.degradation_kinds "
                "contains an unsupported value."
            )
        for key in ("regions_per_sample", "region_size_tokens"):
            limits = tuple(int(value) for value in calibration.get(key, (1, 1)))
            if len(limits) != 2 or limits[0] <= 0 or limits[0] > limits[1]:
                raise ValueError(
                    f"model.token_reliability.calibration.{key} must be a "
                    "positive [min, max] pair."
                )
        severity = tuple(
            float(value) for value in calibration.get("severity", (0.4, 1.0))
        )
        if (
            len(severity) != 2
            or severity[0] <= 0.0
            or severity[0] > severity[1]
            or severity[1] > 1.0
        ):
            raise ValueError(
                "model.token_reliability.calibration.severity must be within "
                "(0, 1] as [min, max]."
            )
        if not 0.0 <= float(
            calibration.get("shared_phase_probability", 0.25)
        ) <= 1.0:
            raise ValueError(
                "model.token_reliability.calibration.shared_phase_probability "
                "must be in [0, 1]."
            )
        if int(calibration.get("blur_kernel_size", 5)) <= 0 or int(
            calibration.get("blur_kernel_size", 5)
        ) % 2 == 0:
            raise ValueError(
                "model.token_reliability.calibration.blur_kernel_size must be "
                "a positive odd number."
            )
        if float(calibration.get("target_temperature", 0.15)) <= 0.0:
            raise ValueError(
                "model.token_reliability.calibration.target_temperature must "
                "be positive."
            )
        if not 0.0 <= float(
            calibration.get("corruption_penalty", 0.75)
        ) <= 1.0:
            raise ValueError(
                "model.token_reliability.calibration.corruption_penalty must "
                "be in [0, 1]."
            )
        for key in ("ranking_margin", "ranking_weight", "noise_std"):
            if float(calibration.get(key, 0.0)) < 0.0:
                raise ValueError(
                    f"model.token_reliability.calibration.{key} cannot be negative."
                )
    ablation = model.get("ablation", {})
    if not isinstance(ablation, Mapping):
        raise ValueError("model.ablation must be a mapping.")
    cross_phase_interaction = ablation.get(
        "cross_phase_interaction",
        True,
    )
    if not isinstance(cross_phase_interaction, bool):
        raise ValueError(
            "model.ablation.cross_phase_interaction must be a boolean."
        )
    semantic_transfer = ablation.get("semantic_transfer", True)
    if not isinstance(semantic_transfer, bool):
        raise ValueError(
            "model.ablation.semantic_transfer must be a boolean."
        )
    time_autonomous_query = ablation.get(
        "time_autonomous_query",
        True,
    )
    if not isinstance(time_autonomous_query, bool):
        raise ValueError(
            "model.ablation.time_autonomous_query must be a boolean."
        )
    semantic_fusion = str(
        ablation.get("semantic_fusion", "proposed_ot")
    ).lower()
    supported_semantic_fusion = {
        "proposed_ot",
        "standard_cross_attention",
        "balanced_ot",
        "conventional_uot",
    }
    if semantic_fusion not in supported_semantic_fusion:
        raise ValueError(
            "model.ablation.semantic_fusion must be one of "
            f"{sorted(supported_semantic_fusion)}."
        )
    if semantic_fusion == "conventional_uot" and float(
        model.get("ot", {}).get(
            "mass_penalty",
            model.get("ot", {}).get("query_balance_weight", 0.5),
        )
    ) <= 0.0:
        raise ValueError(
            "model.ot.mass_penalty must be positive for conventional UOT."
        )
    if int(model.get("image_risk_tokens", 0)) <= 0:
        raise ValueError("model.image_risk_tokens must be positive.")
    matching = model.get("matching", {})
    if not isinstance(matching, Mapping):
        raise ValueError("model.matching must be a mapping.")
    matching_type = str(matching.get("type", "qred")).lower()
    if matching_type != "qred":
        raise ValueError("model.matching.type must be qred.")
    survival_head = model.get("survival_head", {"type": "mtlr"})
    if not isinstance(survival_head, Mapping):
        raise ValueError("model.survival_head must be a mapping.")
    survival_head_type = str(survival_head.get("type", "mtlr")).lower()
    if survival_head_type not in {"mtlr", "piecewise_exponential"}:
        raise ValueError(
            "model.survival_head.type must be mtlr or piecewise_exponential."
        )
    if float(survival_head.get("initial_hazard_rate", 0.01)) <= 0.0:
        raise ValueError("model.survival_head.initial_hazard_rate must be positive.")
    if float(survival_head.get("initial_rate_weight_std", 1.0e-3)) <= 0.0:
        raise ValueError(
            "model.survival_head.initial_rate_weight_std must be positive."
        )
    os_prediction = _require(model, "os_prediction", "model")
    os_prediction_mode = str(
        _require(os_prediction, "mode", "model.os_prediction")
    )
    if os_prediction_mode != "time_query_independent":
        raise ValueError(
            "model.os_prediction.mode must be time_query_independent for "
            "PMF-Surv."
        )
    pfs_detach_shared = os_prediction.get(
        "pfs_detach_shared",
        os_prediction_mode
        in {"time_query_conditional", "time_query_independent"},
    )
    if not isinstance(pfs_detach_shared, bool):
        raise ValueError(
            "model.os_prediction.pfs_detach_shared must be a boolean."
        )
    if not pfs_detach_shared:
        raise ValueError(
            "time_query_independent requires pfs_detach_shared=true."
        )
    time_query = _require(model, "time_query", "model")
    if not isinstance(time_query, Mapping):
        raise ValueError("model.time_query must be a mapping.")
    heads = int(_require(time_query, "attention_heads", "model.time_query"))
    if heads <= 0 or token_dim % heads:
        raise ValueError(
            "model.time_query.attention_heads must be positive and divide "
            "model.token_dim."
        )
    auxiliary = cfg["auxiliary"]
    mask_ratio = float(_require(auxiliary, "reconstruction_mask_ratio", "auxiliary"))
    shared_fraction = float(
        _require(auxiliary, "reconstruction_shared_mask_fraction", "auxiliary")
    )
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("auxiliary.reconstruction_mask_ratio must be between zero and one.")
    if not 0.0 <= shared_fraction <= 1.0:
        raise ValueError(
            "auxiliary.reconstruction_shared_mask_fraction must be between zero and one."
        )
    block_size = tuple(
        int(v) for v in _require(auxiliary, "reconstruction_block_size", "auxiliary")
    )
    if len(block_size) != 2 or block_size[0] <= 0 or block_size[0] > block_size[1]:
        raise ValueError(
            "auxiliary.reconstruction_block_size must be [positive_min, max]."
        )
    if int(auxiliary["spatial_relative_pairs"]) <= 0:
        raise ValueError("auxiliary.spatial_relative_pairs must be positive.")
    dynamic_contrastive_enabled = auxiliary.get(
        "dynamic_contrastive_enabled", True
    )
    if not isinstance(dynamic_contrastive_enabled, bool):
        raise ValueError(
            "auxiliary.dynamic_contrastive_enabled must be a boolean."
        )
    if int(auxiliary["dynamic_contrastive_until_epoch"]) <= 0:
        raise ValueError(
            "auxiliary.dynamic_contrastive_until_epoch must be positive."
        )
    if int(auxiliary["dynamic_contrastive_every_steps"]) <= 0:
        raise ValueError(
            "auxiliary.dynamic_contrastive_every_steps must be positive."
        )
    contrastive_batch_size = int(
        auxiliary["dynamic_contrastive_batch_size_per_gpu"]
    )
    if (
        contrastive_batch_size <= 0
        or contrastive_batch_size > int(training["batch_size_per_gpu"])
    ):
        raise ValueError(
            "auxiliary.dynamic_contrastive_batch_size_per_gpu must be "
            "between 1 and training.batch_size_per_gpu."
        )
    contrastive_microbatch_size = int(
        auxiliary.get(
            "dynamic_contrastive_microbatch_size_per_gpu",
            contrastive_batch_size,
        )
    )
    if (
        contrastive_microbatch_size <= 0
        or contrastive_microbatch_size > contrastive_batch_size
    ):
        raise ValueError(
            "auxiliary.dynamic_contrastive_microbatch_size_per_gpu must be "
            "between 1 and dynamic_contrastive_batch_size_per_gpu."
        )
    if int(training["batch_size_per_gpu"]) <= 0 or int(training["epochs"]) <= 0:
        raise ValueError("training batch size and epochs must be positive.")
    if int(training["gradient_accumulation_steps"]) <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive.")
    if str(training.get("sampler", "")) != "joint_event_batch":
        raise ValueError("The initial validated release supports training.sampler=joint_event_batch only.")
    if str(training["mixed_precision"]).lower() not in {"none", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision must be one of none/fp16/bf16.")
    if not isinstance(training["deterministic"], bool):
        raise ValueError("training.deterministic must be a boolean.")
    checkpoint_selection_metric = str(
        training.get("checkpoint_selection_metric", "os_cindex")
    ).lower()
    if checkpoint_selection_metric != "os_cindex":
        raise ValueError("training.checkpoint_selection_metric must be os_cindex.")
    extra_checkpoint_epochs = training.get("extra_checkpoint_epochs", [])
    if not isinstance(extra_checkpoint_epochs, (list, tuple)):
        raise ValueError(
            "training.extra_checkpoint_epochs must be a list of epochs."
        )
    parsed_extra_epochs = [int(value) for value in extra_checkpoint_epochs]
    if len(set(parsed_extra_epochs)) != len(parsed_extra_epochs) or any(
        value <= 0 or value > int(training["epochs"])
        for value in parsed_extra_epochs
    ):
        raise ValueError(
            "training.extra_checkpoint_epochs must contain unique epochs "
            "between 1 and training.epochs."
        )
    minimum_training_epochs = int(
        training.get("minimum_training_epochs", 0)
    )
    if not 0 <= minimum_training_epochs <= int(training["epochs"]):
        raise ValueError(
            "training.minimum_training_epochs must be between 0 and "
            "training.epochs."
        )
    if bool(data.get("persistent_workers", False)):
        raise ValueError(
            "data.persistent_workers must be false so resumed runs can "
            "reproduce worker random states."
        )
    if float(training["max_stratum_oversample"]) < 1.0:
        raise ValueError("training.max_stratum_oversample must be at least 1.")
    if float(training["stage_boundary_lr"]) <= 0.0:
        raise ValueError("training.stage_boundary_lr must be positive.")
    if int(training["stage1_warmup_steps"]) < 0:
        raise ValueError("training.stage1_warmup_steps cannot be negative.")
    if int(training["stage2_head_warmup_epochs"]) <= 0:
        raise ValueError("training.stage2_head_warmup_epochs must be positive.")
    learning_rate_keys = [
        "encoder_lr",
        "fusion_lr",
        "head_lr",
        "auxiliary_lr",
    ]
    if token_reliability_enabled:
        _require(training, "token_reliability_lr", "training")
        learning_rate_keys.append("token_reliability_lr")
    for key in learning_rate_keys:
        if float(training[key]) < float(training["stage_boundary_lr"]):
            raise ValueError(f"training.{key} must be at least stage_boundary_lr.")
    schedule = cfg["schedule"]
    warmup_end = int(schedule["structure_end_epoch"])
    transition_end = int(schedule["transition_end_epoch"])
    no_staged_schedule = warmup_end == 0 and transition_end == 0
    if not (
        no_staged_schedule
        or 0 <= warmup_end < transition_end <= int(training["epochs"])
    ):
        raise ValueError(
            "Schedule epochs must satisfy either structure_end_epoch=0 and "
            "transition_end_epoch=0, or 0 <= structure_end_epoch < "
            "transition_end_epoch <= epochs."
        )
    warmup_regularization_scale = float(
        _require(
            schedule,
            "warmup_regularization_scale",
            "schedule",
        )
    )
    if not 0.0 < warmup_regularization_scale <= 1.0:
        raise ValueError(
            "schedule.warmup_regularization_scale must be in (0, 1]."
        )
    ecomix = cfg["loss"]["ecomix"]
    detach_image_tokens = _require(
        ecomix,
        "detach_image_tokens",
        "loss.ecomix",
    )
    if not isinstance(detach_image_tokens, bool):
        raise ValueError(
            "loss.ecomix.detach_image_tokens must be a boolean."
        )
    if not bool(detach_image_tokens):
        raise ValueError(
            "Clinical-only ECOMix requires loss.ecomix.detach_image_tokens=true."
        )
    if not 0.0 <= float(ecomix["probability"]) <= 1.0:
        raise ValueError("loss.ecomix.probability must be in [0, 1].")
    if float(ecomix["alpha"]) <= 0.0:
        raise ValueError("loss.ecomix.alpha must be positive.")
    if int(ecomix["top_k"]) <= 0:
        raise ValueError("loss.ecomix.top_k must be positive.")
    if float(ecomix["temperature"]) <= 0.0:
        raise ValueError("loss.ecomix.temperature must be positive.")
    if float(ecomix["endpoint_weight"]) < 0.0:
        raise ValueError("loss.ecomix.endpoint_weight cannot be negative.")
    for key in (
        "os_weight",
        "pfs_weight",
        "os_cox_weight",
        "pfs_cox_weight",
        "os_rank_weight",
        "pfs_rank_weight",
        "brier_weight",
        "endpoint_consistency_weight",
    ):
        if float(cfg["loss"][key]) < 0.0:
            raise ValueError(f"loss.{key} cannot be negative.")
    if float(cfg["loss"]["rank_temperature"]) <= 0.0:
        raise ValueError("loss.rank_temperature must be positive.")
    if int(cfg["loss"]["rank_warmup_epochs"]) <= 0:
        raise ValueError("loss.rank_warmup_epochs must be positive.")
    rank_ipcw = cfg["loss"].get("rank_ipcw", {})
    if not isinstance(rank_ipcw, Mapping):
        raise ValueError("loss.rank_ipcw must be a mapping.")
    for key in ("enabled", "apply_to_pfs"):
        if key in rank_ipcw and not isinstance(rank_ipcw[key], bool):
            raise ValueError(f"loss.rank_ipcw.{key} must be a boolean.")
    if not 0.0 < float(rank_ipcw.get("survival_floor", 0.1)) <= 1.0:
        raise ValueError(
            "loss.rank_ipcw.survival_floor must be in (0, 1]."
        )
    if not 0.0 < float(rank_ipcw.get("cap_quantile", 0.99)) <= 1.0:
        raise ValueError(
            "loss.rank_ipcw.cap_quantile must be in (0, 1]."
        )
    ot = model["ot"]
    semantic_affinity = ot.get("semantic_affinity", {})
    if not isinstance(semantic_affinity, Mapping):
        raise ValueError("model.ot.semantic_affinity must be a mapping.")
    semantic_affinity_mode = str(
        semantic_affinity.get("mode", "cosine")
    ).lower()
    if semantic_affinity_mode != "cosine":
        raise ValueError("model.ot.semantic_affinity.mode must be cosine.")
    if float(ot["epsilon"]) <= 0.0:
        raise ValueError("model.ot.epsilon must be positive.")
    if int(ot["iterations"]) <= 0 or float(ot["tolerance"]) <= 0.0:
        raise ValueError("model.ot iterations and tolerance must be positive.")
    if float(ot.get("query_balance_weight", 0.5)) < 0.0:
        raise ValueError("model.ot.query_balance_weight cannot be negative.")
    normalization = str(data.get("normalization", "zscore_joint"))
    if normalization not in {"none", "zscore_joint", "zscore_per_phase"}:
        raise ValueError(
            "data.normalization must be one of none/zscore_joint/zscore_per_phase."
        )


def save_resolved_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    payload = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
