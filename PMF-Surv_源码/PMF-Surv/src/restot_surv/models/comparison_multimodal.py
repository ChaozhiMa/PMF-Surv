from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from restot_surv.models.survival import MTLRHead


def _conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
) -> nn.Conv3d:
    padding = 0 if kernel_size == 1 else kernel_size // 2
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = _conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                _conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(inputs)
        outputs = self.activation(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.activation(outputs + residual)


class DAFTResidualBlock3D(nn.Module):
    """Dynamic affine conditioning used by DAFT before the final residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        clinical_dim: int,
        bottleneck_dim: int,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conditioner = nn.Sequential(
            nn.Linear(in_channels + clinical_dim, bottleneck_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 2 * in_channels, bias=False),
        )
        self.conv1 = _conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = _conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            _conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm3d(out_channels),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(feature_map).flatten(1)
        affine = self.conditioner(torch.cat([pooled, clinical], dim=1))
        scale, shift = affine.chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        conditioned = scale * feature_map + shift
        residual = self.downsample(conditioned)
        outputs = self.activation(self.bn1(self.conv1(conditioned)))
        outputs = self.bn2(self.conv2(outputs))
        return self.activation(outputs + residual)


class ClinicalOneHot(nn.Module):
    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
    ) -> None:
        super().__init__()
        self.categorical_cardinalities = tuple(
            int(value) for value in categorical_cardinalities
        )
        self.composite_cardinalities = tuple(
            int(value) for value in composite_cardinalities
        )

    @property
    def output_dim(self) -> int:
        return 4 + sum(self.categorical_cardinalities) + sum(
            self.composite_cardinalities
        )

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
    ) -> torch.Tensor:
        values = [continuous.float()]
        for index, cardinality in enumerate(self.categorical_cardinalities):
            values.append(
                F.one_hot(
                    categorical[:, index].long(),
                    num_classes=cardinality,
                ).float()
            )
        for index, cardinality in enumerate(self.composite_cardinalities):
            values.append(
                F.one_hot(
                    composite[:, index].long(),
                    num_classes=cardinality,
                ).float()
            )
        return torch.cat(values, dim=1)


class DAFTMTLR(nn.Module):
    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
        hidden_dim: int,
        num_bins: int,
        dropout: float,
        base_channels: int = 32,
        daft_bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.clinical_vectorizer = ClinicalOneHot(
            categorical_cardinalities,
            composite_cardinalities,
        )
        clinical_dim = self.clinical_vectorizer.output_dim
        self.stem = nn.Sequential(
            _conv3d(3, base_channels),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.block1 = ResidualBlock3D(base_channels, base_channels)
        self.block2 = ResidualBlock3D(base_channels, 2 * base_channels, stride=2)
        self.block3 = ResidualBlock3D(
            2 * base_channels,
            4 * base_channels,
            stride=2,
        )
        self.daft_block = DAFTResidualBlock3D(
            4 * base_channels,
            8 * base_channels,
            clinical_dim=clinical_dim,
            bottleneck_dim=daft_bottleneck_dim,
            stride=2,
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        feature_dim = 8 * base_channels
        self.os_head = MTLRHead(feature_dim, hidden_dim, num_bins, dropout)
        self.pfs_head = MTLRHead(feature_dim, hidden_dim, num_bins, dropout)

    def shared_parameters(self) -> list[nn.Parameter]:
        head_ids = {
            id(parameter)
            for head in (self.os_head, self.pfs_head)
            for parameter in head.parameters()
        }
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in head_ids
        ]

    def forward(
        self,
        image: torch.Tensor,
        clinical_continuous: torch.Tensor,
        clinical_categorical: torch.Tensor,
        clinical_composite: torch.Tensor,
        stage: str = "survival",
        mask_ratio: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        del stage, mask_ratio
        clinical = self.clinical_vectorizer(
            clinical_continuous,
            clinical_categorical,
            clinical_composite,
        )
        features = self.stem(image)
        features = self.block1(features)
        features = self.block2(features)
        features = self.block3(features)
        features = self.daft_block(features, clinical)
        embedding = self.global_pool(features).flatten(1)
        return survival_outputs(self.os_head, self.pfs_head, embedding)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SliceDepthTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.slice_norm = nn.LayerNorm(dim)
        self.slice_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.slice_ffn_norm = nn.LayerNorm(dim)
        self.slice_ffn = FeedForward(dim, dropout)
        self.depth_norm = nn.LayerNorm(dim)
        self.depth_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.depth_ffn_norm = nn.LayerNorm(dim)
        self.depth_ffn = FeedForward(dim, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, depth, spatial, channels = tokens.shape
        slices = tokens.reshape(batch * depth, spatial, channels)
        normalized = self.slice_norm(slices)
        attended, _ = self.slice_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        slices = slices + attended
        slices = slices + self.slice_ffn(self.slice_ffn_norm(slices))
        tokens = slices.reshape(batch, depth, spatial, channels)

        columns = tokens.permute(0, 2, 1, 3).reshape(
            batch * spatial,
            depth,
            channels,
        )
        normalized = self.depth_norm(columns)
        attended, _ = self.depth_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        columns = columns + attended
        columns = columns + self.depth_ffn(self.depth_ffn_norm(columns))
        return columns.reshape(batch, spatial, depth, channels).permute(0, 2, 1, 3)


class VisualSliceDepthEncoder(nn.Module):
    def __init__(
        self,
        image_shape: Sequence[int],
        patch_size: Sequence[int],
        dim: int,
        heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        phases, volume_depth, height, width = (int(value) for value in image_shape)
        patch_depth, patch_height, patch_width = (
            int(value) for value in patch_size
        )
        if (
            volume_depth % patch_depth
            or height % patch_height
            or width % patch_width
        ):
            raise ValueError("Image dimensions must be divisible by the CMC patch size.")
        self.patch_size = (patch_depth, patch_height, patch_width)
        self.grid_depth = volume_depth // patch_depth
        self.grid_height = height // patch_height
        self.grid_width = width // patch_width
        self.spatial_tokens = self.grid_height * self.grid_width
        self.patch_dim = phases * patch_depth * patch_height * patch_width
        self.patch_embedding = nn.Conv3d(
            phases,
            dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.depth_position = nn.Parameter(torch.zeros(1, self.grid_depth, 1, dim))
        self.spatial_position = nn.Parameter(
            torch.zeros(1, 1, self.spatial_tokens, dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.blocks = nn.ModuleList(
            [
                SliceDepthTransformerBlock(dim, heads, dropout)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.depth_position, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, image: torch.Tensor) -> torch.Tensor:
        patch_depth, patch_height, patch_width = self.patch_size
        patches = (
            image.unfold(2, patch_depth, patch_depth)
            .unfold(3, patch_height, patch_height)
            .unfold(4, patch_width, patch_width)
        )
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        batch = patches.shape[0]
        return patches.view(
            batch,
            self.grid_depth,
            self.spatial_tokens,
            self.patch_dim,
        )

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(image)
        tokens = tokens.flatten(3).permute(0, 2, 3, 1)
        return tokens + self.depth_position + self.spatial_position

    def encode(
        self,
        embedded: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = embedded
        if mask is not None:
            tokens = torch.where(
                mask[..., None],
                self.mask_token.to(dtype=tokens.dtype),
                tokens,
            )
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(self.embed(image), mask)


class ClinicalVariableEmbedding(nn.Module):
    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
        dim: int,
    ) -> None:
        super().__init__()
        self.continuous_scale = nn.Parameter(torch.empty(4, dim))
        self.continuous_bias = nn.Parameter(torch.zeros(4, dim))
        cardinalities = tuple(int(v) for v in categorical_cardinalities) + tuple(
            int(v) for v in composite_cardinalities
        )
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dim) for cardinality in cardinalities]
        )
        self.variable_identity = nn.Parameter(
            torch.zeros(1, 4 + len(cardinalities), dim)
        )
        nn.init.normal_(self.continuous_scale, std=0.02)
        nn.init.trunc_normal_(self.variable_identity, std=0.02)

    @property
    def num_variables(self) -> int:
        return self.variable_identity.shape[1]

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
    ) -> torch.Tensor:
        numeric = (
            continuous.float().unsqueeze(-1) * self.continuous_scale.unsqueeze(0)
            + self.continuous_bias.unsqueeze(0)
        )
        category_values = torch.cat([categorical, composite], dim=1).long()
        embedded = [numeric]
        for index, layer in enumerate(self.category_embeddings):
            embedded.append(layer(category_values[:, index])[:, None, :])
        return torch.cat(embedded, dim=1) + self.variable_identity


class GraphTransformerBlock(nn.Module):
    def __init__(self, num_nodes: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.global_nodes = nn.Parameter(torch.zeros(num_nodes, dim))
        self.relation_diagonal = nn.Parameter(torch.ones(dim))
        self.value = nn.Linear(dim, dim, bias=False)
        self.graph_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.global_nodes, std=0.02)

    def forward(
        self,
        nodes: torch.Tensor,
        masked_variables: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample_edges = torch.matmul(
            nodes * self.relation_diagonal,
            nodes.transpose(1, 2),
        )
        global_edges = torch.matmul(
            self.global_nodes,
            self.global_nodes.transpose(0, 1),
        )
        scores = sample_edges + global_edges.unsqueeze(0)
        if masked_variables is not None:
            key_mask = torch.cat(
                [
                    torch.zeros(
                        masked_variables.shape[0],
                        1,
                        dtype=torch.bool,
                        device=masked_variables.device,
                    ),
                    masked_variables,
                ],
                dim=1,
            )
            scores = scores.masked_fill(key_mask[:, None, :], -1.0e4)
        graph = torch.softmax(scores / (nodes.shape[-1] ** 0.5), dim=-1)
        updated = torch.matmul(graph, self.value(nodes))
        nodes = self.graph_norm(nodes + self.dropout(updated))
        return nodes + self.ffn(self.ffn_norm(nodes))


class ClinicalGraphEncoder(nn.Module):
    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
        dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = ClinicalVariableEmbedding(
            categorical_cardinalities,
            composite_cardinalities,
            dim,
        )
        self.num_variables = self.embedding.num_variables
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.variable_mask_tokens = nn.Parameter(
            torch.zeros(1, self.num_variables, dim)
        )
        self.blocks = nn.ModuleList(
            [
                GraphTransformerBlock(self.num_variables + 1, dim, dropout)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.variable_mask_tokens, std=0.02)

    def embed(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(continuous, categorical, composite)

    def encode(
        self,
        embedded: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        variables = embedded
        if mask is not None:
            variables = torch.where(
                mask[..., None],
                self.variable_mask_tokens.to(dtype=variables.dtype),
                variables,
            )
        cls = self.cls_token.expand(variables.shape[0], -1, -1)
        nodes = torch.cat([cls, variables], dim=1)
        for block in self.blocks:
            nodes = block(nodes, mask)
        return self.output_norm(nodes)

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode(
            self.embed(continuous, categorical, composite),
            mask,
        )


class CrossModalityCompletionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.clinical_self_norm = nn.LayerNorm(dim)
        self.clinical_self_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.clinical_cross_norm = nn.LayerNorm(dim)
        self.image_cross_norm = nn.LayerNorm(dim)
        self.clinical_from_image = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_from_clinical = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.clinical_ffn_norm = nn.LayerNorm(dim)
        self.image_ffn_norm = nn.LayerNorm(dim)
        self.clinical_ffn = FeedForward(dim, dropout)
        self.image_ffn = FeedForward(dim, dropout)

    def forward(
        self,
        image_tokens: torch.Tensor,
        clinical_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, depth, spatial, channels = image_tokens.shape
        image_flat = image_tokens.reshape(batch, depth * spatial, channels)

        normalized_clinical = self.clinical_self_norm(clinical_tokens)
        clinical_self, _ = self.clinical_self_attention(
            normalized_clinical,
            normalized_clinical,
            normalized_clinical,
            need_weights=False,
        )
        clinical_tokens = clinical_tokens + clinical_self

        clinical_query = self.clinical_cross_norm(clinical_tokens)
        image_key_value = self.image_cross_norm(image_flat)
        clinical_cross, _ = self.clinical_from_image(
            clinical_query,
            image_key_value,
            image_key_value,
            need_weights=False,
        )
        image_cross, _ = self.image_from_clinical(
            image_key_value,
            clinical_query,
            clinical_query,
            need_weights=False,
        )
        clinical_tokens = clinical_tokens + clinical_cross
        image_flat = image_flat + image_cross
        clinical_tokens = clinical_tokens + self.clinical_ffn(
            self.clinical_ffn_norm(clinical_tokens)
        )
        image_flat = image_flat + self.image_ffn(self.image_ffn_norm(image_flat))
        return (
            image_flat.reshape(batch, depth, spatial, channels),
            clinical_tokens,
        )


class CMCSurvMTLR(nn.Module):
    def __init__(
        self,
        image_shape: Sequence[int],
        categorical_cardinalities: Sequence[int],
        composite_cardinalities: Sequence[int],
        hidden_dim: int,
        num_bins: int,
        dropout: float,
        token_dim: int = 128,
        heads: int = 4,
        encoder_depth: int = 2,
        completion_depth: int = 2,
        patch_size: Sequence[int] = (4, 16, 16),
    ) -> None:
        super().__init__()
        self.visual_encoder = VisualSliceDepthEncoder(
            image_shape,
            patch_size,
            token_dim,
            heads,
            encoder_depth,
            dropout,
        )
        self.clinical_encoder = ClinicalGraphEncoder(
            categorical_cardinalities,
            composite_cardinalities,
            token_dim,
            encoder_depth,
            dropout,
        )
        self.completion_blocks = nn.ModuleList(
            [
                CrossModalityCompletionBlock(token_dim, heads, dropout)
                for _ in range(completion_depth)
            ]
        )
        self.image_decoder = nn.Linear(
            token_dim,
            self.visual_encoder.patch_dim,
        )
        self.continuous_decoders = nn.ModuleList(
            [nn.Linear(token_dim, 1) for _ in range(4)]
        )
        all_cardinalities = tuple(int(v) for v in categorical_cardinalities) + tuple(
            int(v) for v in composite_cardinalities
        )
        self.categorical_decoders = nn.ModuleList(
            [nn.Linear(token_dim, cardinality) for cardinality in all_cardinalities]
        )
        self.os_head = MTLRHead(2 * token_dim, hidden_dim, num_bins, dropout)
        self.pfs_head = MTLRHead(2 * token_dim, hidden_dim, num_bins, dropout)
        self.features_frozen = False

    def feature_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.visual_encoder,
            self.clinical_encoder,
            self.completion_blocks,
        )

    def pretraining_modules(self) -> tuple[nn.Module, ...]:
        return self.feature_modules() + (
            self.image_decoder,
            self.continuous_decoders,
            self.categorical_decoders,
        )

    def set_pretraining_stage(self) -> None:
        self.features_frozen = False
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in self.pretraining_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def set_survival_stage(self) -> None:
        self.features_frozen = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for head in (self.os_head, self.pfs_head):
            for parameter in head.parameters():
                parameter.requires_grad_(True)
        for module in self.feature_modules():
            module.eval()

    def survival_train_mode(self) -> None:
        for module in self.feature_modules():
            module.eval()
        self.os_head.train()
        self.pfs_head.train()

    def pretraining_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def shared_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for module in self.feature_modules()
            for parameter in module.parameters()
        ]

    @staticmethod
    def _random_mask(shape: Sequence[int], ratio: float, device: torch.device) -> torch.Tensor:
        if not 0.0 < ratio < 1.0:
            raise ValueError("CMC mask ratio must be strictly between zero and one.")
        batch = int(shape[0])
        count = int(torch.tensor(shape[1:]).prod().item())
        masked_count = max(1, min(count - 1, round(count * ratio)))
        noise = torch.rand(batch, count, device=device)
        indices = noise.argsort(dim=1)[:, :masked_count]
        mask = torch.zeros(batch, count, dtype=torch.bool, device=device)
        mask.scatter_(1, indices, True)
        return mask.view(*shape)

    def _complete(
        self,
        image_tokens: torch.Tensor,
        clinical_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.completion_blocks:
            image_tokens, clinical_tokens = block(image_tokens, clinical_tokens)
        return image_tokens, clinical_tokens

    def _encode_intact(
        self,
        image: torch.Tensor,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tokens = self.visual_encoder(image)
        clinical_tokens = self.clinical_encoder(
            continuous,
            categorical,
            composite,
        )
        return self._complete(image_tokens, clinical_tokens)

    def pretraining_forward(
        self,
        image: torch.Tensor,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
        mask_ratio: float,
    ) -> dict[str, Any]:
        image_embedded = self.visual_encoder.embed(image)
        clinical_embedded = self.clinical_encoder.embed(
            continuous,
            categorical,
            composite,
        )
        image_mask = self._random_mask(
            image_embedded.shape[:3],
            mask_ratio,
            image.device,
        )
        clinical_mask = self._random_mask(
            clinical_embedded.shape[:2],
            mask_ratio,
            image.device,
        )
        intact_image = self.visual_encoder.encode(image_embedded)
        intact_clinical = self.clinical_encoder.encode(clinical_embedded)
        masked_image = self.visual_encoder.encode(image_embedded, image_mask)
        masked_clinical = self.clinical_encoder.encode(
            clinical_embedded,
            clinical_mask,
        )
        completed_image, _ = self._complete(masked_image, intact_clinical)
        _, completed_clinical = self._complete(intact_image, masked_clinical)
        completed_variables = completed_clinical[:, 1:]
        return {
            "image_prediction": self.image_decoder(completed_image),
            "image_target": self.visual_encoder.patchify(image),
            "image_mask": image_mask,
            "continuous_predictions": [
                decoder(completed_variables[:, index]).squeeze(-1)
                for index, decoder in enumerate(self.continuous_decoders)
            ],
            "categorical_predictions": [
                decoder(completed_variables[:, 4 + index])
                for index, decoder in enumerate(self.categorical_decoders)
            ],
            "clinical_mask": clinical_mask,
            "continuous_target": continuous,
            "categorical_target": torch.cat([categorical, composite], dim=1),
        }

    def pretraining_loss(
        self,
        outputs: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_mask = outputs["image_mask"]
        image_loss = F.mse_loss(
            outputs["image_prediction"][image_mask],
            outputs["image_target"][image_mask],
        )
        clinical_mask = outputs["clinical_mask"]
        continuous_target = outputs["continuous_target"].float()
        categorical_target = outputs["categorical_target"].long()
        continuous_losses = []
        for index, prediction in enumerate(outputs["continuous_predictions"]):
            selected = clinical_mask[:, index]
            if bool(selected.any()):
                continuous_losses.append(
                    F.mse_loss(prediction[selected], continuous_target[selected, index])
                )
        categorical_losses = []
        for index, prediction in enumerate(outputs["categorical_predictions"]):
            variable_index = 4 + index
            selected = clinical_mask[:, variable_index]
            if bool(selected.any()):
                categorical_losses.append(
                    F.cross_entropy(
                        prediction[selected],
                        categorical_target[selected, index],
                    )
                )
        zero = image_loss.new_zeros(())
        continuous_loss = (
            torch.stack(continuous_losses).mean() if continuous_losses else zero
        )
        categorical_loss = (
            torch.stack(categorical_losses).mean() if categorical_losses else zero
        )
        clinical_loss = continuous_loss + categorical_loss
        total = image_loss + clinical_loss
        return total, {
            "image_reconstruction": image_loss,
            "clinical_continuous_reconstruction": continuous_loss,
            "clinical_categorical_reconstruction": categorical_loss,
        }

    def survival_forward(
        self,
        image: torch.Tensor,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        composite: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.features_frozen:
            with torch.no_grad():
                image_tokens, clinical_tokens = self._encode_intact(
                    image,
                    continuous,
                    categorical,
                    composite,
                )
        else:
            image_tokens, clinical_tokens = self._encode_intact(
                image,
                continuous,
                categorical,
                composite,
            )
        image_embedding = image_tokens.mean(dim=(1, 2))
        clinical_embedding = clinical_tokens[:, 0]
        embedding = torch.cat([image_embedding, clinical_embedding], dim=1)
        return survival_outputs(self.os_head, self.pfs_head, embedding)

    def forward(
        self,
        image: torch.Tensor,
        clinical_continuous: torch.Tensor,
        clinical_categorical: torch.Tensor,
        clinical_composite: torch.Tensor,
        stage: str = "survival",
        mask_ratio: float = 0.5,
    ) -> dict[str, Any]:
        if stage == "pretrain":
            reconstruction = self.pretraining_forward(
                image,
                clinical_continuous,
                clinical_categorical,
                clinical_composite,
                mask_ratio,
            )
            loss, components = self.pretraining_loss(reconstruction)
            return {"loss": loss, **components}
        if stage != "survival":
            raise ValueError(f"Unsupported CMC stage: {stage}")
        return self.survival_forward(
            image,
            clinical_continuous,
            clinical_categorical,
            clinical_composite,
        )


def survival_outputs(
    os_head: MTLRHead,
    pfs_head: MTLRHead,
    embedding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    os_logits = os_head(embedding)
    pfs_logits = pfs_head(embedding.detach())
    os_survival = os_head.survival(os_logits)
    pfs_survival = pfs_head.survival(pfs_logits)
    return {
        "embedding": embedding,
        "os_logits": os_logits,
        "pfs_logits": pfs_logits,
        "os_survival": os_survival,
        "pfs_survival": pfs_survival,
        "os_risk": os_head.risk(os_logits),
        "pfs_risk": pfs_head.risk(pfs_logits),
    }
