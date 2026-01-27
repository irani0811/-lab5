# -*- coding: utf-8 -*-
"""
模型结构定义。
文本编码采用预训练 Transformer，图像编码可选 ResNet18/ResNet50 并做线性投影，
支持 Concat/GMU/跨模态注意力/双向协同注意力等多种融合策略，并内置模态 Dropout。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel


class MultiModalSentimentModel(nn.Module):
    """多模态情感模型，支持多种骨干与融合策略。"""

    def __init__(
        self,
        text_model_name: str,
        image_embed_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
        freeze_text: bool,
        freeze_image: bool,
        num_labels: int,
        fusion_method: str = "concat",
        image_backbone: str = "resnet18",
        text_train_layers: int = -1,
        cross_attn_heads: int = 4,
        modality_dropout_prob: float = 0.0,
        ablate_text: bool = False,
        ablate_image: bool = False,
    ) -> None:
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_model.config.hidden_size
        if freeze_text:
            for param in self.text_model.parameters():
                param.requires_grad = False
        elif text_train_layers > 0:
            self._set_text_trainable_layers(text_train_layers)

        resnet = self._build_image_backbone(image_backbone)
        backbone_layers = list(resnet.children())[:-2]
        self.image_backbone = nn.Sequential(*backbone_layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        if freeze_image:
            for param in self.image_backbone.parameters():
                param.requires_grad = False

        self.image_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, image_embed_dim),
            nn.LayerNorm(image_embed_dim),
            nn.GELU(),
        )
        self.backbone_feat_dim = resnet.fc.in_features
        self.modality_dropout_prob = modality_dropout_prob
        self.ablate_text = bool(ablate_text)
        self.ablate_image = bool(ablate_image)
        self.text_contrastive_proj = nn.Sequential(
            nn.Linear(text_dim, image_embed_dim),
            nn.LayerNorm(image_embed_dim),
        )
        self.image_contrastive_proj = nn.Sequential(
            nn.Linear(image_embed_dim, image_embed_dim),
            nn.LayerNorm(image_embed_dim),
        )

        self.fusion_method = fusion_method
        if self.fusion_method == "gmu":
            # Gated Multimodal Unit，参考 Arevalo et al. (2017)
            self.text_fuse = nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, fusion_hidden_dim),
                nn.GELU(),
            )
            self.image_fuse = nn.Sequential(
                nn.LayerNorm(image_embed_dim),
                nn.Linear(image_embed_dim, fusion_hidden_dim),
                nn.GELU(),
            )
            self.gate_layer = nn.Sequential(
                nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
                nn.Sigmoid(),
            )
            classifier_input_dim = fusion_hidden_dim
        elif self.fusion_method == "cross_attn":
            self.image_token_proj = nn.Linear(self.backbone_feat_dim, text_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=cross_attn_heads, batch_first=True, dropout=dropout
            )
            self.cross_norm = nn.LayerNorm(text_dim + image_embed_dim)
            classifier_input_dim = text_dim + image_embed_dim
        elif self.fusion_method == "co_attn":
            self.image_token_proj = nn.Linear(self.backbone_feat_dim, text_dim)
            self.text_to_image_attn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=cross_attn_heads, batch_first=True, dropout=dropout
            )
            self.image_to_text_attn = nn.MultiheadAttention(
                embed_dim=text_dim, num_heads=cross_attn_heads, batch_first=True, dropout=dropout
            )
            self.co_attn_norm = nn.LayerNorm(text_dim * 2 + image_embed_dim)
            classifier_input_dim = text_dim * 2 + image_embed_dim
        else:
            fusion_dim = text_dim + image_embed_dim
            self.concat_norm = nn.LayerNorm(fusion_dim)
            classifier_input_dim = fusion_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_features: bool = False,
    ):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = text_outputs.last_hidden_state
        text_feat = text_hidden[:, 0, :]
        image_features = self.image_backbone(pixel_values)
        text_hidden, text_feat, image_features = self._apply_modality_dropout(text_hidden, text_feat, image_features)

        if self.ablate_text:
            text_hidden = torch.zeros_like(text_hidden)
            text_feat = torch.zeros_like(text_feat)
        if self.ablate_image:
            image_features = torch.zeros_like(image_features)

        pooled = self.global_pool(image_features)
        img_feat = self.image_projection(pooled)
        text_cl = self.text_contrastive_proj(text_feat)
        image_cl = self.image_contrastive_proj(img_feat)
        if self.fusion_method == "gmu":
            t = self.text_fuse(text_feat)
            v = self.image_fuse(img_feat)
            gate = self.gate_layer(torch.cat([t, v], dim=1))
            fused = gate * t + (1 - gate) * v
        elif self.fusion_method == "cross_attn":
            img_tokens = image_features.flatten(2).transpose(1, 2)
            img_tokens = self.image_token_proj(img_tokens)
            cls_query = text_hidden[:, :1, :]
            attn_out, _ = self.cross_attn(cls_query, img_tokens, img_tokens)
            cross_text = attn_out.squeeze(1)
            fused = self.cross_norm(torch.cat([cross_text, img_feat], dim=1))
        elif self.fusion_method == "co_attn":
            img_tokens = image_features.flatten(2).transpose(1, 2)
            img_tokens = self.image_token_proj(img_tokens)
            # 文本从视觉中聚合上下文
            text_context, _ = self.text_to_image_attn(text_hidden, img_tokens, img_tokens)
            # 视觉从文本中聚合线索
            image_context, _ = self.image_to_text_attn(img_tokens, text_hidden, text_hidden)
            text_cls = text_context[:, 0, :]
            image_summary = image_context.mean(dim=1)
            fused = self.co_attn_norm(torch.cat([text_cls, image_summary, img_feat], dim=1))
        else:
            fused = self.concat_norm(torch.cat([text_feat, img_feat], dim=1))
        logits = self.classifier(fused)
        if return_features:
            return logits, text_cl, image_cl
        return logits

    def _set_text_trainable_layers(self, num_layers: int) -> None:
        """仅解冻 Transformer 编码器的最后 N 层，减少训练开销。"""
        encoder = getattr(self.text_model, "encoder", None)
        if encoder is None or not hasattr(encoder, "layer"):
            return
        for param in self.text_model.parameters():
            param.requires_grad = False
        layers = encoder.layer
        total = len(layers)
        start = max(total - num_layers, 0)
        for layer in layers[start:]:
            for param in layer.parameters():
                param.requires_grad = True
        pooler = getattr(self.text_model, "pooler", None)
        if pooler is not None:
            for param in pooler.parameters():
                param.requires_grad = True

    def _build_image_backbone(self, name: str):
        name = name.lower()
        if name == "resnet50":
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if name != "resnet18":
            raise ValueError(f"不支持的图像骨干：{name}")
        return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    def _apply_modality_dropout(
        self,
        text_hidden: torch.Tensor,
        text_feat: torch.Tensor,
        image_features: torch.Tensor,
    ):
        if not self.training or self.modality_dropout_prob <= 0:
            return text_hidden, text_feat, image_features
        batch_size = text_hidden.size(0)
        device = text_hidden.device
        drop_text = torch.rand(batch_size, device=device) < self.modality_dropout_prob
        drop_image = torch.rand(batch_size, device=device) < self.modality_dropout_prob
        both = drop_text & drop_image
        drop_text = drop_text & ~both
        drop_image = drop_image & ~both

        text_mask = (~drop_text).float().view(batch_size, 1, 1)
        image_mask = (~drop_image).float().view(batch_size, 1, 1, 1)

        text_hidden = text_hidden * text_mask
        text_feat = text_feat * text_mask.view(batch_size, 1)
        image_features = image_features * image_mask
        return text_hidden, text_feat, image_features
