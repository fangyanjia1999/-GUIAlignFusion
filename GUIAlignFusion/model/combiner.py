import torch
import torch.nn as nn
import torch.nn.functional as F

#bare_recallat10 = 0.0905
#bare_recallat50 = 0.3850
#form_recallat10 = 0.1064
#form_recallat50 = 0.3802
#gallery_recallat10 = 0.1564
#gallery_recallat50 = 0.5382
#list_recallat10 = 0.0637
#list_recallat50 = 0.1639
#login_recallat10 = 0.1906
#login_recallat50 = 0.9683
#news_recallat10 = 0.1446
#news_recallat50 = 0.6905
#profile_recallat10 = 0.1259
#profile_recallat50 = 0.6209
#search_recallat10 = 0.2287
#search_recallat50 = 1
#settings_recallat10 = 0.0840
#settings_recallat50 = 0.3355
#terms_recallat10 = 0.3407
#terms_recallat50 = 1
#Average recall@10 across GUI types = 0.1532
#Average recall@50 across GUI types = 0.6083
#/home/CLIP4/CLIP4Cir/combiner_training1/RN50x4_2025-07-03_23:51:16/saved_models/combiner3_combiner_best_epoch49_recall0.3792.pt

class Combiner(nn.Module):
    """新的特征融合模块"""

    def __init__(self, feature_dim: int, projection_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim

        # 文本特征投影
        self.text_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(projection_dim)
        )

        # 图像特征投影
        self.image_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(projection_dim)
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # 残差连接适配层
        self.residual_adapter = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(projection_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def combine_features(self, reference_image_features, text_features):
        # 确保输入是浮点类型
        reference_image_features = F.normalize(reference_image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)

        # 投影特征
        proj_text = self.text_projection(text_features)
        proj_ref = self.image_projection(reference_image_features)

        # 融合
        combined = torch.cat([proj_ref, proj_text], dim=1)
        fused_features = self.fusion_layer(combined)

        # 残差连接
        adapted_ref = self.residual_adapter(reference_image_features)
        combined_features = fused_features + adapted_ref

        # 最终输出
        output_features = self.output_layer(combined_features)
        return F.normalize(output_features, dim=-1)

    def forward(self, reference_image_features, text_features, target_image_features=None):
        combined_features = self.combine_features(reference_image_features, text_features)

        if target_image_features is not None:
            # 计算相似度（仅在训练时需要）
            target_features = F.normalize(target_image_features.float(), dim=-1)
            logits = combined_features @ target_features.T
            return logits

        return combined_features
