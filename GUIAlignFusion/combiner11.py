
import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F
#/home/CLIP4/CLIP4Cir/enhanced_combiner_training/RN50x4_2025-07-04_08:01:25/saved_models/enhanced_combiner1_best_epoch49_recall0.4529.pt
# 定义增强版Combiner网络
class Attention(nn.Module):
    """优化后的注意力机制模块"""

    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_head) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], -1, self.num_heads, t.shape[-1] // self.num_heads).transpose(1, 2),
                      (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).reshape(x.shape[0], -1, x.shape[-1])
        return out


class EnhancedDFBlock(nn.Module):
    """优化后的深度融合块"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.3)
        self.gate_network = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.attention = Attention(dim, num_heads=num_heads, dim_head=64, dropout=0.3)

    def forward(self, text, img):
        batch_size = text.size(0)
        combined = torch.stack([text, img], dim=1)  # [B, 2, D]

        # 双向交叉注意力
        attn_output, _ = self.cross_attn(
            combined.view(-1, 1, combined.size(-1)),
            combined.view(-1, 1, combined.size(-1)),
            combined.view(-1, 1, combined.size(-1))
        )
        attn_output = attn_output.view(batch_size, 2, -1)  # [B, 2, D]

        # 残差连接
        text_enhanced = text + attn_output[:, 0]
        img_enhanced = img + attn_output[:, 1]

        # 动态门控融合
        gate = self.gate_network(torch.cat([text_enhanced, img_enhanced], dim=-1))
        fused = gate * text_enhanced + (1 - gate) * img_enhanced

        # 注意力机制
        fused = self.attention(fused.unsqueeze(1)).squeeze(1)

        out = self.proj(fused)
        return out


class EnhancedLocalCache(nn.Module):
    """优化后的局部特征缓存"""

    def __init__(self, num_entries=32, dim=2560):
        super().__init__()
        self.entries = nn.Parameter(torch.randn(num_entries, dim // 2))
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.proj = nn.Sequential(
            nn.Linear(dim // 2, dim),
            nn.GELU()
        )
        self.attention = Attention(dim, num_heads=8, dim_head=64, dropout=0.3)

    def forward(self, features):
        compressed = features[:, :features.size(-1) // 2]

        # 相似度计算
        sim = F.cosine_similarity(
            compressed.unsqueeze(1),
            self.entries.unsqueeze(0),
            dim=-1
        )
        weights = F.softmax(sim * self.temperature, dim=-1)

        # 加权投影
        cached = torch.matmul(weights, self.proj(self.entries))

        # 注意力机制
        cached = self.attention(cached.unsqueeze(1)).squeeze(1)

        out = cached + features
        return out


class EnhancedMultiScaleFusion(nn.Module):
    """优化后的多尺度融合"""

    def __init__(self, dim, scales=[16, 32]):
        super().__init__()
        self.scales = scales
        self.shared_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU()
        )
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool1d(s) for s in scales])
        self.fuse = nn.Sequential(
            nn.Linear(sum(scales), dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.attention = Attention(dim, num_heads=8, dim_head=64, dropout=0.3)

    def forward(self, x):
        projected = self.shared_proj(x)
        features = []

        for pool in self.pools:
            pooled = pool(projected.unsqueeze(1)).squeeze(1)
            features.append(pooled)

        fused = torch.cat(features, dim=-1)
        fused = self.fuse(fused) + x  # 残差连接

        # 注意力机制
        fused = self.attention(fused.unsqueeze(1)).squeeze(1)

        return fused

class Combiner(nn.Module):
    """优化后的检索组合网络"""

    def __init__(self, clip_feature_dim=640, projection_dim=2560, hidden_dim=5120):
        super().__init__()
        self.clip_dim = clip_feature_dim
        self.proj_dim = projection_dim
        self.hidden_dim = hidden_dim

        # 投影层
        self.text_projection = nn.Sequential(
            nn.Linear(clip_feature_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(0.3))
        self.image_projection = nn.Sequential(
            nn.Linear(clip_feature_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(0.3))

        # 融合模块
        self.df_block = EnhancedDFBlock(projection_dim)
        self.local_cache = EnhancedLocalCache(dim=projection_dim)
        self.multiscale = EnhancedMultiScaleFusion(projection_dim)

        # 动态权重生成
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(projection_dim // 4, 1),
            nn.Sigmoid()
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, clip_feature_dim)
        )

        self.logit_scale = 100

    def combine_features(self, image_features, text_features):
        text_proj = self.text_projection(text_features)
        img_proj = self.image_projection(image_features)

        # 深度融合
        fused = self.df_block(text_proj, img_proj)

        # 局部特征增强
        local_feat = self.local_cache(img_proj)

        # 多尺度融合
        multi_input = fused + local_feat
        multi_feat = self.multiscale(multi_input)

        # 动态融合
        combined = torch.cat([multi_feat, text_proj], dim=-1)
        dynamic_weight = self.dynamic_scalar(combined)

        # 残差连接
        output = (self.output_layer(combined) +
                  dynamic_weight * text_features +
                  (1 - dynamic_weight) * image_features)

        return F.normalize(output, dim=-1)

    def forward(self, reference_image_features, text_features, target_image_features):
        combined_features = self.combine_features(reference_image_features, text_features)
        normalized_target = F.normalize(target_image_features, dim=-1)
        return self.logit_scale * combined_features @ normalized_target.T