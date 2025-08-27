
from typing import List, Tuple, Callable

import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import torch
import clip
from data_utils import  gui_transform, GUIDataset
from utils import extract_index_features, device, AttentionGuidedGatedFusion, collate_fn


def compute_gui_val_metrics(dataset: GUIDataset, clip_model: nn.Module,
                            combining_function: Callable,  # 改为 Callable 类型
                            index_features: torch.Tensor,
                            index_names: List[str],  # 确保这是列表
                            batch_size: int = 32) -> Tuple[float, float, float, float, float]:
    """计算GUI验证集的召回率、MRR和准确率"""
    clip_model.eval()

    # 添加类型检查
    if not isinstance(index_names, list):
        print(f"警告: index_names 类型为 {type(index_names)}，应为列表")
        # 尝试转换为列表
        try:
            index_names = list(index_names)
        except TypeError:
            raise TypeError(f"无法将 index_names 转换为列表: {type(index_names)}")

    recall_at10 = 0.0
    recall_at50 = 0.0
    mrr = 0.0
    precision_at10 = 0.0
    precision_at50 = 0.0
    total_samples = 0

    # 创建索引名称到特征的映射
    name_to_feature = {name: feat for name, feat in zip(index_names, index_features)}

    # 创建数据加载器
    val_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,

    )

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            if batch is None:
                continue

            # 现在返回五元组：参考图像、目标图像、描述、参考名称、目标名称
            reference_images, target_images, captions, _, target_names = batch

            reference_images = reference_images.to(device, non_blocking=True)
            text_inputs = clip.tokenize(captions, truncate=True).to(device)

            # 提取特征
            reference_features = clip_model.encode_image(reference_images)
            caption_features = clip_model.encode_text(text_inputs)

            # 使用融合模块
            query_features = combining_function(reference_features, caption_features)
            query_features = F.normalize(query_features, dim=-1)

            # 计算相似度
            sim_matrix = query_features @ index_features.T
            sim_matrix *= clip_model.logit_scale.exp().item()

            # 获取目标索引
            target_indices = []
            for target_name in target_names:  # 使用传入的目标名称
                if target_name in name_to_feature:
                    target_indices.append(index_names.index(target_name))
                else:
                    # 处理缺失的目标
                    target_indices.append(-1)

            # 计算指标
            for i, target_idx in enumerate(target_indices):
                if target_idx == -1:
                    continue  # 跳过缺失目标

                total_samples += 1

                # 获取当前查询的相似度分数
                scores = sim_matrix[i].cpu().numpy()
                sorted_indices = np.argsort(scores)[::-1]  # 从高到低排序

                # 计算Recall@k
                if target_idx in sorted_indices[:10]:
                    recall_at10 += 1
                    precision_at10 += 1 if scores[sorted_indices[0]] > 0.5 else 0
                if target_idx in sorted_indices[:50]:
                    recall_at50 += 1
                    precision_at50 += 1 if scores[sorted_indices[0]] > 0.5 else 0

                # 计算MRR
                rank = np.where(sorted_indices == target_idx)[0][0] + 1
                mrr += 1 / rank

    # 计算平均指标
    recall_at10 /= total_samples
    recall_at50 /= total_samples
    mrr /= total_samples
    precision_at10 /= total_samples
    precision_at50 /= total_samples

    return recall_at10, recall_at50, mrr, precision_at10, precision_at50


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be 'GUI'")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be 'cross_attention'")
    parser.add_argument("--combiner-path", type=Path, help="path to trained combiner model")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--transform", default="guipad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'guipad'] ")

    args = parser.parse_args()

    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution

    if args.clip_model_path:
        print('加载CLIP模型')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["clip_model_state_dict"])
        print('CLIP模型加载成功')

    if args.transform == 'guipad':
        print('使用GUI自适应预处理')
        preprocess = gui_transform(input_dim)
    else:
        print('使用CLIP默认预处理')
        preprocess = clip_preprocess

    if args.combining_function == 'AttentionGuidedGatedFusion':
        feature_dim = clip_model.text_projection.shape[1]
        combining_function = AttentionGuidedGatedFusion(feature_dim).to(device)
        if args.combiner_path:
            print("加载训练好的融合模块")
            state_dict = torch.load(args.combiner_path, map_location=device)
            combining_function.load_state_dict(state_dict["combiner_state_dict"])
    else:
        raise ValueError("combiner_path should be 'AttentionGuidedGatedFusion'")

    if args.dataset.lower() == 'gui':
        # 创建统一验证集
        classic_val_dataset = GUIDataset('val',
                                         ['bare', 'form', 'gallery', 'login', 'news', 'profile', 'terms', 'search'],
                                         'classic', preprocess, input_dim)
        index_features, index_names = extract_index_features(classic_val_dataset, clip_model)

        relative_val_dataset = GUIDataset('val',
                                          ['bare', 'form', 'gallery', 'login', 'news', 'profile', 'terms', 'search'],
                                          'relative', preprocess, input_dim)

        # 计算整体验证指标
        recallat10, recallat50, mrr, precisionat10, precisionat50 = compute_gui_val_metrics(
            relative_val_dataset, clip_model, index_features, index_names, combining_function
        )

        print(f"Overall recall@10 = {recallat10:.4f}")
        print(f"Overall recall@50 = {recallat50:.4f}")
        print(f"Overall mrr = {mrr:.4f}")
        print(f"Overall precision@10 = {precisionat10:.4f}")
        print(f"Overall precision@50 = {precisionat50:.4f}")

    else:
        raise ValueError("Dataset should be 'GUI'")


if __name__ == '__main__':
    main()