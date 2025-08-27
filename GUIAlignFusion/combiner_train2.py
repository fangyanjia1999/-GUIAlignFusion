import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import comet_ml
import clip
import torch
import torch.nn as nn
from combiner7 import Combiner
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# 假设这些模块在项目中存在
from data_utils import base_path, gui_transform, GUIDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description
from utils import extract_index_features, device



# 排序损失函数
def ranking_loss(logits: torch.Tensor, ground_truth: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """排序损失函数"""
    pos_logits = logits[torch.arange(logits.size(0)), ground_truth]

    # 创建掩码排除正样本
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(logits.size(0)), ground_truth] = False
    neg_logits = logits[mask].view(logits.size(0), -1)

    # 获取最难负样本
    hardest_neg, _ = torch.max(neg_logits, dim=1)

    # 计算margin-based损失
    loss = torch.clamp(margin - pos_logits + hardest_neg, min=0)
    return torch.mean(loss)


# 保存模型函数 - 添加错误处理和优化
def save_model(name: str, cur_epoch: int, clip_model: nn.Module, combining_function: nn.Module,
               training_path: Path, is_combiner: bool = False):
    """保存模型"""
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)

    # 根据是否是Combiner模型调整保存名称
    if is_combiner:
        model_path = models_path / f'enhanced_combiner1_{name}.pt'
    else:
        model_path = models_path / f'{name}.pt'

    # 优化保存：使用更高效的方式保存模型
    try:
        # 先尝试保存到临时文件，再重命名，避免写入中断导致文件损坏
        temp_path = model_path.with_suffix('.tmp')

        # 只保存必要的状态字典，而不是整个模型
        state_dict = {
            'epoch': cur_epoch,
            'clip_model_state_dict': clip_model.state_dict(),
            'combiner_state_dict': combining_function.state_dict()
        }

        # 使用高效保存方法
        torch.save(state_dict, str(temp_path), _use_new_zipfile_serialization=True)

        # 如果保存成功，重命名文件
        temp_path.rename(model_path)

        print(f"模型已保存至: {model_path}")
        return model_path
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")

        # 尝试使用更基本的保存方法
        try:
            print("尝试使用基本保存方法...")
            torch.save({
                'epoch': cur_epoch,
                'clip_model_state_dict': clip_model.state_dict(),
                'combiner_state_dict': combining_function.state_dict()
            }, str(model_path))
            print(f"模型已保存至: {model_path}")
            return model_path
        except Exception as e2:
            print(f"基本保存方法也失败: {str(e2)}")

            # 尝试保存到另一个位置
            try:
                alt_path = base_path / f"backup_models/{model_path.name}"
                alt_path.parent.mkdir(exist_ok=True, parents=True)
                print(f"尝试保存到备用位置: {alt_path}")

                torch.save({
                    'epoch': cur_epoch,
                    'clip_model_state_dict': clip_model.state_dict(),
                    'combiner_state_dict': combining_function.state_dict()
                }, str(alt_path))

                print(f"模型已保存至备用位置: {alt_path}")
                return alt_path
            except Exception as e3:
                print(f"所有保存尝试均失败: {str(e3)}")
                return None


def compute_gui_val_metrics(
        dataset: GUIDataset,
        clip_model: nn.Module,
        combining_function: nn.Module,
        index_features: torch.Tensor,
        index_names: List[str]
) -> Tuple[float, float, float, float, float]:
    """
    计算验证指标
    :param dataset: 验证数据集
    :param clip_model: CLIP模型
    :param combining_function: Combiner模型
    :param index_features: 索引特征矩阵
    :param index_names: 索引特征对应的图像名称
    :return: (recall@10, recall@50, mrr, precision@10, precision@50)
    """
    # 准备数据加载器
    val_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False
    )

    # 初始化指标
    recall_at10 = 0.0
    recall_at50 = 0.0
    mrr = 0.0
    precision_at10 = 0.0
    precision_at50 = 0.0
    total_samples = 0

    # 创建从图像名称到索引的映射
    name_to_idx = {name: idx for idx, name in enumerate(index_names)}

    # 进度条
    val_bar = tqdm(val_loader, desc="验证中", ncols=150)

    for batch in val_bar:
        if batch is None:
            continue

        reference_images, target_images, captions, _, target_paths = batch
        batch_size = reference_images.size(0)
        total_samples += batch_size

        # 移动到设备
        reference_images = reference_images.to(device, non_blocking=True)
        text_inputs = clip.tokenize(captions, truncate=True).to(device)

        # 提取特征 - CLIP模型不计算梯度
        with torch.no_grad():
            reference_features = clip_model.encode_image(reference_images)
            caption_features = clip_model.encode_text(text_inputs)

        # 使用combine_features方法获取查询特征
        query_features = combining_function.combine_features(
            reference_features,
            caption_features
        )

        # 计算相似度
        logits = combining_function.logit_scale * query_features @ index_features.T

        # 动态确定最大k值（不超过索引特征数量）
        max_possible_k = logits.size(1)
        k_val = min(50, max_possible_k)  # 确保k值不超过索引特征数量

        # 获取topk结果
        _, topk_indices = torch.topk(logits, k=k_val, dim=1)

        # 获取目标索引
        target_indices = []
        for path in target_paths:
            if path in name_to_idx:
                target_indices.append(name_to_idx[path])
            else:
                # 如果找不到，使用随机索引（这种情况不应该发生）
                target_indices.append(np.random.randint(0, len(index_names)))

        target_indices = torch.tensor(target_indices, dtype=torch.long, device=device)

        # 计算指标
        for i in range(batch_size):
            # 找到正确结果的排名（从1开始）
            rank = (topk_indices[i] == target_indices[i]).nonzero(as_tuple=True)[0]
            if rank.numel() > 0:
                rank = rank.item() + 1  # 转换为1-based索引

                # 更新MRR
                mrr += 1.0 / rank

                # 更新召回率
                if rank <= min(10, k_val):  # 确保不超过实际k_val
                    recall_at10 += 1
                if rank <= k_val:  # 确保不超过实际k_val
                    recall_at50 += 1

                # 更新精确率（分母使用实际的k_val）
                if rank <= min(10, k_val):
                    precision_at10 += 1
                if rank <= k_val:
                    precision_at50 += 1

    # 归一化指标
    recall_at10 /= total_samples
    recall_at50 /= total_samples
    mrr /= total_samples
    precision_at10 /= total_samples * min(10, k_val)  # 分母使用min(10, k_val)
    precision_at50 /= total_samples * k_val  # 分母使用实际的k_val

    return recall_at10, recall_at50, mrr, precision_at10, precision_at50


# 增强版Combiner训练函数
def train_enhanced_combiner(
        train_gui_types: List[str],
        val_gui_types: List[str],
        num_epochs: int,
        clip_model_name: str,
        pretrained_path: str,
        batch_size: int = 32,
        validation_frequency: int = 1,
        transform: str = "guipad",
        save_training: bool = True,
        save_best: bool = True,
        margin: float = 0.2,
        patience: int = 10,
        learning_rate: float = 3e-5,
        experiment: Optional[comet_ml.Experiment] = None,
        **kwargs
):
    """训练增强版Combiner网络"""
    # === 初始化训练环境 ===
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path = Path(base_path / f"enhanced_combiner_training/{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=True, parents=True)

    print(f"增强版Combiner训练开始时间: {training_start}")
    print(f"训练保存路径: {training_path}")
    print(f"预训练模型路径: {pretrained_path}")

    # 保存超参数
    training_hyper_params = {
        "num_epochs": num_epochs,
        "clip_model_name": clip_model_name,
        "pretrained_path": str(pretrained_path),
        "batch_size": batch_size,
        "validation_frequency": validation_frequency,
        "transform": transform,
        "save_training": save_training,
        "save_best": save_best,
        "margin": margin,
        "patience": patience,
        "learning_rate": learning_rate,
        "train_gui_types": train_gui_types,
        "val_gui_types": val_gui_types
    }

    print("\n增强版Combiner训练超参数:")
    for key, value in training_hyper_params.items():
        print(f"  {key}: {value}")

    with open(training_path / "enhanced_combiner_training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # === 模型加载与初始化 ===
    print(f"\n加载基础CLIP模型: {clip_model_name}")
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().to(device)

    # 加载预训练模型
    print(f"加载预训练模型: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)

    # 兼容处理：移除可能存在的模块前缀
    clip_model_state_dict = checkpoint['clip_model_state_dict']
    clip_model_state_dict = {k.replace('module.', ''): v for k, v in clip_model_state_dict.items()}

    # 加载状态字典
    clip_model.load_state_dict(clip_model_state_dict)
    print("预训练CLIP模型加载成功!")

    # 冻结所有CLIP参数
    for param in clip_model.parameters():
        param.requires_grad = False
    print("CLIP模型参数已冻结")

    # 获取特征维度并初始化增强版Combiner
    feature_dim = clip_model.text_projection.shape[1]

    # 增强版Combiner参数
    projection_dim = feature_dim * 4  # 更大的投影维度
    hidden_dim = feature_dim * 8  # 更大的隐藏层维度

    print(f"\n初始化增强版Combiner模块:")
    print(f"  输入特征维度: {feature_dim}")
    print(f"  投影维度: {projection_dim}")
    print(f"  隐藏层维度: {hidden_dim}")

    combining_function = Combiner(
        clip_feature_dim=feature_dim,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim
    ).to(device)

    # 打印Combiner参数量
    combiner_params = sum(p.numel() for p in combining_function.parameters() if p.requires_grad)
    print(f"增强版Combiner可训练参数: {combiner_params:,}")

    # === 数据预处理 ===
    input_dim = clip_model.visual.input_resolution
    print(f"输入分辨率: {input_dim}x{input_dim}")

    if transform == "clip":
        preprocess = clip_preprocess
        print("使用CLIP标准预处理")
    elif transform == "guipad":
        preprocess = gui_transform(input_dim)
        print("使用GUI自适应预处理")
    else:
        raise ValueError("transform参数应为['clip', 'guipad']")

    # === 数据集准备 ===
    print("\n准备数据集:")
    print(f"  训练GUI类型: {', '.join(train_gui_types)}")
    print(f"  验证GUI类型: {', '.join(val_gui_types)}")

    relative_train_dataset = GUIDataset('train', train_gui_types, 'relative', preprocess, input_dim)
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True
    )

    print(f"  训练集大小: {len(relative_train_dataset)} 样本")
    print(f"  训练批次: {len(relative_train_loader)} (批次大小={batch_size})")

    # 验证数据集
    classic_val_dataset = GUIDataset('val', val_gui_types, 'classic', preprocess, input_dim)
    relative_val_dataset = GUIDataset('val', val_gui_types, 'relative', preprocess, input_dim)
    print(f"  验证集大小: {len(relative_val_dataset)} 样本")

    # 为每个GUI类型创建单独的数据集
    gui_type_val_datasets = {}
    for gui_type in val_gui_types:
        gui_classic = GUIDataset('val', [gui_type], 'classic', preprocess, input_dim)
        gui_relative = GUIDataset('val', [gui_type], 'relative', preprocess, input_dim)

        # 检查验证集大小是否足够
        if len(gui_classic) < 10:
            print(f"警告：验证集 '{gui_type}' 太小 ({len(gui_classic)} 样本)，跳过验证")
            continue

        gui_type_val_datasets[gui_type] = (gui_classic, gui_relative)
        print(f"    {gui_type}验证集: {len(gui_relative)} 样本")

    # === 训练配置 ===
    # 优化器使用AdamW，带权重衰减
    optimizer = optim.AdamW(
        combining_function.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )

    # 学习率调度器 - 使用余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01
    )

    scaler = GradScaler()

    # 早停与监控相关变量
    best_avg_recall = 0.0  # 最佳平均召回率
    best_epoch = -1  # 最佳模型所在轮次
    early_stop_triggered = False  # 是否触发早停
    early_stop_count = 0  # 连续无改进轮次计数

    # 平滑召回率相关
    smoothed_recall = 0.0  # 平滑后的召回率
    smoothing_factor = 0.9  # 平滑因子

    # 收敛阈值相关
    min_improvement = 0.0005  # 最小改进阈值 (0.05%)
    min_improvement_decay = 0.95  # 每10轮衰减改进阈值

    # 训练日志
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()
    category_metrics_log = pd.DataFrame()

    # 存储最后一次验证的指标
    last_overall_metrics = None
    last_category_metrics = None

    # 模型保存路径跟踪
    best_model_path = None

    # === 训练循环 ===
    print('\n开始增强版Combiner训练循环')
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        print(f"\n=== 轮次 {epoch + 1}/{num_epochs} ===")

        # 设置模型模式
        clip_model.eval()
        combining_function.train()
        print("训练模式: CLIP冻结 + Combiner训练")

        # === 训练步骤 ===
        train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
        train_bar = tqdm(relative_train_loader, ncols=150, desc=f"训练轮次 {epoch + 1}")

        for idx, batch in enumerate(train_bar):
            if batch is None:
                continue

            reference_images, target_images, captions, _, _ = batch

            # 数据准备
            images_in_batch = reference_images.size(0)
            step = len(train_bar) * epoch + idx

            reference_images = reference_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            text_inputs = clip.tokenize(captions, truncate=True).to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 混合精度训练
            with autocast():
                # 提取特征 - CLIP模型不计算梯度
                with torch.no_grad():
                    reference_features = clip_model.encode_image(reference_images)
                    caption_features = clip_model.encode_text(text_inputs)
                    target_features = clip_model.encode_image(target_images)

                # 使用增强版Combiner进行特征融合和相似度计算
                logits = combining_function(
                    reference_features,
                    caption_features,
                    target_features
                )

                # 计算损失
                ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                loss = ranking_loss(logits, ground_truth, margin)

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                combining_function.parameters(),
                max_norm=0.5
            )

            # 更新参数
            scaler.step(optimizer)
            scaler.update()

            # 日志记录
            if experiment:
                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
            update_train_running_results(train_running_results, loss, images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            if experiment:
                experiment.log_metric('learning_rate', current_lr, step=step)

        # 更新学习率调度器
        scheduler.step()

        # 记录epoch级训练指标
        train_epoch_loss = train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']
        if experiment:
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)
        print(f"训练损失: {train_epoch_loss:.4f}")

        training_log_frame = pd.concat([training_log_frame,
                                        pd.DataFrame({'epoch': epoch, 'train_epoch_loss': train_epoch_loss},
                                                     index=[0])])
        training_log_frame.to_csv(training_path / 'train_metrics.csv', index=False)

        # === 验证步骤 ===
        if (epoch % validation_frequency == 0) or (epoch == num_epochs - 1):
            val_start_time = datetime.now()
            print(f"\n开始验证 (轮次 {epoch + 1})...")

            with torch.no_grad():
                clip_model.eval()
                combining_function.eval()

                # 提取所有验证集的索引特征
                print("提取验证集索引特征...")
                index_features, index_names = extract_index_features(classic_val_dataset, clip_model)

                # 计算整体验证指标
                print("计算整体验证指标...")
                recall_at10, recall_at50, mrr, precision_at10, precision_at50 = compute_gui_val_metrics(
                    relative_val_dataset,
                    clip_model,
                    combining_function,
                    index_features,
                    index_names
                )

                avg_recall_at10 = recall_at10
                avg_recall_at50 = recall_at50
                avg_mrr = mrr
                avg_precision_at10 = precision_at10
                avg_precision_at50 = precision_at50
                avg_recall = (avg_recall_at50 + avg_recall_at10) / 2

                # 计算平滑指标
                if epoch == 0:
                    smoothed_recall = avg_recall
                else:
                    smoothed_recall = (smoothed_recall * smoothing_factor +
                                       avg_recall * (1 - smoothing_factor))

                results_dict = {
                    'epoch': epoch,
                    'recall@10': avg_recall_at10,
                    'recall@50': avg_recall_at50,
                    'mrr': avg_mrr,
                    'precision@10': avg_precision_at10,
                    'precision@50': avg_precision_at50,
                    'average_recall': avg_recall,
                    'smoothed_recall': smoothed_recall
                }

                # 存储最后一次验证的整体指标
                last_overall_metrics = {
                    'recall@10': avg_recall_at10,
                    'recall@50': avg_recall_at50,
                    'mrr': avg_mrr,
                    'precision@10': avg_precision_at10,
                    'precision@50': avg_precision_at50,
                    'average_recall': avg_recall
                }

                print("\n整体验证指标:")
                print(f"  Recall@10: {avg_recall_at10:.4f}")
                print(f"  Recall@50: {avg_recall_at50:.4f}")
                print(f"  MRR: {avg_mrr:.4f}")
                print(f"  Precision@10: {avg_precision_at10:.4f}")
                print(f"  Precision@50: {avg_precision_at50:.4f}")
                print(f"  平均召回率: {avg_recall:.4f}")
                print(f"  平滑召回率: {smoothed_recall:.4f}")

                if experiment:
                    experiment.log_metrics({
                        'recall@10': avg_recall_at10,
                        'recall@50': avg_recall_at50,
                        'mrr': avg_mrr,
                        'precision@10': avg_precision_at10,
                        'precision@50': avg_precision_at50,
                        'average_recall': avg_recall,
                        'smoothed_recall': smoothed_recall
                    }, epoch=epoch)

                # 计算每个类别的验证指标
                category_results = {}
                print("\n计算每个类别的验证指标...")

                for gui_type, (gui_classic, gui_relative) in gui_type_val_datasets.items():
                    # 提取当前类别的索引特征
                    gui_index_features, gui_index_names = extract_index_features(gui_classic, clip_model)

                    # 计算当前类别的指标
                    cat_recall10, cat_recall50, cat_mrr, cat_prec10, cat_prec50 = compute_gui_val_metrics(
                        gui_relative,
                        clip_model,
                        combining_function,
                        gui_index_features,
                        gui_index_names
                    )

                    cat_avg_recall = (cat_recall10 + cat_recall50) / 2

                    # 记录结果
                    category_results[gui_type] = {
                        'recall@10': cat_recall10,
                        'recall@50': cat_recall50,
                        'mrr': cat_mrr,
                        'precision@10': cat_prec10,
                        'precision@50': cat_prec50,
                        'average_recall': cat_avg_recall
                    }

                    # 打印当前类别结果
                    print(f"\n[{gui_type}]")
                    print(f"  Recall@10: {cat_recall10:.4f}")
                    print(f"  Recall@50: {cat_recall50:.4f}")
                    print(f"  MRR: {cat_mrr:.4f}")
                    print(f"  Precision@10: {cat_prec10:.4f}")
                    print(f"  Precision@50: {cat_prec50:.4f}")
                    print(f"  平均召回率: {cat_avg_recall:.4f}")

                    if experiment:
                        experiment.log_metrics({
                            f'{gui_type}_recall@10': cat_recall10,
                            f'{gui_type}_recall@50': cat_recall50,
                            f'{gui_type}_mrr': cat_mrr,
                            f'{gui_type}_precision@10': cat_prec10,
                            f'{gui_type}_precision@50': cat_prec50,
                            f'{gui_type}_avg_recall': cat_avg_recall
                        }, epoch=epoch)

                # 存储最后一次验证的类别指标
                last_category_metrics = category_results

                # 保存类别指标到日志文件
                cat_metrics_row = {'epoch': epoch}
                for gui_type, metrics in category_results.items():
                    for metric_name, value in metrics.items():
                        cat_metrics_row[f'{gui_type}_{metric_name}'] = value

                category_metrics_log = pd.concat([category_metrics_log,
                                                  pd.DataFrame(cat_metrics_row, index=[0])])
                category_metrics_log.to_csv(training_path / 'category_metrics.csv', index=False)

                # 记录验证结果
                validation_log_frame = pd.concat([validation_log_frame,
                                                  pd.DataFrame(results_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                val_duration = datetime.now() - val_start_time
                print(f"验证完成! 耗时: {val_duration}")

            # === 模型保存与早停逻辑 ===
            if save_training:
                if save_best:
                    # 动态调整改进阈值 (随着训练进行，阈值逐渐降低)
                    current_min_improvement = min_improvement * (min_improvement_decay ** (epoch // 10))

                    # 确保阈值不会低于0.0001
                    current_min_improvement = max(current_min_improvement, 0.0001)

                    print(f"当前改进阈值: {current_min_improvement:.5f}")

                    # 检查是否达到新的最佳模型
                    if smoothed_recall > best_avg_recall + current_min_improvement:
                        improvement = smoothed_recall - best_avg_recall
                        best_avg_recall = smoothed_recall
                        best_epoch = epoch
                        model_save_name = f'best_epoch{epoch + 1}_recall{smoothed_recall:.4f}'
                        best_model_path = save_model(
                            model_save_name,
                            epoch,
                            clip_model,
                            combining_function,
                            training_path,
                            is_combiner=True
                        )
                        early_stop_count = 0
                        print(f"新最佳Combiner! 改进: {improvement:.5f}, 平滑召回率: {best_avg_recall:.4f}")
                    else:
                        early_stop_count += 1
                        improvement = smoothed_recall - best_avg_recall
                        print(f"无改进 (差: {improvement:.5f}, 阈值: {current_min_improvement:.5f}), "
                              f"早停计数: {early_stop_count}/{patience}")

                        # 学习率过低时增加早停计数
                        current_lr = optimizer.param_groups[0]['lr']
                        if current_lr < 1e-6:
                            early_stop_count += 1
                            print(f"学习率过低 ({current_lr:.2e})，额外增加早停计数: {early_stop_count}/{patience}")

                    # 判断是否触发早停
                    if early_stop_count >= patience:
                        print(f"早停触发! {patience}轮无改进")
                        early_stop_triggered = True
                else:
                    # 不保存最佳模型，但保存当前模型
                    save_model(
                        f'epoch_{epoch}',
                        epoch,
                        clip_model,
                        combining_function,
                        training_path,
                        is_combiner=True
                    )

        # 如果触发了早停，就跳出循环
        if early_stop_triggered:
            print(f"训练在第{epoch + 1}轮提前终止")
            break

        epoch_duration = datetime.now() - epoch_start_time
        print(f"轮次 {epoch + 1} 完成! 耗时: {epoch_duration}")

    # === 训练结束处理 ===
    print("\n增强版Combiner训练结束!")

    # 保存最终模型
    if save_training:
        if save_best and best_model_path:
            # 复制最佳Combiner为最终模型
            final_model_path = save_model(
                'final',
                best_epoch,
                clip_model,
                combining_function,
                training_path,
                is_combiner=True
            )
            print(f"最终Combiner基于轮次 {best_epoch + 1} 的最佳模型")
        else:
            print("保存当前Combiner为最终模型")
            save_model(
                'final',
                num_epochs - 1,
                clip_model,
                combining_function,
                training_path,
                is_combiner=True
            )

    # 训练总结报告
    total_duration = datetime.now() - datetime.strptime(training_start, "%Y-%m-%d_%H:%M:%S")
    print("\n" + "=" * 60)
    print(f"增强版Combiner训练总结 - 总轮数: {epoch + 1}/{num_epochs}")
    print(f"总耗时: {total_duration}")
    if save_best:
        print(f"最佳Combiner: 轮次 {best_epoch + 1} - 平滑召回率 {best_avg_recall:.4f}")
    print(f"提前停止: {'是' if early_stop_triggered else '否'}")

    # 打印整体验证指标
    if last_overall_metrics:
        print("\n整体验证指标:")
        print(f"  Recall@10: {last_overall_metrics['recall@10']:.4f}")
        print(f"  Recall@50: {last_overall_metrics['recall@50']:.4f}")
        print(f"  MRR: {last_overall_metrics['mrr']:.4f}")
        print(f"  Precision@10: {last_overall_metrics['precision@10']:.4f}")
        print(f"  Precision@50: {last_overall_metrics['precision@50']:.4f}")
        print(f"  平均召回率: {last_overall_metrics['average_recall']:.4f}")

    # 打印每个类别的验证指标
    if last_category_metrics:
        print("\n各类别验证指标:")
        for gui_type, metrics in last_category_metrics.items():
            print(f"\n[{gui_type}]")
            print(f"  Recall@10: {metrics['recall@10']:.4f}")
            print(f"  Recall@50: {metrics['recall@50']:.4f}")
            print(f"  MRR: {metrics['mrr']:.4f}")
            print(f"  Precision@10: {metrics['precision@10']:.4f}")
            print(f"  Precision@50: {metrics['precision@50']:.4f}")
            print(f"  平均召回率: {metrics['average_recall']:.4f}")
    print("=" * 60)

    # 保存最终日志
    training_log_frame.to_csv(training_path / 'train_metrics_final.csv', index=False)
    validation_log_frame.to_csv(training_path / 'validation_metrics_final.csv', index=False)
    category_metrics_log.to_csv(training_path / 'category_metrics_final.csv', index=False)

    return training_path


# 主函数
def main():
    # 参数解析与实验初始化
    parser = argparse.ArgumentParser(description="增强版Combiner训练脚本")
    parser.add_argument("--dataset", type=str, default="GUI", help="数据集名称")
    parser.add_argument("--api-key", type=str, default=None, help="Comet日志API密钥")
    parser.add_argument("--workspace", type=str, default=None, help="Comet工作区")
    parser.add_argument("--experiment-name", type=str, default=None, help="实验名称")
    parser.add_argument("--num-epochs", default=50, type=int, help="总训练轮数")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str,
                        help="CLIP模型名称 (e.g., RN50, ViT-B/32)")
    parser.add_argument("--pretrained-path", type=str, required=True,
                        help="第一阶段预训练模型路径")
    parser.add_argument("--batch-size", default=32, type=int, help="批次大小")
    parser.add_argument("--validation-frequency", default=1, type=int, help="验证频率")
    parser.add_argument("--transform", default="guipad", type=str,
                        choices=["clip", "guipad"], help="预处理方式")
    parser.add_argument("--save-training", action='store_true', help="保存训练过程模型")
    parser.add_argument("--save-best", action='store_true', help="仅保存最佳模型")
    parser.add_argument("--margin", default=0.2, type=float, help="排序损失的边界值")
    parser.add_argument("--patience", default=10, type=int, help="早停轮数")
    parser.add_argument("--learning-rate", default=3e-5, type=float, help="学习率")
    parser.add_argument("--train-types", nargs='+',
                        default=['bare', 'form', 'gallery', 'login', 'news', 'profile', 'terms', 'search'],
                        help="训练使用的GUI类型")
    parser.add_argument("--val-types", nargs='+',
                        default=['bare', 'form', 'gallery', 'login', 'news', 'profile', 'terms', 'search'],
                        help="验证使用的GUI类型")
    args = parser.parse_args()

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "pretrained_path": args.pretrained_path,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "margin": args.margin,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "train_gui_types": args.train_types,
        "val_gui_types": args.val_types
    }

    # 启动训练
    print("=" * 60)
    print("启动增强版Combiner训练")
    print("=" * 60)

    experiment = None
    if args.api_key and args.workspace:
        print("启用Comet日志记录")
        experiment = comet_ml.Experiment(
            api_key=args.api_key,
            project_name=f"GUI_Enhanced_Combiner_Training",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
        else:
            experiment.set_name(f"Enhanced_Combiner_{args.clip_model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    else:
        print("Comet logging DISABLED, in order to enable it you need to provide an api key and a workspace")

    # 记录代码和参数
    if experiment:
        experiment.log_code(folder=str(Path(__file__).parent))
        experiment.log_parameters(training_hyper_params)

    # 打印配置摘要
    print("\n增强版Combiner训练配置摘要:")
    print(f"  总轮数: {args.num_epochs}")
    print(f"  基础模型: {args.clip_model_name}")
    print(f"  预训练模型: {args.pretrained_path}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  早停耐心值: {args.patience}")
    print(f"  保存模式: {'最佳模型' if args.save_best else '所有模型'}")
    print(f"  训练GUI类型: {', '.join(args.train_types)}")
    print(f"  验证GUI类型: {', '.join(args.val_types)}")
    print("=" * 60 + "\n")

    # 启动训练
    training_path = train_enhanced_combiner(
        **training_hyper_params,
        experiment=experiment
    )

    print("\n增强版Combiner训练完成!")
    print(f"模型和日志保存在: {training_path}")
    if experiment:
        experiment.log_asset_folder(str(training_path))
        experiment.end()

    print("=" * 60)
    print("训练脚本结束")


if __name__ == '__main__':
    main()