from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd
import clip
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import base_path, gui_transform, GUIDataset
from utils import (collate_fn, update_train_running_results, set_train_bar_description,
                   extract_index_features,
                   device, AttentionGuidedGatedFusion)
from validate import compute_gui_val_metrics


def unfreeze_clip_layers(model, clip_model_name: str, stage: int, epoch: int, num_epochs: int):
    """分阶段逐步解冻CLIP层"""
    if stage == 1:
        print("第一阶段：冻结所有CLIP参数，仅训练融合模块")
        for param in model.parameters():
            param.requires_grad = False
    elif stage == 2:
        print("第二阶段：逐步解冻视觉编码器层")
        if 'RN' in clip_model_name:  # ResNet架构
            # 根据训练进程逐步解冻层
            if epoch < int(num_epochs * 0.3):
                layers_to_unfreeze = ['layer4']
            elif epoch < int(num_epochs * 0.5):
                layers_to_unfreeze = ['layer3', 'layer4']
            else:
                layers_to_unfreeze = ['layer2', 'layer3', 'layer4']

            for name, param in model.visual.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True
        elif 'ViT' in clip_model_name:  # Vision Transformer架构
            total_blocks = len(model.visual.transformer.resblocks)
            # 根据训练进程逐步解冻层
            if epoch < int(num_epochs * 0.3):
                start_block = total_blocks - 3
            elif epoch < int(num_epochs * 0.5):
                start_block = total_blocks - 5
            else:
                start_block = total_blocks - 7

            for i in range(start_block, total_blocks):
                for param in model.visual.transformer.resblocks[i].parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"不支持的模型类型: {clip_model_name}")


def get_trainable_params(clip_model, combining_function, stage: int, learning_rate: float):
    """获取当前阶段需要训练的参数，并根据阶段调整学习率"""
    params = []
    if stage >= 1:
        params.append({'params': combining_function.parameters(), 'lr': learning_rate, 'weight_decay': 0.01})
    if stage >= 2:
        clip_params = filter(lambda p: p.requires_grad, clip_model.parameters())
        params.append({'params': clip_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.001})
    return params


def save_model(name: str, cur_epoch: int, clip_model: nn.Module, combining_function: nn.Module, training_path: Path):
    """保存模型（包含CLIP和融合模块）"""
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_path = models_path / f'{name}.pt'
    torch.save({
        'epoch': cur_epoch,
        'clip_model_state_dict': clip_model.state_dict(),
        'combiner_state_dict': combining_function.state_dict()
    }, str(model_path))
    print(f"模型已保存至: {model_path}")
    return model_path


def ranking_loss(logits: torch.Tensor, ground_truth: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """修正后的排序损失函数"""
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


def clip_finetune_gui(train_gui_types: List[str], val_gui_types: List[str],
                      num_epochs: int, clip_model_name: str, batch_size: int,
                      validation_frequency: int, transform: str, save_training: bool,
                      save_best: bool, margin=0.2, patience=15, **kwargs):
    """分阶段微调CLIP模型 - 完整优化版"""
    # === 初始化训练环境 ===
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path = Path(base_path / f"models5/clip_finetuned_on_gui_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=True, parents=True)

    print(f"训练开始时间: {training_start}")
    print(f"训练保存路径: {training_path}")

    # 保存超参数
    training_hyper_params = {
        "num_epochs": num_epochs,
        "clip_model_name": clip_model_name,
        "batch_size": batch_size,
        "validation_frequency": validation_frequency,
        "transform": transform,
        "save_training": save_training,
        "save_best": save_best,
        "margin": margin,
        "patience": patience,
        "train_gui_types": train_gui_types,
        "val_gui_types": val_gui_types
    }

    print("\n训练超参数:")
    for key, value in training_hyper_params.items():
        print(f"  {key}: {value}")

    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # === 模型加载与初始化 ===
    print(f"\n加载CLIP模型: {clip_model_name}")
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().to(device)

    # 初始冻结所有CLIP参数
    for param in clip_model.parameters():
        param.requires_grad = False

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
        num_workers=2,
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
        gui_type_val_datasets[gui_type] = (gui_classic, gui_relative)
        print(f"    {gui_type}验证集: {len(gui_relative)} 样本")

    # === 融合模块初始化 ===
    feature_dim = clip_model.text_projection.shape[1]
    combining_function = AttentionGuidedGatedFusion(feature_dim).to(device)
    print(f"\n初始化融合模块: {combining_function.__class__.__name__}")
    print(f"  特征维度: {feature_dim}")

    # === 训练配置 ===
    optimizer = None
    scheduler = None
    scaler = torch.cuda.amp.GradScaler()
    current_stage = 1
    initial_learning_rate = 3e-5

    # 早停与监控相关变量
    best_avg_recall = 0.0
    best_epoch = -1
    early_stop_triggered = False
    early_stop_count = 0
    stage_transition_occurred = False
    smoothed_recall = 0.0
    smoothing_factor = 0.9

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
    print('\n开始训练循环')
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        print(f"\n=== 轮次 {epoch + 1}/{num_epochs} ===")

        # === 动态调整验证频率 ===
        if epoch < num_epochs * 0.3:  # 前30%
            current_val_freq = max(validation_frequency, 5)
        elif epoch < num_epochs * 0.7:  # 30%-70%
            current_val_freq = max(validation_frequency, 2)
        else:  # 后30%
            current_val_freq = 1

        print(f"当前验证频率: 每 {current_val_freq} 轮验证一次")

        # === 阶段转换检测 ===
        if epoch == int(num_epochs * 0.3) and current_stage == 1:
            current_stage = 2
            stage_transition_occurred = True
            print(f"\n*** 切换到第{current_stage}阶段 ***")
            print(f"  解冻CLIP层 (模型: {clip_model_name})")
            unfreeze_clip_layers(clip_model, clip_model_name, current_stage, epoch, num_epochs)

            # 创建新的优化器参数组
            trainable_params = get_trainable_params(clip_model, combining_function, current_stage,
                                                    initial_learning_rate)
            optimizer = optim.AdamW(trainable_params)

            # 重新初始化学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3,
                threshold=0.001, threshold_mode='rel', verbose=True
            )

            # 阶段转换后重置早停计数器
            early_stop_count = 0
            print("  阶段转换完成 - 重置早停计数器和学习率调度器")

        # === 初始化优化器（第一阶段）===
        if epoch == 0 and optimizer is None:
            trainable_params = get_trainable_params(clip_model, combining_function, current_stage,
                                                    initial_learning_rate)
            optimizer = optim.AdamW(trainable_params)

            # 初始化学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3,
                threshold=0.001, threshold_mode='rel', verbose=True
            )
            print(f"初始化优化器 - 学习率: {initial_learning_rate}")

        # === 训练步骤 ===
        with experiment.train():
            # 设置模型模式
            if current_stage >= 2:
                clip_model.train()
                print("训练模式: CLIP模型 + 融合模块")
            else:
                clip_model.eval()
                print("训练模式: 仅融合模块")

            combining_function.train()

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
                with torch.cuda.amp.autocast():
                    # 提取特征
                    reference_features = clip_model.encode_image(reference_images)
                    caption_features = clip_model.encode_text(text_inputs)
                    predicted_features = combining_function(reference_features, caption_features)
                    target_features = clip_model.encode_image(target_images)

                    # 归一化特征
                    predicted_features = F.normalize(predicted_features, dim=-1)
                    target_features = F.normalize(target_features, dim=-1)

                    # 计算相似度
                    logits = predicted_features @ target_features.T * clip_model.logit_scale.exp()

                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = ranking_loss(logits, ground_truth, margin)

                # 反向传播
                scaler.scale(loss).backward()

                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(combining_function.parameters()) +
                    [p for p in clip_model.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                # 更新参数
                scaler.step(optimizer)
                scaler.update()

                # 日志记录
                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

                # 记录学习率
                for i, param_group in enumerate(optimizer.param_groups):
                    experiment.log_metric(f'lr_group_{i}', param_group['lr'], step=step)

            # 记录epoch级训练指标
            train_epoch_loss = train_running_results['accumulated_train_loss'] / train_running_results[
                'images_in_epoch']
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            print(f"训练损失: {train_epoch_loss:.4f}")

            training_log_frame = pd.concat([training_log_frame,
                                            pd.DataFrame({'epoch': epoch, 'train_epoch_loss': train_epoch_loss},
                                                         index=[0])])
            training_log_frame.to_csv(training_path / 'train_metrics.csv', index=False)

        # === 验证步骤 ===
        if (epoch % current_val_freq == 0) or (epoch == num_epochs - 1) or (epoch > num_epochs * 0.7):
            val_start_time = datetime.now()
            print(f"\n开始验证 (轮次 {epoch + 1})...")

            with torch.no_grad():
                clip_model.eval()
                combining_function.eval()

                # 提取所有验证集的索引特征
                print("提取验证集索引特征...")
                index_features, index_names = extract_index_features(classic_val_dataset, clip_model)

                with experiment.validate():
                    # 计算整体验证指标
                    print("计算整体验证指标...")
                    recall_at10, recall_at50, mrr, precision_at10, precision_at50 = compute_gui_val_metrics(
                        relative_val_dataset,
                        clip_model,
                        combining_function,
                        index_features,
                        index_names,

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
                            gui_index_names,

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

                        # 记录到Comet
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

                    # 更新学习率
                    if scheduler:
                        print("\n更新学习率...")
                        scheduler.step(smoothed_recall)
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"当前学习率: {current_lr:.2e}")
                        experiment.log_metric('learning_rate', current_lr, epoch=epoch)

                    # 记录验证结果
                    validation_log_frame = pd.concat([validation_log_frame,
                                                      pd.DataFrame(results_dict, index=[0])])
                    validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                    val_duration = datetime.now() - val_start_time
                    print(f"验证完成! 耗时: {val_duration}")

                # === 模型保存与早停逻辑 ===
                if save_training:
                    # 动态调整耐心值
                    if epoch < num_epochs * 0.4:  # 阶段2早期
                        effective_patience = max(patience, 20)
                    elif epoch > num_epochs * 0.7:  # 后期
                        effective_patience = min(patience, 10)
                    else:
                        effective_patience = patience

                    print(f"当前有效耐心值: {effective_patience} (原始: {patience})")
                    experiment.log_metric('effective_patience', effective_patience, epoch=epoch)

                    if save_best:
                        # 使用平滑指标判断提升
                        improvement_threshold = max(0.0005, 0.005 * (1 - epoch / num_epochs))  # 动态阈值
                        print(f"改进阈值: {improvement_threshold:.5f}")

                        if smoothed_recall > best_avg_recall + improvement_threshold:
                            best_avg_recall = smoothed_recall
                            best_epoch = epoch
                            model_save_name = f'tuned_clip_best_epoch{epoch + 1}_recall{smoothed_recall:.4f}'
                            best_model_path = save_model(model_save_name, epoch,
                                                         clip_model, combining_function,
                                                         training_path)
                            early_stop_count = 0
                            print(f"新最佳模型! 平滑召回率: {best_avg_recall:.4f} (提升: {best_avg_recall - smoothed_recall:.4f})")
                        else:
                            # 阶段转换后给3个epoch的缓冲期
                            if stage_transition_occurred and early_stop_count < 3:
                                print("阶段转换缓冲期，不计入早停计数")
                                early_stop_count = 0
                            else:
                                early_stop_count += 1

                            print(f"无改进，早期停止计数: {early_stop_count}/{effective_patience}")

                            # 学习率过低时增加早停计数
                            if current_lr < 1e-6:
                                early_stop_count += 1
                                print(f"学习率过低 ({current_lr:.2e})，额外增加早停计数: {early_stop_count}/{effective_patience}")

                        # 判断是否触发早停
                        if early_stop_count >= effective_patience:
                            print(f"早停触发! {effective_patience}轮无改进")
                            early_stop_triggered = True
                    else:
                        save_model(f'tuned_clip_{epoch}', epoch, clip_model, combining_function, training_path)

            # 重置阶段转换标志
            if stage_transition_occurred:
                stage_transition_occurred = False
                print("重置阶段转换标志")

        # 如果触发了早停，就跳出循环
        if early_stop_triggered:
            print(f"训练在第{epoch + 1}轮提前终止")
            break

        epoch_duration = datetime.now() - epoch_start_time
        print(f"轮次 {epoch + 1} 完成! 耗时: {epoch_duration}")

    # === 训练结束处理 ===
    print("\n训练结束!")

    # 保存最终模型
    if save_training:
        if save_best:
            if best_model_path:
                # 复制最佳模型为最终模型
                final_model_path = save_model('tuned_clip_final', best_epoch,
                                              clip_model, combining_function,
                                              training_path)
                print(f"最终模型基于轮次 {best_epoch + 1} 的最佳模型")
            else:
                print("未找到最佳模型，保存当前模型为最终模型")
                save_model('tuned_clip_final', num_epochs - 1, clip_model, combining_function, training_path)
        else:
            save_model('tuned_clip_final', num_epochs - 1, clip_model, combining_function, training_path)

    # 训练总结报告
    total_duration = datetime.now() - datetime.strptime(training_start, "%Y-%m-%d_%H:%M:%S")
    print("\n" + "=" * 60)
    print(f"训练总结 - 总轮数: {epoch + 1}/{num_epochs}")
    print(f"总耗时: {total_duration}")
    if save_best:
        print(f"最佳模型: 轮次 {best_epoch + 1} - 平滑召回率 {best_avg_recall:.4f}")
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


if __name__ == '__main__':
    # 参数解析与实验初始化
    parser = ArgumentParser(description="CLIP GUI微调脚本")
    parser.add_argument("--dataset", type=str, default="GUI", help="数据集名称")
    parser.add_argument("--api-key", type=str, help="Comet日志API密钥")
    parser.add_argument("--workspace", type=str, help="Comet工作区")
    parser.add_argument("--experiment-name", type=str, help="实验名称")
    parser.add_argument("--num-epochs", default=100, type=int, help="总训练轮数")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str,
                        help="CLIP模型名称 (e.g., RN50, ViT-B/32)")
    parser.add_argument("--batch-size", default=64, type=int, help="批次大小")
    parser.add_argument("--validation-frequency", default=1, type=int, help="验证频率")
    parser.add_argument("--transform", default="guipad", type=str,
                        help="预处理方式 (clip 或 guipad)")
    parser.add_argument("--save-training", action='store_true', help="保存训练过程模型")
    parser.add_argument("--save-best", action='store_true', help="仅保存最佳模型")
    parser.add_argument("--margin", default=0.2, type=float, help="排序损失的边界值")
    parser.add_argument("--patience", default=15, type=int, help="早停轮数")
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
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "margin": args.margin,
        "patience": args.patience,
        "train_gui_types": args.train_types,
        "val_gui_types": args.val_types
    }

    # 启动训练
    print("=" * 60)
    print("启动CLIP GUI微调训练")
    print("=" * 60)

    if args.api_key and args.workspace:
        print("启用Comet日志记录")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"GUI_CLIP_FineTuning",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
        else:
            experiment.set_name(f"CLIP_{args.clip_model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    else:
        print("Comet logging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    # 记录代码和参数
    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    # 打印配置摘要
    print("\n训练配置摘要:")
    print(f"  总轮数: {args.num_epochs}")
    print(f"  模型: {args.clip_model_name}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  早停耐心值: {args.patience}")
    print(f"  保存模式: {'最佳模型' if args.save_best else '所有模型'}")
    print(f"  训练GUI类型: {', '.join(args.train_types)}")
    print(f"  验证GUI类型: {', '.join(args.val_types)}")
    print("=" * 60 + "\n")

    # 启动训练
    training_path = clip_finetune_gui(**training_hyper_params)

    print("\n训练完成!")
    print(f"模型和日志保存在: {training_path}")
    experiment.log_asset_folder(str(training_path))
    experiment.end()

    print("=" * 60)
    print("训练脚本结束")
    print("=" * 60)