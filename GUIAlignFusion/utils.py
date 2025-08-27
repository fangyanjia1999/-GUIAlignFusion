import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def collate_fn(batch: list):
    """处理各种类型数据的批处理"""
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # 处理 (ref_image, target_image, caption, ref_name, target_name) 类型
    if isinstance(batch[0], tuple) and len(batch[0]) == 5:
        ref_images = []
        target_images = []
        captions = []
        ref_names = []  # 新增
        target_names = []  # 新增

        for item in batch:
            if isinstance(item[0], torch.Tensor) and isinstance(item[1], torch.Tensor):
                ref_images.append(item[0])
                target_images.append(item[1])
                captions.append(item[2])
                ref_names.append(item[3])  # 新增
                target_names.append(item[4])  # 新增

        if len(ref_images) == 0:
            return None

        return (
            torch.stack(ref_images),
            torch.stack(target_images),
            list(captions),
            list(ref_names),  # 新增
            list(target_names)  # 新增
        )

    # 处理 (name, image) 类型
    elif isinstance(batch[0], tuple) and len(batch[0]) == 2:
        names = []
        images = []

        for item in batch:
            if isinstance(item[1], torch.Tensor):
                names.append(item[0])
                images.append(item[1])

        if len(images) == 0:
            return None

        return list(names), torch.stack(images)

    # 处理其他情况
    return torch.utils.data.dataloader.default_collate(batch)


def extract_index_features(dataset: Dataset, clip_model: nn.Module, batch_size: int = 32) -> Tuple[torch.Tensor, list]:
    feature_dim = clip_model.visual.output_dim
    classic_val_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                                    num_workers=4,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []  # 明确初始化为列表

    print(f"Extracting {dataset.gui_types} index features for {dataset.split} split")

    for batch in tqdm(classic_val_loader):
        if batch is None:
            continue

        names, images = batch
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            batch_features = F.normalize(batch_features, dim=-1)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)  # 使用 extend 添加名称

    # 返回前确保 index_names 是列表
    if not isinstance(index_names, list):
        index_names = list(index_names)

    return index_features, index_names

def process_single_caption(caption: str) -> str:
    """处理单个描述：清理和标准化"""
    return caption.strip('.?, ').capitalize()


def update_train_running_results(train_running_results: dict, loss: torch.Tensor, images_in_batch: int):
    train_running_results['accumulated_train_loss'] += loss.detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    avg_loss = train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']
    train_bar.set_description(desc=f"[{epoch}/{num_epochs}] train loss: {avg_loss:.3f}")


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))


class AttentionGuidedGatedFusion(nn.Module):
    """使用交叉注意力的融合模块"""

    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim

        # 文本到图像的交叉注意力
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 图像到文本的交叉注意力
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # 增加维度 (batch, seq_len, feature_dim)
        image_features = image_features.unsqueeze(1)
        text_features = text_features.unsqueeze(1)

        # 文本到图像注意力
        text_attended, _ = self.text_to_image_attn(
            query=text_features,
            key=image_features,
            value=image_features
        )
        text_attended = self.norm1(text_features + text_attended)

        # 图像到文本注意力
        image_attended, _ = self.image_to_text_attn(
            query=image_features,
            key=text_attended,
            value=text_attended
        )
        image_attended = self.norm2(image_features + image_attended)

        # 移除序列维度
        text_attended = text_attended.squeeze(1)
        image_attended = image_attended.squeeze(1)

        # 门控融合
        combined = torch.cat([image_attended, text_attended], dim=1)
        gate_value = self.gate(combined)
        fused = gate_value * image_attended + (1 - gate_value) * text_attended

        # 前馈网络
        fused = fused + self.ffn(fused)

        return fused