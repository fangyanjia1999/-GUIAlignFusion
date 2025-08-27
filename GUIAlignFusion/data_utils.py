import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import List, Callable
from pathlib import Path
import json

import random
import PIL
from PIL import Image, ImageEnhance
# 基础路径设置
base_path = Path('/home/CLIP4/CLIP4Cir')

# 设备选择
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def _convert_image_to_rgb(image):
    """确保处理带透明通道的PNG图像"""
    if image.mode in ('RGBA', 'LA'):
        background = PIL.Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    return image.convert("RGB")


def _enhance_contrast(image):
    """增强图像对比度以突出GUI元素"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)


class GUIEdgePad:
    """GUI专用填充策略，保持宽高比的同时最大化保留布局信息"""

    def __init__(self, target_size: int, pad_color=(255, 255, 255)):
        self.target_size = target_size
        self.pad_color = pad_color

    def __call__(self, image):
        w, h = image.size
        scale = min(self.target_size / w, self.target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = image.resize((new_w, new_h), PIL.Image.LANCZOS)

        padding_left = (self.target_size - new_w) // 2
        padding_top = (self.target_size - new_h) // 2

        # 创建新图像并粘贴调整大小后的图像
        if resized_image.mode == 'RGBA':
            new_image = PIL.Image.new('RGBA', (self.target_size, self.target_size), (*self.pad_color, 0))
            new_image.paste(resized_image, (padding_left, padding_top), resized_image)
        else:
            new_image = PIL.Image.new('RGB', (self.target_size, self.target_size), self.pad_color)
            new_image.paste(resized_image, (padding_left, padding_top))

        return new_image


def gui_transform(dim: int):
    """适用于GUI布局图的预处理流程"""
    return transforms.Compose([
        GUIEdgePad(dim),
        transforms.Lambda(_enhance_contrast),
        transforms.Resize((dim, dim), interpolation=PIL.Image.LANCZOS),
        transforms.Lambda(_convert_image_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


class GUIDataset(Dataset):
    valid_categories = ['bare', 'form', 'gallery', 'list', 'login', 'news', 'profile', 'terms', 'search', 'settings']

    def __init__(self, split: str, gui_types: List[str], mode: str, preprocess: Callable, image_size: int = 288):
        self.mode = mode
        self.gui_types = gui_types
        self.split = split
        self.image_size = image_size

        # 验证输入参数
        if mode not in ['relative', 'classic']:
            raise ValueError("Mode must be 'relative' or 'classic'")
        if split not in ['test', 'train', 'val']:
            raise ValueError("Split must be 'test', 'train' or 'val'")
        for gui_type in gui_types:
            if gui_type not in self.valid_categories:
                raise ValueError(f"Invalid GUI type: {gui_type}")

        self.preprocess = preprocess
        self.triplets: List[dict] = []
        self.image_names: list = []

        # 加载数据
        for gui_type in gui_types:
            try:
                # 加载三元组数据
                cap_path = base_path / 'GUI' / 'captions' / f'cap.{gui_type}.{split}.json'
                if cap_path.exists():
                    with open(cap_path) as f:
                        data = json.load(f)
                        self.triplets.extend(data)

                # 加载图像列表
                split_path = base_path / 'GUI' / 'image_splits' / f'split.{gui_type}.{split}.json'
                if split_path.exists():
                    with open(split_path) as f:
                        self.image_names.extend(json.load(f))
            except Exception as e:
                print(f"Error loading {gui_type}.{split}: {str(e)}")
                continue

        print(f"GUI {split} - {gui_types} dataset in {mode} mode initialized. "
              f"Triplets: {len(self.triplets)}, Images: {len(self.image_names)}")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                # 获取三元组数据
                item = self.triplets[index]
                caption = item['captions']
                reference_name = item['candidate'].strip()
                target_name = item['target'].strip()

                # 加载参考图像
                reference_image_path = base_path / 'GUI' / 'images' / f"{reference_name}.png"
                if not reference_image_path.exists():
                    return None
                reference_image = PIL.Image.open(reference_image_path).convert('RGBA')

                # 加载目标图像
                target_image_path = base_path / 'GUI' / 'images' / f"{target_name}.png"
                if not target_image_path.exists():
                    return None
                target_image = PIL.Image.open(target_image_path).convert('RGBA')

                # 仅在训练时应用数据增强
                if self.split == 'train':
                    reference_image = self._random_augmentation(reference_image)
                    target_image = self._random_augmentation(target_image)

                # 预处理图像
                reference_image = self.preprocess(reference_image)
                target_image = self.preprocess(target_image)

                # 返回图像张量、名称和描述
                return (
                    reference_image,
                    target_image,
                    caption,
                    reference_name,
                    target_name
                )

            elif self.mode == 'classic':
                # 获取单个图像数据
                image_name = self.image_names[index].strip()
                image_path = base_path / 'GUI' / 'images' / f"{image_name}.png"
                if not image_path.exists():
                    return None

                image = PIL.Image.open(image_path).convert('RGBA')

                if self.split == 'train':
                    image = self._random_augmentation(image)

                image = self.preprocess(image)
                return image_name, image

        except Exception as e:
            print(f"Error loading item {index}: {str(e)}")
            return None

    def _random_augmentation(self, image):
        """对图像进行随机增强"""
        try:
            if image is None:
                return None

            # 随机颜色调整
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.9, 1.1))

            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.9, 1.1))

            if random.random() > 0.7:
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))

            return image
        except Exception as e:
            print(f"Error during augmentation: {str(e)}")
            return image

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

