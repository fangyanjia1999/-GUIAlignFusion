import os
import json
import torch
import clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from combiner7 import Combiner  # 使用增强版Combiner
from datetime import datetime
import shutil

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"


# 配置参数
class Config:
    data_root = "/home/CLIP4/CLIP4Cir/GUI"
    image_size = (224, 224)
    mapping_file = "/home/CLIP4/CLIP4Cir/GUI/mapping.csv"
    number_image_folder = "/home/CLIP4/CLIP4Cir/GUI/image3"
    output_dir = "/home/CLIP4/CLIP4Cir/GUI/Results"
    model_path = "/home/CLIP4/CLIP4Cir/enhanced_combiner_training/RN50x4_2025-07-05_10:57:59/saved_models/enhanced_combiner1_final.pt"  # 替换为增强版模型路径
    clip_model_name = "RN50x4"
    image_lib_path = "/home/CLIP4/CLIP4Cir/GUI/images"


def load_mapping():
    """加载CSV映射表"""
    try:
        df = pd.read_csv(Config.mapping_file)
    except Exception as e:
        print(f"加载映射表失败: {e}")
        return {}

    # 检查必要的列
    required_columns = ['subfolder', 'original', 'new', 'order']
    for col in required_columns:
        if col not in df.columns:
            print(f"CSV文件缺少必需的列: '{col}'")
            return {}

    mapping_dict = {}

    # 构建映射关系
    for _, row in df.iterrows():
        # 处理new列 - 作为键
        new_key = str(row['new']).strip()
        # 移除扩展名
        if new_key.lower().endswith('.png'):
            new_key = new_key[:-4]
        elif new_key.lower().endswith('.jpg'):
            new_key = new_key[:-4]

        # 获取original列的值
        original_value = str(row['original']).strip()
        # 移除扩展名
        if original_value.lower().endswith('.png'):
            original_value = original_value[:-4]
        elif original_value.lower().endswith('.jpg'):
            original_value = original_value[:-4]

        # 添加到映射字典
        mapping_dict[new_key] = original_value

    print(f"已加载 {len(mapping_dict)} 个映射项")
    return mapping_dict


def get_number_image_path(number):
    """获取数字对应的图像路径"""
    if number == "N/A" or not number:
        return None

    # 移除扩展名
    number = str(number).split('.')[0]

    # 尝试不同格式
    jpg_path = os.path.join(Config.number_image_folder, f"{number}.jpg")
    if os.path.exists(jpg_path):
        return jpg_path

    # 尝试去掉前导零
    if number.startswith('0'):
        stripped_number = number.lstrip('0')
        if stripped_number:
            stripped_jpg_path = os.path.join(Config.number_image_folder, f"{stripped_number}.jpg")
            if os.path.exists(stripped_jpg_path):
                return stripped_jpg_path

    # 尝试添加前导零
    if len(number) < 4:
        padded_number = number.zfill(4)
        padded_jpg_path = os.path.join(Config.number_image_folder, f"{padded_number}.jpg")
        if os.path.exists(padded_jpg_path):
            return padded_jpg_path

    return None


def load_model():
    """加载CLIP模型和增强版Combiner"""
    print(f"加载模型: {Config.model_path}")
    checkpoint = torch.load(Config.model_path, map_location=device)

    # 加载CLIP模型
    clip_model, _ = clip.load(Config.clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().to(device)

    # 加载CLIP状态字典（兼容处理）
    clip_state_dict = checkpoint['clip_model_state_dict']
    clip_state_dict = {k.replace('module.', ''): v for k, v in clip_state_dict.items()}
    clip_model.load_state_dict(clip_state_dict)

    # 获取特征维度并计算增强版Combiner参数
    feature_dim = clip_model.text_projection.shape[1]
    projection_dim = feature_dim * 4  # 增强版参数
    hidden_dim = feature_dim * 8  # 增强版参数

    # 加载增强版Combiner
    combiner = Combiner(
        clip_feature_dim=feature_dim,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim
    ).to(device)

    # 加载Combiner状态字典
    combiner_state_dict = checkpoint.get('combiner_state_dict', checkpoint.get('model_state_dict', {}))
    combiner.load_state_dict(combiner_state_dict)

    # 修复logit_scale处理 - 确保它是张量
    if 'logit_scale' in checkpoint:
        # 确保logit_scale是张量
        if not isinstance(checkpoint['logit_scale'], torch.Tensor):
            # 将标量转换为张量
            logit_scale_tensor = torch.tensor(
                checkpoint['logit_scale'],
                dtype=torch.float32,
                device=device
            )
        else:
            logit_scale_tensor = checkpoint['logit_scale'].to(device)

        # 修改Combiner的logit_scale属性为张量
        combiner.logit_scale = logit_scale_tensor
    else:
        # 使用CLIP默认值
        combiner.logit_scale = torch.tensor(np.log(1 / 0.07), device=device)

    print("模型加载成功!")
    return clip_model, combiner


def extract_index_features(image_paths, clip_model, preprocess):
    """提取图像库特征"""
    print(f"提取图像库特征 ({len(image_paths)} 张图像)...")
    index_features = []
    index_names = []

    with torch.no_grad():
        for img_path in tqdm(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                features = clip_model.encode_image(img_tensor)
                index_features.append(features.cpu())
                index_names.append(Path(img_path).stem)  # 只保存文件名（不带扩展名）
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")

    return torch.cat(index_features, dim=0), index_names


def retrieve_images(reference_image, text, index_features, index_names, clip_model, combiner, preprocess, k=10):
    """检索前k个相似图像"""
    # 处理参考图像
    reference_image = preprocess(reference_image).unsqueeze(0).to(device)

    # 处理文本
    text_input = clip.tokenize([text], truncate=True).to(device)

    with torch.no_grad():
        # 提取特征
        reference_features = clip_model.encode_image(reference_image)
        text_features = clip_model.encode_text(text_input)

        # 使用增强版Combiner融合特征
        query_features = combiner.combine_features(reference_features, text_features)

        # 计算相似度 - 确保logit_scale是张量
        if isinstance(combiner.logit_scale, (int, float)):
            logit_scale = torch.tensor(combiner.logit_scale, device=device)
        else:
            logit_scale = combiner.logit_scale

        index_features = index_features.to(device)
        similarities = logit_scale * (query_features @ index_features.T)
        similarities = similarities.squeeze(0).cpu().numpy()

    # 获取前k个结果
    topk_indices = np.argsort(similarities)[::-1][:k]
    topk_ids = [index_names[i] for i in topk_indices]  # 图像ID
    topk_scores = [similarities[i] for i in topk_indices]

    # 获取完整图像路径
    topk_paths = []
    for img_id in topk_ids:
        # 尝试不同格式
        png_path = os.path.join(Config.image_lib_path, f"{img_id}.png")
        jpg_path = os.path.join(Config.image_lib_path, f"{img_id}.jpg")

        if os.path.exists(png_path):
            topk_paths.append(png_path)
        elif os.path.exists(jpg_path):
            topk_paths.append(jpg_path)
        else:
            print(f"警告: 找不到图像 {img_id}")

    return topk_paths, topk_ids, topk_scores


def create_clean_result_image(input_image_path, input_text, results, mapping_dict, output_path):
    """
    创建简洁的结果图像:
    - 第一行: 输入图像 + 输入数字图像 + 文本描述
    - 第二行: 检索结果图像
    - 第三行: 检索结果对应的数字图像
    """
    # 加载映射表
    mapping_dict = mapping_dict or {}

    # 从输入图像路径中提取ID
    input_id = Path(input_image_path).stem

    # 获取输入图像对应的数字
    input_number = mapping_dict.get(input_id, "N/A")

    # 获取输入数字图像路径
    input_number_image_path = get_number_image_path(input_number)

    # 加载输入图像
    input_img = Image.open(input_image_path).convert("RGB")

    # 加载输入数字图像（如果存在）
    input_num_img = None
    if input_number_image_path:
        try:
            input_num_img = Image.open(input_number_image_path).convert("RGB")
        except Exception as e:
            print(f"无法加载输入数字图像: {e}")

    # 加载结果图像
    result_images = []
    result_num_images = []

    for img_path in results:
        img = Image.open(img_path).convert("RGB")
        result_images.append(img)

        # 获取结果图像ID
        img_id = Path(img_path).stem

        # 获取结果图像的数字
        result_number = mapping_dict.get(img_id, "N/A")
        result_num_img_path = get_number_image_path(result_number)

        # 加载数字图像（如果存在）
        result_num_img = None
        if result_num_img_path:
            try:
                result_num_img = Image.open(result_num_img_path).convert("RGB")
            except Exception as e:
                print(f"无法加载结果数字图像: {e}")

        result_num_images.append(result_num_img)

    # 设置图像大小
    thumb_size = (180, 180)
    num_results = len(results)

    # 计算网格列数
    ncols = max(5, num_results)

    # 创建画布
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, ncols, height_ratios=[1, 1, 1])

    # === 第一行: 输入信息 ===
    # 输入图像
    ax_input_img = fig.add_subplot(gs[0, 0])
    ax_input_img.imshow(input_img)
    ax_input_img.axis('off')

    # 输入数字图像
    if input_num_img:
        ax_input_num = fig.add_subplot(gs[0, 1])
        ax_input_num.imshow(input_num_img)
        ax_input_num.axis('off')
    else:
        # 如果没有数字图像，留空
        ax_input_num = fig.add_subplot(gs[0, 1])
        ax_input_num.axis('off')

    # 文本描述 (占据剩余列)
    ax_text = fig.add_subplot(gs[0, 2:])
    ax_text.axis('off')
    ax_text.text(0.02, 0.5, input_text,
                 fontsize=14, ha='left', va='center',
                 wrap=True, color='black')

    # === 第二行: 检索结果图像 ===
    for i, img in enumerate(result_images):
        ax_result = fig.add_subplot(gs[1, i])
        ax_result.imshow(img)
        ax_result.axis('off')

    # 填充空白 (第二行)
    for i in range(num_results, ncols):
        ax_empty = fig.add_subplot(gs[1, i])
        ax_empty.axis('off')

    # === 第三行: 数字图像 ===
    for i, num_img in enumerate(result_num_images):
        if i < ncols:  # 确保不会超出网格范围
            if num_img:
                ax_num = fig.add_subplot(gs[2, i])
                ax_num.imshow(num_img)
                ax_num.axis('off')
            else:
                # 如果没有数字图像，留空
                ax_empty = fig.add_subplot(gs[2, i])
                ax_empty.axis('off')

    # 填充空白 (第三行)
    for i in range(num_results, ncols):
        ax_empty = fig.add_subplot(gs[2, i])
        ax_empty.axis('off')

    # 调整布局并保存
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已保存简洁结果图像: {output_path}")


def main():
    # 创建输出目录
    os.makedirs(Config.output_dir, exist_ok=True)

    # 加载映射表
    mapping_dict = load_mapping()

    # 加载模型
    clip_model, combiner = load_model()

    # 准备图像预处理
    input_dim = clip_model.visual.input_resolution
    print(f"输入分辨率: {input_dim}x{input_dim}")

    preprocess = Compose([
        Resize(input_dim, interpolation=Image.BICUBIC),
        CenterCrop(input_dim),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # 准备图像库
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_paths.extend(Path(Config.image_lib_path).glob(ext))
    print(f"图像库中有 {len(image_paths)} 张图像")

    # 提取图像库特征
    index_features, index_names = extract_index_features(image_paths, clip_model, preprocess)

    # 示例查询 (可替换为实际输入)
    input_image_path = "/home/CLIP4/CLIP4Cir/GUI/images/2ACA62E113.png"
    input_text = "Added new ImageView elements"


    # 打开输入图像
    input_image = Image.open(input_image_path).convert("RGB")

    # 检索图像
    results, result_ids, scores = retrieve_images(
        input_image,
        input_text,
        index_features,
        index_names,
        clip_model,
         combiner,
        preprocess,
        k=10
    )

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"   )
    results_dir = Path(Config.output_dir) / f"retrieval_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 保存输入信息
    shutil.copy(input_image_path, results_dir / "input_image.png")
    with open(results_dir / "input_text.txt", "w") as f:
        f.write(input_text)

    # 保存检索结果
    result_info = []
    for i, (img_path, img_id, score) in enumerate(zip(results, result_ids, scores)):
        dest_path = results_dir / f"result_{i + 1}_{img_id}.png"
        shutil.copy(img_path, dest_path)
        result_info.append({
            "rank": i + 1,
            "id": img_id,
            "score": float(score),
            "path": str(dest_path)
        })

    # 保存元数据
    with open(results_dir / "results.json", "w") as f:
        json.dump({
            "input_image": str(results_dir / "input_image.png"),
            "input_text": input_text,
            "results": result_info
        }, f, indent=2)

    # 创建并保存简洁结果图像
    result_image_path = results_dir / "1.png"
    create_clean_result_image(
        input_image_path,
        input_text,
        results,
        mapping_dict,
        result_image_path
    )

    print(f"\n所有结果已保存到: {results_dir}")


if __name__ == "__main__":
    main()