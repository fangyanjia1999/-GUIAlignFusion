import os
import json
import random
import numpy as np

# 设置输入和输出目录
input_dir = "/home/CLIP4/CLIP4Cir/image_splits"
output_dir = "/home/CLIP4/CLIP4Cir/captions"
os.makedirs(output_dir, exist_ok=True)

# 获取所有 JSON 文件
json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

# 统计原始文件中的图像数量
original_counts = {}
for json_file in json_files:
    file_path = os.path.join(input_dir, json_file)
    with open(file_path, 'r') as f:
        data = json.load(f)
    original_counts[json_file] = len(data)
    print(f"{json_file}: 原始图像数量 = {len(data)}")

# 根据原始数量分配配对数 (合理分配策略)
total_target_pairs = 62530
original_total_images = sum(original_counts.values())
pair_allocation = {}

# 按比例分配配对数
for json_file, count in original_counts.items():
    # 每个文件至少生成 count 对（保证每个图像至少配对一次）
    min_pairs = count
    # 剩余配对数按比例分配
    remaining_pairs = total_target_pairs - sum(pair_allocation.values()) - min_pairs
    if remaining_pairs <= 0:
        pair_allocation[json_file] = min_pairs
        continue

    # 计算比例
    proportion = count / original_total_images
    allocated_pairs = int(proportion * remaining_pairs) + min_pairs
    pair_allocation[json_file] = allocated_pairs

# 调整确保不超出总对数
current_total = sum(pair_allocation.values())
if current_total > total_target_pairs:
    # 如果超出，按比例减少
    for json_file in pair_allocation:
        pair_allocation[json_file] = int(pair_allocation[json_file] * total_target_pairs / current_total)
elif current_total < total_target_pairs:
    # 如果不足，随机分配剩余对数
    remaining = total_target_pairs - current_total
    for json_file in np.random.choice(list(pair_allocation.keys()), remaining, replace=True):
        pair_allocation[json_file] += 1

# 打印分配结果
for json_file, pairs in pair_allocation.items():
    print(f"{json_file}: 生成配对数 = {pairs}")

# 验证总对数
print(f"总配对数: {sum(pair_allocation.values())}")

# 生成配对数据
for json_file, pair_count in pair_allocation.items():
    file_path = os.path.join(input_dir, json_file)
    with open(file_path, 'r') as f:
        images = json.load(f)

    # 生成配对
    pairs = []
    image_ids = [item for item in images if isinstance(item, str)]  # 假设直接存储的是图像标识

    for _ in range(pair_count):
        # 随机选择两个不同的图像
        target = random.choice(image_ids)
        candidate = target
        while candidate == target:
            candidate = random.choice(image_ids)
        pairs.append({
            "target": target,
            "candidate": candidate,
            "captions": []
        })

    # 保存结果
    output_file = f"cap{json_file[4:]}"  # 将 split 替换为 cap
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=4)

    print(f"已生成 {output_file}，包含 {len(pairs)} 对")