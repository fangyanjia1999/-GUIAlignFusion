import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import re
import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import defaultdict

# 配置路径
IMAGE_DIR = Path("/home/CLIP4/CLIP4Cir/GUI/images")
JSON_DIR = Path("/home/CLIP4/CLIP4Cir/GUI/captions")

# 颜色到组件的映射 (BGR格式) - 增加组件重要性权重
COLOR_MAPPING = {
    (0, 255, 0): ("TextView", 1.2),  # 绿色 - 重要
    (0, 0, 255): ("ImageView", 1.5),  # 红色 - 非常重要
    (198, 204, 79): ("CheckedTextView", 1.0),
    (93, 47, 207): ("WebView", 1.5),  # 重要
    (187, 187, 187): ("View", 1.5),  # 次要
    (255, 0, 0): ("EditText", 1.5),  # 非常重要
    (238, 179, 142): ("ToggleButton", 1.0),
    (150, 105, 72): ("ToggleButtonOutline", 0.7),
    (0, 165, 255): ("RadioButton", 1.1),
    (0, 255, 255): ("Button", 1.3),  # 重要
    (15, 196, 241): ("CheckBox", 1.0),
    (139, 125, 96): ("SwitchMain", 1.0),
    (56, 234, 251): ("SwitchSlider", 1.0),
    (203, 192, 255): ("Widget", 1.0)  # 次要
}

# 检查可用的 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化GPT-2模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
model.eval()

# 设置pad_token
tokenizer.pad_token = tokenizer.eos_token

# 预先计算所有颜色的 HSV 值 - 缩小容差范围
COLOR_HSV_MAPPING = {}
for color_bgr, (name, weight) in COLOR_MAPPING.items():
    color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # 根据颜色特性动态调整容差范围
    hue_range = 10  # 缩小色调范围
    sat_range = 40 if color_hsv[1] > 100 else 60  # 高饱和度颜色更严格
    val_range = 40 if color_hsv[2] > 100 else 60  # 高亮度颜色更严格

    lower = np.array([
        max(0, color_hsv[0] - hue_range),
        max(0, color_hsv[1] - sat_range),
        max(0, color_hsv[2] - val_range)
    ])
    upper = np.array([
        min(179, color_hsv[0] + hue_range),
        min(255, color_hsv[1] + sat_range),
        min(255, color_hsv[2] + val_range)
    ])

    COLOR_HSV_MAPPING[color_bgr] = (lower, upper)


def detect_components(image_path, min_area=25):
    """Detect UI components in image with enhanced color accuracy"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Unable to read image: {image_path}")
        return {}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    components = {}

    for color_bgr, (name, weight) in COLOR_MAPPING.items():
        # 获取预计算的HSV范围
        lower, upper = COLOR_HSV_MAPPING[color_bgr]

        mask = cv2.inRange(hsv, lower, upper)

        # 形态学操作增强掩码质量
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 面积过滤 - 忽略小区域
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # 添加组件权重信息
            components.setdefault(name, []).append({
                "position": (x, y),
                "size": (w, h),
                "area": w * h,
                "weight": weight
            })

    return components


def generate_caption_with_gpt(components, min_size=10):
    """Generate natural language description of UI components using GPT-2"""
    # 首先使用规则方法生成基础描述
    base_description = generate_caption_rule_based(components, min_size)

    # 使用GPT-2改进描述
    prompt = f"Describe the user interface with these components: {base_description}"

    # 编码输入
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

    # 生成描述
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # 解码输出
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 移除提示部分，只保留生成的描述
    if caption.startswith(prompt):
        caption = caption[len(prompt):].strip()

    return caption


def generate_caption_rule_based(components, min_size=10):
    """Generate natural language description of UI components using rule-based approach"""
    # 按类型分组组件
    type_counts = defaultdict(int)
    for comp_name, comp_list in components.items():
        type_counts[comp_name] = len(comp_list)

    # 生成组件统计描述
    description_parts = []
    if type_counts:
        count_descriptions = []
        for comp_type, count in type_counts.items():
            if count == 1:
                count_descriptions.append(f"1 {comp_type}")
            else:
                count_descriptions.append(f"{count} {comp_type}s")

        if len(count_descriptions) == 1:
            description = f"There is {count_descriptions[0]}."
        else:
            description = f"There are {', '.join(count_descriptions[:-1])} and {count_descriptions[-1]}."
        description_parts.append(description)

    # 为每种类型选择1-2个重要组件进行详细描述
    for comp_type, comp_list in components.items():
        # 按面积排序，选择最重要的组件
        comp_list.sort(key=lambda x: x["area"], reverse=True)

        # 最多描述2个实例
        for i, comp in enumerate(comp_list[:min(2, len(comp_list))]):
            x, y = comp["position"]
            w, h = comp["size"]

            # 忽略小尺寸组件
            if w < min_size and h < min_size:
                continue

            # 生成位置描述
            if i == 0 and type_counts[comp_type] > 1:
                prefix = f"One {comp_type} is"
            elif i == 1:
                prefix = f"Another {comp_type} is"
            else:
                prefix = f"The {comp_type} is"

            comp_desc = f"{prefix} at position ({x},{y}) with size {w}x{h}."
            description_parts.append(comp_desc)

    # 如果没有检测到组件
    if not description_parts:
        return "No UI components detected."

    return " ".join(description_parts)


def analyze_image(image_path):
    """Analyze image and generate description"""
    print(f"Analyzing image: {image_path}")

    components = detect_components(image_path)
    print(f"Detected {sum(len(v) for v in components.values())} components")

    caption = generate_caption_with_gpt(components)
    return caption


# ================== 改进的解析和比较函数 ==================
def parse_component_description(desc):
    """Parse description string into structured data"""
    components = {}
    # 改进正则表达式匹配
    pattern = r'(\w+) is at position \((\d+),(\d+)\) with size (\d+)x(\d+)\.'

    # 查找所有组件实例
    matches = re.findall(pattern, desc)
    for match in matches:
        comp_name = match[0]
        x, y = int(match[1]), int(match[2])
        w, h = int(match[3]), int(match[4])

        if comp_name not in components:
            components[comp_name] = []

        components[comp_name].append({
            "position": (x, y),
            "size": (w, h),
            "area": w * h
        })

    # 尝试解析统计行
    stats_pattern = r'There (?:is|are) (.*?)\.'
    stats_match = re.search(stats_pattern, desc)
    if stats_match:
        stats_str = stats_match.group(1)
        stat_items = re.findall(r'(\d+) (\w+)s?', stats_str)
        for count, comp_type in stat_items:
            count = int(count)
            # 确保组件类型在字典中
            if comp_type not in components:
                components[comp_type] = []
            # 如果解析的实例数量少于统计数量，添加占位符
            while len(components[comp_type]) < count:
                components[comp_type].append({
                    "position": (-1, -1),
                    "size": (0, 0),
                    "area": 0,
                    "placeholder": True
                })

    return components


def position_distance(pos1, pos2, img_width, img_height):
    """Calculate normalized position distance"""
    # 防止除以零错误
    img_width = max(img_width, 1)
    img_height = max(img_height, 1)

    dx = abs(pos1[0] - pos2[0]) / img_width
    dy = abs(pos1[1] - pos2[1]) / img_height
    return math.sqrt(dx * dx + dy * dy)


def size_similarity(size1, size2):
    """Calculate size similarity (0-1)"""
    w1, h1 = size1
    w2, h2 = size2

    # 处理零尺寸情况
    w1 = max(w1, 1)
    h1 = max(h1, 1)
    w2 = max(w2, 1)
    h2 = max(h2, 1)

    w_sim = min(w1, w2) / max(w1, w2)
    h_sim = min(h1, h2) / max(h1, h2)
    return (w_sim + h_sim) / 2


def match_instances(target_instances, candidate_instances, img_size, pos_threshold=0.05):
    """Improved instance matching algorithm"""
    matched_pairs = []
    unmatched_targets = []
    unmatched_candidates = list(candidate_instances)

    img_width, img_height = img_size

    # 优先匹配位置最接近的实例
    for t_inst in target_instances:
        best_match = None
        min_dist = float('inf')

        for c_idx, c_inst in enumerate(unmatched_candidates):
            dist = position_distance(t_inst["position"], c_inst["position"], img_width, img_height)
            if dist < min_dist and dist < pos_threshold:
                min_dist = dist
                best_match = c_idx

        if best_match is not None:
            matched_pairs.append((t_inst, unmatched_candidates[best_match]))
            del unmatched_candidates[best_match]
        else:
            unmatched_targets.append(t_inst)

    return matched_pairs, unmatched_targets, unmatched_candidates


def calculate_change_significance(change_type, comp_type, t_inst=None, c_inst=None,
                                  count_diff=0, img_size=(1440, 2560)):
    """Calculate change significance score with component weighting"""
    # 获取组件权重
    weight = next((w for color, (name, w) in COLOR_MAPPING.items() if name == comp_type), 1.0)

    # 基本分数
    base_score = 0

    # 计算图像面积（修复变量未定义问题）
    img_area = img_size[0] * img_size[1] if img_size[0] > 0 and img_size[1] > 0 else 1

    if change_type == "type_add":
        base_score = 80
    elif change_type == "type_remove":
        base_score = 80
    elif change_type == "count_change":
        base_score = min(70, abs(count_diff) * 10)
    elif change_type == "add":
        # 基于相对面积
        rel_area = c_inst["area"] / img_area if img_area > 0 else 0
        base_score = min(100, int(rel_area * 1000))  # 占屏幕1%得10分
    elif change_type == "remove":
        rel_area = t_inst["area"] / img_area if img_area > 0 else 0
        base_score = min(100, int(rel_area * 1000))
    elif change_type == "move":
        dist = position_distance(t_inst["position"], c_inst["position"], img_size[0], img_size[1])
        base_score = min(80, int(dist * 500))  # 移动距离5%得40分
    elif change_type == "resize":
        size_sim = size_similarity(t_inst["size"], c_inst["size"])
        base_score = min(70, int((1 - size_sim) * 100))

    # 应用组件权重
    weighted_score = base_score * weight

    return min(weighted_score, 100)


def compare_ui_descriptions(target_desc, candidate_desc, img_size=(1440, 2560),
                            pos_threshold=0.05, size_sim_threshold=0.8,
                            min_size=10, max_significant_changes=3):
    """Improved UI comparison, keep only top 3 significant changes"""
    # 解析描述
    target_comps = parse_component_description(target_desc)
    candidate_comps = parse_component_description(candidate_desc)

    # 收集所有组件类型
    all_types = set(target_comps.keys()) | set(candidate_comps.keys())

    all_changes = []

    # 1. 检查组件类型级别的差异
    for comp_type in all_types:
        # 过滤掉占位符
        target_list = [item for item in target_comps.get(comp_type, []) if not item.get("placeholder")]
        candidate_list = [item for item in candidate_comps.get(comp_type, []) if not item.get("placeholder")]

        target_count = len(target_list)
        candidate_count = len(candidate_list)

        # 检查组件类型是否存在
        if comp_type not in target_comps:
            # 新增组件类型
            score = calculate_change_significance(
                "type_add", comp_type,
                count_diff=candidate_count,
                img_size=img_size
            )
            all_changes.append({
                "type": "type_add",
                "comp_type": comp_type,
                "count": candidate_count,
                "score": score
            })
            continue

        if comp_type not in candidate_comps:
            # 移除组件类型
            score = calculate_change_significance(
                "type_remove", comp_type,
                count_diff=target_count,
                img_size=img_size
            )
            all_changes.append({
                "type": "type_remove",
                "comp_type": comp_type,
                "count": target_count,
                "score": score
            })
            continue

        # 2. 检查实例数量变化
        count_diff = abs(target_count - candidate_count)
        if count_diff > 0:
            score = calculate_change_significance(
                "count_change", comp_type,
                count_diff=count_diff,
                img_size=img_size
            )
            all_changes.append({
                "type": "count_change",
                "comp_type": comp_type,
                "target_count": target_count,
                "candidate_count": candidate_count,
                "score": score
            })

        # 3. 匹配实例并比较细节
        matched_pairs, unmatched_targets, unmatched_candidates = match_instances(
            target_list, candidate_list, img_size, pos_threshold
        )

        # 4. 过滤小尺寸组件
        def is_large_component(comp):
            w, h = comp["size"]
            return w >= min_size or h >= min_size

        large_unmatched_targets = [c for c in unmatched_targets if is_large_component(c)]
        large_unmatched_candidates = [c for c in unmatched_candidates if is_large_component(c)]

        # 报告未匹配的移除实例
        for t_inst in large_unmatched_targets:
            x, y = t_inst['position']
            w, h = t_inst['size']
            score = calculate_change_significance(
                "remove", comp_type, t_inst=t_inst, img_size=img_size)
            all_changes.append({
                "type": "remove",
                "comp_type": comp_type,
                "position": (x, y),
                "size": (w, h),
                "score": score
            })

        # 报告未匹配的新增实例
        for c_inst in large_unmatched_candidates:
            x, y = c_inst['position']
            w, h = c_inst['size']
            score = calculate_change_significance(
                "add", comp_type, c_inst=c_inst, img_size=img_size)
            all_changes.append({
                "type": "add",
                "comp_type": comp_type,
                "position": (x, y),
                "size": (w, h),
                "score": score
            })

        # 5. 比较匹配的实例
        for t_inst, c_inst in matched_pairs:
            # 检查位置变化
            pos_dist = position_distance(t_inst["position"], c_inst["position"], img_size[0], img_size[1])
            if pos_dist > pos_threshold:
                tx, ty = t_inst['position']
                cx, cy = c_inst['position']
                score = calculate_change_significance(
                    "move", comp_type, t_inst=t_inst, c_inst=c_inst, img_size=img_size)
                all_changes.append({
                    "type": "move",
                    "comp_type": comp_type,
                    "from_position": (tx, ty),
                    "to_position": (cx, cy),
                    "distance": pos_dist,
                    "score": score
                })

            # 检查尺寸变化
            size_sim = size_similarity(t_inst["size"], c_inst["size"])
            if size_sim < size_sim_threshold:
                tw, th = t_inst['size']
                cw, ch = c_inst['size']
                score = calculate_change_significance(
                    "resize", comp_type, t_inst=t_inst, c_inst=c_inst, img_size=img_size)
                all_changes.append({
                    "type": "resize",
                    "comp_type": comp_type,
                    "from_size": (tw, th),
                    "to_size": (cw, ch),
                    "similarity": size_sim,
                    "score": score
                })

    # 如果没有检测到变化
    if not all_changes:
        return "No significant changes detected"

    # 按重要性排序并选择最显著的变化
    all_changes.sort(key=lambda x: x["score"], reverse=True)
    top_changes = all_changes[:max_significant_changes]

    # 生成简洁的英文差异描述
    differences = []
    for change in top_changes:
        if change["type"] == "type_add":
            differences.append(f"Added new {change['comp_type']} elements ({change['count']} places)")
        elif change["type"] == "type_remove":
            differences.append(f"Removed all {change['comp_type']} elements")
        elif change["type"] == "count_change":
            diff = change['candidate_count'] - change['target_count']
            if diff > 0:
                differences.append(f"{abs(diff)} additional {change['comp_type']} added")
            else:
                differences.append(f"{abs(diff)} {change['comp_type']} removed")
        elif change["type"] == "add":
            x, y = change["position"]
            differences.append(f"Added {change['comp_type']} at position ({x},{y})")
        elif change["type"] == "remove":
            x, y = change["position"]
            differences.append(f"Removed {change['comp_type']} from position ({x},{y})")
        elif change["type"] == "move":
            tx, ty = change["from_position"]
            cx, cy = change["to_position"]
            differences.append(f"{change['comp_type']} moved from ({tx},{ty}) to ({cx},{cy})")
        elif change["type"] == "resize":
            tw, th = change["from_size"]
            cw, ch = change["to_size"]
            differences.append(f"{change['comp_type']} resized from {tw}x{th} to {cw}x{ch}")

    # 生成自然语言描述
    if not differences:
        return "No changes"

    if len(differences) == 1:
        return differences[0]

    return "Changes: " + "; ".join(differences[:-1]) + " and " + differences[-1]


def get_image_size(image_path):
    """Get image dimensions"""
    img = cv2.imread(str(image_path))
    if img is not None:
        return img.shape[1], img.shape[0]  # (width, height)
    return (1440, 2560)  # default dimensions


def process_item(item):
    """Process single image pair, keep only top 3 significant changes"""
    target_id = item.get("target")
    candidate_id = item.get("candidate")

    if not target_id or not candidate_id:
        print(f"Invalid item format: {item}")
        item["captions"] = "Invalid item format"
        return item

    target_path = IMAGE_DIR / f"{target_id}.png"
    target_caption = ""
    img_size = (1440, 2560)  # default dimensions

    if target_path.exists():
        img_size = get_image_size(target_path)
        target_caption = analyze_image(str(target_path))
    else:
        print(f"Target image not found: {target_path}")
        target_caption = "Target image not found"

    candidate_path = IMAGE_DIR / f"{candidate_id}.png"
    candidate_caption = ""
    if candidate_path.exists():
        candidate_caption = analyze_image(str(candidate_path))
    else:
        print(f"Candidate image not found: {candidate_path}")
        candidate_caption = "Candidate image not found"

    # 检查描述是否有效
    if "not found" in target_caption or "not found" in candidate_caption:
        item["captions"] = "Comparison failed: missing image"
        return item

    if target_caption and candidate_caption:
        # 使用新的比较函数，只保留3个最显著变化
        differences = compare_ui_descriptions(
            target_caption,
            candidate_caption,
            img_size=img_size,
            max_significant_changes=3
        )
        item["captions"] = differences
    else:
        print("One or both descriptions are empty, cannot compare")
        item["captions"] = "No changes detected"

    return item


def process_all_pairs(json_file_path):
    """Process all image pairs in JSON file"""
    try:
        # 创建备份文件
        backup_path = json_file_path.with_suffix(".backup.json")
        if json_file_path.exists():
            json_file_path.rename(backup_path)
        else:
            print(f"JSON file not found: {json_file_path}")
            return

        with open(backup_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            print("JSON file is empty")
            return

        # 创建临时文件路径
        temp_path = json_file_path.with_suffix(".temp.json")

        try:
            processed_data = []  # Store all processed data
            for index, item in enumerate(tqdm(data, desc=f"Processing {json_file_path.name}")):
                processed_item = process_item(item)
                processed_data.append(processed_item)

                # Print differences for current item
                if "captions" in processed_item:
                    print(f"Item {index + 1} differences: {processed_item['captions']}")

            # Write all processed data to temp file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)

            # Rename temp file to original if successful
            if temp_path.exists():
                temp_path.rename(json_file_path)

        except Exception as e:
            print(f"Processing error: {e}")
            # On error, delete temp and restore backup
            if temp_path.exists():
                temp_path.unlink()
            if backup_path.exists():
                backup_path.rename(json_file_path)
            return

        # Delete backup if exists
        if backup_path.exists():
            backup_path.unlink()

    except Exception as e:
        print(f"JSON processing error: {e}")
        # Restore backup on error
        if backup_path.exists():
            backup_path.rename(json_file_path)
        return


def main():
    json_files = [
        "cap.bare.test.json", "cap.bare.train.json", "cap.bare.val.json",
        "cap.form.test.json", "cap.form.train.json", "cap.form.val.json",
        "cap.gallery.test.json", "cap.gallery.train.json", "cap.gallery.val.json",
        "cap.list.test.json", "cap.list.train.json", "cap.list.val.json",
        "cap.login.test.json", "cap.login.train.json", "cap.login.val.json",
        "cap.news.test.json", "cap.news.train.json", "cap.news.val.json",
        "cap.profile.test.json", "cap.profile.train.json", "cap.profile.val.json",
        "cap.search.test.json", "cap.search.train.json", "cap.search.val.json",
        "cap.settings.test.json", "cap.settings.train.json", "cap.settings.val.json",
        "cap.terms.test.json", "cap.terms.train.json", "cap.terms.val.json"
    ]

    for json_file_name in json_files:
        json_file_path = JSON_DIR / json_file_name

        if json_file_path.exists():
            print(f"\nProcessing JSON file: {json_file_name}...")
            process_all_pairs(json_file_path)
        else:
            print(f"JSON file not found: {json_file_name}")


if __name__ == "__main__":
    main()