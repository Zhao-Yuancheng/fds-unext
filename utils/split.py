"""三划分清单（train/val/test）的生成与持久化。

设计要点（附录 Q81）：
- 清单只生成一次（split.json），训练与评估全程复用同一份，保证
  多次实验/消融组严格同一切分；换数据集需删除清单重建
- 切分单位抽象为 unit：默认单瓦片；group_pattern 提供时为"景"
  （正则捕获组聚合的瓦片组），整组进入同一边——多景瓦片数据防
  跨景空间自相关泄漏
- 种子随机洗牌：同 seed + 同输入 → 完全相同的划分；比例作用于
  unit 数（景级切分时瓦片级比例为近似值）
"""

import json
import os
import random
import re


def create_split(images, gts, ratios=(0.7, 0.1, 0.2), seed=42,
                 group_pattern=None):
    """生成 train/val/test 三划分。

    Args:
        images / gts: 等长路径列表
        ratios: (train, val, test) 比例，自动归一化
        seed: 切分随机种子
        group_pattern: 景级切分的文件名正则（自文件名起始 match，
            第 1 个捕获组为景标识，如 r'(scene\\d+)_t\\d+'）；
            None 时逐瓦片切分
    Returns:
        dict: {'seed', 'ratios', 'group_pattern',
               'train'/'val'/'test': {'images': [...], 'gts': [...]}}
    """
    assert len(images) == len(gts)
    total = sum(ratios)
    fr = [r / total for r in ratios]
    rng = random.Random(seed)

    if group_pattern:
        pat = re.compile(group_pattern)
        groups = {}
        for im, gt in zip(images, gts):
            m = pat.match(os.path.basename(im))
            if not m or not m.groups():
                raise ValueError(f"group_pattern 无法从文件名提取捕获组: {im}")
            groups.setdefault(m.group(1), []).append((im, gt))
        units = list(groups.values())
    else:
        units = [[p] for p in sorted(zip(images, gts))]
    rng.shuffle(units)

    n = len(units)
    n_train = round(n * fr[0])
    n_val = round(n * fr[1])
    if n >= 3:   # 保证三边非空（优先保 train 上限）
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)

    def flat(us):
        return [p for u in us for p in u]

    parts = {'train': flat(units[:n_train]),
             'val': flat(units[n_train:n_train + n_val]),
             'test': flat(units[n_train + n_val:])}
    return {'seed': seed, 'ratios': list(ratios),
            'group_pattern': group_pattern,
            **{k: {'images': [p[0] for p in v], 'gts': [p[1] for p in v]}
               for k, v in parts.items()}}


def save_split(split, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=1, ensure_ascii=False)


def load_split(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
