import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def compute_class_sdf(mask, num_classes=3):
    """预计算逐类归一化 SDF（附录 Q20/Q30）。

    对每个类别 c 独立计算有符号距离函数：
      phi = distance_transform_edt(1 - mask_c)   # 背景到前景距离
          - distance_transform_edt(mask_c)       # 前景到背景距离
      负 = 类别内部，0 = 边界，正 = 类别外部（Kervadec 约定）
    逐类独立归一化到 [-1, 1]，保证等权平均 /num_classes 时
    大类别（背景）与小类别量级一致（附录 Q30）。

    Args:
        mask: (H, W) numpy，整型标签 {0, 1, ..., C-1}
        num_classes: 类别数
    Returns:
        sdf: (num_classes, H, W) float32，已归一化
    """
    sdf = np.zeros((num_classes,) + mask.shape, dtype=np.float32)
    for c in range(num_classes):
        mask_c = (mask == c).astype(np.uint8)
        area = int(mask_c.sum())
        if area == 0:
            # 类全空：处处在类别外部（phi=+1）。显式处理：scipy 对
            # 全 1 输入返回以数组边缘为锚的伪距离场（有限但语义错误，
            # 会注入边缘结构偏置），公共数据集大量"无目标瓦片"命中此情形
            sdf[c] = 1.0
            continue
        if area == mask_c.size:
            # 类全满：处处在类别内部（phi=-1），同理显式处理
            sdf[c] = -1.0
            continue
        pos_dist = distance_transform_edt(mask_c)
        neg_dist = distance_transform_edt(1 - mask_c)
        phi = neg_dist - pos_dist
        sdf[c] = phi / max(np.abs(phi).max(), 1e-6)
    return sdf
