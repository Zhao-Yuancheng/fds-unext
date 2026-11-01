"""动态数据增强套件（附录 Q89）。

研究依据与设计决策：
- Mosaic9（YOLOv9）：9 图 3×3 网格 + 灰底(114)填充 + 随机裁剪；
  分割适配为"瓦片限制在各自网格单元内、尺寸在 scale_range 内随机"，
  避免检测版重叠粘贴对密集标签造成的物体切断伪影
  （网格口径与 Improved YOLOv10 论文的 Mosaic9 描述一致）。
  训练最后 close_epochs 轮自动关闭（ultralytics close_mosaic=10 默认
  惯例：末期在真实数据分布下微调，避免马赛克人造特征干扰收敛）。
- 稀有类 Copy-Paste（IGARSS 2024, arXiv 2404.05693）：连通域级实例
  粘贴的语义分割适配，DynamicEarthNet+U-Net 上 mIoU 37.9→44.1；
  类别按 1/出现瓦片数 逆频率加权选择，直接服务瓦片类等稀有类别。
- 分辨率模拟 1.25×/0.8×（本项目核心设计）：上采样=更细 GSD（随机裁回），
  下采样=更粗 GSD（随机位置贴回 + 灰底填充）。
- 光度增强仅用乘性增益（保反照率比值——高/低反照率类别由亮度比定义，
  色相/伽马会破坏类别语义，故默认关闭）。

SDF 一致性不变量：任何改变标签几何的复合操作（Mosaic/粘贴/尺度/填充）
之后 SDF 一律由组合标签重算，保证边界损失的 SDF-标签严格一致；
翻转/旋转为等距变换，SDF 同步变换即精确成立，无需重算。
"""
import random
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from utils.sdf import compute_class_sdf

FILL = 114.0 / 255.0  # YOLO 惯例灰底，填充区域标签为背景


@dataclass
class AugmentConfig:
    mosaic: bool = True
    mosaic_prob: float = 1.0
    mosaic_close_epochs: int = 10      # 最后 N 轮关闭（YOLO close_mosaic）
    mosaic_open_epoch: int = 1         # 首开轮（1-based，默认 1=全程开）
    # (0.7, 1.0)：瓦片占格期望面积 73%（灰底 ~27%）；取 (0.5, 1.0) 时
    # 灰底升至 ~42%，对密集分割的背景先验偏强，不建议
    mosaic_scale: Tuple[float, float] = (0.7, 1.0)
    copy_paste: bool = True
    copy_paste_prob: float = 0.3
    copy_paste_min_area: int = 16      # 过小连通域不粘贴（噪声级碎片）
    res_scale: bool = True
    res_scale_prob: float = 0.5
    res_up: float = 1.25
    res_down: float = 0.8
    rot90: bool = True
    photo_metric: bool = False
    photo_gain: Tuple[float, float] = (0.9, 1.1)

    def summary(self, total_epochs=None):
        sched = ""
        if total_epochs is not None:
            if self.mosaic_open_epoch > 1:
                sched = (f"，epoch {self.mosaic_open_epoch} 起启用、epoch "
                         f"{total_epochs - self.mosaic_close_epochs} 起关闭")
            else:
                sched = f"，epoch {total_epochs - self.mosaic_close_epochs} 起关闭"
        return (f"mosaic={self.mosaic}(p={self.mosaic_prob}, "
                f"scale={self.mosaic_scale}{sched}) "
                f"copy_paste={self.copy_paste}(p={self.copy_paste_prob}) "
                f"res_scale={self.res_scale}(p={self.res_scale_prob}, "
                f"{self.res_down}/{self.res_up}x) rot90={self.rot90} "
                f"photo={self.photo_metric}")


def _resize_label(label, size):
    """标签 NEAREST 缩放，返回 (size, size) int64 numpy。"""
    t = torch.from_numpy(np.ascontiguousarray(label))[None, None].float()
    return (F.interpolate(t, size=size, mode='nearest')[0, 0]
            .to(torch.int64).numpy())


def mosaic9_compose(pairs, size, rng=random, scale_range=(0.5, 1.0)):
    """9 对 (img, label) 组合为 3×3 画布后随机裁剪 size×size。

    每瓦片缩放到 uniform(scale_range)×size，随机放置于各自网格单元内
    （不越界、不重叠），空隙填灰底（标签=背景）。
    """
    canvas_img = torch.full((3, size * 3, size * 3), FILL,
                            dtype=torch.float32)
    canvas_lab = np.zeros((size * 3, size * 3), dtype=np.int64)
    for k, (img, lab) in enumerate(pairs):
        ts = max(1, int(round(size * rng.uniform(*scale_range))))
        img_r = F.interpolate(img[None], size=(ts, ts), mode='bilinear',
                              align_corners=False)[0]
        lab_r = _resize_label(lab, (ts, ts))
        ci, cj = divmod(k, 3)
        y0 = ci * size + rng.randint(0, size - ts)
        x0 = cj * size + rng.randint(0, size - ts)
        canvas_img[:, y0:y0 + ts, x0:x0 + ts] = img_r
        canvas_lab[y0:y0 + ts, x0:x0 + ts] = lab_r
    ty, tx = rng.randint(0, 2 * size), rng.randint(0, 2 * size)
    return (canvas_img[:, ty:ty + size, tx:tx + size],
            np.ascontiguousarray(canvas_lab[ty:ty + size, tx:tx + size]))


def copy_paste(image, label, donor_img, donor_label, cls, rng=random,
               min_area=16):
    """连通域级稀有类粘贴（IGARSS 2024 Cut-and-Paste 分割适配）。

    从 donor 标签提取 cls 的随机连通域（面积 ≥ min_area），仅粘贴该
    连通域的像素（非包围盒），不破坏目标图其余内容。返回新数组，
    不修改入参；无有效连通域时返回 None。
    """
    comp, n = ndimage.label(donor_label == cls)
    if n == 0:
        return None
    areas = np.bincount(comp.ravel())
    valid = np.nonzero(areas[1:] >= min_area)[0] + 1
    if len(valid) == 0:
        return None
    k = int(rng.choice(valid))
    ys, xs = np.nonzero(comp == k)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    H, W = label.shape
    if h > H or w > W:
        return None
    image = image.clone()
    label = label.copy()
    ty, tx = rng.randint(0, H - h), rng.randint(0, W - w)
    m = torch.from_numpy(comp[y0:y1, x0:x1] == k)
    image[:, ty:ty + h, tx:tx + w][:, m] = \
        donor_img[:, y0:y1, x0:x1][:, m]
    label[ty:ty + h, tx:tx + w][m] = cls
    return image, label


def random_res_scale(image, label, rng=random, up=1.25, down=0.8):
    """分辨率模拟（核心设计）：上采样=细 GSD 随机裁回，下采样=粗 GSD
    随机位置贴回灰底画布。等概率二选一。"""
    size = image.shape[-1]
    if rng.random() < 0.5:
        us = int(round(size * up))
        img = F.interpolate(image[None], size=(us, us), mode='bilinear',
                            align_corners=False)[0]
        lab = _resize_label(label, (us, us))
        y0, x0 = rng.randint(0, us - size), rng.randint(0, us - size)
        return (img[:, y0:y0 + size, x0:x0 + size],
                np.ascontiguousarray(lab[y0:y0 + size, x0:x0 + size]))
    ds = int(round(size * down))
    img = F.interpolate(image[None], size=(ds, ds), mode='bilinear',
                        align_corners=False)[0]
    lab = _resize_label(label, (ds, ds))
    canvas_img = torch.full((3, size, size), FILL, dtype=torch.float32)
    canvas_lab = np.zeros((size, size), dtype=np.int64)
    y0, x0 = rng.randint(0, size - ds), rng.randint(0, size - ds)
    canvas_img[:, y0:y0 + ds, x0:x0 + ds] = img
    canvas_lab[y0:y0 + ds, x0:x0 + ds] = lab
    return canvas_img, canvas_lab


def photometric_gain(image, rng=random, gain_range=(0.9, 1.1)):
    """乘性增益光度增强：高/低反照率类别由亮度比定义，乘性增益
    保持任意两像素亮度比值不变，类别语义无损（色相/伽马则不然）。"""
    return (image * rng.uniform(*gain_range)).clamp(0.0, 1.0)


def recompute_sdf(label, num_classes):
    """组合标签 → 逐类归一化 SDF（复合增广后的 SDF-标签一致性重算）。"""
    return torch.from_numpy(
        compute_class_sdf(label.astype(np.int64), num_classes))


def apply_isometric(image, label, sdf, rng=random, rot90=False):
    """随机水平/垂直翻转 +（可选）随机 90° 旋转。等距变换下 SDF
    同步变换即精确成立（距离对翻转/旋转不变），无需重算。"""
    if rng.random() < 0.5:
        image = torch.flip(image, dims=[-1])
        label = np.flip(label, -1)
        if sdf is not None:
            sdf = torch.flip(sdf, dims=[-1])
    if rng.random() < 0.5:
        image = torch.flip(image, dims=[-2])
        label = np.flip(label, -2)
        if sdf is not None:
            sdf = torch.flip(sdf, dims=[-2])
    if rot90:
        k = rng.randrange(4)
        if k:
            image = torch.rot90(image, k, dims=(1, 2))
            label = np.rot90(label, k)
            if sdf is not None:
                sdf = torch.rot90(sdf, k, dims=(1, 2))
    return image, np.ascontiguousarray(label), sdf
