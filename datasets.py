import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from utils.sdf import compute_class_sdf
from utils.augment import (AugmentConfig, apply_isometric, copy_paste,
                           mosaic9_compose, photometric_gain,
                           random_res_scale, recompute_sdf)


IMG_EXTS = ('.jpg', '.png', '.tif', '.tiff')


class FullDataset(Dataset):
    """语义分割训练/验证数据集（标签单通道 PNG/TIF，{0=背景,1..C-1}，
    num_classes 可为 2（二值）或任意多分类）。

    - 标签 Resize 用 NEAREST（原版对 label 用 BICUBIC 会引入类别索引
      之外的插值值，属于隐患，此处修正）
    - label_mapping：公共数据集标签值重映射（LUT 方式），如二值
      0/255 数据集传 {255: 1}，ISPRS 杂乱类传 {6: 0}；初始化时
      抽样校验映射后取值均在 [0, num_classes)，越界即报错
    - SDF 逐类归一化预计算，缓存在内存 RAM（附录 Q21），
      与图像/标签同步 Resize + 翻转（bilinear / 整轴翻转）
    - train / val 模式计算并缓存 SDF（val 也需边界损失项做验证损失），
      test 模式跳过；增广仅 train 模式启用
    - 动态增强（附录 Q89，augment=AugmentConfig 时启用）：
      Mosaic9（9 宫格拼接，YOLOv4 风格；默认全程开、最后 close_epochs 轮
      自动关闭；可配 mosaic_open_epoch 延后首开（前 N-1 轮干净热身）；
      train 循环每轮调 set_epoch(epoch, total) 传播调度）→ 稀有类连通域 Copy-Paste
      （Simple Copy-Paste, CVPR 2021）→ 1.25×/0.8× 分辨率模拟（核心设计）→
      SDF 由组合标签重算（一致性不变量）→ 翻转/rot90（等距，SDF
      同步变换）→ 光度增益。augment=None 保留旧行为（仅随机翻转）
    - image_list / gt_list 支持从同一目录确定性切分子集
      （split.json 三划分清单模式）
    """

    def __init__(self, image_root, gt_root, size, mode='train', num_classes=3,
                 image_list=None, gt_list=None, label_mapping=None,
                 augment: AugmentConfig = None):
        if image_list is None:
            self.images = sorted(
                os.path.join(image_root, f) for f in os.listdir(image_root)
                if f.lower().endswith(IMG_EXTS))
            self.gts = sorted(
                os.path.join(gt_root, f) for f in os.listdir(gt_root)
                if f.lower().endswith(IMG_EXTS))
        else:
            self.images, self.gts = list(image_list), list(gt_list)
        assert len(self.images) == len(self.gts), \
            f"images({len(self.images)}) and gts({len(self.gts)}) mismatch"
        self.size = size
        self.mode = mode
        self.num_classes = num_classes
        self.augment = augment
        self._epoch = 0
        self._total_epochs = None

        self.label_lut = None
        if label_mapping:
            lut = np.arange(256, dtype=np.uint8)
            for k, v in label_mapping.items():
                lut[k] = v
            self.label_lut = lut

        self._validate_labels()
        self.sdf_cache = None
        self._tiles_with_class = {}
        if mode in ('train', 'val'):
            sdf_cache = []
            for i, p in enumerate(self.gts):
                arr = self._gt_array(p)
                sdf_cache.append(compute_class_sdf(arr, num_classes))
                if mode == 'train':
                    # Copy-Paste 稀有类索引：类别 → 含该类的瓦片下标
                    for c in np.unique(arr):
                        self._tiles_with_class.setdefault(
                            int(c), []).append(i)
            self.sdf_cache = sdf_cache

    def set_epoch(self, epoch, total_epochs=None):
        """每轮训练前由 train.py 调用，传播 Mosaic 关闭调度
        （DataLoader 默认每轮重建 worker，状态随之传播）。"""
        self._epoch = epoch
        self._total_epochs = total_epochs

    def _mosaic_active(self):
        if not (self.augment and self.augment.mosaic):
            return False
        if self._total_epochs is None:
            return True
        # 首开轮（1-based，默认 1=全程开）≤ 当前轮 < 总轮数 − 关闭轮数
        return (self.augment.mosaic_open_epoch - 1 <= self._epoch
                < self._total_epochs - self.augment.mosaic_close_epochs)

    def _gt_array(self, path):
        """读取标签并应用重映射，返回 (H, W) uint8 numpy。"""
        with open(path, 'rb') as f:
            arr = np.array(Image.open(f).convert('L'))
        if self.label_lut is not None:
            arr = self.label_lut[arr]
        return arr

    def _validate_labels(self):
        """抽样校验标签取值域（系统边界校验，前 16 张）。"""
        for p in self.gts[:16]:
            arr = self._gt_array(p)
            vmax = int(arr.max())
            if vmax >= self.num_classes:
                found = np.unique(arr[:200, :200]).tolist()
                raise ValueError(
                    f"标签值 {vmax} 超出 [0, {self.num_classes})，"
                    f"首张抽样取值 {found}。若为 0/255 二值数据集请传 "
                    f"label_mapping={{255: 1}} 并设 num_classes=2；"
                    f"多分类杂乱/忽略类同理映射到 0")

    def _load_pair(self, idx):
        """供 Mosaic/Copy-Paste 使用的对齐样本（图像 [0,1] 张量 +
        NEAREST 缩放标签），不带任何随机增广。"""
        image = TF.to_tensor(
            TF.resize(self._rgb_loader(self.images[idx]),
                      (self.size, self.size)))
        return image, self._label_np(idx)

    def _label_np(self, idx):
        label = Image.fromarray(self._gt_array(self.gts[idx]))
        label = TF.resize(label, (self.size, self.size),
                          interpolation=TF.InterpolationMode.NEAREST)
        return np.array(label, dtype=np.int64)

    def _cached_sdf(self, idx):
        if self.sdf_cache is None:
            return None
        sdf = torch.from_numpy(self.sdf_cache[idx])
        if sdf.shape[-2:] != (self.size, self.size):
            sdf = F.interpolate(sdf.unsqueeze(0),
                                size=(self.size, self.size),
                                mode='bilinear', align_corners=False)[0]
        return sdf

    def _apply_copy_paste(self, image, label, rng):
        """稀有类选择按 1/出现瓦片数逆频率加权（越稀有越常被粘贴）。"""
        classes = [c for c in range(1, self.num_classes)
                   if self._tiles_with_class.get(c)]
        if not classes:
            return None
        weights = [1.0 / len(self._tiles_with_class[c]) for c in classes]
        cls = rng.choices(classes, weights=weights, k=1)[0]
        donor_idx = rng.choice(self._tiles_with_class[cls])
        donor_img, donor_label = self._load_pair(donor_idx)
        return copy_paste(image, label, donor_img, donor_label, cls, rng,
                          min_area=self.augment.copy_paste_min_area)

    def _augment(self, idx, image, label):
        aug = self.augment
        rng = random
        dirty = False
        if self._mosaic_active() and rng.random() < aug.mosaic_prob:
            pairs = [(image, label)]
            pairs += [self._load_pair(rng.randrange(len(self)))
                      for _ in range(8)]
            image, label = mosaic9_compose(pairs, self.size, rng,
                                           aug.mosaic_scale)
            dirty = True
        if aug.copy_paste and rng.random() < aug.copy_paste_prob:
            out = self._apply_copy_paste(image, label, rng)
            if out is not None:
                image, label = out
                dirty = True
        if aug.res_scale and rng.random() < aug.res_scale_prob:
            image, label = random_res_scale(image, label, rng,
                                            aug.res_up, aug.res_down)
            dirty = True
        # 复合增广后 SDF 一律重算（SDF-标签一致性不变量）
        sdf = recompute_sdf(label, self.num_classes) if dirty \
            else self._cached_sdf(idx)
        image, label, sdf = apply_isometric(image, label, sdf, rng,
                                            rot90=aug.rot90)
        if aug.photo_metric:
            image = photometric_gain(image, rng, aug.photo_gain)
        return image, label, sdf

    def __getitem__(self, idx):
        image, label = self._load_pair(idx)

        if self.mode == 'train':
            if self.augment is None:
                sdf = self._cached_sdf(idx)
                image, label, sdf = apply_isometric(image, label, sdf,
                                                     rng=random, rot90=False)
            else:
                image, label, sdf = self._augment(idx, image, label)
        else:
            sdf = self._cached_sdf(idx)

        image = TF.normalize(image, [0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        return {'image': image, 'label': torch.from_numpy(label).unsqueeze(0),
                'sdf': sdf}

    @staticmethod
    def _rgb_loader(path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def __len__(self):
        return len(self.images)


class TestDataset:
    """推理数据集（无需 SDF）。"""

    def __init__(self, image_root, size):
        self.images = sorted(
            os.path.join(image_root, f) for f in os.listdir(image_root)
            if f.lower().endswith(IMG_EXTS))
        self.size = size
        self.index = 0

    def load_data(self):
        import torchvision.transforms as T
        with open(self.images[self.index], 'rb') as f:
            image = Image.open(f).convert('RGB')
        transform = T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        name = os.path.basename(self.images[self.index])
        self.index += 1
        return transform(image).unsqueeze(0), name

    def __len__(self):
        return len(self.images)
