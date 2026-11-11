"""对比模型统一训练/评估脚本（README §7 优化版，附录 Q111）。

设计要点（与主项目严格对齐，保证对比公平）
------------------------------------------
1. **同一数据管线**：复用本项目 `datasets.FullDataset`——同一 256→1024
   resize、同一 NEAREST 标签、同一 ImageNet 归一化；复用主 run 的
   split.json（seed 42 省级切分，与 FDS-UNeXT 逐条一致）。
2. **同一评估代码**：复用 `utils.metrics`（ConfusionMatrix 同口径导出
   mIoU/OA/FWIoU/Kappa/mF1；BF-Score 同实现，公共集容差 8px）。
3. **同预算**：默认 30 epoch（与主 run 一致），val mIoU 选 best，
   test 仅评一次。
4. **增广公平性**：对比模型仅随机翻转（= 上游基线 E0' 的原版口径，
   FlipTrainSet 与 augment=None 同分布）；本项目的 Mosaic9/Copy-Paste/
   分辨率模拟套件是方法贡献的一部分，不入对比（其增益由 A 系列消融
   单独量化）。裁剪类增广一律不用——瓦片即样本（256 原生），随机裁剪
   会破坏整瓦片协议。
5. **模型**：smp（U-Net/U-Net++/DeepLabV3+/SegFormer）+ 自实现
   Attention U-Net (ResNet34) 与 HRNet-W32+FCN，全部 ImageNet 预训练
   编码器、端到端训练（Go 对比模型不冻结、不加 SDF/边界监督）。

用法
----
    python train_compare.py --model unet_r34            # 完整训练+test
    python train_compare.py --model unet_r34 --limit 160 --epoch 1  # 冒烟
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

import timm
import segmentation_models_pytorch as smp

from datasets import FullDataset
from models.unetformer import UNetFormer
from models.segnext import SegNeXt
from utils.augment import (AugmentConfig, apply_isometric, mosaic9_compose,
                           photometric_gain, random_res_scale)
from utils.metrics import ConfusionMatrix, boundary_f_score


# ---------------------------------------------------------------- 模型定义
class _ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class AttentionGate(nn.Module):
    """Attention U-Net（Oktay 2018）注意力门：g=粗分辨率门控信号，
    x=编码器 skip；输出 x 的软注意力加权。"""

    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False),
                                 nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_x, f_int, 1, bias=False),
                                 nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=False),
                                 nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.psi(self.relu(self.w_g(g) + self.w_x(x)))
        return x * a


class _UpAG(nn.Module):
    def __init__(self, c_dec, c_skip, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=False)
        self.ag = AttentionGate(c_dec, c_skip, max(c_skip // 2, 8))
        self.conv = nn.Sequential(_ConvBN(c_dec + c_skip, c_out),
                                  _ConvBN(c_out, c_out))

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, self.ag(g=x, x=skip)], dim=1))


class AttentionUNet(nn.Module):
    """Attention U-Net（ResNet34 编码器 + 注意力门跳跃连接）。

    pretrained=False 供纯评估路径使用（ckpt 自带全部权重，离线集群
    不必连 HF Hub 下载 ImageNet 预训练，2026-11-05 gpu1 断链教训）。
    """

    def __init__(self, num_classes, encoder='resnet34', pretrained=True):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=pretrained,
                                     features_only=True,
                                     out_indices=(0, 1, 2, 3, 4))
        c = self.enc.feature_info.channels()      # /2,/4,/8,/16,/32
        self.d4 = _UpAG(c[4], c[3], 256)
        self.d3 = _UpAG(256, c[2], 128)
        self.d2 = _UpAG(128, c[1], 64)
        self.d1 = _UpAG(64, c[0], 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        f = self.enc(x)
        y = self.d4(f[4], f[3])
        y = self.d3(y, f[2])
        y = self.d2(y, f[1])
        y = self.d1(y, f[0])
        y = F.interpolate(y, scale_factor=2, mode='bilinear',
                          align_corners=False)
        return self.head(y)


class HRNetFCN(nn.Module):
    """HRNet-W32 + FCN 头（官方 HRNet-seg 的 HRNetV2 表示口径）。

    timm 的 HighResolutionNet.forward_features 会把四分支合并到 /32 的
    2048ch 分类特征（不能直接用于分割）；分割应使用 **HRNetV2 表示** =
    四分支（/4,/8,/16,/32）上采样至 /4 后拼接（w32 → 480ch）。此处
    复刻其 stem + stages 调用取分支输出，再按官方 FCN 头（1×1 conv +
    双线性上采样）出全分辨率 logits。"""

    def __init__(self, num_classes, variant='hrnet_w32', pretrained=True):
        super().__init__()
        # pretrained=False 供纯评估路径（同 AttentionUNet 注释）
        self.body = timm.create_model(variant, pretrained=pretrained)
        with torch.no_grad():          # 探测 HRNetV2 拼接通道数
            fl = self._features(torch.zeros(1, 3, 256, 256))
        self.head = nn.Conv2d(sum(f.shape[1] for f in fl), num_classes, 1)

    def _features(self, x):
        b = self.body
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.act1(x)
        x = b.conv2(x)
        x = b.bn2(x)
        x = b.act2(x)
        return b.stages(x)             # [f1(/4), f2(/8), f3(/16), f4(/32)]

    def forward(self, x):
        fl = self._features(x)
        size = fl[0].shape[-2:]
        y = torch.cat([F.interpolate(f, size=size, mode='bilinear',
                                     align_corners=False) for f in fl], dim=1)
        y = self.head(y)
        return F.interpolate(y, size=x.shape[-2:], mode='bilinear',
                             align_corners=False)


MODEL_ZOO = {
    'unet_r34': dict(desc='U-Net (ResNet34)', lr=1e-4, batch=4, wd=1e-4),
    'unetpp_r34': dict(desc='U-Net++ (ResNet34)', lr=1e-4, batch=4, wd=1e-4),
    'attunet_r34': dict(desc='Attention U-Net (ResNet34)', lr=1e-4,
                        batch=4, wd=1e-4),
    'deeplabv3plus_r50': dict(desc='DeepLabV3+ (ResNet50)', lr=1e-4,
                              batch=4, wd=1e-4),
    # SegFormer 原论文（ADE20K，AdamW）：lr 6e-5 / wd 0.01
    'segformer_b2': dict(desc='SegFormer-B2 (mit_b2)', lr=6e-5, batch=4,
                         wd=1e-2),
    'hrnet_w32': dict(desc='HRNet-W32 + FCN head', lr=1e-4, batch=2, wd=1e-4),
    # UNetFormer（Wang et al., 2022, ResNet18 主干）；协议与其他 CNN 对照一致
    'unetformer_r18': dict(desc='UNetFormer (ResNet18)', lr=1e-4, batch=4,
                           wd=1e-4),
    # SegNeXt-T：官方 ADE20K 配方 lr 6e-5 / wd 1e-2；batch 4 与 SegFormer 对齐
    'segnext_t': dict(desc='SegNeXt-T (MSCAN-T)', lr=6e-5, batch=4,
                      wd=1e-2),
}


def build_model(name, num_classes=4, abl=False):
    """对比模型工厂。abl=True 时给 SegFormer 挂边界头（P2 边界感知基线）。"""
    if name == 'unet_r34':
        return smp.Unet('resnet34', encoder_weights='imagenet',
                        classes=num_classes)
    if name == 'unetpp_r34':
        return smp.UnetPlusPlus('resnet34', encoder_weights='imagenet',
                                classes=num_classes)
    if name == 'attunet_r34':
        return AttentionUNet(num_classes)
    if name == 'deeplabv3plus_r50':
        return smp.DeepLabV3Plus('resnet50', encoder_weights='imagenet',
                                 classes=num_classes)
    if name == 'segformer_b2':
        m = smp.Segformer('mit_b2', encoder_weights='imagenet',
                          classes=num_classes)
        return SegFormerABL(m, num_classes) if abl else m
    if name == 'hrnet_w32':
        return HRNetFCN(num_classes)
    if name == 'unetformer_r18':
        return UNetFormer(num_classes=num_classes, backbone_name='resnet18',
                          pretrained=True)
    if name == 'segnext_t':
        ckpt = None
        for cand in ('weights/segnext_t_ade20k.pth',
                     'weights/segnext_t_geonusaf.pth',
                     'weights/mscan_t.pth',
                     'pretrained/mscan_t.pth'):
            if os.path.isfile(cand):
                ckpt = cand
                break
        if ckpt is None:
            print('[segnext] WARNING: no pretrained checkpoint found; '
                  'this run would not be a fair pretrained baseline',
                  flush=True)
        return SegNeXt(num_classes=num_classes, pretrained=ckpt)
    raise SystemExit(f"未知模型 {name}；可选：{list(MODEL_ZOO)}")


class SegFormerABL(nn.Module):
    """SegFormer-B2 + 轻量边界头（P2 边界感知基线，2026-11-05）。

    动机（评审 R1-MAJOR）：论文主张边界优越，但 7 个比较器全是区域导向模型，
    从未与边界感知设计对照。本包装给最强对照 SegFormer-B2 加上边界监督，
    使"架构优势"与"是否做边界监督"两个因素可分离。

    设计（ABL 思想的可行简化，引用 wang2022abl）：
      - 边界头 bhead 从主 logits 派生 1 通道边界 logit（不侵入 smp 内部特征，
        计算量可忽略：1x1 -> 3x3 -> 1x1）
      - 边界 GT = 逐类 3x3 膨胀 − 腐蚀 的并集（与 FDS 的 boundary_head_loss 同定义）
      - 损失 = BCEWithLogits(bhead, GT_b) 按 **到 GT 边界的切比雪夫距离加权**
        （w = 1 + alpha * d/d_max，上限 20 步）：离真值边界越远的预测边界
        单元被惩罚越重，即 ABL "把错位边界拉回"的核心机制
      - 这是 re-implementation，不是原论文逐行复刻；论文中如实标注
    """

    def __init__(self, segformer, num_classes):
        super().__init__()
        self.seg = segformer
        self.bhead = nn.Sequential(
            nn.Conv2d(num_classes, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))

    def forward(self, x):
        logits = self.seg(x)
        return logits, self.bhead(logits)


def boundary_band(target, kernel_size=3):
    """逐类 3x3 膨胀-腐蚀边界带（与 FDS 的 extract_boundary 同定义）。"""
    nc = int(target.max().item()) + 1 if target.numel() else 1
    nc = max(nc, 2)
    t = target.squeeze(1).long()
    onehot = F.one_hot(t, nc).permute(0, 3, 1, 2).float()
    k = dict(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    dilated = F.max_pool2d(onehot, **k)
    eroded = -F.max_pool2d(-onehot, **k)
    return ((dilated - eroded).sum(dim=1, keepdim=True) > 0).float()


def dist_to_band(band, iters=20):
    """到边界带的切比雪夫距离（以 3x3 膨胀步数计，上限 iters）。

    用迭代 max_pool 近似欧氏距离变换：GPU 上无需 scipy/cv2，且只作用于
    无需梯度的 GT，开销很小（20 次 3x3 pooling）。
    """
    dist = torch.zeros_like(band)
    cur = band.clone()
    for i in range(1, iters + 1):
        nxt = F.max_pool2d(cur, 3, 1, 1)
        newly = (nxt > 0) & (cur == 0)
        dist = dist + newly.float() * i
        cur = nxt
        if bool((cur > 0).all()):
            break
    return dist


def abl_loss(bnd_logits, target, alpha=2.0, iters=20):
    """距离加权边界 BCE（ABL 简化实现）。"""
    gt_b = boundary_band(target)
    gt_b = F.interpolate(gt_b, size=bnd_logits.shape[-2:], mode='nearest')
    d = dist_to_band(gt_b, iters=iters)
    w = 1.0 + alpha * d / float(iters)
    return F.binary_cross_entropy_with_logits(bnd_logits, gt_b, weight=w)


# ---------------------------------------------------------------- 工具
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _collate(batch):
    """只保留 image/label —— FullDataset 的 sdf 在 test 模式为 None，
    默认 collate 会拒绝；对比模型不使用 SDF，直接丢弃该字段。"""
    return {'image': torch.stack([b['image'] for b in batch]),
            'label': torch.stack([b['label'] for b in batch])}


class FlipTrainSet(FullDataset):
    """对比模型训练集：mode='test'（跳过 SDF 预计算与缓存——对比模型
    不用 SDF，避免 9972 张 × 逐类 SDF 的 ~10GB RAM 与每样本
    256→1024 SDF 插值开销）+ 随机水平/垂直翻转（与 E0'/原版
    SAM2-UNeXT 的 augment=None 口径同分布：各 p=0.5，python random
    随 worker 播种）。"""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)      # mode='test'：无增广无 SDF
        image, label = item['image'], item['label']
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            label = torch.flip(label, dims=[-1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-2])
            label = torch.flip(label, dims=[-2])
        return {'image': image.contiguous(),
                'label': label.contiguous()}


class AugTrainSet(FullDataset):
    """对比模型完整增强训练集（批 B / P2-2，2026-11-04，评审 DA-C2）。

    与 FlipTrainSet 同为 mode='test' 底座（跳过 SDF 预计算与缓存——对比
    模型不用 SDF，避免逐类 SDF 的 RAM 占用与每样本重算开销），但应用与
    train.py 的 FullDataset(mode='train', augment=...) 完全同分布的动态
    增强管线：Mosaic9 → 稀有类 Copy-Paste → 1.25x/0.8x 分辨率模拟 →
    翻转/rot90 → 光度增益。与 train.py 的唯一差别是不计算/返回 SDF
    （对比模型经 _collate 本就丢弃该字段），增强分布逐字一致。

    注意：FullDataset 只在 mode='train' 时自动构建 Copy-Paste 的稀有类
    索引（_tiles_with_class），这里在 __init__ 手动补建（逐张读 native
    标签取 unique，一次性开销，241 张 1024^2 约数十秒）。
    """

    def __init__(self, image_root, gt_root, size, num_classes,
                 image_list=None, gt_list=None, augment=None):
        super().__init__(image_root, gt_root, size, mode='test',
                         num_classes=num_classes, image_list=image_list,
                         gt_list=gt_list, augment=augment)
        if augment is not None and augment.copy_paste:
            for i, p in enumerate(self.gts):
                for c in np.unique(self._gt_array(p)):
                    self._tiles_with_class.setdefault(int(c), []).append(i)

    def __getitem__(self, idx):
        # 管线与 FullDataset._augment 逐步对应（除 SDF）；rng=random 与
        # train.py 相同（worker 经 seed_worker 播种，随机流同源）
        image, label = self._load_pair(idx)
        aug = self.augment
        rng = random
        if self._mosaic_active() and rng.random() < aug.mosaic_prob:
            pairs = [(image, label)]
            pairs += [self._load_pair(rng.randrange(len(self)))
                      for _ in range(8)]
            image, label = mosaic9_compose(pairs, self.size, rng,
                                           aug.mosaic_scale)
        if aug.copy_paste and rng.random() < aug.copy_paste_prob:
            out = self._apply_copy_paste(image, label, rng)
            if out is not None:
                image, label = out
        if aug.res_scale and rng.random() < aug.res_scale_prob:
            image, label = random_res_scale(image, label, rng,
                                            aug.res_up, aug.res_down)
        # sdf=None：apply_isometric 显式支持（仅翻转/旋转，无 SDF 可转）
        image, label, _ = apply_isometric(image, label, None, rng,
                                          rot90=aug.rot90)
        if aug.photo_metric:
            image = photometric_gain(image, rng, aug.photo_gain)
        image = TF.normalize(image, [0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        return {'image': image.contiguous(),
                'label': torch.from_numpy(
                    np.ascontiguousarray(label)).unsqueeze(0).contiguous()}


def run_pass(model, loader, device, optim=None, amp=False, scaler=None,
             aux_weight=0.3, abl_weight=0.0):
    """单轮训练/验证；复用 ConfusionMatrix（与主项目同指标口径）。
    amp 开启时必须配 GradScaler（fp16 梯度下溢防护）。

    双输出模型的两种情形按通道数区分（2026-11-05 新增 ABL）：
      - extra.shape[1] == 1  -> ABL 边界头，用距离加权 BCE（abl_weight）
      - extra.shape[1] == C  -> UNetFormer 辅助分割头，用 CE（aux_weight）
    """
    train = optim is not None
    model.train(train)
    cm = ConfusionMatrix(loader.dataset.num_classes, device=device)
    loss_sum, n_seen = 0.0, 0
    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        pbar = tqdm(loader, dynamic_ncols=True, leave=False,
                    desc='train' if train else 'val')
        for batch in pbar:
            x = batch['image'].to(device, non_blocking=True)
            t = batch['label'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp):
                out = model(x)
                if isinstance(out, tuple):      # (main, aux/boundary)
                    logits, extra = out
                    loss = F.cross_entropy(logits, t.squeeze(1).long())
                    if extra.shape[1] == 1:
                        if abl_weight > 0:
                            loss = loss + abl_weight * abl_loss(extra, t)
                    elif aux_weight > 0:
                        loss = loss + aux_weight * F.cross_entropy(
                            extra, t.squeeze(1).long())
                else:
                    logits = out
                    loss = F.cross_entropy(logits, t.squeeze(1).long())
            if train:
                optim.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            bs = x.size(0)
            loss_sum += float(loss.item()) * bs
            n_seen += bs
            cm.update_from_logits(logits.detach(), t)
            pbar.set_postfix(loss=f"{loss_sum / n_seen:.4f}")
    return loss_sum / max(n_seen, 1), cm


@torch.no_grad()
def eval_test(model, dataset, device, num_classes, batch, workers,
              bf_tol=8, class_names=None):
    """test 单次评估：全局指标 + 逐类 + BF-Score（与 eval.py 同口径）。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        collate_fn=_collate)
    cm = ConfusionMatrix(num_classes, device=device)
    bf_all = []
    for batch_ in tqdm(loader, dynamic_ncols=True, leave=False, desc='test'):
        x = batch_['image'].to(device, non_blocking=True)
        t = batch_['label'].to(device, non_blocking=True)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        cm.update_from_logits(logits, t)
        pred = logits.argmax(1)
        for i in range(x.size(0)):
            bf_all.append(boundary_f_score(pred[i], t[i].squeeze(0),
                                           num_classes, tolerance=bf_tol))
    s = cm.summary()
    pc = cm.per_class()
    bf = np.array(bf_all, dtype=float)
    bf_mean = np.nanmean(bf, axis=0)
    names = class_names or [f'class{c}' for c in range(num_classes)]
    print(f"\nOA {s['oa']:.4f} | mIoU {s['miou']:.4f} | "
          f"FWIoU {s['fwiou']:.4f} | Kappa {s['kappa']:.4f} | "
          f"mF1 {s['mf1']:.4f}")
    print(f"{'class':<18}{'IoU':>9}{'Prec':>9}{'Recall':>9}{'F1':>9}"
          f"{'BF':>9}")
    for c in range(num_classes):
        print(f"{names[c]:<18}{pc['iou'][c]:>9.4f}{pc['precision'][c]:>9.4f}"
              f"{pc['recall'][c]:>9.4f}{pc['f1'][c]:>9.4f}{bf_mean[c]:>9.4f}")
    print(f"{'macro mean':<18}{s['miou']:>9.4f}{s['mprecision']:>9.4f}"
          f"{s['mrecall']:>9.4f}{s['mf1']:>9.4f}{np.nanmean(bf_mean):>9.4f}")
    return s, pc, bf_mean


def main(args):
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    zoo = MODEL_ZOO[args.model]
    lr = args.lr or zoo['lr']
    batch = args.batch_size or zoo['batch']
    wd = args.weight_decay if args.weight_decay is not None \
        else zoo.get('wd', 1e-4)
    # 显存预探覆盖（Q112）：_probe_memory.py 在本节点实测后写入
    # runs/cmp_batch.json，OOM 的模型自动降 batch（用户显式传参优先）
    probe_path = os.path.join(os.path.dirname(args.split_path) or '.',
                              'cmp_batch.json')
    if args.batch_size is None and os.path.exists(probe_path):
        with open(probe_path, 'r', encoding='utf-8') as f:
            ov = json.load(f)
        if args.model in ov:
            batch = int(ov[args.model])
            print(f"[cmp] 使用显存预探 batch={batch}（{probe_path}）")

    with open(args.split_path, 'r', encoding='utf-8') as f:
        split = json.load(f)

    def sub(part):
        imgs, gts = split[part]['images'], split[part]['gts']
        if args.limit > 0:
            imgs, gts = imgs[:args.limit], gts[:args.limit]
        return imgs, gts

    ti, tg = sub('train')
    vi, vg = sub('val')
    # 训练集：FlipTrainSet = 无 SDF 路径 + 随机翻转（E0' 口径同分布）；
    # AugTrainSet（批 B / P2-2）= 无 SDF 路径 + 与 train.py 同分布的完整
    # 增强——回答评审 DA-C2"对比方法获得同等增强后边际还剩多少"；
    # 验证/测试：mode='test'（干净、无 SDF）
    augment = None
    if args.augment:
        augment = AugmentConfig(
            mosaic=args.mosaic, mosaic_prob=args.mosaic_prob,
            mosaic_close_epochs=args.mosaic_close_epochs,
            mosaic_open_epoch=args.mosaic_open_epoch,
            mosaic_scale=tuple(args.mosaic_scale),
            copy_paste=args.copy_paste, copy_paste_prob=args.copy_paste_prob,
            res_scale=args.res_scale, res_scale_prob=args.res_scale_prob,
            res_up=1.25, res_down=0.8, rot90=args.rot90,
            photo_metric=args.photo_metric,
            photo_gain=tuple(args.photo_gain))
        if args.mosaic_open_epoch > args.epoch - args.mosaic_close_epochs:
            raise SystemExit(
                f"--mosaic_open_epoch({args.mosaic_open_epoch}) 晚于关闭轮"
                f"({args.epoch - args.mosaic_close_epochs})，Mosaic 永不启用")
    if augment is not None:
        train_set = AugTrainSet('', '', 1024, num_classes=args.num_classes,
                                image_list=ti, gt_list=tg, augment=augment)
    else:
        train_set = FlipTrainSet('', '', 1024, mode='test',
                                 num_classes=args.num_classes,
                                 image_list=ti, gt_list=tg)
    val_set = FullDataset('', '', 1024, mode='test',
                          num_classes=args.num_classes,
                          image_list=vi, gt_list=vg)
    print(f"[cmp] model={args.model} ({zoo['desc']}) | lr={lr:g} | "
          f"batch={batch} | wd={wd:g} | amp={args.amp} | "
          f"train: {len(train_set)} | val: {len(val_set)}")
    if augment is not None:
        print(f"[cmp] augment: {augment.summary(args.epoch)}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              generator=g, worker_init_fn=seed_worker,
                              drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=_collate)

    model = build_model(args.model, args.num_classes,
                        abl=args.abl_weight > 0).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cmp] params: total {sum(p.numel() for p in model.parameters())/1e6:.1f}M"
          f" (trainable {n_tr/1e6:.1f}M)")

    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, args.epoch, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    os.makedirs(args.save_path, exist_ok=True)
    best_path = os.path.join(args.save_path, 'best.pth')
    csv_path = os.path.join(args.save_path, 'results.csv')
    csv_f = open(csv_path, 'w', encoding='utf-8')
    csv_f.write('epoch,time_s,lr,train_loss,train_miou,val_loss,val_miou,'
                'val_oa,val_mf1\n')

    best_miou, best_epoch = -1.0, 0
    for epoch in range(args.epoch):
        # 传播 Mosaic 关闭调度（FlipTrainSet 无 augment 时为无害 no-op；
        # DataLoader 每轮重建 worker，主进程状态随之传播）
        train_set.set_epoch(epoch, args.epoch)
        t0 = time.time()
        lr_now = optim.param_groups[0]['lr']
        tr_loss, cm_tr = run_pass(model, train_loader, device, optim=optim,
                                  amp=args.amp, scaler=scaler,
                                  aux_weight=args.aux_weight,
                                  abl_weight=args.abl_weight)
        va_loss, cm_va = run_pass(model, val_loader, device, amp=args.amp,
                                  aux_weight=args.aux_weight,
                                  abl_weight=args.abl_weight)
        sched.step()
        tr, va = cm_tr.summary(), cm_va.summary()
        dt = time.time() - t0
        tag = ''
        if not math.isnan(va['miou']) and va['miou'] > best_miou:
            best_miou, best_epoch = va['miou'], epoch + 1
            torch.save(model.state_dict(), best_path)
            tag = '  *best*'
        print(f"[Epoch {epoch+1:02d}/{args.epoch}] {dt:.1f}s lr {lr_now:.1e}"
              f"  train loss {tr_loss:.4f} mIoU {tr['miou']:.4f} | "
              f"val loss {va_loss:.4f} mIoU {va['miou']:.4f} "
              f"OA {va['oa']:.4f} mF1 {va['mf1']:.4f}{tag}")
        csv_f.write(f"{epoch+1},{dt:.1f},{lr_now:.2e},{tr_loss:.6f},"
                    f"{tr['miou']:.6f},{va_loss:.6f},{va['miou']:.6f},"
                    f"{va['oa']:.6f},{va['mf1']:.6f}\n")
        csv_f.flush()
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epoch:
            p = os.path.join(args.save_path, f'snapshot-{epoch+1}.pth')
            torch.save(model.state_dict(), p)
    csv_f.close()
    print(f"[cmp] best val mIoU {best_miou:.4f} (epoch {best_epoch}) -> "
          f"{best_path}")

    if args.eval_test:
        tei, teg = sub('test')
        test_set = FullDataset('', '', 1024, mode='test',
                               num_classes=args.num_classes,
                               image_list=tei, gt_list=teg)
        print(f"[cmp] loading best for test eval ({len(test_set)} imgs)")
        model.load_state_dict(torch.load(best_path, map_location=device))
        s, pc, bf = eval_test(model, test_set, device, args.num_classes,
                              batch, args.num_workers, bf_tol=args.bf_tolerance,
                              class_names=(args.class_names.split(',')
                                           if args.class_names else None))
        with open(os.path.join(args.save_path, 'test_metrics.json'), 'w') as f:
            json.dump({'summary': s, 'per_class': pc,
                       'bf_mean': [float(x) for x in bf],
                       'best_epoch': best_epoch,
                       'best_val_miou': best_miou}, f, indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser("对比模型统一训练/评估（README §7）")
    ap.add_argument('--model', required=True, choices=list(MODEL_ZOO))
    ap.add_argument('--split_path',
                    default='runs/full_public_egcs_full/split.json',
                    help='与主 run 共用的省级切分清单（seed 42）')
    ap.add_argument('--save_path', default=None,
                    help='默认 runs/cmp_<model>')
    ap.add_argument('--epoch', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=None,
                    help='默认取模型档案（多为 4，HRNet 为 2）')
    ap.add_argument('--lr', type=float, default=None,
                    help='默认取模型档案（CNN 1e-4 / SegFormer 6e-5）')
    ap.add_argument('--weight_decay', type=float, default=None,
                    help='默认取模型档案（SegFormer-B2 为 1e-2 按原论文，'
                         '其余 1e-4）')
    ap.add_argument('--num_classes', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0,
                    help='仅用前 N 张（冒烟/调试）')
    ap.add_argument('--amp', action='store_true',
                    help='混合精度（默认 fp32——与主 run FDS-UNeXT 同精度'
                         '口径；开启时自动配 GradScaler）')
    ap.add_argument('--bf_tolerance', type=int, default=8,
                    help='公共集 E-GCS 用 8px（原生 2px × 4 倍上采样）')
    ap.add_argument('--aux_weight', type=float, default=0.3,
                    help='UNetFormer 辅助头损失权重（0 = 关闭）')
    ap.add_argument('--class_names', type=str,
                    default='background,blue-steel,red-steel,white-steel')
    ap.add_argument('--no-eval_test', dest='eval_test',
                    action='store_false')
    # ---- 批 B（P2-2，2026-11-04）：完整增强开关组，语义/默认值与
    # train.py 逐字一致；唯一差别是总开关 --augment 默认 False（保持
    # 对比方法 flip-only 旧行为，历史 run 复现不受影响）
    ap.add_argument('--augment', action=argparse.BooleanOptionalAction,
                    default=False,
                    help='完整增强总开关（Mosaic9+稀有类Copy-Paste+分辨率'
                         '模拟+rot90，与 train.py 同分布）；默认 False = '
                         '旧行为（仅随机翻转）')
    ap.add_argument('--mosaic', action=argparse.BooleanOptionalAction,
                    default=True, help='Mosaic9 九图马赛克')
    ap.add_argument('--mosaic_prob', type=float, default=1.0)
    ap.add_argument('--mosaic_close_epochs', type=int, default=10,
                    help='最后 N 轮关闭 Mosaic（FDS full 协议为 50）')
    ap.add_argument('--mosaic_open_epoch', type=int, default=1)
    ap.add_argument('--mosaic_scale', type=float, nargs=2, default=[0.7, 1.0])
    ap.add_argument('--copy_paste', action=argparse.BooleanOptionalAction,
                    default=True, help='稀有类连通域粘贴')
    ap.add_argument('--copy_paste_prob', type=float, default=0.3)
    ap.add_argument('--res_scale', action=argparse.BooleanOptionalAction,
                    default=True, help='1.25x/0.8x 分辨率模拟')
    ap.add_argument('--res_scale_prob', type=float, default=0.5)
    ap.add_argument('--rot90', action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument('--photo_metric', action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument('--photo_gain', type=float, nargs=2, default=[0.9, 1.1])
    # ---- P2 边界感知基线（2026-11-05，评审 R1-MAJOR）----
    ap.add_argument('--abl_weight', type=float, default=0.0,
                    help='边界感知基线：给 SegFormer-B2 挂边界头并加距离加权 '
                         'BCE（ABL 简化实现）。0 = 关闭（默认，历史 run 不受 '
                         '影响）；建议 0.5')
    args = ap.parse_args()
    if args.save_path is None:
        args.save_path = os.path.join('runs', f'cmp_{args.model}')
    main(args)