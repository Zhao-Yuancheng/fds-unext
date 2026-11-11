"""SAM2-UNeXT 官方基线训练脚本（对照基线）。

与 `train.py`（FDS-UNeXT）的区别：
  - 模型 = `models/sam2unext_ref.py`（官方架构，BN 解码器、bilinear 上采样、
    concat 胶水层、无 FDM/DySample/边界分支）
  - 损失 = `structure_loss_multiclass`（官方 weit k=31 同时作用于 wBCE
    与 wIoU；FDS 用 k=7 且 wIoU 不加权）
  - 增强 = 仅随机水平/垂直翻转（与官方 dataset.py 的 train transform 一致）
  - 无 SDF 缓存（val/test 用 mode='test'，省掉 EDT 预计算）

协议其余部分与 FDS 严格同口径：同一 split.json、1024²、AdamW + cosine→1e-7、
固定 epoch 无早停、按 val mIoU 选点（同时另存 best-bf）、bf_tol=8。

用法示例（LRMD fold0）：
  python train_ref.py --hiera_path weights/sam2.1_hiera_large.pt \
    --dinov2_path weights/dinov2_vitl14_pretrain.pth \
    --train_image_path dataset/roof_dataset_v2/images \
    --train_mask_path dataset/roof_dataset_v2/masks \
    --split_path dataset/roof_dataset_v2/split_fold0.json \
    --num_classes 4 --epoch 100 --lr 1e-4 --batch_size 2 \
    --save_path runs/roof_v2_ref/sam2unext_ref_e100_lr1e4_fold0
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import FullDataset
from models.sam2unext_ref import SAM2UNeXTRef
from utils.augment import AugmentConfig
from utils.metrics import ConfusionMatrix, boundary_f_score
from utils.split import load_split
from train_compare import AugTrainSet

WT_K = 31          # 官方 weit 卷积核（显著性检测文献默认值）
WT_COEF = 5.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def structure_loss_multiclass(pred, target, num_classes):
    """官方 structure_loss 的多分类直译版。

    pred: (B,C,H,W) logits；target: (B,1,H,W) 类别索引。
    官方二值式为 weit(k=31) 加权 BCE + weit 加权 soft-IoU；
    多分类下 weit 由各类 one-hot 的 |avg_pool − onehot| 求和得到（与官方
    二值情形逐字一致），wIoU 逐类计算后取类平均。
    """
    tgt = target.squeeze(1).long()
    onehot = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()
    smooth = F.avg_pool2d(onehot, kernel_size=WT_K, stride=1,
                          padding=WT_K // 2)
    weit = 1 + WT_COEF * (smooth - onehot).abs().sum(dim=1, keepdim=True)

    ce = F.cross_entropy(pred, tgt, reduction='none').unsqueeze(1)
    wbce = (weit * ce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp_min(1e-6)

    prob = torch.softmax(pred, dim=1)
    inter = ((prob * onehot) * weit).sum(dim=(2, 3))
    union = ((prob + onehot) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou.mean(dim=1, keepdim=True)).mean()


def parse_label_mapping(s):
    if not s:
        return None
    m = {}
    for kv in s.split(','):
        k, v = kv.split(':')
        m[int(k)] = int(v)
    return m


class _ImgLabel(torch.utils.data.Dataset):
    """只保留 image/label 的轻包装。

    `FullDataset` 在 mode='test' 下 `sdf=None`，直接进 DataLoader 会在
    collate 处报 "found <class 'NoneType'>"；官方基线不需要 SDF。
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        return {'image': s['image'], 'label': s['label']}


def evaluate(model, loader, device, num_classes, bf_tol, amp=False):
    model.eval()
    cm = ConfusionMatrix(num_classes, device='cpu')
    bf_all = []
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device, non_blocking=True)
            tgt = batch['label'].squeeze(1).long()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp):
                logits = model(x)
            pred = logits.float().argmax(dim=1).cpu()
            cm.update(pred, tgt)
            for i in range(pred.size(0)):
                # boundary_f_score 返回逐类列表 → 展平后取宏平均
                bf_all.extend(boundary_f_score(pred[i], tgt[i],
                                               num_classes, bf_tol))
    summ = cm.summary()
    bf = float(np.nanmean([v for v in bf_all if v is not None]))
    return summ['miou'], bf, summ


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.save_path, exist_ok=True)

    split = load_split(args.split_path)
    lm = parse_label_mapping(args.label_mapping)
    if args.augment:
        aug = AugmentConfig(mosaic_close_epochs=args.mosaic_close_epochs,
                            mosaic_open_epoch=args.mosaic_open_epoch,
                            copy_paste_prob=args.copy_paste_prob,
                            res_scale_prob=args.res_scale_prob,
                            photo_metric=args.photo_metric)
        train_set = AugTrainSet(args.train_image_path, args.train_mask_path, 1024,
                                num_classes=args.num_classes,
                                image_list=split['train']['images'],
                                gt_list=split['train']['gts'],
                                augment=aug)
    else:
        train_set = FullDataset(args.train_image_path, args.train_mask_path, 1024,
                                mode='train', num_classes=args.num_classes,
                                image_list=split['train']['images'],
                                gt_list=split['train']['gts'],
                                label_mapping=lm, augment=None)
    val_set = _ImgLabel(FullDataset(
        args.train_image_path, args.train_mask_path, 1024,
        mode='test', num_classes=args.num_classes,
        image_list=split['val']['images'],
        gt_list=split['val']['gts'], label_mapping=lm))
    print(f"[ref] train {len(train_set)} | val {len(val_set)} | "
          f"test {len(split['test']['images'])} (held out)", flush=True)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, generator=g,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    model = SAM2UNeXTRef(args.hiera_path, args.dinov2_path,
                         num_classes=args.num_classes,
                         hiera_size=args.hiera_size,
                         dino_size=args.dino_size,
                         norm=args.norm).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[ref] params total {n_all / 1e6:.2f}M | trainable {n_tr / 1e6:.2f}M",
          flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = opt.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    csv_path = os.path.join(args.save_path, 'results.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'time_s', 'lr', 'train_loss',
                                    'val_miou', 'val_bf'])
    best_miou, best_bf = -1.0, -1.0

    for epoch in range(1, args.epoch + 1):
        model.train()
        if getattr(args, 'augment', False):
            train_set.set_epoch(epoch, args.epoch)
        t0 = time.time()
        loss_sum, n_seen = 0.0, 0
        for batch in train_loader:
            x = batch['image'].to(device, non_blocking=True)
            tgt = batch['label'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16, enabled=args.amp):
                pred = model(x)
                loss = structure_loss_multiclass(pred.float(), tgt,
                                                 args.num_classes)
            if not torch.isfinite(loss):
                optim.zero_grad(set_to_none=True)
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            optim.step()
            loss_sum += float(loss.detach()) * x.size(0)
            n_seen += x.size(0)
        sched.step()

        miou, bf, summ = evaluate(model, val_loader, device,
                                  args.num_classes, args.val_bf_tol,
                                  amp=args.amp)
        dt = time.time() - t0
        print(f"[ref] ep{epoch}/{args.epoch} loss={loss_sum / max(n_seen, 1):.4f}"
              f" val_mIoU={miou:.4f} val_BF={bf:.4f} lr={sched.get_last_lr()[0]:.2e}"
              f" {dt:.0f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, round(dt, 1),
                                    f"{sched.get_last_lr()[0]:.3e}",
                                    f"{loss_sum / max(n_seen, 1):.6f}",
                                    f"{miou:.6f}",
                                    "" if np.isnan(bf) else f"{bf:.6f}"])
        sd = model.state_dict()
        if miou > best_miou:
            best_miou = miou
            torch.save(sd, os.path.join(args.save_path, 'SAM2-UNeXT-best.pth'))
            print(f"  [save] best val mIoU {miou:.4f}", flush=True)
        if not np.isnan(bf) and bf > best_bf:
            best_bf = bf
            torch.save(sd, os.path.join(args.save_path,
                                        'SAM2-UNeXT-best-bf.pth'))
            print(f"  [save] best val BF {bf:.4f}", flush=True)

    print(f"[ref] DONE best_mIoU={best_miou:.4f} best_BF={best_bf:.4f}",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser("SAM2-UNeXT reference baseline")
    p.add_argument("--hiera_path", type=str, required=True)
    p.add_argument("--dinov2_path", type=str, required=True)
    p.add_argument("--train_image_path", type=str, required=True)
    p.add_argument("--train_mask_path", type=str, required=True)
    p.add_argument("--split_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--label_mapping", type=str, default=None)
    p.add_argument("--val_bf_tol", type=int, default=8)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="autocast fp16 前向（与 FDS-UNeXT 训练精度口径一致，"
                        "无 GradScaler/无 loss scaling）")
    p.add_argument("--hiera_size", type=str, default="large")
    p.add_argument("--dino_size", type=str, default="large")
    p.add_argument("--norm", type=str, default="gn", choices=["gn", "bn"],
                   help="解码器归一化：gn=数值修正版（默认），bn=官方原样")
    p.add_argument("--augment", action="store_true",
                   help="同协议对照用：应用与 train.py 逐字同分布的完整增强 "
                        "(Mosaic9/Copy-Paste/分辨率模拟/翻转)；默认关闭=官方 "
                        "flip-only 口径，历史 run 不受影响")
    p.add_argument("--mosaic_close_epochs", type=int, default=50)
    p.add_argument("--mosaic_open_epoch", type=int, default=1)
    p.add_argument("--copy_paste_prob", type=float, default=0.3)
    p.add_argument("--res_scale_prob", type=float, default=0.5)
    p.add_argument("--photo_metric", action="store_true")
    main(p.parse_args())
