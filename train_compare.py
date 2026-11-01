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

import timm
import segmentation_models_pytorch as smp

from datasets import FullDataset
from models.unetformer import UNetFormer
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
    """Attention U-Net（ResNet34 编码器 + 注意力门跳跃连接）。"""

    def __init__(self, num_classes, encoder='resnet34'):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=True,
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

    def __init__(self, num_classes, variant='hrnet_w32'):
        super().__init__()
        self.body = timm.create_model(variant, pretrained=True)
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
}


def build_model(name, num_classes=4):
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
        return smp.Segformer('mit_b2', encoder_weights='imagenet',
                             classes=num_classes)
    if name == 'hrnet_w32':
        return HRNetFCN(num_classes)
    if name == 'unetformer_r18':
        return UNetFormer(num_classes=num_classes, backbone_name='resnet18',
                          pretrained=True)
    raise SystemExit(f"未知模型 {name}；可选：{list(MODEL_ZOO)}")


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


def run_pass(model, loader, device, optim=None, amp=False, scaler=None,
             aux_weight=0.3):
    """单轮训练/验证；复用 ConfusionMatrix（与主项目同指标口径）。
    amp 开启时必须配 GradScaler（fp16 梯度下溢防护）。"""
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
                if isinstance(out, tuple):      # UNetFormer: (main, aux)
                    logits, aux = out
                    loss = F.cross_entropy(logits, t.squeeze(1).long())
                    if aux_weight > 0:
                        loss = loss + aux_weight * F.cross_entropy(
                            aux, t.squeeze(1).long())
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
        logits = model(x)
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
    # 验证/测试：mode='test'（干净、无 SDF）
    train_set = FlipTrainSet('', '', 1024, mode='test',
                             num_classes=args.num_classes,
                             image_list=ti, gt_list=tg)
    val_set = FullDataset('', '', 1024, mode='test',
                          num_classes=args.num_classes,
                          image_list=vi, gt_list=vg)
    print(f"[cmp] model={args.model} ({zoo['desc']}) | lr={lr:g} | "
          f"batch={batch} | wd={wd:g} | amp={args.amp} | "
          f"train: {len(train_set)} | val: {len(val_set)}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              generator=g, worker_init_fn=seed_worker,
                              drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=_collate)

    model = build_model(args.model, args.num_classes).to(device)
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
        t0 = time.time()
        lr_now = optim.param_groups[0]['lr']
        tr_loss, cm_tr = run_pass(model, train_loader, device, optim=optim,
                                  amp=args.amp, scaler=scaler,
                                  aux_weight=args.aux_weight)
        va_loss, cm_va = run_pass(model, val_loader, device, amp=args.amp,
                                  aux_weight=args.aux_weight)
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
    args = ap.parse_args()
    if args.save_path is None:
        args.save_path = os.path.join('runs', f'cmp_{args.model}')
    main(args)