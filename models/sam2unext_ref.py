"""SAM2-UNeXT 官方实现的忠实移植（论文对照基线）。

来源：`SAM2-UNeXT-main/SAM2UNeXT.py`（arXiv:2508.03566，WZH0120 官方仓库），
仅做两处**必要**改动，其余逐字保留：
  1. `head` 输出通道 1 → `num_classes`（原实现是二值单通道，
     无法直接用于 4 类 LRMD / 6 类 Potsdam）；
  2. 主干构建复用本项目 `build_sam2` + `_load_hiera_trunk`（本项目需兼容
     SAM2/SAM2.1 checkpoint；官方是 build_sam2 直接严格加载全 ckpt），
     权重张量与官方同源同键。

保留的官方特征（与 FDS-UNeXT 的差异，写作时必须如实说明）：
  - 解码器 `DoubleConv` 用 **BatchNorm2d**（本项目改用 GroupNorm）
  - 全部上采样为 **bilinear(align_corners=True)**
  - DINOv2 特征按 stage **双线性**拉伸后 concat（无 FDM、无 DySample）
  - 无边界分支、无 SDF/不确定性损失（损失见 train_ref.py 的
    `structure_loss_multiclass`，weit k=31 同时作用于 wBCE 与 wIoU）
  - 官方 forward 里 head 被调用两次（一次丢弃），此处修正为一次；
    数值上无差异（1×1 conv 无副作用、无 BN）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from sam2.build_sam import build_sam2

from .fds_unext import _DINO_SPECS, _HIERA_SPECS, Adapter, _load_hiera_trunk


class RefDoubleConv(nn.Module):
    """官方版 (conv → BN → ReLU) × 2。

    `norm='bn'` 为官方原样；`norm='gn'` 为数值修正版（见下方说明）。
    在原仓库设置下（batch=1）BN 可训练；但在 bs=2、1024² 时本项目的
    复现出现 BN running stats 爆炸、逐轮 mIoU 在 0.21–0.53 间震荡、
    若干轮整幅退化为单一类别（val_BF=0），与本项目早前诊断的 BN 病理一致。
    因此默认改用 GroupNorm（与 FDS-UNeXT 一致），BN 版本保留可复现。
    """

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 norm='gn', groups=32):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        def n(c):
            if norm == 'bn':
                return nn.BatchNorm2d(c)
            return nn.GroupNorm(groups, c)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            n(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            n(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class RefUp(nn.Module):
    """官方版上采样块：先 pad 对齐 → concat → bilinear ×2 → DoubleConv。"""

    def __init__(self, in_channels, out_channels, norm='gn'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.conv = RefDoubleConv(in_channels, out_channels,
                                  in_channels // 2, norm=norm)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diff_y = x1.size(2) - x2.size(2)
            diff_x = x1.size(3) - x2.size(3)
            x2 = F.pad(x2, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        return self.conv(self.up(x))


class SAM2UNeXTRef(nn.Module):
    """官方 SAM2-UNeXT（双冻结主干 + concat 胶水层 + UNet 解码器）。"""

    def __init__(self, checkpoint_path=None, dinov2_path=None,
                 num_classes=2, hiera_size='large', dino_size='large',
                 pretrained=True, norm='gn'):
        super().__init__()
        hiera_cfg, (c1, c2, c3, c4) = _HIERA_SPECS[hiera_size]
        dino_name, dino_ch = _DINO_SPECS[dino_size]

        model = build_sam2(hiera_cfg, device="cpu")
        for attr in ["sam_mask_decoder", "sam_prompt_encoder", "memory_encoder",
                     "memory_attention", "mask_downsample",
                     "obj_ptr_tpos_proj", "obj_ptr_proj"]:
            delattr(model, attr)
        del model.image_encoder.neck
        self.sam = model.image_encoder.trunk
        if checkpoint_path:
            _load_hiera_trunk(self.sam, checkpoint_path)
        for param in self.sam.parameters():
            param.requires_grad = False
        self.sam.blocks = nn.Sequential(
            *[Adapter(block) for block in self.sam.blocks])

        if dinov2_path:
            self.dino = timm.create_model(
                dino_name, features_only=True, img_size=(448, 448),
                pretrained=True,
                pretrained_cfg_overlay=dict(file=dinov2_path))
        else:
            self.dino = timm.create_model(
                dino_name, features_only=True, img_size=(448, 448),
                pretrained=pretrained)
        for param in self.dino.parameters():
            param.requires_grad = False

        self.align1 = nn.Conv2d(dino_ch, c1, 1)
        self.align2 = nn.Conv2d(dino_ch, c2, 1)
        self.align3 = nn.Conv2d(dino_ch, c3, 1)
        self.align4 = nn.Conv2d(dino_ch, c4, 1)

        self.reduce1 = nn.Conv2d(c1 * 2, 128, 1)
        self.reduce2 = nn.Conv2d(c2 * 2, 128, 1)
        self.reduce3 = nn.Conv2d(c3 * 2, 128, 1)
        self.reduce4 = nn.Conv2d(c4 * 2, 128, 1)

        self.up1 = RefUp(256, 128, norm=norm)
        self.up2 = RefUp(256, 128, norm=norm)
        self.up3 = RefUp(256, 128, norm=norm)
        self.up4 = RefUp(128, 128, norm=norm)
        self.head = nn.Conv2d(128, num_classes, 1)
        self.num_classes = num_classes

    def forward(self, x):
        x1_s, x2_s, x3_s, x4_s = self.sam(x)
        x_d = self.dino(F.interpolate(x, size=(448, 448),
                                      mode='bilinear'))[-1]

        x1_d = F.interpolate(self.align1(x_d), size=x1_s.shape[-2:],
                             mode='bilinear')
        x2_d = F.interpolate(self.align2(x_d), size=x2_s.shape[-2:],
                             mode='bilinear')
        x3_d = F.interpolate(self.align3(x_d), size=x3_s.shape[-2:],
                             mode='bilinear')
        x4_d = F.interpolate(self.align4(x_d), size=x4_s.shape[-2:],
                             mode='bilinear')

        x1 = self.reduce1(torch.cat([x1_s, x1_d], dim=1))
        x2 = self.reduce2(torch.cat([x2_s, x2_d], dim=1))
        x3 = self.reduce3(torch.cat([x3_s, x3_d], dim=1))
        x4 = self.reduce4(torch.cat([x4_s, x4_d], dim=1))

        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        return F.interpolate(out, scale_factor=2, mode='bilinear')
