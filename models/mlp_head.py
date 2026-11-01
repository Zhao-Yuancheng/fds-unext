import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm_utils import gn


class MLPFusionHead(nn.Module):
    """方案 A（Q123）：全局语义 MLP 融合头，SegFormer All-MLP head 的并联适配版。

    五路输入：四级融合特征(x1-x4, 已含 SAM2+DINOv2+FDM/BA) + DINOv2 原始终层
    特征(x_d, 未经任何 align/FDM 变换——全局语义直连，核心增量路)。
    工作尺度 1/8 (128×128)：定位为"全局语义偏置"，细节由主路径卷积链负责。
    末层 cls 1×1 conv 零初始化 → 开启瞬间 logits_mlp ≡ 0 → out ≡ out_main
    （与 DySample staged 零初始化 Q119 同哲学；不用可学 λ=0——那会使
    mlp_head 内部梯度恒 0 死锁，末层零 conv 保证 cls 梯度从第一步就活跃）。

    logit 级相加（不碰主路径特征空间）：BA/DCN/BF 机制零风险，消融最干净。

    10-02 修订（v2）：fuse1/fuse2 的 BatchNorm→GroupNorm（gn，与 Q107 同款）。
    首次 R2-MLP 实测（bs=2）BN 失稳：val 自 E02 塌缩，E08 非有限批串被防线
    中止；[repair] 报告 4 个 BN running stats（fuse1/fuse2 的 mean/var）被
    坏批前向污染——与 Q107 记录的「batch=2 下 running stats 无界膨胀」同机
    理。GN 无 running stats、train/eval 行为一致，根除该类失败。
    """

    def __init__(self, num_classes, dim=128, fuse_dim=384):
        super().__init__()
        # 通道统一（1×1 conv = 逐像素 Linear；x_d 路 1024→dim 为主压缩）
        self.w1 = nn.Conv2d(dim, dim, 1)
        self.w2 = nn.Conv2d(dim, dim, 1)
        self.w3 = nn.Conv2d(dim, dim, 1)
        self.w4 = nn.Conv2d(dim, dim, 1)
        self.wd = nn.Conv2d(1024, dim, 1)
        # SegFormer 式 MLP 融合（1×1 conv + GN + GELU；10-02 修订 BN→gn）
        self.fuse1 = nn.Sequential(nn.Conv2d(dim * 5, fuse_dim, 1),
                                   gn(fuse_dim), nn.GELU())
        self.fuse2 = nn.Sequential(nn.Conv2d(fuse_dim, fuse_dim, 1),
                                   gn(fuse_dim), nn.GELU())
        # 分类层：零初始化（ResNet zero-gamma / ControlNet zero-conv 同款）
        self.cls = nn.Conv2d(fuse_dim, num_classes, 1)
        nn.init.zeros_(self.cls.weight)
        nn.init.zeros_(self.cls.bias)

    def forward(self, x1, x2, x3, x4, x_d, out_size):
        # x1 (B,128,256,256)  x2 (B,128,128,128)  x3 (B,128,64,64)
        # x4 (B,128,32,32)    x_d (B,1024,32,32)  out_size=(1024,1024)
        s = (128, 128)                                    # 1/8 工作尺度
        p1 = self.w1(F.interpolate(x1, size=s, mode='bilinear', align_corners=False))
        p2 = self.w2(x2)                                  # 已在 1/8，恒等尺度
        p3 = self.w3(F.interpolate(x3, size=s, mode='bilinear', align_corners=False))
        p4 = self.w4(F.interpolate(x4, size=s, mode='bilinear', align_corners=False))
        pd = self.wd(F.interpolate(x_d, size=s, mode='bilinear', align_corners=False))
        z = torch.cat([p1, p2, p3, p4, pd], dim=1)        # (B, 640, 128, 128)
        z = self.fuse2(self.fuse1(z))                     # (B, 384, 128, 128)
        logits = self.cls(z)                              # (B, C, 128, 128)
        return F.interpolate(logits, size=out_size, mode='bilinear',
                             align_corners=False)         # (B, C, 1024, 1024)
