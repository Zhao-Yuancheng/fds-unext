import torch
import torch.nn as nn

from .wavelet import dwt_haar, iwt_haar


class ChannelAttention(nn.Module):
    """SE 式通道注意力，用于低频(LL)子带的精炼。

    瓶颈维度取 min(channels // 4, 64)（附录 Q58）：小通道保持 4:1 压缩，
    大通道硬限制 64，防止深层 Stage 参数爆炸。
    """

    def __init__(self, channels):
        super().__init__()
        bottleneck = min(channels // 4, 64)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3)))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class FrequencyDecompositionModule(nn.Module):
    """频率感知特征分解模块（FDM，简化版）。

    替代 SAM2-UNeXT 的 Dense Glue Layer（concat + 1x1 卷积）：
      1. 两路特征分别 Haar DWT 分解为 {LL, LH, HL, HH}
      2. 低频 LL：通道注意力精炼（结构/语义信息）
      3. 高频注入：SAM2 的高频（边缘/纹理）注入 DINOv2 高频，
         注入前置在 DINOv2 原始 32x32 分辨率上（附录 Q57），
         保证频带空间一致（均为 16x16）且 IWT 无损重建
      4. IWT 重建得到两路频率增强特征

    高频注入门控为逐通道可学习向量（初始 0，即初始无注入、渐进学习）。

    Args:
        channels: 该 Stage 的 SAM2 通道数（DINOv2 已对齐到同一通道数）
    """

    def __init__(self, channels):
        super().__init__()
        self.ll_attn_sam = ChannelAttention(channels)
        self.ll_attn_dino = ChannelAttention(channels)
        self.inject_lh = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.inject_hl = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.inject_hh = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, feat_sam, feat_dino):
        """
        Args:
            feat_sam: (B, C, Hs, Ws) SAM2 Hiera 特征（原始 Stage 分辨率）
            feat_dino: (B, C, 32, 32) DINOv2 特征（1x1 conv 通道对齐后）
        Returns:
            feat_sam_freq: (B, C, Hs, Ws) SAM2 频率增强特征
            feat_dino_freq: (B, C, 32, 32) DINOv2 高频注入增强特征
        """
        # ---- SAM2 支路：低频通道注意力精炼，高频直接保留 ----
        ll_s, lh_s, hl_s, hh_s = dwt_haar(feat_sam)
        ll_s = self.ll_attn_sam(ll_s)
        feat_sam_freq = iwt_haar(ll_s, lh_s, hl_s, hh_s)

        # ---- DINOv2 支路（32x32，频带 16x16）----
        ll_d, lh_d, hl_d, hh_d = dwt_haar(feat_dino)
        ll_d = self.ll_attn_dino(ll_d)

        # 高频注入：SAM2 高频自适应池化到与 DINOv2 频带相同尺寸后相加
        band_size = ll_d.shape[-2:]
        lh_d = lh_d + self.inject_lh * F_adaptive_pool(lh_s, band_size)
        hl_d = hl_d + self.inject_hl * F_adaptive_pool(hl_s, band_size)
        hh_d = hh_d + self.inject_hh * F_adaptive_pool(hh_s, band_size)
        feat_dino_freq = iwt_haar(ll_d, lh_d, hl_d, hh_d)

        return feat_sam_freq, feat_dino_freq


def F_adaptive_pool(x, size):
    """同尺寸时恒等返回，避免无谓的池化算子调用。"""
    if x.shape[-2:] == tuple(size):
        return x
    return torch.nn.functional.adaptive_avg_pool2d(x, size)
