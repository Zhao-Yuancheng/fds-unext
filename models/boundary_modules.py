import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from .norm_utils import gn


class BoundaryAttentionGate(nn.Module):
    """方案 A：边界注意力门（论文 §3.3.1）。

    生成边界注意力图，增强 skip connection 特征的边界响应：
      - 学习式分支：3x3 conv -> GN -> SiLU -> 1x1 -> sigmoid
        （SiLU 平滑非线性在零点附近保留细梯度，边界特征恰分布在
        低激活区，ReLU 的死区会截断这部分信号）
      - 固定式分支：Sobel 边缘检测（buffer，不参与训练）
      - 融合：attn * (1 + sobel_edge)，逐像素乘回特征

    附录约束：仅用于浅层 skip 特征（x1, x2）；深层特征的 Sobel
    边缘与真实边界错位（附录教训），x3/x4 不加。

    与上游实现的差异：sobel_edge 按样本内最大值归一化（上游用整个
    batch 的 max，跨样本耦合，batch_size=1 时无差异，>1 时更稳）。
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            gn(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        # 教程 8.1 写的是 transpose(1,2)，会得到 (1,3,1,3) 的错误形状；
        # 正确做法是核内 2D 转置 transpose(2,3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        attn = torch.sigmoid(self.boundary_conv(x))

        x_gray = x.mean(dim=1, keepdim=True)
        edge_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        sobel_edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        sobel_edge = sobel_edge / (sobel_edge.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

        boundary_attn = attn * (1 + sobel_edge)
        return x * boundary_attn


class BoundaryRefinement(nn.Module):
    """方案 C：可变形卷积边界精炼（论文 §3.3.3）。

    offset_conv 预测 3x3 核 9 个采样点的 (dx, dy) 偏移（2*9=18 通道），
    DeformConv2d 在偏移位置采样卷积；GN + SiLU + 残差（SiLU 平滑
    非线性对边界精炼这类细粒度任务优于 ReLU 的硬零区）。

    offset_conv 零初始化：初始偏移全 0，精确退化为标准 3x3 卷积，
    训练从标准卷积舒适区渐进学习采样点偏移（与 DySample 初始化
    哲学一致）；残差连接进一步保证最差情况不劣于输入。

    附录约束：部署在最终上采样（ds_final）之前的特征精炼位置。
    """

    def __init__(self, channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(channels, 2 * 9, 3, padding=1)
        self.deform_conv = DeformConv2d(channels, channels, 3, padding=1)
        self.bn = gn(channels)
        self.act = nn.SiLU(inplace=True)
        # 严格零初始化（Q72）：std=0.001 的近似零（教程原稿）会残留
        # 1e-3 量级随机偏移，初始不再精确等于标准 3x3 卷积，冒烟测试
        # 的回归断言（err < 1e-4）无法成立
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        refined = self.deform_conv(x, offset)
        return self.act(self.bn(refined)) + x
