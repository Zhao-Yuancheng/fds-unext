import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """轻量级动态上采样器（ICCV 2023, arXiv:2308.15085）。

    将上采样重新定义为"学习采样点位置"：

        y[i,j] = x( u_ij ),   u_ij = b_ij + Δ_ij
        Δ_ij   = offset(x)_{ij} · sigmoid( scope(x)_{ij} ) · 0.5

    基础坐标 b_ij 取双线性上采样的源坐标，偏移 Δ_ij 由 1×1 卷积从特征
    自身预测——采样点位置随内容变化，即"内容感知"。offset/scope 权重
    全部零初始化：offset(x)=0 且 sigmoid(0)=0.5，故 Δ = 0·0.5 = 0，
    初始输出严格等于 align_corners=True 的 bilinear 上采样，训练从
    双线性基线出发渐进学习偏移。

    与官方 tiny-smart/dysample 的关系（附录 Q75 逐行核对，2026-08）：
    - 结构等价官方 DySample+（'lp' 风格 + groups + dyscope 变体）；
    - 两处自洽的工程差异（已确认保留，Q76 决策）：
      ① 格点约定 AC=True（官方 AC=False）：初始 ≡ AC=True 双线性，与
        基线解码器 nn.Upsample(align_corners=True) 严格一致，替换点
        零扰动；官方初始 ≡ PyTorch 默认 AC=False 双线性。两套约定各自
        自洽，仅基底格点差 ≤0.25 LR 像素（边界附近最大、中部为 0）。
      ② 偏移单位为归一化坐标（1 单位 = (W-1)/2 个 LR 像素；官方为 LR
        像素）。对可学习卷积仅是参数化缩放差异，可被权重吸收。

    Args:
        in_channels: 输入通道数，需能被 groups 整除
        scale: 上采样倍率（论文验证至 4，工程上可用于 2/4/8）
        groups: 采样分组数，每组独立预测采样点（g=1 即论文 DySample-1P）
        offset_pixels: 偏移预算（LR 像素）——tanh 软界的语义上界，
            采样点最多偏离双线性格点 offset_pixels 个 LR 像素
    """

    def __init__(self, in_channels, scale=2, groups=4, offset_pixels=4.0):
        super().__init__()
        assert in_channels % groups == 0
        self.scale = scale
        self.groups = groups
        # 输出通道 2·g·s² 的因子分解：坐标方向 xy(2) × 分组(g) × 子像素(s²)。
        # 即每个分组、每个输出子像素都持有独立的偏移量——逐子像素参数化
        # （等价官方 'lp' 风格），偏移场不经任何插值放大
        self.offset = nn.Conv2d(in_channels, 2 * groups * scale ** 2, 1)
        # scope：偏移的逐元素动态范围门控（官方 dyscope 变体），
        # 与 offset 同形状
        self.scope = nn.Conv2d(in_channels, 2 * groups * scale ** 2, 1)
        self.offset_pixels = offset_pixels
        self.init_weights()

    def init_weights(self):
        # 权重、偏置全部零初始化：
        #   Δ = 0 · sigmoid(0) · 0.5 = 0  →  初始 ≡ bilinear(AC=True)
        # 零权重不阻塞训练：∂L/∂W = (∂L/∂y)·x ≠ 0，梯度正常流动
        # （冒烟测试 Test 3 断言 offset/scope 梯度非零）
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.scope.weight)
        nn.init.zeros_(self.scope.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale
        g = self.groups

        # ---- 1. 偏移预测：整理为 (b, g, h, w, s_h, s_w, 2) ----
        # 卷积输出 (b, 2gs², h, w) 的通道维按 (g, s_h, s_w, xy) 拆分，
        # permute 后空间维 (h, w) 提前、子像素维 (s_h, s_w) 靠后，
        # 与 _base_grid 返回的 (h, w, s, s, 2) 逐元素对齐。
        # sigmoid·0.5 把偏移乘性压缩到 (0, 0.5) 的门控范围内：采样点
        # 只在基础格点邻域做局部微调，避免大偏移造成跨区域的双线性混合
        offset = self.offset(x).view(b, g, s, s, 2, h, w)
        # 语义级偏移界（Q113 终版）：DySample 设计本义是对双线性采样
        # 坐标做局部微调（边缘拟合/特征对齐，论文量级为几个 LR 像素），
        # 而非跨像素远程取值。归一化坐标 1 单位 = (w-1)/2 个 LR 像素，
        # 故 k 像素预算 → T = 2k/(w-1)。实测无约束时 up1.up offset 达
        # 20 单位（≈1275 LR 像素 @ w=256，为图像宽度 5 倍），全部被
        # border-clamp 成边界复制——通道相干化 → up1.conv 被动相干放大
        # 42 倍（Q113 第三次塌缩根因）。界内近恒等（健康实例偏移远小
        # 于 T，tanh 线性区失真 <1%），仅约束病态大偏移；不加参数，
        # 旧 checkpoint 直接兼容（与 Adapter prompt 软界同法）
        T = 2.0 * self.offset_pixels / max(w - 1, 1)
        offset = T * torch.tanh(offset / T)
        scope = torch.sigmoid(self.scope(x)).view(b, g, s, s, 2, h, w)
        offset = (offset * scope * 0.5).permute(0, 1, 5, 6, 2, 3, 4)

        # ---- 2. 合成采样坐标：u = b + Δ ----
        base = self._base_grid(h, w, s, x.dtype, x.device)      # (h, w, s, s, 2)
        pos = base + offset                                      # (b, g, h, w, s, s, 2)

        # ---- 3. 重排到输出分辨率 ----
        # 输出像素 (H_out, W_out) 满足 H_out = h·s + s_h、W_out = w·s + s_w；
        # reshape 到 (h·s, w·s) 的行主序内存布局要求维度顺序为
        # (h, s_h, w, s_w)——s_h 必须比 w 变化更慢。交织顺序由显式
        # permute 声明（附录 Q66 交织 bug 的修复点），不依赖隐式约定
        pos = pos.permute(0, 1, 2, 4, 3, 5, 6).reshape(b * g, h * s, w * s, 2)

        # ---- 4. 分组内容感知采样 ----
        # 通道按组拆分并折叠进 batch 维，每组用本组自己的采样场独立
        # grid_sample（官方分组机制原义：不同通道组可学习不同的采样模式）
        x = x.view(b, g, c // g, h, w).reshape(b * g, c // g, h, w)
        out = F.grid_sample(x, pos, mode='bilinear',
                            padding_mode='border', align_corners=True)
        return out.reshape(b, c, h * s, w * s)

    @staticmethod
    def _base_grid(h, w, s, dtype, device):
        """零偏移时的采样坐标（align_corners=True 约定），返回 (h, w, s, s, 2)。

        输出像素 j ∈ [0, sW) 的归一化坐标 x_j = -1 + 2j/(sW-1)（角像素
        中心对齐）；AC=True 的反映射 x_src = (x_j+1)/2·(W-1) =
        j(W-1)/(sW-1)，正是 F.interpolate(bilinear, align_corners=True)
        的源坐标——因此 Δ=0 时输出与双线性上采样逐像素一致（数值验证
        err 5.2e-6~7.4e-6，附录 Q68 Test 2）。

        布局：输出网格 (h·s, w·s) 先 view 为 (h, s_h, w, s_w)，再
        permute 成 (h, w, s_h, s_w)，把索引 (h, w) 处的 s×s 个子像素
        分到同一组，与偏移张量 (b, g, h, w, s, s, 2) 相加时严格对齐；
        该 permute 与 forward 第 3 步的 permute 互为逆操作。
        """
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, h * s, dtype=dtype, device=device),
            torch.linspace(-1, 1, w * s, dtype=dtype, device=device),
            indexing='ij')
        grid = torch.stack([gx, gy], dim=-1)                     # (h*s, w*s, 2)
        return grid.view(h, s, w, s, 2).permute(0, 2, 1, 3, 4)   # (h, w, s, s, 2)
