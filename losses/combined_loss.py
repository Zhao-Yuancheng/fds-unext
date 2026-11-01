import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """组合损失（改进三，支持任意类别数；论文 §3.4）。

    组成：
      total = structure_loss                                # 全程满权（Q26）
            + gate · [precision_b · boundary_loss           # SDF 边界损失
                      + 0.5 · log_var_b]                    # Kendall 正则项

    1. 结构损失 = wBCE(weit) + IoU(无 weit)
       - weit: 1 + 5*|avg_pool(onehot, k=7) - onehot|，k=7 精准覆盖
         mask 边界 ±3px（附录 Q11，固定 k=7，不用多尺度组合）
       - wBCE 用 weit 加权边界，IoU 不用 weit 负责区域完整性（附录 Q2）
    2. gate（边界损失何时启用，两个门相乘）：
       - epoch 门：前 warmup_epochs 轮为 0（附录 Q19）
       - Dice 门：smoothstep(Dice; dice_start, dice_end)（附录 Q22 方案 1，
         经 Q24 修正：sigmoid 永不为 0，低 Dice 区泄漏非零权重——Dice=0.3
         时仍 16.8%，且 ramp<0.01 守卫在 Dice≥0 下永不触发；smoothstep
         在 Dice < dice_start 时精确为 0，参数语义变为真正的起点/满权点，
         C1 连续）
       - 就绪度指标用宏平均 Dice 而非 IoU——低重叠区灵敏度更高、数值
         更稳（附录 Q23）
    3. SDF 边界损失（全类监督，附录 Q29 方案 B）
       - 使用数据集预计算的逐类归一化 SDF（[-1,1]，附录 Q30）
       - 逐类 (phi_c * p_c).mean() 后等权平均 /num_classes
       - phi 符号: 负=内部、正=外部（Kervadec 约定，附录 Q15 修正）
    4. 不确定性加权（附录 Q26/Q27，uncertainty=True 时）：
       - 边界损失权重由可学习 precision_b = exp(-log_var_b) 给出，
         结构损失固定满权 1.0（Q26：门控只乘辅助损失）
       - log_var_b 初值 = -ln(boundary_weight)，即 precision 初值等于
         经验权重 β（Q27），不做全 0 冷启动等权
       - log_var_b 每 step clamp 到 [-2, 4]，权重范围 [e^-4, e^2]≈
         [0.018, 7.4]，防 exp 爆炸（Q27）
       - gate=0 时整个边界项（含正则项）不参与前向，σ 自然不被早期
         "假损失"污染，等价于 Q26 所述的 σ 学习 warmup

    Args:
        num_classes: 类别数（含背景）
        weit_k: weit 的 avg_pool 核大小（固定 7）
        weit_coeff: weit 边界提升系数（固定 5）
        boundary_weight: 边界损失权重 β。uncertainty=False 时是固定权重；
            uncertainty=True 时仅作为 precision_b 的初值（Q28：主损失已含
            weit，β 取 0.2 而非通用文献的 0.5~1.0）
        warmup_epochs: epoch 门轮数（0 = 不设 epoch 门）
        adaptive: 是否启用 Dice 自适应调度（Q22 方案 1）
        dice_start: Dice 起始点，宏平均 Dice 低于此值边界权重精确为 0
            （默认 0.5，保留 Q22"预测对了一半以上"语义为真零点，Q24）
        dice_end: Dice 满权点，高于此值边界权重满额（默认 0.7，过渡
            宽度与 Q22 原 ramp_speed=8 的 sigmoid 相当）
        uncertainty: 是否启用不确定性加权（Q26/Q27）
        log_var_range: log_var_b 的 clamp 范围（Q27 默认 [-2, 4]）
    """

    def __init__(self, num_classes=3, weit_k=7, weit_coeff=5,
                 boundary_weight=0.2, warmup_epochs=0,
                 adaptive=True, dice_start=0.5, dice_end=0.7,
                 uncertainty=True, log_var_range=(-2.0, 4.0)):
        super().__init__()
        self.num_classes = num_classes
        self.weit_k = weit_k
        self.weit_coeff = weit_coeff
        self.boundary_weight = boundary_weight
        self.warmup_epochs = warmup_epochs
        self.adaptive = adaptive
        self.dice_start = dice_start
        self.dice_end = dice_end
        self.uncertainty = uncertainty
        self.log_var_range = log_var_range
        pad = weit_k // 2
        self._pool_kwargs = dict(kernel_size=weit_k, stride=1, padding=pad)
        self.log_var_b = None
        if uncertainty:
            init = -math.log(boundary_weight)
            self.log_var_b = nn.Parameter(torch.tensor(float(init)))
        self.last_stats = {}

    def _onehot(self, target):
        """target: (B,1,H,W) 值 {0,1,2} -> (B,C,H,W) float"""
        t = target.squeeze(1).long()
        return F.one_hot(t, self.num_classes).permute(0, 3, 1, 2).float()

    def compute_weit(self, onehot):
        """多分类 weit：任一类别的边界像素都被 boost。返回 (B,1,H,W)"""
        smooth = F.avg_pool2d(onehot, **self._pool_kwargs)
        boundary = (smooth - onehot).abs().sum(dim=1, keepdim=True)
        return 1 + self.weit_coeff * boundary

    @staticmethod
    def smoothstep(x, start, end):
        """Q24 调度核：x<start 精确 0；start≤x≤end 平滑 0→1（C1 连续）；
        x>end 满权 1。"""
        t = torch.clamp((x - start) / (end - start), 0.0, 1.0)
        return t * t * (3 - 2 * t)

    @torch.no_grad()
    def dice_score(self, pred, target):
        """宏平均 Dice（逐类 Dice 的等权平均），作就绪度指标（Q23）。"""
        pred_cls = pred.argmax(dim=1)
        t = target.squeeze(1).long()
        dices = []
        for c in range(self.num_classes):
            p = (pred_cls == c).float()
            m = (t == c).float()
            inter = (p * m).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + m.sum(dim=(1, 2))
            dices.append(((2 * inter + 1e-8) / (denom + 1e-8)).mean())
        return torch.stack(dices).mean()

    def structure_loss(self, pred, target):
        """wBCE(weit, k=7) + IoU(无 weit)。"""
        onehot = self._onehot(target)

        weit = self.compute_weit(onehot)                     # (B,1,H,W)
        wbce = F.cross_entropy(pred, target.squeeze(1).long(), reduction='none')
        wbce = (wbce * weit.squeeze(1)).sum(dim=(1, 2)) / weit.squeeze(1).sum(dim=(1, 2))

        probs = F.softmax(pred, dim=1)
        inter = (probs * onehot).sum(dim=(2, 3))
        union = (probs + onehot).sum(dim=(2, 3))
        iou = 1 - (inter + 1) / (union - inter + 1)          # (B,C) 逐类，无 weit
        iou = iou.mean()

        return wbce.mean() + iou

    def boundary_loss(self, pred, sdf):
        """SDF 边界损失（全类监督，Q29 方案 B）。
        sdf: (B,C,H,W) 预计算逐类归一化 SDF，负=内部。"""
        probs = F.softmax(pred, dim=1)
        loss = 0.0
        for c in range(self.num_classes):
            loss = loss + (sdf[:, c:c + 1] * probs[:, c:c + 1]).mean()
        return loss / self.num_classes

    def extract_boundary(self, target, kernel_size=3):
        """方案 B 的边界 GT：逐类"膨胀-腐蚀"取并集（多分类版，教程 8.2）。

        target: (B,1,H,W) {0..C-1} -> (B,1,H,W) float，1=边界带
        """
        onehot = self._onehot(target)
        k = dict(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        dilated = F.max_pool2d(onehot, **k)
        eroded = -F.max_pool2d(-onehot, **k)
        return ((dilated - eroded).sum(dim=1, keepdim=True) > 0).float()

    def boundary_head_loss(self, boundary_pred, target):
        """方案 B 辅助边界头的监督损失。

        边界 GT 从整幅标签提取后下采样到 boundary_pred 的分辨率；
        边界像素占比随屋顶密度在 0.3%~5% 间波动（稀疏工业大屋顶瓦片
        可低于 1%），用 pos_weight = N_neg/N_pos 平衡正负样本。
        上限 200（附录 Q82）：r≤200 时严格教科书平衡（HED/GSCNN 口径），
        r∈(200,400] 保持正样本梯度占比 ≥1/3；保留上限是因为逐批比例在
        batch=2 下方差大，近空瓦片 r 可达 10³+，无上限会以千倍权重放大
        少数像素的梯度尖峰——对分布尾部的防护，而非对典型瓦片的保守。
        """
        gt = self.extract_boundary(target)
        gt = F.interpolate(gt, size=boundary_pred.shape[-2:], mode='nearest')
        pos = gt.sum()
        neg = gt.numel() - pos
        pos_weight = (neg / pos.clamp(min=1.0)).clamp(1.0, 200.0)
        return F.binary_cross_entropy_with_logits(
            boundary_pred, gt, pos_weight=pos_weight)

    def forward(self, pred, target, sdf, current_epoch=None):
        """
        Args:
            pred: (B,C,H,W) logits
            target: (B,1,H,W) {0,1,2}
            sdf: (B,C,H,W) 预计算 SDF（逐类归一化 [-1,1]）
            current_epoch: 当前轮次（0 起），用于 epoch 门
        """
        loss_s = self.structure_loss(pred, target)

        # epoch 门（Q19 固定 warmup 语义）
        epoch_gate = 1.0
        if current_epoch is not None and self.warmup_epochs > 0 \
                and current_epoch < self.warmup_epochs:
            epoch_gate = 0.0

        # Dice 门（Q22 方案 1 + Q24 smoothstep 修正）
        dice = None
        ramp = 1.0
        if self.adaptive:
            dice = self.dice_score(pred, target)
            ramp = self.smoothstep(dice, self.dice_start, self.dice_end)

        gate = epoch_gate * ramp
        if float(gate) < 1e-3:
            self.last_stats = {'gate': float(gate), 'dice': None if dice is None
                               else float(dice), 'beta_eff': 0.0}
            return loss_s

        loss_b = self.boundary_loss(pred, sdf)
        if self.uncertainty:
            with torch.no_grad():
                self.log_var_b.clamp_(*self.log_var_range)
            precision_b = torch.exp(-self.log_var_b)
            total = loss_s + gate * (precision_b * loss_b + 0.5 * self.log_var_b)
            beta_eff = float(precision_b.detach())
        else:
            beta_eff = self.boundary_weight * float(gate)
            total = loss_s + beta_eff * loss_b

        self.last_stats = {'gate': float(gate),
                           'dice': None if dice is None else float(dice),
                           'beta_eff': beta_eff}
        return total
