"""像素分类指标统一实现（训练监控与离线评估共用）。

训练期只累积 C×C 混淆矩阵——update 用 bincount 单 kernel 在 GPU 完成，
相对 522M 参数的前向开销可忽略；mIoU/OA/mF1 等全部由 C×C 矩阵导出，
无逐像素重算。逐像素的边界指标（BF-Score）仅在离线评估时计算。
"""

import torch
import torch.nn.functional as F


class ConfusionMatrix:
    """C×C 混淆矩阵累积器（行 = GT，列 = 预测）。

    mIoU / mPrecision / mRecall / mF1 取宏平均（逐类等权，符合项目约定：
    对小类别公平，不使用 FWIoU 作主指标）；GT 与预测中均未出现的类
    记 NaN，由 nanmean 跳过（缺席类不惩罚也不奖励）。
    """

    def __init__(self, num_classes, device='cpu'):
        self.n = num_classes
        self.mat = torch.zeros(num_classes, num_classes,
                               dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred, target):
        """pred / target: 形状一致（(B,H,W) 或 (H,W)）的类别索引张量。"""
        p = pred.reshape(-1).long().to(self.mat.device)
        t = target.reshape(-1).long().to(self.mat.device)
        self.mat += torch.bincount(t * self.n + p,
                                   minlength=self.n ** 2).reshape(self.n, self.n)

    @torch.no_grad()
    def update_from_logits(self, logits, target):
        """logits: (B,C,H,W)，target: (B,1,H,W)。"""
        self.update(logits.argmax(dim=1), target.squeeze(1))

    def reset(self):
        self.mat.zero_()

    def matrix(self):
        return self.mat.cpu().numpy()

    def _counts(self):
        mat = self.mat.double()
        tp = torch.diagonal(mat)
        fp = mat.sum(0) - tp    # 预测为 c、GT 非 c（列和 − 对角）
        fn = mat.sum(1) - tp    # GT 为 c、预测非 c（行和 − 对角）
        return mat, tp, fp, fn

    @staticmethod
    def _div(a, b):
        return torch.where(b > 0, a / b, torch.full_like(a, float('nan')))

    def per_class(self):
        """逐类 IoU / Precision / Recall / F1（缺席类为 NaN）。"""
        _, tp, fp, fn = self._counts()
        return {
            'present': ((tp + fp + fn) > 0).tolist(),
            'iou': self._div(tp, tp + fp + fn).tolist(),
            'precision': self._div(tp, tp + fp).tolist(),
            'recall': self._div(tp, tp + fn).tolist(),
            'f1': self._div(2 * tp, 2 * tp + fp + fn).tolist(),
        }

    def summary(self):
        """全局指标：OA / mIoU / FWIoU / Kappa / mPrecision / mRecall / mF1。"""
        mat, tp, fp, fn = self._counts()
        total = mat.sum()
        keys = ['oa', 'miou', 'fwiou', 'kappa',
                'mprecision', 'mrecall', 'mf1']
        if total == 0:
            return dict.fromkeys(keys, float('nan'))

        def macro(x):
            return float(torch.nanmean(x))

        iou = self._div(tp, tp + fp + fn)
        rowsum, colsum = mat.sum(1), mat.sum(0)
        oa = float(torch.diagonal(mat).sum() / total)
        # FWIoU：逐类频率 × IoU；缺席类频率为 0，其 NaN IoU 置 0 后不参与
        fwiou = float(((rowsum / total) * torch.nan_to_num(iou, 0.0)).sum())
        pe = float((rowsum * colsum).sum() / total ** 2)
        kappa = float('nan') if pe >= 1.0 else (oa - pe) / (1 - pe)
        return {
            'oa': oa,
            'miou': macro(iou),
            'fwiou': fwiou,
            'kappa': kappa,
            'mprecision': macro(self._div(tp, tp + fp)),
            'mrecall': macro(self._div(tp, tp + fn)),
            'mf1': macro(self._div(2 * tp, 2 * tp + fp + fn)),
        }


def _boundary_band(mask):
    """二值 mask 的 3×3 "膨胀−腐蚀"边界带（与方案 B 边界 GT 同一定义）。"""
    x = mask[None, None].float()
    d = F.max_pool2d(x, 3, stride=1, padding=1)
    e = -F.max_pool2d(-x, 3, stride=1, padding=1)
    return (d - e)[0, 0] > 0


def _dilate(band, k):
    x = band[None, None].float()
    return F.max_pool2d(x, k, stride=1, padding=k // 2)[0, 0] > 0


@torch.no_grad()
def boundary_f_score(pred, target, num_classes, tolerance=2):
    """逐类 BF-Score：容差 tolerance 像素内的边界匹配 F1（离线评估用）。

    约定（主流边界 F1 实现的常见口径）：
    - 边界带 = 逐类区域的 3×3 膨胀−腐蚀（同方案 B 边界 GT、weit 的边界定义）
    - 命中 = 预测边界像素落在 GT 边界 dilate(2t+1) 带内，反之亦然
    - GT 缺席、或 GT 类覆盖整幅图像（膨胀−腐蚀边界带为空，真实边界
      被图像边缘裁掉）时记 NaN（边界未定义），由调用方 nanmean 跳过；
      GT 有而预测整类缺失 → recall=0 → F=0
    - 跨图聚合采用 image-mean（逐图 F 先平均），而非全库 micro 平均

    Args:
        pred / target: (H,W) long 张量（同一设备）
    Returns:
        长度 num_classes 的列表，逐类 F 值
    """
    scores = []
    k = 2 * tolerance + 1
    for c in range(num_classes):
        g = target == c
        if not g.any():
            scores.append(float('nan'))
            continue
        p = pred == c
        g_b = _boundary_band(g)
        if not g_b.any():
            # 类覆盖整幅图像：边界带为空（真实边界在图像边缘被裁掉），
            # 边界未定义，与缺席类同口径记 NaN
            scores.append(float('nan'))
            continue
        p_b = _boundary_band(p)
        g_d = _dilate(g_b, k)
        p_d = _dilate(p_b, k)
        n_p = int(p_b.sum())
        # 无预测边界时不惩罚 precision（空集上的 precision 无定义，取 1），
        # recall 已为 0，F 自然为 0
        precision = float((p_b & g_d).sum()) / n_p if n_p > 0 else 1.0
        recall = float((g_b & p_d).sum()) / float(g_b.sum())
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall > 0 else 0.0)
    return scores
