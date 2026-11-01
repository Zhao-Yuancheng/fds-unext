import torch


def dwt_haar(x):
    """单级 Haar 离散小波变换（完全重构正交基，系数归一化 1/2）。

    Args:
        x: (B, C, H, W)，H、W 需为偶数
    Returns:
        LL, LH, HL, HH: 各 (B, C, H/2, W/2)
        LH 对应水平方向差（行差），HL 对应垂直方向差（列差）
    """
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, f"H/W must be even, got {H}x{W}"
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    x00 = x[:, :, :, 0, :, 0]
    x01 = x[:, :, :, 0, :, 1]
    x10 = x[:, :, :, 1, :, 0]
    x11 = x[:, :, :, 1, :, 1]
    ll = (x00 + x01 + x10 + x11) / 2
    lh = (-x00 - x01 + x10 + x11) / 2
    hl = (-x00 + x01 - x10 + x11) / 2
    hh = (x00 - x01 - x10 + x11) / 2
    return ll, lh, hl, hh


def iwt_haar(ll, lh, hl, hh):
    """单级 Haar 逆小波变换，与 dwt_haar 互为精确逆变换。

    Args:
        LL, LH, HL, HH: 各 (B, C, H/2, W/2)
    Returns:
        x: (B, C, H, W)
    """
    B, C, H2, W2 = ll.shape
    a = (ll - lh - hl + hh) / 2
    b = (ll - lh + hl - hh) / 2
    c = (ll + lh - hl - hh) / 2
    d = (ll + lh + hl + hh) / 2
    # 布局 (B, C, H2, h_in, W2, w_in)：外层新维度须插在 H2 与 W2 之间
    x = torch.stack([torch.stack([a, b], dim=-1),
                     torch.stack([c, d], dim=-1)], dim=-3)
    return x.reshape(B, C, H2 * 2, W2 * 2)
