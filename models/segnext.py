"""SegNeXt-T (MSCAN + LightHamHead) 独立实现，供 train_compare.py 使用。

来源
----
官方实现 https://github.com/Visual-Attention-Network/SegNeXt
  - mmseg/models/backbones/mscan.py
  - mmseg/models/decode_heads/ham_head.py
  - local_configs/segnext/tiny/segnext.tiny.512x512.ade.160k.py

本文件只做一件事：把 mmcv / mmseg 依赖替换为纯 PyTorch，
结构与超参保持官方 SegNeXt-T：
  embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2],
  mlp_ratios=[8, 8, 4, 4], drop_path_rate=0.1,
  head: in_channels=[64,160,256], channels=256, ham_channels=256,
        ham_kwargs=dict(MD_R=16), dropout=0.1

与论文口径的约定
----------------
- 训练/测试均复用 train_compare.py 的同一数据管线与 utils.metrics；
  BF 的 class-equal 口径由 train_compare 的 macro mean 给出，与
  Table 3 的 SegFormer/SAM2/UNetFormer 行一致。
- 归一化层用 BatchNorm2d（官方为 SyncBN；单卡训练下二者统计口径一致，
  但 BN 在小 batch 下更 noisy，需在文中披露）。
- 预训练权重默认读取 weights/segnext_t_ade20k.pth（ADE20K 全模型，
  只取 backbone.*）；也兼容 weights/mscan_t.pth（IN-1K backbone）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


def _norm(dim):
    # 官方为 SyncBN；单卡用 BN，state_dict 键名一致，可直接加载官方权重。
    return nn.BatchNorm2d(dim)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1)),
            _norm(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1)),
            _norm(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn = self.conv3(attn + attn_0 + attn_1 + attn_2)
        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = _norm(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = _norm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
            * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
            * self.mlp(self.norm2(x)))
        return x.view(B, C, N).permute(0, 2, 1)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = _norm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x.flatten(2).transpose(1, 2), H, W


class MSCAN(nn.Module):
    def __init__(self, in_chans=3, embed_dims=(32, 64, 160, 256),
                 mlp_ratios=(8, 8, 4, 4), drop_rate=0.,
                 drop_path_rate=0.1, depths=(3, 3, 5, 2), num_stages=4):
        super().__init__()
        self.depths = list(depths)
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]
        cur = 0
        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_chans, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3, stride=2, in_chans=embed_dims[i - 1],
                    embed_dim=embed_dims[i])
            block = nn.ModuleList([
                Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                      drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.apply(_init_weights)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# LightHamHead（Hamburger / NMF2D）
# ---------------------------------------------------------------------------

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        args = {} if args is None else dict(args)
        self.spatial = args.setdefault('SPATIAL', True)
        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)
        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)
        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)
        self.rand_init = args.setdefault('RAND_INIT', True)

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.shape
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)
        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer('bases', bases)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)
        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)
        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=None):
        super().__init__(args)
        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        bases = torch.rand((B * S, D, R), device=device)
        return F.normalize(bases, dim=1)

    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        return coef * numerator / (denominator + 1e-6)


class ConvBlock(nn.Module):
    """ConvModule 的最小替代：conv + GN(32) + ReLU。"""
    def __init__(self, in_ch, out_ch, k=1, norm=True, act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, 1, k // 2, bias=not norm)]
        if norm:
            layers.append(nn.GroupNorm(32, out_ch))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Hamburger(nn.Module):
    def __init__(self, ham_channels=256, ham_kwargs=None):
        super().__init__()
        self.ham_in = ConvBlock(ham_channels, ham_channels, 1,
                                norm=False, act=False)
        self.ham = NMF2D(ham_kwargs)
        self.ham_out = ConvBlock(ham_channels, ham_channels, 1,
                                 norm=True, act=False)

    def forward(self, x):
        enjoy = F.relu(self.ham_in(x), inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        return F.relu(x + enjoy, inplace=True)


class LightHamHead(nn.Module):
    def __init__(self, in_channels=(64, 160, 256), in_index=(1, 2, 3),
                 channels=256, ham_channels=256, num_classes=4,
                 dropout=0.1, ham_kwargs=None):
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.in_index = tuple(in_index)
        self.squeeze = ConvBlock(sum(in_channels), ham_channels, 1)
        self.hamburger = Hamburger(ham_channels, ham_kwargs)
        self.align = ConvBlock(ham_channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs):
        feats = [inputs[i] for i in self.in_index]
        size = feats[0].shape[2:]
        feats = [F.interpolate(f, size=size, mode='bilinear',
                               align_corners=False) for f in feats]
        x = self.squeeze(torch.cat(feats, dim=1))
        x = self.hamburger(x)
        x = self.align(x)
        x = self.dropout(x)
        return self.cls_seg(x)


class SegNeXt(nn.Module):
    """SegNeXt-T：MSCAN-T + LightHamHead，输出 1024² logits。"""
    def __init__(self, num_classes=4, pretrained=None):
        super().__init__()
        self.backbone = MSCAN()
        self.decode_head = LightHamHead(
            in_channels=(64, 160, 256), in_index=(1, 2, 3),
            channels=256, ham_channels=256, num_classes=num_classes,
            dropout=0.1, ham_kwargs=dict(MD_R=16))
        if pretrained:
            self.load_pretrained(pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear',
                             align_corners=False)

    def load_pretrained(self, path):
        try:
            sd = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict):
            sd = sd.get('model_state', sd.get('state_dict', sd))
        clean = {}
        for k, v in sd.items():
            k = k.replace('module.', '')
            if k.startswith('model.'):
                k = k[len('model.'):]
            if k.startswith('backbone.'):
                k = k[len('backbone.'):]
            # GeoNUSAF 的 DWConv 键名少一层 .dwconv；对齐到官方 MSCAN 结构
            if ('.mlp.dwconv.weight' in k
                    and '.mlp.dwconv.dwconv.weight' not in k):
                k = k.replace('.mlp.dwconv.weight',
                              '.mlp.dwconv.dwconv.weight')
            if ('.mlp.dwconv.bias' in k
                    and '.mlp.dwconv.dwconv.bias' not in k):
                k = k.replace('.mlp.dwconv.bias',
                              '.mlp.dwconv.dwconv.bias')
            clean[k] = v
        missing, unexpected = self.backbone.load_state_dict(clean,
                                                            strict=False)
        print(f'[segnext] pretrained={path} missing={len(missing)} '
              f'unexpected={len(unexpected)}', flush=True)
