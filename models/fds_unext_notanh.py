import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import trunc_normal_

from sam2.build_sam import build_sam2
from .dysample import DySample
from .mlp_head import MLPFusionHead
from .fdm import FrequencyDecompositionModule
from .boundary_modules import BoundaryAttentionGate, BoundaryRefinement
from .norm_utils import gn

# ===== 无 tanh 副本（消融：adapter 残差不做有界约束）=====
# 本文件 = models/fds_unext.py 的逐字节副本，唯一差异 = Adapter.forward
# 移除 Q113 的 prompt tanh 软界（用户指令：复制后修改，原文件不动）。
# state_dict 键与原版完全一致（软界无参数），checkpoint 互通。


def _load_hiera_trunk(trunk, ckpt_path):
    """SAM2 / SAM2.1 checkpoint 通用的 trunk-only 权重加载（附录 Q86）。

    两代官方 checkpoint 的 Hiera trunk 配置与权重键完全一致（差异全在
    memory/obj_ptr 等本模型已删除的模块），故只抽取 `image_encoder.trunk.*`
    子字典 strict 灌入：既兼容 2.1 多出的 obj_ptr_tpos_proj 等键，也免去
    加载即将 delattr 的 decoder/memory 权重（快且省 RAM）。也接受
    trunk-only 格式（无前缀）的 state dict。
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    prefix = "image_encoder.trunk."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items()
              if k.startswith(prefix)}
    missing, unexpected = trunk.load_state_dict(sd, strict=True)
    return len(sd)


class Adapter(nn.Module):
    """Hiera block 的 Prompt 微调适配器（沿用 SAM2-UNeXT 原设计）。"""

    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        # 无 tanh 变体（本文件唯一行为差异）：Q113
        # 软界是 E-GCS lr3e-5 三连崩后的应急防护——彼时 BN→GN（Q107）
        # 已落地仍爆，根因为 48 层加性注入的结构性无界累加，非 BN。
        # Potsdam 冠军协议（R3-DY）amax 尾部 560-620 无漂移、总步数
        # 14.6k 远低于 E-GCS 崩溃阈值（25-40k），本副本去掉软界检验：
        # (a) amax 稳定性 (b) 界内失真（<8%）去除后的精度变化。
        # 原版逐字见 models/fds_unext.py；软界无参数，state_dict 互通
        return self.block(x + prompt)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.prompt_learn.apply(_init_weights)


class DoubleConv(nn.Module):
    """(conv => GN => ReLU) * 2（Q107：BN→GN，见 _gn 注释）"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            gn(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            gn(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """解码器上采样块：DySample 内容感知上采样替代 bilinear（改进二）。

    先 concat 再上采样（沿用原版数据流）；skip 与 decoder 特征分辨率一致，
    原版 F.pad 为死代码，已移除（附录工程约定）。
    """

    def __init__(self, in_channels, out_channels, use_dysample=True,
                 offset_pixels=4.0):
        super().__init__()
        if use_dysample:
            self.up = DySample(in_channels, scale=2,
                               offset_pixels=offset_pixels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


# ===== 编码器规格表（轻量化变体：替换 Hiera 与 DINOv2 的尺寸）=====
# hiera_size -> (build_sam2 配置名, 各 stage 通道 c1-c4)；
#   Hiera trunk 各 stage 通道 = embed_dim × [1,2,4,8]
# dino_size -> (timm 模型名, 输出通道；448²/patch14 → 32×32 tokens)
# 默认 large/large = 改造前逐字行为（state_dict 键与形状均不变）
_HIERA_SPECS = {
    # SAM2 仓库 Hiera-T 与 S 同宽（embed 96，仅深度差：T=[1,2,7,2] vs
    # S=[1,2,11,2]，D5 冒烟实测 t.yaml stage1=96ch 修正，非 Hiera 论文 64 系）
    "tiny":      ("sam2_hiera_t.yaml",   [96, 192, 384, 768]),
    "small":     ("sam2_hiera_s.yaml",   [96, 192, 384, 768]),
    "base_plus": ("sam2_hiera_b+.yaml", [112, 224, 448, 896]),
    "large":     ("sam2_hiera_l.yaml",  [144, 288, 576, 1152]),
}
_DINO_SPECS = {
    "small": ("vit_small_patch14_dinov2", 384),
    "base":  ("vit_base_patch14_dinov2",  768),
    "large": ("vit_large_patch14_dinov2", 1024),
}


class FDSUNeXT(nn.Module):
    """FDS-UNeXT: Frequency Decomposition and dynamic Sampling UNeXT.

    在 SAM2-UNeXT（Hiera-L + DINOv2 双编码器 + UNet 解码器）基础上的
    三项改进（对应论文方法部分）：
      改进一 FDM：频率分解模块替代 Dense Glue Layer 的 concat+1x1
      改进二 DySample：解码器 Up / DINOv2 对齐 / 最终输出三处的
                       bilinear 上采样全部替换为内容感知动态采样
      改进三 组合损失：wBCE(weit,k=7) + IoU(无 weit) + SDF 边界损失
                       （损失见 losses/combined_loss.py）

    另有三个边界监督分支（论文 §3.3）：
      方案 A 边界注意力门：仅作用于浅层 skip 特征 x1/x2（深层特征
               Sobel 边缘与真实边界错位，不加）
      方案 B 辅助边界预测头：1x1 卷积多任务学习，训练时经
               with_boundary=True 返回辅助输出，推理时丢弃
      方案 C 可变形卷积精炼：放在最终上采样 ds_final 之前，
               offset 零初始化（初始退化为标准 3x3 卷积）+ 残差

    Args:
        checkpoint_path: SAM2/SAM2.1 Hiera 预训练权重路径（trunk-only
          加载，附录 Q86，sam2_hiera_large.pt 与 sam2.1_hiera_large.pt
          均可；尺寸须与 hiera_size 匹配；None 则随机初始化）
        dinov2_path: DINOv2 权重路径（ViT-L/14 或 ViT-B/14，须与
          dino_size 匹配）
        num_classes: 类别数（默认 3：0 背景 / 1 高反照率彩钢板 / 2 低反照率
          彩钢板；4 分类时 3 = 瓦片）
        pretrained: 是否加载 DINOv2 ImageNet 预训练（冒烟测试可关）
        use_fdm: 启用改进一（消融开关）
        use_dysample: 启用改进二（消融开关）
        use_ba: 启用方案 A（消融开关）
        use_boundary_head: 启用方案 B（消融开关）
        use_dcn: 启用方案 C（消融开关）
        mlp_fuse_dim: MLP 融合头隐层宽（v3-c 减容候选；默认 384）
        dy_edge_offset: 贴边四模块（ds_align1a/1b/2、up1.up）偏移预算
          LR 像素（R3-DY 候选 6；默认 4 = 既有 run 逐字）
        hiera_size: SAM2 Hiera 编码器尺寸（D3 轻量化实验）——"large"
          默认 = 既有 run 逐字（stage 通道 144/288/576/1152）；
          "base_plus"=112/224/448/896、"small"=96/192/384/768、
          "tiny"=96/192/384/768（与 small 同宽，差在 trunk 深度）
        dino_size: DINOv2 编码器尺寸——"large" 默认（1024ch）/
          "base"（768ch）
    """

    def __init__(self, checkpoint_path=None, dinov2_path=None, num_classes=3,
                 pretrained=True, use_fdm=True, use_dysample=True,
                 use_ba=True, use_boundary_head=True, use_dcn=True,
                 use_mlp_head=False, mlp_fuse_dim=384, dy_edge_offset=4.0,
                 hiera_size="large", dino_size="large"):
        super().__init__()
        self.use_fdm = use_fdm
        self.use_dysample = use_dysample
        self.use_ba = use_ba
        self.use_boundary_head = use_boundary_head
        self.use_dcn = use_dcn
        self.use_mlp_head = use_mlp_head

        # ===== SAM2 Hiera 编码器（冻结 + Adapter 微调；尺寸见规格表）=====
        hiera_cfg, (c1, c2, c3, c4) = _HIERA_SPECS[hiera_size]
        dino_name, dino_ch = _DINO_SPECS[dino_size]
        # device 统一在构造完成后由外部 .to(device) 指定，构建期固定 CPU。
        # 权重经 _load_hiera_trunk 单独灌入 trunk（附录 Q86）：SAM2 与
        # SAM2.1 checkpoint 通吃，绕开 build_sam 的全模型严格匹配
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
            *[Adapter(block) for block in self.sam.blocks]
        )

        # ===== DINOv2 编码器（冻结，448x448 -> dino_ch @ 32x32）=====
        if dinov2_path:
            self.dino = timm.create_model(
                dino_name, features_only=True,
                img_size=(448, 448), pretrained=True,
                pretrained_cfg_overlay=dict(file=dinov2_path))
        else:
            self.dino = timm.create_model(
                dino_name, features_only=True,
                img_size=(448, 448), pretrained=pretrained)
        for param in self.dino.parameters():
            param.requires_grad = False

        # ===== 通道对齐（DINOv2 dino_ch -> 各 Stage 通道数）=====
        self.align1 = nn.Conv2d(dino_ch, c1, 1)
        self.align2 = nn.Conv2d(dino_ch, c2, 1)
        self.align3 = nn.Conv2d(dino_ch, c3, 1)
        self.align4 = nn.Conv2d(dino_ch, c4, 1)

        # ===== 改进一：FDM（32x32 上做频率分解与高频注入）=====
        if use_fdm:
            self.fdm1 = FrequencyDecompositionModule(c1)
            self.fdm2 = FrequencyDecompositionModule(c2)
            self.fdm3 = FrequencyDecompositionModule(c3)
            self.fdm4 = FrequencyDecompositionModule(c4)

        # ===== 改进二：DINOv2 对齐上采样（DySample）=====
        # 修正后 Hiera 维度: x1=256x256, x2=128x128, x3=64x64, x4=32x32
        # align1 32->256 采用两步 DySample(4x)+DySample(2x)（附录 Q35，
        #   仅使用论文已验证的 scale；中间 128x128 与 align2 目标一致）
        # align2 32->128 单步 DySample(4x)（论文已验证）
        # align3 32->64 DySample(2x)（部署优先级最低但坚持使用，参数
        #   0.037M，见教程 §5.3.3 CRAM3）
        # dy_edge_offset（R3-DY，Q124 贴边探针后续）：贴边四模块
        #   （ds_align1a/1b/2 与 up1.up——探针实测 EDGE 饱和 93-99%）
        #   的偏移预算；默认 4.0 = 既有 run 逐字（上限 2 LR 像素）
        if use_dysample:
            self.ds_align1a = DySample(c1, scale=4,
                                       offset_pixels=dy_edge_offset)
            self.ds_align1b = DySample(c1, scale=2,
                                       offset_pixels=dy_edge_offset)
            self.ds_align2 = DySample(c2, scale=4,
                                      offset_pixels=dy_edge_offset)
            self.ds_align3 = DySample(c3, scale=2)  # 32 -> 64

        # ===== 融合降维 =====
        self.reduce1 = nn.Conv2d(2 * c1, 128, 1)
        self.reduce2 = nn.Conv2d(2 * c2, 128, 1)
        self.reduce3 = nn.Conv2d(2 * c3, 128, 1)
        self.reduce4 = nn.Conv2d(2 * c4, 128, 1)

        # ===== UNet 解码器（Up 内 DySample）=====
        # up1.up 为贴边四模块之一（R3-DY 预算候选，见 ds_align 段注释）；
        # up2-4 维持默认 4.0
        self.up1 = Up(256, 128, use_dysample=use_dysample,
                      offset_pixels=dy_edge_offset)
        self.up2 = Up(256, 128, use_dysample=use_dysample)
        self.up3 = Up(256, 128, use_dysample=use_dysample)
        self.up4 = Up(128, 128, use_dysample=use_dysample)

        # ===== 最终输出上采样：head 之前对 128ch 特征 DySample 2x =====
        # 附录 Q36：head 后 bilinear 2x 是无参数模糊操作，改为 head 前
        # DySample 可提升边界锐利度；head 为 1x1 卷积，FLOPs 增量可忽略。
        # 消融回退 bilinear（与原版 head 后 2x 数学等价：1x1 卷积与
        # 双线性插值可交换），保证输出恒为 1024x1024
        self.ds_final = (DySample(128, scale=2) if use_dysample else
                         nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True))

        # ===== 方案 A：边界注意力门（仅浅层 skip x1/x2，附录约束）=====
        if use_ba:
            self.bag1 = BoundaryAttentionGate(128)
            self.bag2 = BoundaryAttentionGate(128)

        # ===== 方案 C：可变形卷积精炼（最终上采样之前，附录约束）=====
        self.dcn_refine = BoundaryRefinement(128) if use_dcn else None

        # ===== 方案 B：辅助边界预测头（1x1，推理时丢弃）=====
        self.boundary_head = nn.Conv2d(128, 1, 1) if use_boundary_head else None

        self.head = nn.Conv2d(128, num_classes, 1)

        # ===== Q123：MLP 全局语义融合头（logit 级并联，cls 零初始化 =====
        # → 开启瞬间 ≡ 基线；五路输入见 MLPFusionHead docstring）
        self.mlp_head = (MLPFusionHead(num_classes, fuse_dim=mlp_fuse_dim)
                         if use_mlp_head else None)
        # mlp_head 五路输入按 DINOv2-L（x_d 1024ch）设计且该线已终止
        # 存档（Q132），与轻量 dino_size 的组合不经支持，构造期拦截
        if use_mlp_head and dino_ch != 1024:
            raise NotImplementedError(
                "mlp_head 仅支持 dino_size='large'（x_d 1024ch；"
                "mlp 线已终止存档，见《MLP融合头研究存档.md》）")

    def forward(self, x, with_boundary=False):
        # 编码器
        x1_s, x2_s, x3_s, x4_s = self.sam(x)
        x_d = self.dino(F.interpolate(x, size=(448, 448), mode='bilinear'))[-1]

        d1 = self.align1(x_d)
        d2 = self.align2(x_d)
        d3 = self.align3(x_d)
        d4 = self.align4(x_d)

        # 改进一：FDM（高频注入前置在 DINOv2 原始 32x32 分辨率，附录 Q57）
        if self.use_fdm:
            x1_s, d1 = self.fdm1(x1_s, d1)
            x2_s, d2 = self.fdm2(x2_s, d2)
            x3_s, d3 = self.fdm3(x3_s, d3)
            x4_s, d4 = self.fdm4(x4_s, d4)

        # 改进二：DINOv2 空间对齐上采样
        if self.use_dysample:
            x1_d = self.ds_align1b(self.ds_align1a(d1))   # 32 -> 128 -> 256
            x2_d = self.ds_align2(d2)                     # 32 -> 128
            x3_d = self.ds_align3(d3)                     # 32 -> 64
        else:
            x1_d = F.interpolate(d1, size=x1_s.shape[-2:], mode='bilinear')
            x2_d = F.interpolate(d2, size=x2_s.shape[-2:], mode='bilinear')
            x3_d = F.interpolate(d3, size=x3_s.shape[-2:], mode='bilinear')
        x4_d = d4                                        # 32x32 恒等

        # 融合 + 解码
        x1 = self.reduce1(torch.cat([x1_s, x1_d], dim=1))
        x2 = self.reduce2(torch.cat([x2_s, x2_d], dim=1))
        x3 = self.reduce3(torch.cat([x3_s, x3_d], dim=1))
        x4 = self.reduce4(torch.cat([x4_s, x4_d], dim=1))

        # 方案 A：浅层 skip 特征边界增强（x3/x4 深层不加，附录约束）
        if self.use_ba:
            x1 = self.bag1(x1)
            x2 = self.bag2(x2)

        x = self.up4(x4)          # 32  -> 64
        x = self.up3(x, x3)       # 64  -> 128
        x = self.up2(x, x2)       # 128 -> 256
        x = self.up1(x, x1)       # 256 -> 512

        # 方案 C：最终上采样前的可变形卷积特征精炼
        if self.dcn_refine is not None:
            x = self.dcn_refine(x)

        # 方案 B：辅助边界预测（512x512，推理时丢弃）
        boundary_pred = None
        if self.boundary_head is not None:
            boundary_pred = self.boundary_head(x)

        x = self.ds_final(x)    # 512 -> 1024（DySample 或 bilinear 消融回退）
        out = self.head(x)      # (B, num_classes, 1024, 1024)，head 只调用一次
        # Q123：MLP 融合头 logit 级并联（零初始化 → 开启瞬间 out 不变；
        # x1-x4 为 reduce(+BA) 后融合特征，x_d 为 DINOv2 原始终层特征）
        if self.mlp_head is not None:
            out = out + self.mlp_head(x1, x2, x3, x4, x_d, out.shape[-2:])
        if with_boundary:
            return out, boundary_pred # boundary_pred: (B,1,H_1024,W_1024)
        return out
