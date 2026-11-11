import math
import os
import random
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets import FullDataset
from models import FDSUNeXT
from losses import CombinedLoss
from utils.augment import AugmentConfig
from utils.metrics import ConfusionMatrix, boundary_f_score
from utils.split import create_split, save_split, load_split


parser = argparse.ArgumentParser("FDS-UNeXT")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--dinov2_path", type=str, required=True,
                    help="path to the pretrained dinov2")
parser.add_argument("--train_image_path", type=str, required=True)
parser.add_argument("--train_mask_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--num_classes", type=int, default=4,
                    help="类别数（含背景；私有数据集现为 4 分类："
                         "0 背景/1 高反照率彩钢板/2 低反照率彩钢板/3 瓦片，"
                         "公共数据集显式指定，如二值传 2）")
parser.add_argument("--boundary_weight", type=float, default=0.2,
                    help="β 初值；uncertainty 开启时仅为 precision_b 初值（附录 Q28）")
parser.add_argument("--weit_coeff", type=float, default=5.0,
                    help="wBCE weit 边界系数（0 = weit≡1 退化为纯 CE，nosup "
                         "消融开关；默认 5 = 既有行为。2026-11-03 评审 P0-1A 新增）")
parser.add_argument("--warmup_epochs", type=int, default=0,
                    help="epoch 门轮数，0 = epoch//2（附录 Q19）")
parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True,
                    help="Dice 自适应调度边界损失（附录 Q22 方案1/Q23）")
parser.add_argument("--uncertainty", action=argparse.BooleanOptionalAction, default=True,
                    help="不确定性加权（附录 Q26/Q27）")
parser.add_argument("--dice_start", type=float, default=0.5,
                    help="Dice 门起点：宏平均 Dice 低于此值边界损失精确关闭"
                         "（Q24 smoothstep，真零点）")
parser.add_argument("--dice_end", type=float, default=0.7,
                    help="Dice 门满权点：宏平均 Dice 高于此值边界损失满权（Q24）")
parser.add_argument("--aux_boundary_weight", type=float, default=0.5,
                    help="方案 B 辅助边界头权重（教程 8.2 默认 0.5）")
parser.add_argument("--use_fdm", action=argparse.BooleanOptionalAction,
                    default=True, help="改进一 FDM 频率分解（消融开关）")
parser.add_argument("--use_dysample", action=argparse.BooleanOptionalAction,
                    default=True, help="改进二 DySample 动态上采样（消融开关）")
parser.add_argument("--use_ba", action=argparse.BooleanOptionalAction, default=True,
                    help="方案 A 边界注意力门（仅浅层 skip x1/x2）")
parser.add_argument("--use_boundary_head", action=argparse.BooleanOptionalAction,
                    default=True, help="方案 B 辅助边界预测头")
parser.add_argument("--use_dcn", action=argparse.BooleanOptionalAction, default=True,
                    help="方案 C 可变形卷积精炼")
# ---- 单流消融（2026-11-05，评审 P2「单流对照」；详见 models/fds_unext.py）----
# 与结构开关同风格：默认双流 = 既有 run 零影响；单流必须同时 --no-use_fdm
parser.add_argument("--use_dino", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="单流消融：--no-use_dino = SAM2-only（DINOv2 流与 "
                         "align/ds_align 支路整体移除，FDM 须同时关闭）")
parser.add_argument("--use_sam", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="单流消融：--no-use_sam = DINOv2-only（不构建 SAM2 "
                         "编码器；解码器容量与 full 一致）")
parser.add_argument("--use_mlp_head", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="MLP 全局语义融合头（Q123，零初始化 logit 并联）")
parser.add_argument("--mlp_fuse_dim", type=int, default=384,
                    help="MLP 融合头隐层宽（v3-c 减容候选 256；"
                         "默认 384 = v2 逐字）")
parser.add_argument("--mlp_lr_scale", type=float, default=1.0,
                    help="mlp_head 分支 lr 系数（× --lr；默认 1.0 = 单组"
                         "与 v2 逐字一致；v3-a 低 lr 候选 0.1）")
parser.add_argument("--dy_edge_offset", type=float, default=4.0,
                    help="贴边四模块（ds_align1a/1b/2、up1.up）偏移预算"
                         " LR 像素（R3-DY 候选 6；默认 4 = 既有 run 逐字，"
                         "上限 2 LR 像素）")
parser.add_argument("--hiera_size", type=str, default="large",
                    choices=["tiny", "small", "base_plus", "large"],
                    help="SAM2 Hiera 编码器尺寸（D3 轻量化实验；默认 "
                         "large = 既有 run 逐字，stage 通道 144/288/576/"
                         "1152；base_plus = 112/224/448/896，权重须匹配）")
parser.add_argument("--dino_size", type=str, default="large",
                    choices=["small", "base", "large"],
                    help="DINOv2 编码器尺寸（默认 large = 1024ch 既有 run "
                         "逐字；base = 768ch，权重须匹配）")
parser.add_argument("--seed", type=int, default=42,
                    help="全局随机种子：切分/洗牌/增广全链路固定"
                         "（GPU 上 cudnn 非确定性为尽力复现）")
parser.add_argument("--val_split", type=float, default=0.1,
                    help="验证集比例（仅在 split.json 不存在时用于生成）")
parser.add_argument("--test_split", type=float, default=0.2,
                    help="测试集比例（仅写入清单，训练全程不触碰）")
parser.add_argument("--group_pattern", type=str, default=None,
                    help="景级切分：文件名正则第 1 捕获组为景标识"
                         "（如 '(scene\\\\d+)_'），整景进入同一边防跨景泄漏")
parser.add_argument("--label_mapping", type=str, default=None,
                    help="公共数据集标签重映射，'k:v,k:v' 格式（LUT）。"
                         "二值 0/255 数据集：'255:1'（配 --num_classes 2）；"
                         "ISPRS 杂乱类并背景：'6:0'")
parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                    help="混合精度训练（V100 fp16 加速约 2x；首次基线"
                         "建议 fp32 复现，验证收敛后再开启）")
parser.add_argument("--fp32_forward", action="store_true",
                    help="诊断用：真正关闭 autocast（绕开 bool(scaler) 恒真）。"
                         "默认关闭，历史 run 行为不变")
parser.add_argument("--clip_grad", type=float, default=1.0,
                    help="梯度范数裁剪上限（inf norm），0 = 关闭。默认"
                         "开启：无裁剪时单步梯度尖峰会永久 NaN 化权重且"
                         "无法恢复（公共集 V100 首跑 epoch 7 实测，附录 Q93）")
parser.add_argument("--split_path", type=str, default=None,
                    help="三划分清单路径；默认 <save_path>/split.json，"
                         "已存在则复用（保证多次训练/评估同一切分）")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--augment", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="动态数据增强总开关（附录 Q89）；关闭则仅保留"
                         "随机翻转（旧行为）")
parser.add_argument("--mosaic", action=argparse.BooleanOptionalAction,
                    default=True, help="Mosaic9 九图马赛克（YOLOv9）；"
                    "训练最后 --mosaic_close_epochs 轮自动关闭"
                    "（ultralytics close_mosaic 惯例：末期在真实分布下微调）")
parser.add_argument("--mosaic_prob", type=float, default=1.0)
parser.add_argument("--mosaic_close_epochs", type=int, default=10)
parser.add_argument("--mosaic_open_epoch", type=int, default=1,
                    help="Mosaic9 首开轮（1-based，默认 1=全程开；"
                         "取 6 则前 5 轮干净热身、E06 起启用）")
parser.add_argument("--mosaic_scale", type=float, nargs=2,
                    default=[0.7, 1.0], metavar=('MIN', 'MAX'),
                    help="马赛克瓦片随机缩放范围（相对训练尺寸）；"
                    "灰底填充占比随下界降低而增大")
parser.add_argument("--copy_paste", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="稀有类连通域粘贴（IGARSS 2024 Cut-and-Paste，"
                    "按 1/瓦片数逆频率加权选类，利好瓦片类等稀有类）")
parser.add_argument("--copy_paste_prob", type=float, default=0.3)
parser.add_argument("--res_scale", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="1.25×/0.8× 分辨率模拟增广（核心设计要素）")
parser.add_argument("--res_scale_prob", type=float, default=0.5)
parser.add_argument("--rot90", action=argparse.BooleanOptionalAction,
                    default=True, help="随机 90° 旋转（正射影像无方向先验）")
parser.add_argument("--photo_metric", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="乘性增益光度增强；默认关——高/低反照率类别由"
                    "亮度比定义，增益保比值但保守起见不默认启用")
parser.add_argument("--photo_gain", type=float, nargs=2,
                    default=[0.9, 1.1], metavar=('MIN', 'MAX'))
parser.add_argument("--resume", default=None,
                    help="断点续跑：加载的 checkpoint 路径（仅模型权重；"
                         "优化器动量冷启动，配合 --resume_epoch）")
parser.add_argument("--resume_epoch", type=int, default=0,
                    help="续跑起始 epoch（0 基，如 ep5 快照传 5）；"
                         "cosine 调度快进相应步数，results.csv 追加")
parser.add_argument("--val_bf_tol", type=int, default=8,
                    help="验证集逐轮 Boundary F-score 容差（像素），"
                         "0 = 关闭（省时）。与 eval.py / train_compare.py "
                         "的 bf_tol 同口径，便于追踪边缘类改进（Q114）")
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader worker 内的 python/numpy 随机源（翻转增广用）。"""
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_label_mapping(s):
    """'255:1,6:0' -> {255: 1, 6: 0}"""
    if not s:
        return None
    mapping = {}
    for kv in s.split(','):
        k, v = kv.split(':')
        mapping[int(k)] = int(v)
    return mapping


def list_pairs(image_root, gt_root):
    images = sorted(os.path.join(image_root, f) for f in os.listdir(image_root)
                    if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff')))
    gts = sorted(os.path.join(gt_root, f) for f in os.listdir(gt_root)
                 if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff')))
    assert len(images) == len(gts), \
        f"images({len(images)}) and gts({len(gts)}) mismatch"
    return images, gts


def build_split(args):
    """三划分清单：已存在则复用，否则按 seed 生成并落盘。"""
    path = args.split_path or os.path.join(args.save_path, 'split.json')
    if os.path.exists(path):
        print(f"reusing split: {path}")
        return load_split(path)
    images, gts = list_pairs(args.train_image_path, args.train_mask_path)
    train_ratio = 1.0 - args.val_split - args.test_split
    split = create_split(images, gts,
                         ratios=(train_ratio, args.val_split, args.test_split),
                         seed=args.seed, group_pattern=args.group_pattern)
    save_split(split, path)
    print(f"created split: {path} (seed={args.seed}, "
          f"group_pattern={args.group_pattern})")
    return split


def trace_first_nonfinite(model, x):
    """坏批定位（Q104）：带前向钩子重跑一次，返回第一个输出非有限的模块。"""
    first = [None]

    def make_hook(name):
        def hook(mod, inp, out):
            if first[0] is None:
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for o in outs:
                    if (torch.is_tensor(o) and o.is_floating_point()
                            and not bool(torch.isfinite(o).all())):
                        first[0] = f"{name} ({mod.__class__.__name__})"
                        break
        return hook

    handles = [m.register_forward_hook(make_hook(n))
               for n, m in model.named_modules()]
    try:
        with torch.no_grad():
            model(x, with_boundary=True)
    finally:
        for h in handles:
            h.remove()
    return first[0] or '未定位（模块输出均有限——疑似损失函数内部溢出）'


def repair_bn_buffers(model, bad_names, snapshot):
    """回滚被坏批前向污染的非有限 buffer；无快照项重置为中性值。"""
    bufs = dict(model.named_buffers())
    for n in bad_names:
        if snapshot and n in snapshot:
            bufs[n].copy_(snapshot[n])
        elif n.endswith('running_mean'):
            bufs[n].zero_()
        elif n.endswith('running_var'):
            bufs[n].fill_(1.0)


def run_pass(model, loader, criterion, device, epoch, args, optim=None,
             scaler=None, canary_x=None, guard_state=None):
    """单轮训练（optim 非 None）或验证（optim=None）。

    混淆矩阵复用损失计算已有的 logits：argmax + bincount 单 kernel，
    相对前向开销可忽略，训练集 mIoU 不产生任何额外前向。验证损失与
    训练损失同口径（含门控边界项与方案 B 辅助项）。
    """
    train = optim is not None
    model.train(train)
    cm = ConfusionMatrix(args.num_classes, device=device)
    # 验证集逐轮 Boundary F-score（Q114：公共集 mIoU 已饱和而边界指标远
    # 未饱和——UNet-R34 test mIoU 0.936 而 macro BF 仅 0.679，边缘类改进
    # 须在训练期可见）。train 恒 0，不产生额外开销
    bf_tol = 0 if train else getattr(args, 'val_bf_tol', 0)
    bf_all = []
    # 注意（论文 §3.5.2 已披露）：这里用 bool(scaler) 判定 autocast，而
    # GradScaler 只要创建即为真——因此无论 --amp 是否开启，前向都在 fp16
    # 下执行，参数/梯度/优化器状态仍为 fp32。论文报告的所有数值都在这一
    # 状态下得到。--fp32_forward 仅用于精度对照 run：显式关闭 autocast，
    # 默认 False 时行为与历史 run 完全一致。
    fp32_forward = getattr(args, 'fp32_forward', False)
    amp_ctx = torch.amp.autocast('cuda',
                                 enabled=bool(scaler) and not fp32_forward)
    loss_sum, aux_sum, n_seen, nan_skips, consec_bad = 0.0, 0.0, 0, 0, 0
    canary_used = 0
    traced = False
    trace_done = False
    oom_skips, consec_oom = 0, 0
    params = ([p for g in optim.param_groups for p in g['params']]
              if train else [])
    ctx = nullcontext() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"E{epoch + 1:02d} {'train' if train else 'val'}",
                    dynamic_ncols=True, leave=False)
        for batch_idx, batch in enumerate(pbar):
            x = batch['image'].to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)
            sdf = batch['sdf'].to(device, non_blocking=True)

            if train:
                optim.zero_grad()
            try:
                with amp_ctx:
                    pred, boundary_pred = model(x, with_boundary=True)
                    loss = criterion(pred, target, sdf, current_epoch=epoch)
                    aux = None
                    if boundary_pred is not None:
                        aux = criterion.boundary_head_loss(boundary_pred,
                                                           target)
                        loss = loss + args.aux_boundary_weight * aux
                if train:
                    bad = not math.isfinite(loss.item())
                    fail = 'loss'
                    if not bad:
                        scaler.scale(loss).backward()
                        # amp 时必须先 unscale 再裁剪/检查（enabled=False 时
                        # unscale_/step 均为直通）
                        scaler.unscale_(optim)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            params, args.clip_grad if args.clip_grad > 0
                            else float('inf'))
                        # 损失有限 ≠ 梯度有限：梯度含 NaN/Inf 时范数非有限，
                        # 裁剪退化为 NaN 乘法（Inf×0=NaN）无法净化，必须
                        # 整步跳过（Q96：b2@5e-5 epoch 7 有限损失 + 非有限
                        # 梯度毒化权重）
                        if bool(torch.isfinite(grad_norm)):
                            scaler.step(optim)
                            scaler.update()
                            consec_bad = 0
                        else:
                            bad = True
                            fail = 'grad'
                if train and bad:
                    # 跳过该优化步并清梯度：amp=False 时 GradScaler 无
                    # inf 梯子防护，单批坏梯度会永久 NaN 化权重
                    optim.zero_grad(set_to_none=True)
                    scaler.update()
                    nan_skips += 1
                    consec_bad += 1
                    print(f"  [skip] ep{epoch + 1} 批 {batch_idx + 1}/"
                          f"{len(loader)}：{fail}非有限")
                    if fail == 'loss':
                        if not traced:
                            # 分量级定位（Q105）：trace 只看模块输出，损失
                            # 内部溢出时抓不到——逐分量重算并打印，区分
                            # NaN/±Inf、pred 幅值与各损失项（先用本批张量）
                            traced = True
                            with torch.no_grad():
                                lv = float(loss.item())
                                kind = ('nan' if math.isnan(lv)
                                        else ('+inf' if lv > 0 else '-inf'))
                                pd = pred.detach()
                                pa = pd.abs().max().item()
                                l_s = criterion.structure_loss(
                                    pd, target).item()
                                a_v = (criterion.boundary_head_loss(
                                           boundary_pred.detach(),
                                           target).item()
                                       if boundary_pred is not None else None)
                            print(f"  [comp] ep{epoch + 1}：loss={kind} "
                                  f"pred_absmax={pa:.3e} "
                                  f"pred_finite="
                                  f"{bool(torch.isfinite(pd).all())} "
                                  f"structure={l_s} aux={a_v}")
                        # 显存卫生（Q113）：loss 非有限的批未 backward，
                        # 其计算图（全模型 fp32@1024 激活，10+GB）被
                        # pred 引用持有，活到下一批前向完成才释放——
                        # 连续坏批时新旧两份激活图叠加 → V100 32GB OOM
                        # （E06 批 512-517 实测：27.45GB allocated）。
                        # 必须在 continue 前立即置空释放
                        pred = boundary_pred = None
                        loss = aux = None
                        if not trace_done:
                            # 首个坏批带钩子重跑前向，定位第一个非有限
                            # 输出的模块（Q104：数值级根因定位）——放在
                            # 释放之后：重放峰值 = 一份 no_grad 前向，
                            # 不与坏批计算图叠加
                            trace_done = True
                            site = trace_first_nonfinite(model, x)
                            print(f"  [trace] ep{epoch + 1}：首个非有限输出"
                                  f"位于 {site}")
                    if consec_bad >= 20:
                        # 状态三分支排查（Q104）：
                        # ① 参数非有限 → 真权重中毒，不可恢复，终止
                        # ② 仅 BN running stats 非有限 → 坏批的 train 前
                        #    向在损失检查之前就更新了它们（防护盲区），
                        #    eval 路径被污染而训练路径健康——回滚快照继续
                        # ③ 全有限 → 病态数据/输入触发的前向溢出，跳批继续
                        # 每轮最多放行 3 次，防僵尸化
                        with torch.no_grad():
                            params_bad = any(not torch.isfinite(p).all()
                                             for p in model.parameters())
                        if params_bad:
                            crash_path = os.path.join(args.save_path,
                                                      'FDS-UNeXT-crash.pth')
                            torch.save(model.state_dict(), crash_path)
                            raise SystemExit(
                                f"epoch {epoch + 1}：连续 20 批非有限且参数"
                                "已含 NaN——真权重中毒，训练不可恢复。爆点权"
                                f"重已存 {crash_path}；请携带 [trace]/[comp]/"
                                "[skip] 日志归档本次 save_path 后重跑")
                        bad_bufs = [n for n, b in model.named_buffers()
                                    if b.is_floating_point()
                                    and not torch.isfinite(b).all()]
                        if bad_bufs:
                            repair_bn_buffers(model, bad_bufs,
                                              (guard_state or {})
                                              .get('snap'))
                            print(f"  [repair] ep{epoch + 1} 批 "
                                  f"{batch_idx + 1}：{len(bad_bufs)} 个 BN "
                                  f"running stats 被坏批前向污染（如 "
                                  f"{bad_bufs[0]}），已回滚最近全有限快照")
                        if canary_x is not None:
                            model.eval()
                            with torch.no_grad():
                                c_pred, _ = model(canary_x,
                                                  with_boundary=True)
                            ok = bool(torch.isfinite(c_pred).all().item())
                            model.train()
                            print(f"  [canary] ep{epoch + 1} 批 "
                                  f"{batch_idx + 1}：eval 前向有限={ok}"
                                  "（False=溢出与 stats 无关，为输入/权重"
                                  "幅值触发）")
                        canary_used += 1
                        if canary_used > 3:
                            crash_path = os.path.join(args.save_path,
                                                      'FDS-UNeXT-crash.pth')
                            torch.save(model.state_dict(), crash_path)
                            raise SystemExit(
                                f"epoch {epoch + 1}：非有限批串反复触发超 "
                                "3 次——疑似系统性前向溢出而非偶发病态批，"
                                f"爆点权重已存 {crash_path}（供离线复现），"
                                "请携带 [trace]/[comp]/[skip]/[repair] 日志"
                                "复盘")
                        consec_bad = 0
                        print(f"  [resume] ep{epoch + 1} 批 {batch_idx + 1}"
                              f"：参数有限，跳批继续（{canary_used}/3）")
                    if fail == 'loss':
                        # 非有限损失不进入轮统计：Q103 前跳批的 NaN loss
                        # 会计入 loss_sum 污染轮均值
                        consec_oom = 0    # 本批前向/诊断均未 OOM，打断连击
                        continue
            except torch.cuda.OutOfMemoryError:
                if not train:
                    # val 前向 OOM：无优化器状态风险，保留原崩溃行为
                    raise
                # 训练批 OOM（Q113）：若发生在 step 之前权重完好；即使
                # 在 backward 中途，zero_grad 也足以清理半算梯度——
                # 释放本批引用与缓存后跳批，不再让单次 OOM 毁掉整轮
                optim.zero_grad(set_to_none=True)
                pred = boundary_pred = None
                loss = aux = None
                torch.cuda.empty_cache()
                oom_skips += 1
                consec_oom += 1
                if consec_oom >= 10:
                    crash_path = os.path.join(args.save_path,
                                              'FDS-UNeXT-crash.pth')
                    torch.save(model.state_dict(), crash_path)
                    raise SystemExit(
                        f"epoch {epoch + 1}：连续 10 批 CUDA OOM——疑似"
                        "显存碎片化死锁而非偶发批，权重已存 "
                        f"{crash_path}；请携带 [oom-skip] 日志复盘")
                print(f"  [oom-skip] ep{epoch + 1} 批 {batch_idx + 1}/"
                      f"{len(loader)}：CUDA OOM，已释放本批与缓存后跳过"
                      f"（权重未更新，连击 {consec_oom}/10）")
                continue
            consec_oom = 0

            bs = x.size(0)
            loss_sum += loss.item() * bs
            if aux is not None:
                aux_sum += aux.item() * bs
            n_seen += bs
            cm.update_from_logits(pred.detach(), target)

            if bf_tol:
                p = pred.detach().argmax(1)
                for i in range(p.size(0)):
                    bf_all.append(boundary_f_score(
                        p[i], target[i].squeeze(0), args.num_classes, bf_tol))

            if train:
                st = criterion.last_stats
                dice = 'n/a' if st['dice'] is None else f"{st['dice']:.3f}"
                aw = getattr(model, '_amax_watch', None)
                amax = (f"{aw['v']:.1e}" if aw else 'n/a')
                pbar.set_postfix(loss=f"{loss_sum / n_seen:.4f}", dice=dice,
                                 aux=f"{aux_sum / n_seen:.3f}",
                                 gate=f"{st['gate']:.3f}",
                                 amax=amax,
                                 lr=f"{optim.param_groups[0]['lr']:.1e}")
            else:
                pbar.set_postfix(loss=f"{loss_sum / n_seen:.4f}")
    if train and nan_skips:
        print(f"  [warn] {nan_skips}/{len(loader)} 批损失/梯度非有限，已跳"
              f"过其优化步（附录 Q93/Q96/Q104 防护；偶发可忽略，持续出现"
              f"请携带 [trace] 日志复盘）")
    if train and oom_skips:
        print(f"  [warn] {oom_skips}/{len(loader)} 批 CUDA OOM 已跳过"
              f"（Q113 防护；偶发可忽略，连续出现请携带 [oom-skip] 日志"
              f"复盘显存碎片化）")
    if train and guard_state is not None:
        # 轮末 BN 状态维护（Q104）：非有限 running stats 回滚快照（散点
        # 污染未达 20 连触发也要在 val 前修复），全有限则更新快照
        polluted = [n for n, b in model.named_buffers()
                    if b.is_floating_point() and not torch.isfinite(b).all()]
        if polluted:
            repair_bn_buffers(model, polluted, guard_state['snap'])
            print(f"  [repair] 轮末：{len(polluted)} 个 BN running stats 非"
                  f"有限，已回滚（如 {polluted[0]}）")
        else:
            guard_state['snap'] = {n: b.detach().clone()
                                   for n, b in model.named_buffers()
                                   if b.is_floating_point()}
    bf_macro = None
    if bf_all:
        # 与 eval.py 同口径：全图非 NaN 逐类 F 的 nanmean（= nanmean(arr)）
        arr = np.array(bf_all, dtype=float)
        flat = arr[~np.isnan(arr)]
        bf_macro = float(flat.mean()) if flat.size else float('nan')
    return loss_sum / max(n_seen, 1), cm, bf_macro


def build_optimizer(model, criterion, args):
    """优化器构建（v3 参数组拆分；Q130 候选）。

    默认单组（与 v2 逐字一致）；--mlp_lr_scale ≠ 1.0 时把 mlp_head 拆为
    独立参数组（initial_lr = args.lr × scale），其余参数组保持 args.lr。
    cosine 调度按乘法因子同步衰减，两组 lr 比例全程不变。
    """
    if not (args.use_mlp_head and args.mlp_lr_scale != 1.0):
        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable += list(criterion.parameters())   # log_var_b 需随训练更新
        return opt.AdamW([{"params": trainable, "initial_lr": args.lr}],
                         lr=args.lr, weight_decay=args.weight_decay)
    mlp_params = [p for p in model.mlp_head.parameters() if p.requires_grad]
    mlp_ids = {id(p) for p in mlp_params}
    base = [p for p in model.parameters()
            if p.requires_grad and id(p) not in mlp_ids]
    base += list(criterion.parameters())   # log_var_b 需随训练更新
    print(f"[optim] mlp 参数组拆分: base initial_lr={args.lr:g} | "
          f"mlp initial_lr={args.lr * args.mlp_lr_scale:g} "
          f"(scale={args.mlp_lr_scale}, {len(mlp_params)} tensors)")
    return opt.AdamW([
        {"params": base, "initial_lr": args.lr},
        {"params": mlp_params, "initial_lr": args.lr * args.mlp_lr_scale},
    ], lr=args.lr, weight_decay=args.weight_decay)


def main(args):
    warmup = args.warmup_epochs or args.epoch // 2
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # 输入尺寸恒为 1024x1024，benchmark 启发式选核无形状抖动代价
        torch.backends.cudnn.benchmark = True
    # enabled=False 时 GradScaler/autocast 均为无操作直通
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    label_mapping = parse_label_mapping(args.label_mapping)
    augment = None
    if args.augment:
        augment = AugmentConfig(
            mosaic=args.mosaic, mosaic_prob=args.mosaic_prob,
            mosaic_close_epochs=args.mosaic_close_epochs,
            mosaic_open_epoch=args.mosaic_open_epoch,
            mosaic_scale=tuple(args.mosaic_scale),
            copy_paste=args.copy_paste, copy_paste_prob=args.copy_paste_prob,
            res_scale=args.res_scale, res_scale_prob=args.res_scale_prob,
            res_up=1.25, res_down=0.8, rot90=args.rot90,
            photo_metric=args.photo_metric,
            photo_gain=tuple(args.photo_gain))
    split = build_split(args)
    train_set = FullDataset(args.train_image_path, args.train_mask_path,
                            1024, mode='train', num_classes=args.num_classes,
                            image_list=split['train']['images'],
                            gt_list=split['train']['gts'],
                            label_mapping=label_mapping,
                            augment=augment)
    val_set = FullDataset(args.train_image_path, args.train_mask_path,
                          1024, mode='val', num_classes=args.num_classes,
                          image_list=split['val']['images'],
                          gt_list=split['val']['gts'],
                          label_mapping=label_mapping)
    print(f"train: {len(train_set)} | val: {len(val_set)} | "
          f"test: {len(split['test']['images'])} (held out) | "
          f"warmup={warmup} epochs | seed={args.seed} | amp={args.amp}")
    if augment is not None:
        if args.mosaic_open_epoch > args.epoch - args.mosaic_close_epochs:
            raise SystemExit(
                f"--mosaic_open_epoch({args.mosaic_open_epoch}) 晚于关闭轮"
                f"({args.epoch - args.mosaic_close_epochs})，Mosaic 永不启用")
        print(f"augment: {augment.summary(args.epoch)}")
        open_s = ("" if args.mosaic_open_epoch <= 1 else
                  f"，epoch {args.mosaic_open_epoch} 起启用")
        print(f"  Mosaic9{open_s}，epoch {args.epoch - args.mosaic_close_epochs} "
              f"起关闭（共 {args.epoch} 轮）")

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, generator=loader_gen,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    # canary 批（Q103）：val 首个干净样本常驻 GPU，连续跳批达阈值时
    # 前向一次以区分“权重中毒”（→终止）与“病态数据批串”（→继续）
    canary_x = val_set[0]['image'].unsqueeze(0).to(device)
    # BN running stats 快照（Q104）：轮末全有限时更新，坏批污染时回滚
    guard_state = {'snap': None}

    model = FDSUNeXT(args.hiera_path, args.dinov2_path,
                     num_classes=args.num_classes,
                     use_fdm=args.use_fdm,
                     use_dysample=args.use_dysample,
                     use_ba=args.use_ba,
                     use_boundary_head=args.use_boundary_head,
                     use_dcn=args.use_dcn,
                     use_mlp_head=args.use_mlp_head,
                     mlp_fuse_dim=args.mlp_fuse_dim,
                     dy_edge_offset=args.dy_edge_offset,
                     hiera_size=args.hiera_size,
                     dino_size=args.dino_size,
                     use_dino=args.use_dino,
                     use_sam=args.use_sam).to(device)
    if args.use_mlp_head:
        _n_mlp = sum(p.numel() for p in model.mlp_head.parameters())
        print(f"[mlp_head] fuse_dim={args.mlp_fuse_dim} "
              f"params={_n_mlp / 1e6:.3f}M")
    # 激活监控（Q105）：污染重灾 BN（up1.conv.double_conv.1）的**输入**
    # 幅值常驻 postfix——输入 ~1.8e19 时其平方上溢（E[x²]>f32 上限），
    # var=Inf→running stats 污染；监控输入才能看到爬向阈值的过程
    model._amax_watch = {'v': 0.0, 'name': ''}
    for _n, _m in model.named_modules():
        if _n.endswith('up1.conv.double_conv.1'):
            def _watch(mod, inp, out, name=_n):
                model._amax_watch['v'] = inp[0].detach().abs().max().item()
                model._amax_watch['name'] = name
            _m.register_forward_hook(_watch)
            break
    criterion = CombinedLoss(num_classes=args.num_classes,
                             boundary_weight=args.boundary_weight,
                             weit_coeff=args.weit_coeff,
                             warmup_epochs=warmup,
                             adaptive=args.adaptive,
                             dice_start=args.dice_start,
                             dice_end=args.dice_end,
                             uncertainty=args.uncertainty).to(device)

    optim = build_optimizer(model, criterion, args)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    # 断点续跑（Q113）：加载模型权重（优化器动量冷启动，AdamW 一阶
    # 动量数批内重建，影响可忽略）；cosine 快进 start_epoch 步恢复 lr
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        # staged 消融开关（Q119/E1）：允许从「无该模块」的权重 resume 到
        # 「有该模块」的架构（如 B 臂 E20 快照 + --use_dysample 开启）。
        # DySample 零初始化（初始输出严格等于 bilinear AC=True，dysample.py
        # L53-61）保证缺失参数以零值引入是前向无损的；其余模块（FDM/BA/
        # 边界头/DCN）构造时随机初始化，静默缺失 = 随机参数混入，必须拦截。
        _staged_ok = ('.up.offset.', '.up.scope.', 'ds_align', 'ds_final',
                  'mlp_head.')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise SystemExit(f"[resume] 权重含当前架构没有的参数（resume "
                             f"了错误的 checkpoint？）: {unexpected[:6]}")
        bad = [k for k in missing if not any(s in k for s in _staged_ok)]
        if bad:
            raise SystemExit(f"[resume] 缺失参数超出 staged 白名单（DySample "
                             f"offset/scope/ds_*），架构与权重不匹配: {bad[:6]}")
        if missing:
            print(f"[resume] staged 引入 {len(missing)} 个 DySample 参数"
                  f"（零初始化，前向 ≡ bilinear 基线）")
        start_epoch = max(0, min(args.resume_epoch, args.epoch))
        for _ in range(start_epoch):
            scheduler.step()
        print(f"[resume] 已加载 {args.resume}，从 epoch {start_epoch + 1} "
              f"继续（cosine 快进 {start_epoch} 步，"
              f"lr={optim.param_groups[0]['lr']:.2e}）")

    os.makedirs(args.save_path, exist_ok=True)
    # 两份权重：FDS-UNeXT-best.pth 按验证 mIoU 选点（论文所有表格都用这份），
    # FDS-UNeXT-best-bf.pth 按验证边界 F-score 选点（仅作参考）
    best_path = os.path.join(args.save_path, 'FDS-UNeXT-best.pth')
    csv_path = os.path.join(args.save_path, 'results.csv')
    # 续跑时 csv 追加（保留崩前各轮记录），冷启动覆盖写
    resume_csv = bool(args.resume) and os.path.exists(csv_path)
    csv_file = open(csv_path, 'a' if resume_csv else 'w',
                    encoding='utf-8')
    if not resume_csv:
        csv_file.write("epoch,time_s,lr,train_loss,train_miou,train_oa,"
                       "val_loss,val_miou,val_oa,val_mf1,val_bf\n")

    best_miou, best_epoch = -1.0, 0
    best_bf, best_bf_epoch = float('-inf'), 0
    if resume_csv:
        # 从既有 csv 恢复 best（防止重启后较低的 val 覆盖 best.pth）
        import csv as _csv
        with open(csv_path, encoding='utf-8') as f:
            for row in _csv.DictReader(f):
                try:
                    v = float(row['val_miou'])
                    if not math.isnan(v) and v > best_miou:
                        best_miou, best_epoch = v, int(row['epoch'])
                    b = float(row.get('val_bf', 'nan'))
                    if not math.isnan(b) and b > best_bf:
                        best_bf, best_bf_epoch = b, int(row['epoch'])
                except (ValueError, KeyError, TypeError):
                    pass
        print(f"[resume] best val mIoU {best_miou:.4f} (epoch "
              f"{best_epoch})，续跑需超过此值才覆盖 best.pth")
    for epoch in range(start_epoch, args.epoch):
        t0 = time.time()
        # 传播 Mosaic 关闭调度（worker 每轮重建，状态随数据集拷贝生效）
        train_set.set_epoch(epoch, args.epoch)
        lr = optim.param_groups[0]['lr']

        train_loss, cm_train, _ = run_pass(model, train_loader, criterion,
                                           device, epoch, args, optim=optim,
                                           scaler=scaler, canary_x=canary_x,
                                           guard_state=guard_state)
        val_loss, cm_val, val_bf = run_pass(model, val_loader, criterion,
                                            device, epoch, args, optim=None,
                                            scaler=scaler)
        scheduler.step()

        tr, va = cm_train.summary(), cm_val.summary()
        dt = time.time() - t0

        tag = ''
        if not math.isnan(va['miou']) and va['miou'] > best_miou:
            best_miou, best_epoch = va['miou'], epoch + 1
            torch.save(model.state_dict(), best_path)
            tag = f'  *best* saved -> {best_path}'
        # best-by-valBF 旁路存档（Q117）：公共集 mIoU 已饱和（~0.94），
        # mIoU-best 未必是边界最优；额外按 val BF 存 FDS-UNeXT-best-bf.pth，
        # 不动原判据，评估期按指标口径各取所需
        if val_bf is not None and not math.isnan(val_bf) and val_bf > best_bf:
            best_bf, best_bf_epoch = val_bf, epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(args.save_path,
                                    'FDS-UNeXT-best-bf.pth'))
            tag += f'  *best-bf* {best_bf:.4f}'

        print(f"[Epoch {epoch + 1:02d}/{args.epoch}] {dt:.1f}s  lr {lr:.1e}")
        print(f"  train | loss {train_loss:.4f} | mIoU {tr['miou']:.4f} | "
              f"OA {tr['oa']:.4f}")
        bf_str = (f" | BF {val_bf:.4f}"
                  if val_bf is not None and not math.isnan(val_bf) else '')
        print(f"  val   | loss {val_loss:.4f} | mIoU {va['miou']:.4f} | "
              f"OA {va['oa']:.4f} | mF1 {va['mf1']:.4f}{bf_str}{tag}")
        bf_best = (f" | BF {best_bf:.4f} (epoch {best_bf_epoch})"
                   if best_bf_epoch else '')
        print(f"  best  | val mIoU {best_miou:.4f} (epoch {best_epoch})"
              f"{bf_best}")

        csv_file.write(f"{epoch + 1},{dt:.1f},{lr:.2e},{train_loss:.6f},"
                       f"{tr['miou']:.6f},{tr['oa']:.6f},"
                       f"{val_loss:.6f},{va['miou']:.6f},"
                       f"{va['oa']:.6f},{va['mf1']:.6f},"
                       f"{val_bf if val_bf is not None else float('nan'):.6f}\n")
        csv_file.flush()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            path = os.path.join(args.save_path, 'FDS-UNeXT-%d.pth' % (epoch + 1))
            torch.save(model.state_dict(), path)
            print('[Saving Snapshot:]', path)

    csv_file.close()


if __name__ == "__main__":
    main(args)
