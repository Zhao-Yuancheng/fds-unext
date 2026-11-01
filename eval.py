import argparse
import warnings

import numpy as np
import torch
from tqdm import tqdm

from datasets import FullDataset
from models import FDSUNeXT
from utils.metrics import ConfusionMatrix, boundary_f_score
from utils.split import load_split


parser = argparse.ArgumentParser("FDS-UNeXT evaluation")
parser.add_argument("--hiera_path", type=str, required=True)
parser.add_argument("--dinov2_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_classes", type=int, default=4,
                    help="类别数（含背景，须与训练一致）")
parser.add_argument("--label_mapping", type=str, default=None,
                    help="标签重映射 'k:v,k:v'（公共数据集，如二值 0/255："
                         "'255:1' 配 --num_classes 2）")
parser.add_argument("--bf_tolerance", type=int, default=8,
                    help="BF-Score 边界匹配容差（像素，默认 8，与论文一致）")
parser.add_argument("--split_path", type=str, default=None,
                    help="三划分清单 split.json；提供后评估 --split 指定子集"
                         "（图像/GT 路径以清单记录为准，与训练严格同源）")
parser.add_argument("--split", type=str, default='test',
                    choices=['train', 'val', 'test'],
                    help="清单模式下评估的子集（默认 test）")
parser.add_argument("--test_image_path", type=str, default=None,
                    help="目录模式：测试图像目录（与 --test_mask_path 同用）")
parser.add_argument("--test_mask_path", type=str, default=None,
                    help="目录模式：测试标签目录")
parser.add_argument("--class_names", type=str, default=None,
                    help="逗号分隔的类别名（按索引 0..C-1），覆盖默认私有"
                         "集命名；公共集 E-GCS 用 "
                         "'background,blue-steel,red-steel,white-steel'")
for flag, help_ in [
        ("use_fdm", "改进一 FDM（消融模型评估须与训练一致）"),
        ("use_dysample", "改进二 DySample"),
        ("use_ba", "方案 A 边界注意力门"),
        ("use_boundary_head", "方案 B 辅助边界预测头"),
        ("use_dcn", "方案 C 可变形卷积精炼"),
        ("use_mlp_head", "MLP 全局语义融合头（评估须与训练一致）")]:
    parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction,
                        default=True, help=help_)
parser.add_argument("--mlp_fuse_dim", type=int, default=384,
                    help="MLP 融合头隐层宽（评估须与训练一致；"
                         "默认 384 = v2）")
parser.add_argument("--dy_edge_offset", type=float, default=4.0,
                    help="贴边四模块偏移预算 LR 像素（评估须与训练一致；"
                         "默认 4 = 既有 run；R3-DY 为 6）")
parser.add_argument("--hiera_size", type=str, default="large",
                    choices=["tiny", "small", "base_plus", "large"],
                    help="SAM2 Hiera 编码器尺寸（评估须与训练一致；"
                         "默认 large = 既有 run）")
parser.add_argument("--dino_size", type=str, default="large",
                    choices=["small", "base", "large"],
                    help="DINOv2 编码器尺寸（评估须与训练一致）")
args = parser.parse_args()

CLASS_NAMES = {
    3: ['background', 'high-albedo-steel', 'low-albedo-steel'],
    4: ['background', 'high-albedo-steel', 'low-albedo-steel', 'rusty-steel'],
}


def class_names(n, override=None):
    """本项目 3/4 分类用固定命名，其余自动命名（公共数据集场景）。

    --class_names 提供逗号分隔覆盖（Q92 待办落地：公共集 E-GCS 的
    蓝/红/白类名与私有集反照率口径不同，默认名会误导报告）。
    """
    if override:
        names = [s.strip() for s in override.split(',')]
        assert len(names) == n, \
            f"--class_names 给了 {len(names)} 个名，应为 {n} 个"
        return names
    return CLASS_NAMES.get(n) or [f'class{c}' for c in range(n)]


def parse_label_mapping(s):
    if not s:
        return None
    return {int(k): int(v) for k, v in (kv.split(':') for kv in s.split(','))}


def build_dataset(args):
    names = class_names(args.num_classes, args.class_names)
    if args.split_path:
        part = load_split(args.split_path)[args.split]
        dataset = FullDataset('', '', 1024, mode='test',
                              num_classes=args.num_classes,
                              image_list=part['images'],
                              gt_list=part['gts'],
                              label_mapping=parse_label_mapping(args.label_mapping))
        print(f"split mode: '{args.split}' subset, {len(dataset)} images "
              f"from {args.split_path}")
        return dataset, names
    if args.test_image_path and args.test_mask_path:
        return FullDataset(args.test_image_path, args.test_mask_path, 1024,
                           mode='test', num_classes=args.num_classes,
                           label_mapping=parse_label_mapping(
                               args.label_mapping)), names
    raise SystemExit("需要 --split_path（清单模式）或 --test_image_path + "
                     "--test_mask_path（目录模式）")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                     dino_size=args.dino_size).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    if not args.use_mlp_head:
        # 诊断模式（v3 病理分析）：允许以"关闭 mlp 头"的模型评估训练时
        # 含 mlp_head.* 键的 checkpoint——剥离这些键；正常路径不受影响
        sd = {k: v for k, v in sd.items() if not k.startswith('mlp_head.')}
    model.load_state_dict(sd)
    model.eval()

    dataset, names = build_dataset(args)

    # 指标口径：mIoU 为逐图逐类 IoU 的宏平均，参考图中缺席的类记为 NaN 并
    # 跳过（与论文一致）；BF-Score 为逐类边界 F1 的宏平均，容差默认 8 px，
    # 与论文报告的数值口径相同
    cm = ConfusionMatrix(args.num_classes)   # CPU 累积，与离线评估节奏匹配
    bf_scores = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc='eval', dynamic_ncols=True):
            sample = dataset[i]
            x = sample['image'].unsqueeze(0).to(device)
            label = sample['label'].squeeze(0)
            pred = model(x).argmax(dim=1).squeeze(0).cpu()
            cm.update(pred, label)
            bf_scores.append(boundary_f_score(pred, label, args.num_classes,
                                              args.bf_tolerance))

    per = cm.per_class()
    summ = cm.summary()

    print(f"\n{'class':<18}{'IoU':>9}{'Prec':>9}{'Recall':>9}{'F1':>9}")
    for c in range(args.num_classes):
        print(f"{names[c]:<18}{per['iou'][c]:>9.4f}{per['precision'][c]:>9.4f}"
              f"{per['recall'][c]:>9.4f}{per['f1'][c]:>9.4f}")
    print(f"{'macro mean':<18}{summ['miou']:>9.4f}{summ['mprecision']:>9.4f}"
          f"{summ['mrecall']:>9.4f}{summ['mf1']:>9.4f}")

    print(f"\nOA {summ['oa']:.4f} | mIoU {summ['miou']:.4f} | "
          f"FWIoU {summ['fwiou']:.4f} | Kappa {summ['kappa']:.4f}")

    bf_arr = np.array(bf_scores)   # (N, C)，GT 缺席类为 NaN
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bf_cls = np.nanmean(bf_arr, axis=0)
        bf_mean = np.nanmean(bf_arr)
    print(f"\nBoundary F-score (tolerance = {args.bf_tolerance}px)")
    print(f"{'class':<18}{'BF':>9}")
    for c in range(args.num_classes):
        print(f"{names[c]:<18}{bf_cls[c]:>9.4f}")
    print(f"{'mean BF':<18}{bf_mean:>9.4f}")

    print("\nConfusion matrix (rows = GT, cols = Pred)")
    print(cm.matrix())


if __name__ == "__main__":
    main(args)
