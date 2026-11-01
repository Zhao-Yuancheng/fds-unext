import colorsys
import os
import argparse
import numpy as np
import torch
from PIL import Image

from datasets import TestDataset
from models import FDSUNeXT


parser = argparse.ArgumentParser("FDS-UNeXT inference")
parser.add_argument("--hiera_path", type=str, required=True)
parser.add_argument("--dinov2_path", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--test_image_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--num_classes", type=int, default=4,
                    help="类别数（须与训练一致；0 恒为背景）")
for flag, help_ in [
        ("use_fdm", "改进一 FDM（消融模型推理须与训练一致）"),
        ("use_dysample", "改进二 DySample"),
        ("use_ba", "方案 A 边界注意力门"),
        ("use_boundary_head", "方案 B 辅助边界预测头"),
        ("use_dcn", "方案 C 可变形卷积精炼")]:
    parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction,
                        default=True, help=help_)
args = parser.parse_args()


def make_palette(n):
    """0=黑色背景；本项目 3 分类红/绿、4 分类红/绿/蓝（瓦片），其余 HSV 均匀取色。"""
    if n == 3:
        return [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    if n == 4:
        return [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return [(0, 0, 0)] + [
        tuple(int(v * 255) for v in colorsys.hsv_to_rgb(i / max(n - 1, 1), 0.9, 0.9))
        for i in range(1, n)]


PALETTE = make_palette(args.num_classes)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FDSUNeXT(args.hiera_path, args.dinov2_path,
                     num_classes=args.num_classes,
                     use_fdm=args.use_fdm,
                     use_dysample=args.use_dysample,
                     use_ba=args.use_ba,
                     use_boundary_head=args.use_boundary_head,
                     use_dcn=args.use_dcn).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = TestDataset(args.test_image_path, 1024)
    os.makedirs(args.save_path, exist_ok=True)

    with torch.no_grad():
        for _ in range(len(dataset)):
            x, name = dataset.load_data()
            pred = model(x.to(device)).argmax(dim=1).squeeze(0).cpu().numpy()

            color = np.zeros(pred.shape + (3,), dtype=np.uint8)
            for c, rgb in enumerate(PALETTE):
                color[pred == c] = rgb
            name = os.path.splitext(name)[0] + '.png'
            Image.fromarray(color).save(os.path.join(args.save_path, name))
            print('saved:', name)


if __name__ == "__main__":
    main(args)
