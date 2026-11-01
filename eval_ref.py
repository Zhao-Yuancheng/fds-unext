"""SAM2-UNeXT 官方基线的离线评估（对照基线）。

与 `eval.py` 等价的口径（同一 `utils.metrics`、同一 bf_tol、同一 split），
只是模型类换成 `SAM2UNeXTRef`，因此不触碰既有 eval.py。

用法：
  python eval_ref.py --hiera_path ... --dinov2_path ... --checkpoint <pth> \
    --split_path <split.json> --split test --num_classes 4 --bf_tolerance 8 \
    --class_names background,high-albedo-steel,low-albedo-steel,tile
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import FullDataset
from models.sam2unext_ref import SAM2UNeXTRef
from utils.metrics import ConfusionMatrix, boundary_f_score
from utils.split import load_split


class _ImgLabel(torch.utils.data.Dataset):
    """只保留 image/label（FullDataset 的 test 模式 sdf=None，无法 collate）。"""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        return {'image': s['image'], 'label': s['label']}


def main(a):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    part = load_split(a.split_path)[a.split]
    ds = _ImgLabel(FullDataset(
        '', '', 1024, mode='test', num_classes=a.num_classes,
        image_list=part['images'], gt_list=part['gts'],
        label_mapping=None))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=True)
    model = SAM2UNeXTRef(a.hiera_path, a.dinov2_path,
                         num_classes=a.num_classes,
                         hiera_size=a.hiera_size,
                         dino_size=a.dino_size,
                         norm=a.norm).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    cm = ConfusionMatrix(a.num_classes, device='cpu')
    per_class_bf = [[] for _ in range(a.num_classes)]
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b['image'].to(device)
            tgt = b['label'].squeeze(1).long()
            pred = model(x).argmax(dim=1).cpu()
            cm.update(pred, tgt)
            for j in range(pred.size(0)):
                v = boundary_f_score(pred[j], tgt[j], a.num_classes,
                                     a.bf_tolerance)
                for c in range(a.num_classes):
                    per_class_bf[c].append(v[c])
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(loader)}', flush=True)

    s = cm.summary()
    names = (a.class_names.split(',') if a.class_names
             else [f'class{c}' for c in range(a.num_classes)])
    print(f"\ncheckpoint: {os.path.basename(a.checkpoint)}  split={a.split}"
          f"  n={len(ds)}")
    print(f"{'class':<20}{'IoU':>8}{'BF':>8}")
    per = cm.per_class()
    for c in range(a.num_classes):
        v = np.nanmean([x for x in per_class_bf[c] if x is not None])
        iou = per['iou'][c]
        print(f"{names[c]:<20}{iou:>8.4f}{v:>8.4f}")
    print(f"\nmIoU={s['miou']:.4f}  OA={s['oa']:.4f}  FWIoU={s['fwiou']:.4f}"
          f"  Kappa={s['kappa']:.4f}  mF1={s['mf1']:.4f}")
    cls_mean = [np.nanmean([x for x in per_class_bf[c] if x is not None])
                for c in range(a.num_classes)]
    print(f"BF-Score (macro over classes, tol={a.bf_tolerance}px) ="
          f" {np.nanmean(cls_mean):.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser('SAM2-UNeXT reference evaluation')
    p.add_argument('--hiera_path', type=str, required=True)
    p.add_argument('--dinov2_path', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--split_path', type=str, required=True)
    p.add_argument('--split', type=str, default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--num_classes', type=int, default=4)
    p.add_argument('--bf_tolerance', type=int, default=8)
    p.add_argument('--class_names', type=str, default=None)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--hiera_size', type=str, default='large')
    p.add_argument('--dino_size', type=str, default='large')
    p.add_argument('--norm', type=str, default='gn', choices=['gn', 'bn'])
    main(p.parse_args())
