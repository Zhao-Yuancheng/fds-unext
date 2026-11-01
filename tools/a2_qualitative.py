"""A2: qualitative comparison strips: Input | GT | FDS-UNeXT | FDS-bare | SegFormer | HRNet.

Usage (roof):
  python tools/a2_qualitative.py --dataset roof --nc 4 \
      --hiera_path weights/sam2.1_hiera_large.pt \
      --dinov2_path weights/dinov2_vitl14_pretrain.pth \
      --ckpt_full  runs/roof_v2/fds_unext_full_..._fold0/FDS-UNeXT-best-bf.pth \
      --ckpt_bare  runs/roof_v2/fds_unext_bare_..._fold0/FDS-UNeXT-best-bf.pth \
      --ckpt_seg   runs/roof_v2/cmp_segformer_b2/best.pth \
      --ckpt_hrnet runs/roof_v2/cmp_hrnet_w32/best.pth \
      --split_path dataset/roof_dataset_v2/split_fold0.json --split test \
      --out runs/a2_qualitative_roof --num_images 4
Empty/missing ckpt args skip that panel gracefully.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import segmentation_models_pytorch as smp  # noqa: E402

from datasets import FullDataset  # noqa: E402
from models import FDSUNeXT as FDSFull  # noqa: E402
from train_compare import AttentionUNet, HRNetFCN  # noqa: E402
from utils.split import load_split  # noqa: E402

PALETTES = {
    'roof': [(0, 0, 0), (230, 60, 60), (255, 190, 60), (70, 130, 255)],
    'egcs': [(0, 0, 0), (30, 90, 255), (220, 40, 40), (240, 240, 240)],
    'potsdam': [(255, 255, 255), (0, 0, 255), (0, 255, 0), (0, 255, 255),
                (255, 255, 0), (255, 0, 0)],
}
BARE_KW = dict(use_fdm=False, use_dysample=False, use_ba=False,
               use_boundary_head=False, use_dcn=False)


def build_fds(args, bare, ckpt):
    kw = dict(num_classes=args.nc, use_mlp_head=False, dy_edge_offset=6)
    if bare:
        kw.update(BARE_KW)
    model = FDSFull(args.hiera_path, args.dinov2_path, **kw)
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k: v for k, v in sd.items() if not k.startswith('mlp_head.')}
    model.load_state_dict(sd)
    return model


def build_cmp(name, args, ckpt):
    nc = args.nc
    if name == 'segformer_b2':
        model = smp.Segformer('mit_b2', encoder_weights=None, classes=nc)
    elif name == 'hrnet_w32':
        model = HRNetFCN(nc)
    else:
        raise SystemExit(name)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model


def colorize(mask, palette):
    m = np.asarray(mask).astype(np.int64)
    m = np.clip(m, 0, len(palette) - 1)
    out = np.zeros(m.shape[:2] + (3,), dtype=np.uint8)
    for c, col in enumerate(palette):
        out[m == c] = col
    return Image.fromarray(out)


def error_map(pred, gt):
    """Black=correct BG, gray=correct FG, red=FP, blue=FN, orange=class confusion."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    out = np.zeros(gt.shape + (3,), dtype=np.uint8)
    correct_fg = (pred == gt) & (gt > 0)
    out[correct_fg] = (90, 90, 90)
    fp = (gt == 0) & (pred > 0)
    fn = (gt > 0) & (pred == 0)
    conf = (gt > 0) & (pred > 0) & (pred != gt)
    out[fp] = (255, 40, 40)
    out[fn] = (40, 80, 255)
    out[conf] = (255, 160, 0)
    return Image.fromarray(out)


def strip(images, labels, cell=320, gap=8, bar=30):
    w = cell * len(images) + gap * (len(images) + 1)
    canvas = Image.new('RGB', (w, bar + cell + gap), (255, 255, 255))
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(canvas)
    x = gap
    for img, lab in zip(images, labels):
        d.text((x + 4, 6), lab, fill=(20, 20, 20), font=font)
        rs = Image.LANCZOS if lab == 'Input' else Image.NEAREST
        canvas.paste(img.resize((cell, cell), rs), (x, bar))
        x += cell + gap
    return canvas


def pick_indices(ds, want, scan):
    scores = []
    for i in range(min(scan, len(ds))):
        u = torch.unique(ds[i]['label'])
        scores.append((len(u), -i, i))
    scores.sort(reverse=True)
    return [i for _, _, i in scores[:want]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['roof', 'potsdam', 'egcs'])
    ap.add_argument('--nc', type=int, required=True)
    ap.add_argument('--hiera_path', required=True)
    ap.add_argument('--dinov2_path', required=True)
    ap.add_argument('--ckpt_full', required=True)
    ap.add_argument('--ckpt_bare', default='')
    ap.add_argument('--ckpt_seg', required=True)
    ap.add_argument('--ckpt_hrnet', required=True)
    ap.add_argument('--split_path', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--error_out', default='',
                    help='if set, also save FDS/SegFormer error maps (FP red, '
                         'FN blue, class confusion orange)')
    ap.add_argument('--num_images', type=int, default=4)
    ap.add_argument('--scan', type=int, default=40)
    args = ap.parse_args()

    device = torch.device('cuda')
    os.makedirs(args.out, exist_ok=True)
    part = load_split(args.split_path)[args.split]
    ds = FullDataset('', '', 1024, mode='test', num_classes=args.nc,
                     image_list=part['images'], gt_list=part['gts'])
    print(f'dataset {args.dataset}: {len(ds)} test images '
          f'({args.split_path})', flush=True)

    models = []
    for label, kind, ckpt in [
            ('FDS-UNeXT', 'full', args.ckpt_full),
            ('FDS-bare', 'bare', args.ckpt_bare),
            ('SegFormer-B2', 'cmp_segformer_b2', args.ckpt_seg),
            ('HRNet-W32', 'cmp_hrnet_w32', args.ckpt_hrnet)]:
        if not ckpt or not os.path.isfile(ckpt):
            print(f'skip panel {label}: ckpt missing ({ckpt!r})', flush=True)
            continue
        if kind in ('full', 'bare'):
            m = build_fds(args, bare=(kind == 'bare'), ckpt=ckpt)
        else:
            m = build_cmp(kind.replace('cmp_', ''), args, ckpt)
        m.to(device).eval()
        models.append((label, m))
        print(f'loaded {label}', flush=True)

    labels = ['Input', 'GT'] + [lab for lab, _ in models]
    palette = PALETTES[args.dataset]
    idxs = pick_indices(ds, args.num_images, args.scan)
    print(f'picked test indices: {idxs}', flush=True)
    if args.error_out:
        os.makedirs(args.error_out, exist_ok=True)

    with torch.no_grad():
        for k, i in enumerate(idxs):
            sample = ds[i]
            x = sample['image'].unsqueeze(0).to(device)
            gt = sample['label'].squeeze(0).cpu().numpy()
            img = Image.open(ds.images[i]).convert('RGB').resize(
                (1024, 1024), Image.LANCZOS)
            panels = [img, colorize(gt, palette)]
            preds = {}
            for label, m in models:
                pred = m(x).argmax(dim=1).squeeze(0).cpu().numpy()
                preds[label] = pred
                panels.append(colorize(pred, palette))
            out_path = os.path.join(args.out, f'{args.dataset}_qual_{k:02d}.png')
            strip(panels, labels).save(out_path)
            print(f'saved {out_path}', flush=True)
            if args.error_out:
                ep = [img, colorize(gt, palette)]
                el = ['Input', 'GT']
                for label in ('FDS-UNeXT', 'SegFormer-B2'):
                    if label in preds:
                        ep.append(error_map(preds[label], gt))
                        el.append(label + ' err')
                epath = os.path.join(args.error_out,
                                     f'{args.dataset}_err_{k:02d}.png')
                strip(ep, el).save(epath)
                print(f'saved {epath}', flush=True)
    print('A2_DONE', flush=True)


if __name__ == '__main__':
    main()
