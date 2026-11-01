"""A3: representative-tile qualitative panels with fold-matched checkpoints.

每个瓦片只用「该瓦片位于测试集的那一折」的权重推理，因此图中任何一个
瓦片都没有被任何被展示的模型在训练中见过。对比方法同样按折取权重。

用法（在仓库根目录下）：
  python tools/a3_representative.py --out runs/a3_rep --which all
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import segmentation_models_pytorch as smp  # noqa: E402

from models import FDSUNeXT as FDSFull  # noqa: E402
from train_compare import HRNetFCN  # noqa: E402
from utils.metrics import boundary_f_score  # noqa: E402

# HRNetFCN 内部用 timm.create_model(pretrained=True)，会去 HuggingFace 拉
# ImageNet 权重；这里最终加载的是各折 best.pth，骨架权重无用，而在离线集群上
# （实测报 'Device or resource busy' 重试 5 次后失败）。改为不下载。
import timm  # noqa: E402

_timm_create_model = timm.create_model


def _create_model_no_pretrain(*a, **k):
    k['pretrained'] = False
    return _timm_create_model(*a, **k)


timm.create_model = _create_model_no_pretrain

PALETTE = [(0, 0, 0), (230, 60, 60), (255, 190, 60), (70, 130, 255)]

# (tile, category label)
MAIN = [
    ('1652001_822746', 'Urban village, many roof types'),
    ('1652057_822682', 'Poor illumination'),
    ('1652501_822570', 'Vehicles, urban village'),
    ('1652473_822534', 'Vegetation and mixed roofs'),
]
ERR = [
    ('1651945_822774', 'Urban village, many roof types'),
    ('1651893_822362', 'Urban village, mixed materials'),
    ('1652057_822678', 'Poor illumination'),
    ('1653829_823122', 'Vehicles on road'),
    ('1653409_822678', 'Poor illumination + vehicles'),
    ('1652845_822974', 'Vehicles, rural'),
    ('1652845_822526', 'Vehicles, urban'),
]
COLS = ['Input', 'Reference', 'FDS-UNeXT', 'SegFormer-B2', 'HRNet-W32']
ERR_COLS = ['Input', 'Reference', 'FDS-UNeXT err', 'SegFormer-B2 err']


def font(size, bold=False):
    cands = [
        '/usr/share/fonts/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
        '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf'
        % ('-Bold' if bold else ''),
    ]
    for p in cands:
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def colorize(mask):
    m = np.clip(np.asarray(mask).astype(np.int64), 0, len(PALETTE) - 1)
    out = np.zeros(m.shape + (3,), dtype=np.uint8)
    for c, col in enumerate(PALETTE):
        out[m == c] = col
    return Image.fromarray(out)


def error_map(pred, gt):
    """Black=correct BG, gray=correct roof, red=FP, blue=FN, orange=confusion."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    out = np.zeros(gt.shape + (3,), dtype=np.uint8)
    out[(pred == gt) & (gt > 0)] = (90, 90, 90)
    out[(gt == 0) & (pred > 0)] = (255, 40, 40)
    out[(gt > 0) & (pred == 0)] = (40, 80, 255)
    out[(gt > 0) & (pred > 0) & (pred != gt)] = (255, 160, 0)
    return Image.fromarray(out)


def normalize(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
        np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def ckpt_paths(runs, fold):
    sfx = '' if fold == 0 else f'_fold{fold}'
    return {
        'fds': f'{runs}/fds_unext_full_e100_w10mc50_dy6_lr3e5_fold{fold}/'
               f'FDS-UNeXT-best.pth',
        'seg': f'{runs}/cmp_segformer_b2{sfx}/best.pth',
        'hrn': f'{runs}/cmp_hrnet_w32{sfx}/best.pth',
    }


def build_fds(hiera, dinov2, nc, ckpt):
    model = FDSFull(hiera, dinov2, num_classes=nc, use_mlp_head=False,
                    dy_edge_offset=6)
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k: v for k, v in sd.items() if not k.startswith('mlp_head.')}
    model.load_state_dict(sd)
    return model


def build_cmp(kind, nc, ckpt):
    if kind == 'seg':
        model = smp.Segformer('mit_b2', encoder_weights=None, classes=nc)
    elif kind == 'hrn':
        model = HRNetFCN(nc)
    else:
        raise SystemExit(kind)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model


def load_models(args, fold, device):
    paths = ckpt_paths(args.runs, fold)
    for p in paths.values():
        if not os.path.isfile(p):
            raise SystemExit(f'missing checkpoint: {p}')
    models = {
        'fds': build_fds(args.hiera, args.dinov2, args.nc, paths['fds']),
        'seg': build_cmp('seg', args.nc, paths['seg']),
        'hrn': build_cmp('hrn', args.nc, paths['hrn']),
    }
    for k in models:
        models[k].to(device).eval()
    print(f'fold {fold}: loaded fds/seg/hrn', flush=True)
    return models


def test_fold_map(split_dir):
    """tile -> fold in which the tile is in the test split."""
    out = {}
    for k in range(5):
        with open(os.path.join(split_dir, f'split_fold{k}.json')) as f:
            sp = json.load(f)
        for p in sp['test']['images']:
            out[os.path.splitext(os.path.basename(p))[0]] = k
    return out


def row_image(pairs, cell, gap=8, head=30, bar=44, title=None, sub=None,
              title_size=24, sub_size=19, head_size=19):
    w = gap + len(pairs) * cell + (len(pairs) - 1) * gap + gap
    top = (bar if title else 0) + head
    h = top + cell
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((gap, 6), title, fill=(0, 0, 0), font=font(title_size, True))
        if sub:
            tw = d.textlength(title, font=font(title_size, True))
            d.text((gap + tw + 16, 11), sub, fill=(90, 90, 90),
                   font=font(sub_size))
    x = gap
    for img, lab in pairs:
        d.text((x + 3, top - head + 5), lab, fill=(20, 20, 20),
               font=font(head_size))
        rs = Image.LANCZOS if lab == 'Input' else Image.NEAREST
        canvas.paste(img.resize((cell, cell), rs), (x, top))
        x += cell + gap
    return canvas


def stack(rows, gap=18):
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + gap * (len(rows) - 1)
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + gap
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='dataset/roof_dataset_v2')
    ap.add_argument('--split_dir', default='dataset/roof_dataset_v2')
    ap.add_argument('--runs', default='runs/roof_v2')
    ap.add_argument('--hiera', default='weights/sam2.1_hiera_large.pt')
    ap.add_argument('--dinov2', default='weights/dinov2_vitl14_pretrain.pth')
    ap.add_argument('--nc', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--which', default='all', choices=['main', 'err', 'all'])
    ap.add_argument('--cell', type=int, default=350)
    ap.add_argument('--stats', action='store_true',
                    help='also dump per-tile mIoU/BF/error-pixel counts '
                         'and the raw predictions (npz)')
    args = ap.parse_args()

    device = torch.device('cuda')
    os.makedirs(args.out, exist_ok=True)
    fmap = test_fold_map(args.split_dir)

    jobs = []
    if args.which in ('main', 'all'):
        jobs += [('main', t, lab) for t, lab in MAIN]
    if args.which in ('err', 'all'):
        jobs += [('err', t, lab) for t, lab in ERR]

    for kind, tile, lab in jobs:
        if tile not in fmap:
            raise SystemExit(f'{tile} not found in any test split')
        print(f'{kind} {tile} -> fold {fmap[tile]} ({lab})', flush=True)

    # 按折排序，减少权重反复装卸
    jobs.sort(key=lambda j: fmap[j[1]])

    preds = {}
    stats = {}
    cache_fold = None
    models = None
    for kind, tile, lab in jobs:
        fold = fmap[tile]
        if fold != cache_fold:
            models = None
            torch.cuda.empty_cache()
            models = load_models(args, fold, device)
            cache_fold = fold
        img = Image.open(
            os.path.join(args.data_dir, 'images', f'{tile}.png')).convert('RGB')
        gt = np.array(
            Image.open(os.path.join(args.data_dir, 'masks', f'{tile}.png')))
        if img.size != (1024, 1024):
            img = img.resize((1024, 1024), Image.LANCZOS)
        x = normalize(img).unsqueeze(0).to(device)
        out = {}
        with torch.no_grad():
            for k, m in models.items():
                out[k] = m(x).argmax(dim=1).squeeze(0).cpu().numpy()
        preds[tile] = out
        if args.stats:
            gt_t = torch.from_numpy(gt)
            rec = {'fold': fold, 'label': lab,
                   'gt_px': [int((gt == c).sum()) for c in range(args.nc)]}
            for k, p in out.items():
                ious = []
                for c in range(args.nc):
                    g = gt == c
                    pr = p == c
                    if not g.any():
                        # 与工程口径一致：GT 缺席的类不计入 mIoU
                        ious.append(float('nan'))
                        continue
                    u = int((g | pr).sum())
                    ious.append(float((g & pr).sum()) / u if u
                                else float('nan'))
                rec[k] = {
                    'miou': float(np.nanmean(ious)),
                    'bf': float(np.nanmean(np.asarray(
                        boundary_f_score(torch.from_numpy(p), gt_t,
                                         args.nc, 8), dtype=float))),
                    'fp': int(((gt == 0) & (p > 0)).sum()),
                    'fn': int(((gt > 0) & (p == 0)).sum()),
                    'conf': int(((gt > 0) & (p > 0) & (p != gt)).sum()),
                }
            stats[tile] = rec
            np.savez_compressed(
                os.path.join(args.out, f'pred_{tile}.npz'), gt=gt,
                **{f'pred_{k}': v for k, v in out.items()})
        print(f'done {tile}', flush=True)
        if kind == 'main':
            pairs = [(img, 'Input'), (colorize(gt), 'Reference'),
                     (colorize(out['fds']), 'FDS-UNeXT'),
                     (colorize(out['seg']), 'SegFormer-B2'),
                     (colorize(out['hrn']), 'HRNet-W32')]
            row_image(pairs, args.cell, head=30, bar=44,
                      title=lab, sub=f'tile {tile} · fold {fold}').save(
                os.path.join(args.out, f'main_{tile}.png'))
        else:
            pairs = [(img, 'Input'), (colorize(gt), 'Reference'),
                     (error_map(out['fds'], gt), 'FDS-UNeXT err'),
                     (error_map(out['seg'], gt), 'SegFormer-B2 err')]
            row_image(pairs, args.cell, head=30, bar=44,
                      title=lab, sub=f'tile {tile} · fold {fold}').save(
                os.path.join(args.out, f'err_{tile}.png'))

    if args.which in ('main', 'all'):
        rows = [Image.open(os.path.join(args.out, f'main_{t}.png'))
                for t, _ in MAIN]
        stack(rows).save(os.path.join(args.out, 'grid_main.png'))
        print('saved grid_main.png', flush=True)
    if args.which in ('err', 'all'):
        rows = [Image.open(os.path.join(args.out, f'err_{t}.png'))
                for t, _ in ERR]
        stack(rows).save(os.path.join(args.out, 'grid_err.png'))
        print('saved grid_err.png', flush=True)
    print('A3_DONE', flush=True)

    if args.stats:
        with open(os.path.join(args.out, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=1)
        print('\ntile                 fold  mIoU(F/S/H)      BF(F/S/H)'
              '        gt px (bg/high/low/tile)', flush=True)
        for t, r in stats.items():
            print('%-20s %4d  %5.1f/%5.1f/%5.1f  %5.1f/%5.1f/%5.1f  %s' % (
                t, r['fold'],
                r['fds']['miou'] * 100, r['seg']['miou'] * 100,
                r['hrn']['miou'] * 100,
                r['fds']['bf'] * 100, r['seg']['bf'] * 100,
                r['hrn']['bf'] * 100,
                '/'.join(str(v // 1000) + 'k' for v in r['gt_px'])),
                flush=True)


if __name__ == '__main__':
    main()
