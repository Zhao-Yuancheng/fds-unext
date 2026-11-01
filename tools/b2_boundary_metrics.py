"""B2: Boundary IoU + HD95 (+ optional BF cross-check) full re-evaluation.

Boundary IoU (Cheng et al., CVPR 2021): per-class boundary region
  B(G) = {p in G : dist(p, complement(G)) <= d},  d = ratio * image diagonal
  BIoU = |B(pred) ∩ B(gt)| / |B(pred) ∪ B(gt)|
HD95: 95th percentile of symmetric surface distances (surface = 1-px inner
  boundary), per image per class. Primary value excludes class-instances with
  empty prediction (rate reported separately as empty_pred_rate); a secondary
  penalized value assigns the image diagonal to those instances.
Aggregation follows the project convention: per-image values, then nanmean
(absent classes / degenerate boundaries = NaN, skipped).

Usage:
  python tools/b2_boundary_metrics.py --group roof_fds --out runs/b2_metrics
  python tools/b2_boundary_metrics.py --group all --only roof_fds_f0_best
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import segmentation_models_pytorch as smp  # noqa: E402

from datasets import FullDataset  # noqa: E402
from models import FDSUNeXT as FDSFull  # noqa: E402
from models.fds_unext_notanh import FDSUNeXT as FDSNoTanh  # noqa: E402
from models.unetformer import UNetFormer  # noqa: E402
from train_compare import AttentionUNet, HRNetFCN  # noqa: E402
from utils.metrics import boundary_f_score  # noqa: E402
from utils.split import load_split  # noqa: E402

DS = dict(
    roof=dict(nc=4, split='dataset/roof_dataset_v2/split_fold{}.json',
              cls='background,high-albedo-steel,low-albedo-steel,tile'),
    potsdam=dict(nc=6, split='dataset/potsdam_1024/split.json',
                 cls='impervious,building,low-veg,tree,car,clutter'),
    egcs=dict(nc=4, split='runs/full_public_egcs_full/split.json',
              cls='background,blue-steel,red-steel,white-steel'),
)
BARE_KW = dict(use_fdm=False, use_dysample=False, use_ba=False,
               use_boundary_head=False, use_dcn=False)


def build_jobs():
    J = []

    def fds(group, tag, ckpt, ds, fold=None, arch='full',
            hiera='large', dino='large', dino_path=None):
        J.append(dict(group=group, name=tag, arch=arch, ckpt=ckpt, ds=ds,
                      split=DS[ds]['split'].format(fold),
                      nc=DS[ds]['nc'], cls=DS[ds]['cls'],
                      hiera_size=hiera, dino_size=dino, dino_path=dino_path))

    def cmp(group, tag, name, ckpt, ds, fold=None):
        J.append(dict(group=group, name=tag, arch='cmp:' + name, ckpt=ckpt,
                      ds=ds, split=DS[ds]['split'].format(fold),
                      nc=DS[ds]['nc'], cls=DS[ds]['cls']))

    # --- roof_v2: FDS five folds, both selection criteria ---
    for f in range(5):
        base = f'runs/roof_v2/fds_unext_full_e100_w10mc50_dy6_lr3e5_fold{f}'
        fds('roof_fds', f'roof_fds_f{f}_best', base + '/FDS-UNeXT-best.pth',
            'roof', f)
        fds('roof_fds', f'roof_fds_f{f}_bestbf', base + '/FDS-UNeXT-best-bf.pth',
            'roof', f)
    # notanh fold0 (representative)
    nb = 'runs/roof_v2/fds_unext_full_e100_w10mc50_dy6_lr3e5_notanh_fold0'
    fds('roof_notanh', 'roof_notanh_f0_bestbf', nb + '/FDS-UNeXT-best-bf.pth',
        'roof', 0, arch='notanh')
    # --- roof comparisons: SegFormer five folds + six-model fold0 ---
    for f in range(5):
        d = 'runs/roof_v2/cmp_segformer_b2' + ('' if f == 0 else f'_fold{f}')
        cmp('roof_cmp', f'roof_segformer_f{f}', 'segformer_b2',
            d + '/best.pth', 'roof', f)
    for m in ('unet_r34', 'unetpp_r34', 'attunet_r34', 'deeplabv3plus_r50',
              'hrnet_w32'):
        cmp('roof_cmp', f'roof_{m}_f0', m,
            f'runs/roof_v2/cmp_{m}/best.pth', 'roof', 0)
    # UNetFormer（ckpt 由 train_compare.py 训练产出）
    cmp('roof_cmp', 'roof_unetformer_f0', 'unetformer_r18',
        'runs/roof_v2/cmp_unetformer_r18/best.pth', 'roof', 0)

    # --- Potsdam ---
    p = 'runs/potsdam/'
    fds('potsdam', 'potsdam_fds_best',
        p + 'fds_unext_full_w3mc15_dy6_lr5e5/FDS-UNeXT-best.pth', 'potsdam')
    fds('potsdam', 'potsdam_fds_bestbf',
        p + 'fds_unext_full_w3mc15_dy6_lr5e5/FDS-UNeXT-best-bf.pth', 'potsdam')
    fds('potsdam', 'potsdam_notanh_best',
        p + 'fds_unext_full_w3mc15_dy6_lr5e5_notanh/FDS-UNeXT-best.pth',
        'potsdam', arch='notanh')
    fds('potsdam', 'potsdam_bare_best',
        p + 'fds_unext_bare_flip_lr5e5/FDS-UNeXT-best.pth', 'potsdam',
        arch='bare')
    cmp('potsdam', 'potsdam_segformer', 'segformer_b2',
        p + 'cmp_segformer_b2/best.pth', 'potsdam')
    cmp('potsdam', 'potsdam_hrnet', 'hrnet_w32',
        p + 'cmp_hrnet_w32/best.pth', 'potsdam')

    # --- E-GCS ---
    e = 'runs/'
    fds('egcs', 'egcs_fds_best',
        e + 'full_public_egcs_full_v6_w3mc15_dy6_lr1e5/FDS-UNeXT-best.pth',
        'egcs')
    fds('egcs', 'egcs_fds_bestbf',
        e + 'full_public_egcs_full_v6_w3mc15_dy6_lr1e5/FDS-UNeXT-best-bf.pth',
        'egcs')
    fds('egcs', 'egcs_notanh_best',
        e + 'full_public_egcs_full_v6_w3mc15_dy6_lr1e5_notanh/FDS-UNeXT-best.pth',
        'egcs', arch='notanh')
    fds('egcs', 'egcs_bare_best',
        e + 'full_public_egcs_full_bare_flip_lr1e5/FDS-UNeXT-best.pth',
        'egcs', arch='bare')
    fds('egcs', 'egcs_Lb_best',
        e + 'full_public_egcs_full_v6_dinob_w3mc15_dy6_lr1e5/FDS-UNeXT-best.pth',
        'egcs', hiera='large', dino='base',
        dino_path='weights/dinov2_vitb14_pretrain.pth')
    cmp('egcs', 'egcs_segformer', 'segformer_b2',
        e + 'cmp_segformer_b2/best.pth', 'egcs')
    return J


_CACHE = {}


def get_model(job, args, device):
    key = (job['arch'], job['nc'], job.get('hiera_size', 'large'),
           job.get('dino_size', 'large'))
    if key in _CACHE:
        return _CACHE[key]
    arch = job['arch']
    hs = job.get('hiera_size', 'large')
    ds_ = job.get('dino_size', 'large')
    dpath = job.get('dino_path') or args.dinov2_path
    if arch == 'full':
        m = FDSFull(args.hiera_path, dpath, num_classes=job['nc'],
                    use_mlp_head=False, dy_edge_offset=6,
                    hiera_size=hs, dino_size=ds_)
    elif arch == 'bare':
        m = FDSFull(args.hiera_path, dpath, num_classes=job['nc'],
                    use_mlp_head=False, dy_edge_offset=6,
                    hiera_size=hs, dino_size=ds_, **BARE_KW)
    elif arch == 'notanh':
        m = FDSNoTanh(args.hiera_path, dpath, num_classes=job['nc'],
                      use_mlp_head=False, dy_edge_offset=6,
                      hiera_size=hs, dino_size=ds_)
    else:
        name = arch.split(':', 1)[1]
        nc = job['nc']
        if name == 'unet_r34':
            m = smp.Unet('resnet34', encoder_weights=None, classes=nc)
        elif name == 'unetpp_r34':
            m = smp.UnetPlusPlus('resnet34', encoder_weights=None, classes=nc)
        elif name == 'attunet_r34':
            m = AttentionUNet(nc)
        elif name == 'deeplabv3plus_r50':
            m = smp.DeepLabV3Plus('resnet50', encoder_weights=None, classes=nc)
        elif name == 'segformer_b2':
            m = smp.Segformer('mit_b2', encoder_weights=None, classes=nc)
        elif name == 'hrnet_w32':
            m = HRNetFCN(nc)
        elif name == 'unetformer_r18':
            m = UNetFormer(num_classes=nc, backbone_name='resnet18',
                           pretrained=False)
        else:
            raise SystemExit('unknown arch ' + name)
    m.to(device).eval()
    _CACHE[key] = m
    return m


def dist_to_bg(mask):
    """distance from each True pixel to nearest False pixel (0 outside mask)."""
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2,
                                 cv2.DIST_MASK_PRECISE)


def per_image_metrics(pred, gt, nc, d_biou, diag, with_bf):
    """Returns (biou, hd95_valid, hd95_penalized, bf, n_empty_pred).

    hd95_valid excludes class-instances whose prediction is empty (rate is
    reported separately); hd95_penalized assigns the image diagonal to those
    instances (medpy-inf analogue) for a conservative secondary view.
    """
    biou, hd95, hd95_pen = [], [], []
    n_empty = 0
    for c in range(nc):
        g = gt == c
        p = pred == c
        if not g.any():
            biou.append(np.nan)
            hd95.append(np.nan)
            hd95_pen.append(np.nan)
            continue
        gd = dist_to_bg(g)
        g_b = g & (gd <= d_biou)
        if not g_b.any():
            biou.append(np.nan)
            hd95.append(np.nan)
            hd95_pen.append(np.nan)
            continue
        if not p.any():
            biou.append(0.0)
            hd95.append(np.nan)
            hd95_pen.append(float(diag))
            n_empty += 1
            continue
        pd = dist_to_bg(p)
        p_b = p & (pd <= d_biou)
        union = int((g_b | p_b).sum())
        biou.append(float((g_b & p_b).sum()) / union if union else np.nan)

        g_s = g & (gd <= 1.0)
        p_s = p & (pd <= 1.0)
        if not g_s.any() or not p_s.any():
            hd95.append(np.nan)
            hd95_pen.append(np.nan)
            continue
        dg = cv2.distanceTransform((~g_s).astype(np.uint8), cv2.DIST_L2,
                                   cv2.DIST_MASK_PRECISE)
        dp = cv2.distanceTransform((~p_s).astype(np.uint8), cv2.DIST_L2,
                                   cv2.DIST_MASK_PRECISE)
        d1 = dg[p_s]
        d2 = dp[g_s]
        v = float(np.percentile(np.concatenate([d1, d2]), 95))
        hd95.append(v)
        hd95_pen.append(v)
    bf = boundary_f_score(torch.from_numpy(pred), torch.from_numpy(gt),
                          nc, 8) if with_bf else None
    return biou, hd95, hd95_pen, bf, n_empty


def nanmean(a, axis=None):
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.nanmean(np.asarray(a, dtype=float), axis=axis)


def run_job(job, args, device):
    out = os.path.join(args.out_dir, job['name'] + '.json')
    if os.path.isfile(out) and not args.force:
        print(f'SKIP {job["name"]} (exists)', flush=True)
        return
    part = load_split(job['split'])[args.split]
    ds = FullDataset('', '', 1024, mode='test', num_classes=job['nc'],
                     image_list=part['images'], gt_list=part['gts'])
    model = get_model(job, args, device)
    sd = torch.load(job['ckpt'], map_location=device)
    sd = {k: v for k, v in sd.items() if not k.startswith('mlp_head.')}
    model.load_state_dict(sd)
    model.eval()
    diag = float(np.hypot(*ds[0]['image'].shape[-2:]))
    d_biou = args.biou_ratio * diag
    B, H, HP, BF = [], [], [], []
    n_empty_total = 0
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x = s['image'].unsqueeze(0).to(device)
            gt = s['label'].squeeze(0).numpy().astype(np.int64)
            pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
            b, h, hp, f, ne = per_image_metrics(pred, gt, job['nc'], d_biou,
                                                diag, args.with_bf)
            B.append(b)
            H.append(h)
            HP.append(hp)
            n_empty_total += ne
            if f is not None:
                BF.append(f)
            if (i + 1) % 200 == 0:
                print(f'  {job["name"]} {i + 1}/{len(ds)}', flush=True)
    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)
    HP = np.asarray(HP, dtype=float)
    res = dict(
        job=job['name'], group=job['group'], arch=job['arch'],
        ckpt=job['ckpt'], ds=job['ds'], split=job['split'], nc=job['nc'],
        n_images=len(ds), biou_ratio=args.biou_ratio,
        biou_dilation_px=round(d_biou, 2),
        biou_percent=float(100 * nanmean(B)),
        hd95=float(nanmean(H)),
        hd95_penalized=float(nanmean(HP)),
        empty_pred_instances=int(n_empty_total),
        empty_pred_rate=float(n_empty_total / max(1, len(ds) * job['nc'])),
        biou_per_class=[None if np.isnan(x) else float(x)
                        for x in nanmean(B, axis=0)],
        hd95_per_class=[None if np.isnan(x) else float(x)
                        for x in nanmean(H, axis=0)],
        biou_per_image=B.tolist(), hd95_per_image=H.tolist(),
    )
    if args.with_bf:
        BFa = np.asarray(BF, dtype=float)
        res['bf_percent'] = float(100 * nanmean(BFa))
        res['bf_per_class'] = [None if np.isnan(x) else float(x)
                               for x in nanmean(BFa, axis=0)]
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)
    extra = f" BF={res['bf_percent']:.2f}" if args.with_bf else ''
    print(f"DONE {job['name']}: BIoU={res['biou_percent']:.2f} "
          f"HD95={res['hd95']:.2f}px (pen={res['hd95_penalized']:.1f}, "
          f"empty={n_empty_total}){extra} (n={len(ds)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='all',
                    choices=['all', 'roof_fds', 'roof_notanh', 'roof_cmp',
                             'potsdam', 'egcs'])
    ap.add_argument('--only', default='', help='substring filter on job name')
    ap.add_argument('--split', default='test')
    ap.add_argument('--biou_ratio', type=float, default=0.005,
                    help='boundary band width as fraction of image diagonal')
    ap.add_argument('--with_bf', action='store_true',
                    help='also recompute BF (cross-check against eval logs)')
    ap.add_argument('--hiera_path', required=True)
    ap.add_argument('--dinov2_path', required=True)
    ap.add_argument('--out_dir', default='runs/b2_metrics')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')
    jobs = [j for j in build_jobs()
            if (args.group == 'all' or j['group'] == args.group)
            and (not args.only or args.only in j['name'])]
    print(f'jobs={len(jobs)} group={args.group} only={args.only!r}', flush=True)
    for j in jobs:
        try:
            run_job(j, args, device)
        except Exception as exc:
            print(f'FAIL {j["name"]}: {exc!r}', flush=True)
    print('B2_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
