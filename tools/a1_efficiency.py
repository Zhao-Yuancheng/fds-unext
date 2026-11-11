"""A1: efficiency benchmark (params / GFLOPs / FPS / peak VRAM) @1024x1024, batch1 fp32.

Base specs: FDS-UNeXT (full, roof cfg), FDS-UNeXT (bare), and the six cmp baselines
from train_compare.MODEL_ZOO (random init, structure-only).
Lightweight FDS variants via repeatable --fds_variant "label|hiera_path|dino_path|nc"
(full module config, matching the trained potsdam lightweight ladder).

GFLOPs via torch.profiler with_flops (attention kernels may be undercounted;
params/FPS/VRAM are exact).

Usage:
  python tools/a1_efficiency.py --hiera_path weights/sam2.1_hiera_large.pt \
      --dinov2_path weights/dinov2_vitl14_pretrain.pth --out runs/a1_efficiency \
      --fds_variant "FDS-L+B(hieraL,dinoB)|weights/sam2.1_hiera_large.pt|weights/dinov2_vitb14_pretrain.pth|6" \
      --fds_variant "FDS-L+S(hieraL,dinoS)|weights/sam2.1_hiera_large.pt|weights/dinov2_vits14_pretrain.pth|6" \
      --fds_variant "FDS-BP+B(base+,dinoB)|weights/sam2.1_hiera_base_plus.pt|weights/dinov2_vitb14_pretrain.pth|6" \
      --fds_variant "FDS-BP+S(base+,dinoS)|weights/sam2.1_hiera_base_plus.pt|weights/dinov2_vits14_pretrain.pth|6" \
      --fds_variant "FDS-T+S(tiny,dinoS)|weights/sam2.1_hiera_tiny.pt|weights/dinov2_vits14_pretrain.pth|6"
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import segmentation_models_pytorch as smp  # noqa: E402

from train_compare import AttentionUNet, HRNetFCN  # noqa: E402
from models import FDSUNeXT as FDSFull  # noqa: E402

BARE_KW = dict(use_fdm=False, use_dysample=False, use_ba=False,
               use_boundary_head=False, use_dcn=False)


def build_fds(args, kind='full', hiera=None, dino=None, nc=4,
              hiera_size='large', dino_size='large'):
    kw = dict(num_classes=nc, use_mlp_head=False, dy_edge_offset=6,
              hiera_size=hiera_size, dino_size=dino_size)
    if kind == 'bare':
        kw.update(BARE_KW)
    return FDSFull(hiera or args.hiera_path, dino or args.dinov2_path, **kw)


def build_cmp(name, nc=4):
    if name == 'unet_r34':
        return smp.Unet('resnet34', encoder_weights=None, classes=nc)
    if name == 'unetpp_r34':
        return smp.UnetPlusPlus('resnet34', encoder_weights=None, classes=nc)
    if name == 'attunet_r34':
        return AttentionUNet(nc)
    if name == 'deeplabv3plus_r50':
        return smp.DeepLabV3Plus('resnet50', encoder_weights=None, classes=nc)
    if name == 'segformer_b2':
        return smp.Segformer('mit_b2', encoder_weights=None, classes=nc)
    if name == 'hrnet_w32':
        return HRNetFCN(nc)
    raise SystemExit(f'unknown cmp model {name}')


def measure(model, device, warmup=10, iters=30, amp=False):
    torch.backends.cudnn.benchmark = True
    x = torch.randn(1, 3, 1024, 1024, device=device)
    ctx = torch.autocast('cuda', dtype=torch.float16, enabled=amp)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with ctx:
                model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            with ctx:
                model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        peak_mib = torch.cuda.max_memory_allocated() / 2 ** 20
        with torch.no_grad(), ctx, profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True) as prof:
            model(x)
    flops = sum(ev.flops for ev in prof.key_averages())
    return {
        'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        'gflops': round(flops / 1e9, 1),
        'fps': round(1.0 / dt, 2),
        'latency_ms': round(dt * 1e3, 1),
        'peak_vram_MiB': round(peak_mib),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hiera_path', required=True)
    ap.add_argument('--dinov2_path', required=True)
    ap.add_argument('--num_classes', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--out', default='runs/a1_efficiency')
    ap.add_argument('--amp', action='store_true',
                    help='fp16 autocast 推理测速（默认 fp32）')
    ap.add_argument('--only', default=None,
                    help='仅测名称包含该子串的模型（如 "FDS-UNeXT (full)"）')
    ap.add_argument('--fds_variant', action='append', default=[],
                    help='label|hiera_path|dino_path|nc|hiera_size|dino_size '
                         '(repeatable, full cfg; sizes: large/base_plus/tiny, '
                         'large/base/small)')
    args = ap.parse_args()

    device = torch.device('cuda')
    os.makedirs(args.out, exist_ok=True)

    specs = [('FDS-UNeXT (full)', lambda: build_fds(args)),
             ('FDS-UNeXT (bare)', lambda: build_fds(args, kind='bare')),
             ('U-Net (R34)', lambda: build_cmp('unet_r34')),
             ('U-Net++ (R34)', lambda: build_cmp('unetpp_r34')),
             ('AttUNet (R34)', lambda: build_cmp('attunet_r34')),
             ('DeepLabV3+ (R50)', lambda: build_cmp('deeplabv3plus_r50')),
             ('SegFormer-B2', lambda: build_cmp('segformer_b2')),
             ('HRNet-W32', lambda: build_cmp('hrnet_w32'))]
    for spec in args.fds_variant:
        label, hiera, dino, nc, hsize, dsize = spec.split('|')
        specs.append((label, lambda h=hiera, d=dino, n=int(nc), hs=hsize,
                      ds=dsize: build_fds(args, hiera=h, dino=d, nc=n,
                                          hiera_size=hs, dino_size=ds)))

    results = {}
    print('model\tparams(M)\tGFLOPs\tFPS\tlatency(ms)\tpeakVRAM(MiB)', flush=True)
    for name, fn in specs:
        if args.only and args.only not in name:
            continue
        model = None
        try:
            model = fn().to(device)
            row = measure(model, device, args.warmup, args.iters, amp=args.amp)
            results[name] = row
            print(f"{name}\t{row['params_M']}\t{row['gflops']}\t{row['fps']}"
                  f"\t{row['latency_ms']}\t{row['peak_vram_MiB']}", flush=True)
        except Exception as exc:  # chain continues
            results[name] = {'error': str(exc)[:300]}
            print(f"{name}\tERROR\t{str(exc)[:300]}", flush=True)
        finally:
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()

    meta = {'gpu': torch.cuda.get_device_name(0),
            'torch': torch.__version__, 'input': '1024x1024x3 batch1 fp32',
            'gflops_note': 'torch.profiler with_flops; attention kernels may be '
                           'undercounted'}
    with open(os.path.join(args.out, 'efficiency.json'), 'w') as f:
        json.dump({'meta': meta, 'results': results}, f, indent=2)
    print('A1_DONE', flush=True)


if __name__ == '__main__':
    main()
