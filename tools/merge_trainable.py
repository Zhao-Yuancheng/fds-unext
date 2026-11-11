"""Rebuild a full FDS-UNeXT checkpoint from a trainable-only file.

`tools/export_trainable.py` stores the 6.97 M parameters that training updates.
This script constructs the model with the two frozen trunks, loads those
tensors, and writes a complete `FDS-UNeXT-best.pth` that `eval.py` can read.

Usage:
  python tools/merge_trainable.py --delta deltas/lrmd-fold0.pth \
      --hiera_path weights/sam2.1_hiera_large.pt \
      --dinov2_path weights/dinov2_vitl14_pretrain.pth \
      --out runs/fold0/FDS-UNeXT-best.pth
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FDSUNeXT  # noqa: E402


def main():
    ap = argparse.ArgumentParser('merge trainable tensors into a full model')
    ap.add_argument('--delta', required=True)
    ap.add_argument('--hiera_path', required=True)
    ap.add_argument('--dinov2_path', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_classes', type=int, default=None,
                    help='defaults to the value stored in the delta file')
    ap.add_argument('--dy_edge_offset', type=float, default=None)
    ap.add_argument('--hiera_size', type=str, default=None)
    ap.add_argument('--dino_size', type=str, default=None)
    args = ap.parse_args()

    blob = torch.load(args.delta, map_location='cpu')
    cfg = dict(blob['config'])
    cfg['num_classes'] = args.num_classes or cfg['num_classes']
    cfg['dy_edge_offset'] = (args.dy_edge_offset
                             if args.dy_edge_offset is not None
                             else cfg['dy_edge_offset'])
    cfg['hiera_size'] = args.hiera_size or cfg['hiera_size']
    cfg['dino_size'] = args.dino_size or cfg['dino_size']

    model = FDSUNeXT(args.hiera_path, args.dinov2_path, **cfg)
    sd = model.state_dict()
    sub = blob['state_dict']
    unknown = sorted(set(sub) - set(sd))
    if unknown:
        raise SystemExit(f'{len(unknown)} tensors not in the model, '
                         f'e.g. {unknown[:3]}')
    for k, v in sub.items():
        if sd[k].shape != v.shape:
            raise SystemExit(f'shape mismatch on {k}: '
                             f'{tuple(v.shape)} vs {tuple(sd[k].shape)}')
    sd.update(sub)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    torch.save(sd, args.out)
    print(f'{args.out}: {len(sd)} tensors, '
          f'{sum(v.numel() for v in sd.values()) / 1e6:.2f} M parameters, '
          f'{os.path.getsize(args.out) / 1e6:.1f} MB '
          f'(from {blob.get("source", "delta")})', flush=True)


if __name__ == '__main__':
    main()
