"""Export only the trainable tensors of a FDS-UNeXT checkpoint.

A trained checkpoint holds 523.1 M parameters, but 516.2 M of them belong to the
two frozen trunks and never change. This script keeps the 6.97 M that the
optimizer actually updated, which shrinks a 2.0 GB file to about 28 MB.

Usage:
  python tools/export_trainable.py --checkpoint runs/fold0/FDS-UNeXT-best.pth \
      --out deltas/lrmd-fold0.pth

The output can be turned back into a full checkpoint with
`tools/merge_trainable.py`, which is what `eval.py` expects.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FDSUNeXT  # noqa: E402


def main():
    ap = argparse.ArgumentParser('export trainable tensors')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num_classes', type=int, default=4)
    ap.add_argument('--dy_edge_offset', type=float, default=6.0)
    ap.add_argument('--hiera_size', type=str, default='large')
    ap.add_argument('--dino_size', type=str, default='large')
    ap.add_argument('--use_mlp_head', action=argparse.BooleanOptionalAction,
                    default=False)
    args = ap.parse_args()

    # 不读两主干权重：只需要结构来判定哪些参数可训练（pretrained=False）
    model = FDSUNeXT(None, None, num_classes=args.num_classes,
                     pretrained=False, use_mlp_head=args.use_mlp_head,
                     dy_edge_offset=args.dy_edge_offset,
                     hiera_size=args.hiera_size, dino_size=args.dino_size)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    full = torch.load(args.checkpoint, map_location='cpu')
    sub = {k: v for k, v in full.items() if k in trainable}
    missing = sorted(trainable - set(sub))
    if missing:
        raise SystemExit(f'checkpoint lacks {len(missing)} trainable tensors, '
                         f'e.g. {missing[:3]}')

    cfg = dict(num_classes=args.num_classes,
               dy_edge_offset=args.dy_edge_offset,
               hiera_size=args.hiera_size, dino_size=args.dino_size,
               use_mlp_head=args.use_mlp_head)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    torch.save({'state_dict': sub, 'config': cfg,
                'source': os.path.basename(args.checkpoint)}, args.out)
    n = sum(v.numel() for v in sub.values())
    print(f'{args.out}: {len(sub)} tensors, {n / 1e6:.2f} M parameters, '
          f'{os.path.getsize(args.out) / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
