# -*- coding: utf-8 -*-
"""Potsdam 模糊退化数据生成

做法：把 1024² 影像缩小 f 倍再双线性放大回 1024，
制造不同退化程度；标签（mask）保持不变，只新建图像目录与对应 split。

退化方式：PIL BOX（区域平均）下采样 → BILINEAR 上采样（等价 area-interp 流程）。
输出：dataset/potsdam_1024_x{f}/images + dataset/potsdam_1024_x{f}/split.json
      （split 中仅替换 images 路径，masks 仍指向原始标签目录）
绝不覆盖源数据。
"""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'dataset' / 'potsdam_1024'


def degrade(src: Path, dst: Path, f: int):
    im = Image.open(src).convert('RGB')
    w, h = im.size
    small = im.resize((max(1, w // f), max(1, h // f)),
                      Image.BOX if hasattr(Image, 'BOX') else Image.BILINEAR)
    out = small.resize((w, h), Image.BILINEAR)
    out.save(dst)


def main(factors):
    split = json.load(open(SRC / 'split.json', encoding='utf-8'))
    src_imgs = sorted((SRC / 'images').glob('*.png'))
    print(f'source images: {len(src_imgs)}')
    for f in factors:
        out_root = ROOT / 'dataset' / f'potsdam_1024_x{f}'
        out_imgs = out_root / 'images'
        out_imgs.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(src_imgs, 1):
            degrade(p, out_imgs / p.name, f)
            if i % 200 == 0:
                print(f'  x{f}: {i}/{len(src_imgs)}', flush=True)
        new = dict(split)
        for part in ('train', 'val', 'test'):
            if part not in split:
                continue
            new[part] = dict(split[part])
            if 'images' in split[part]:
                new[part]['images'] = [
                    q.replace(str(SRC / 'images'), str(out_imgs))
                    for q in split[part]['images']]
        json.dump(new, open(out_root / 'split.json', 'w', encoding='utf-8'), indent=1)
        print(f'x{f}: {len(list(out_imgs.glob("*.png")))} images -> {out_root}')


if __name__ == '__main__':
    fs = [int(a) for a in sys.argv[1:]] or [2, 4]
    main(fs)
