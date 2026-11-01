"""ISPRS Potsdam 1024² 锚定裁剪 + 景级切分（制备第三步）。

输入：dataset/potsdam_raw/{2_Ortho_RGB/*.tif, 5_Labels_index/*.png}
输出：dataset/potsdam_1024/{images,masks}/*.png + split.json

裁剪：6000² → 每轴偏移 [0,1024,2048,3072,4096,4976]（末块回贴对齐
右下角，无丢弃像素），36 块/景，景内 144px 重叠——景级切分保证重叠
不跨 train/val/test。38 景 × 36 = 1368 块。
切分：景级 7:1:2，seed 42，group_pattern 捕获 top_potsdam_X_Y。
在仓库根目录运行（用 utils.split 的相对导入）。
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.split import create_split, save_split  # noqa: E402

RAW = 'dataset/potsdam_raw'
DST = 'dataset/potsdam_1024'
OFFS = [0, 1024, 2048, 3072, 4096, 4976]
T = 1024

os.makedirs(f'{DST}/images', exist_ok=True)
os.makedirs(f'{DST}/masks', exist_ok=True)

rgbs = sorted(glob.glob(f'{RAW}/2_Ortho_RGB/*.tif'))
idx = {os.path.basename(p)[:-4]: p
       for p in glob.glob(f'{RAW}/5_Labels_index/*.png')}
print(f'{len(rgbs)} rgb, {len(idx)} idx', flush=True)
assert len(rgbs) == len(idx) == 38, '文件数不是 38，上传可能不完整'

n = 0
for i, rp in enumerate(rgbs):
    scene = '_'.join(os.path.basename(rp).split('_')[:4])
    lp = idx.get(scene)
    assert lp, f'no index label for {scene}'
    img = np.asarray(Image.open(rp).convert('RGB'))
    lab = np.asarray(Image.open(lp))
    H, W = lab.shape
    assert img.shape[:2] == (H, W), f'{scene}: rgb/lab shape mismatch'
    for r, oy in enumerate(OFFS):
        for c, ox in enumerate(OFFS):
            name = f'{scene}_r{r}c{c}.png'
            pi, pm = f'{DST}/images/{name}', f'{DST}/masks/{name}'
            if not os.path.exists(pi):
                Image.fromarray(img[oy:oy + T, ox:ox + T]).save(pi)
            if not os.path.exists(pm):
                Image.fromarray(lab[oy:oy + T, ox:ox + T],
                                mode='L').save(pm)
            n += 1
    print(f'[{i + 1}/{len(rgbs)}] {scene} -> 36 tiles', flush=True)

images = sorted(os.path.abspath(p) for p in glob.glob(f'{DST}/images/*.png'))
gts = sorted(os.path.abspath(p) for p in glob.glob(f'{DST}/masks/*.png'))
assert len(images) == len(gts) == n == 1368, (len(images), len(gts), n)
split = create_split(images, gts, ratios=(0.7, 0.1, 0.2), seed=42,
                     group_pattern=r'^(top_potsdam_\d+_\d+)')
save_split(split, f'{DST}/split.json')
for k in ('train', 'val', 'test'):
    print(f"{k}: {len(split[k]['images'])} tiles", flush=True)
print('ALL DONE', flush=True)
