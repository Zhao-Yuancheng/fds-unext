"""ISPRS Potsdam 标签 RGB→索引转换（制备第一步）。

输入：../数据集制备/ISPRS Potsdam/5_Labels_all_noBoundary/*.tif
      （RGB 彩色编码，6000×6000，实测恰好 6 色）
输出：../数据集制备/ISPRS Potsdam/5_Labels_index/<scene_id>.png
      （单波段 uint8 0-5，scene_id 如 top_potsdam_2_10）

类别序（ISPRS 标准口径，训练直接 --num_classes 6 无需 label_mapping）：
    0 不透水面 (255,255,255)   1 建筑 (0,0,255)   2 低植被 (0,255,255)
    3 树 (0,255,0)             4 车 (255,255,0)   5 杂乱 (255,0,0)

noBoundary 版无黑色边界线，应恰好 6 色；颜色重叠直接抛错。非精确匹配
像素按最近色吸附：int32 距离平方 <= 70^2（欧氏 70px）内吸附到该色，
逐文件报告吸附占比；超出阈值仍硬失败（真实污染，不静默吞掉）。

实测（int32 距离普查）：37/38 景 100% 纯净或近纯净；仅 4_12 全图有损
压缩污染（med d2≈1890，全部在 70px 内）、6_7 轻微噪声（0.69%），
二者吸附即可，其余景吸附占比应为 0。
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

SRC = os.environ.get('POTSDAM_LABEL_RGB',
                     'dataset/potsdam_raw/5_Labels_all_noBoundary')
DST = os.environ.get('POTSDAM_LABEL_INDEX',
                     'dataset/potsdam_raw/5_Labels_index')

COLORS = {
    (255, 255, 255): 0,   # impervious surface
    (0, 0, 255): 1,       # building
    (0, 255, 255): 2,     # low vegetation
    (0, 255, 0): 3,       # tree
    (255, 255, 0): 4,     # car
    (255, 0, 0): 5,       # clutter
}

SNAP_D2 = 72 ** 2   # 最近色吸附阈值：欧氏 72px（差值<=255，平方和<=195075，
                    # int32 无溢出；int16 会回绕成负数，Q117 实测教训）。
                    # 定 72 而非 70：4_12 有损压缩尾部在 36M 像素中有 2 个
                    # 落在 70.0-70.5px（如 d2=4962），仍属伪影；调色板缺
                    # 黑/品红角落，真实污染（近黑或近品红像素）最近色
                    # d2 可达 255^2=65025，远超此界，硬失败保护仍有效


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, '*.tif')))
    print(f'{len(files)} label tiles', flush=True)
    for i, p in enumerate(files):
        scene = '_'.join(os.path.basename(p).split('_')[:4])
        out = os.path.join(DST, scene + '.png')
        if os.path.exists(out):
            print(f'[{i + 1}/{len(files)}] skip {scene}', flush=True)
            continue
        a = np.asarray(Image.open(p).convert('RGB'))
        h, w, _ = a.shape
        idx = np.full((h, w), 255, np.uint8)
        matched = np.zeros((h, w), bool)
        for (r, g, b), c in COLORS.items():
            m = (a[..., 0] == r) & (a[..., 1] == g) & (a[..., 2] == b)
            if (matched & m).any():
                sys.exit(f'FATAL: color overlap in {p}')
            matched |= m
            idx[m] = c
        n_snap = 0
        if not matched.all():
            # 非精确像素：int32 最近色距离，阈值内吸附、超出硬失败
            sub = a[~matched].astype(np.int32)          # (N,3)
            best_d2 = np.full(len(sub), 1 << 30, np.int64)
            best_c = np.full(len(sub), 255, np.uint8)
            for (r, g, b), c in COLORS.items():
                d2 = ((sub[:, 0] - r) ** 2 + (sub[:, 1] - g) ** 2
                      + (sub[:, 2] - b) ** 2)           # int32，无溢出
                upd = d2 < best_d2
                best_d2[upd] = d2[upd]
                best_c[upd] = c
            bad = best_d2 > SNAP_D2
            if bad.any():
                j = int(np.argmax(bad))
                sys.exit(f'FATAL: {int(bad.sum())} px beyond d2 {SNAP_D2} '
                         f'in {p}, e.g. '
                         f'rgb={tuple(int(v) for v in sub[j])} '
                         f'd2={int(best_d2[j])}')
            idx[~matched] = best_c
            n_snap = int(len(sub))
        Image.fromarray(idx, mode='L').save(out)
        snap_pct = 100.0 * n_snap / (h * w)
        print(f'[{i + 1}/{len(files)}] {scene} {h}x{w} ok'
              + (f' snap={snap_pct:.2f}%({n_snap})' if n_snap else ''),
              flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
