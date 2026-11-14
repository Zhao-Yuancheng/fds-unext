# FDS-UNeXT

Code for the paper **"Boundary Fidelity in Frozen Foundation-Model Adaptation:
When It Holds and When It Fails for Fine-Grained Rooftop Segmentation"**
(submitted to *Remote Sensing*).

FDS-UNeXT turns two frozen foundation models, the SAM 2.1 Hiera-L image encoder
and DINOv2 ViT-L/14, into a four-class rooftop material segmenter with 6.97 M
trainable parameters (1.33 % of the 523.1 M total). On the Lanzhou Rooftop
Material Dataset (LRMD, 0.14 m GSD) it reaches **70.00 % mIoU** and
**68.58 % boundary F-score** (class-equal weighting).

![Architecture of FDS-UNeXT](assets/architecture.webp)

*Architecture. Snowflake markers denote frozen parameters and flame markers
trainable ones: the two trunks stay frozen, while the 48 Hiera adapters, the
four FDM blocks, the DySample units, the two boundary attention gates, the
U-Net decoder, the deformable-convolution refinement, and the segmentation head
are trained.*

Four components define the pipeline:

| Component | Module | What it replaces / adds |
|---|---|---|
| Frequency-decoupling module (FDM) | `models/fdm.py` | Replaces the dense glue layer between the two encoders with an invertible Haar DWT/IWT pair, an SE recalibration of the low-frequency band, and a zero-initialized gate that injects the high-frequency bands of the SAM stream into the DINOv2 stream. |
| DySample | `models/dysample.py` | Replaces every bilinear upsampling in the decoder with content-aware resampling (zero-initialized offset and scope heads, bounded offsets `T·tanh(o/T)`). |
| Boundary supervision | `models/boundary_modules.py` | A boundary attention gate on the shallow skips (fixed Sobel branch × learned attention), a training-only auxiliary boundary head on 3-px bands, and a deformable-convolution refinement before the final upsampling. |
| Combined loss | `losses/combined_loss.py` | Class-balanced BCE (k = 7) + IoU + SDF boundary term, with an uncertainty-weighted multi-task balance and a Dice-epoch gate. |

The ablation does **not** credit all four with the comparison margin. Removing
the boundary supervision, the three boundary branches, or both together leaves
both metrics within the fold-to-fold spread; DySample is the one architectural
component whose removal is resolved by five folds (BF −2.38, same sign on all
five folds). The paper therefore attributes the margin to the recipe as a
whole, not to a single module.

## Results

| Dataset | Setting | mIoU (%) | BF (%) |
|---|---|---|---|
| LRMD | five folds, 0.14 m | 70.00 ± 6.58 | 68.58 ± 5.08 |
| Potsdam | two seeds, 0.05 m | 75.56 ± 0.49 | 73.18 ± 0.65 |
| E-GCS | three seeds, 0.3 m | 95.15 ± 0.37 | 73.05 ± 1.30 |

All BF values use **class-equal weighting** (per-class mean over images, then
mean over classes) and an 8-pixel tolerance. Do not compare them with
element-weighted BF from earlier releases (Potsdam seed 42: 73.14; E-GCS
three-seed: 76.17 ± 0.59); the two conventions must not be mixed or subtracted.
Under protocol matching, the region margin over SegFormer-B2 disappears and the
LRMD boundary margin narrows from 7.47 to 2.44 points; adding the same boundary
loss to that comparator gives 1.73, and transplanting DySample into it lowers
its BF by 4.39 points over five folds. See the paper for the conditional
reading.

## Repository layout

```
models/          FDS-UNeXT, FDM, DySample, boundary modules, the SAM2-UNeXT
                 reference model, and the U-Net++/UNetFormer baselines
losses/          combined loss (BCE + IoU + SDF, uncertainty weighting)
utils/           dataset split, augmentation (Mosaic-9, copy-paste, resolution
                 perturbation, flips), metrics (mIoU, boundary F-score), SDF
datasets.py      torch Dataset and the train/test transforms
train.py         five-fold training, checkpoint selection on validation mIoU
eval.py          test-set evaluation (mIoU, per-class IoU, boundary F-score)
train_compare.py comparators (U-Net, U-Net++, Attention U-Net, DeepLabV3+,
                 SegFormer-B2, HRNet-W32, UNetFormer-R18)
eval_ref.py      evaluation of the official SAM2-UNeXT baseline
tools/           boundary IoU / HD95, throughput, qualitative and error maps,
                 checkpoint export/merge
sam2/, sam2_configs/   minimal vendored SAM 2 package (Apache-2.0)
assets/          architecture figure shown above
```

## Installation

```bash
conda create -n fds-unext python=3.10 -y
conda activate fds-unext
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the two frozen trunks and place them in `weights/`:

```bash
mkdir -p weights
# SAM 2.1 Hiera-L (Meta, Apache-2.0)
wget -P weights https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# DINOv2 ViT-L/14 (Meta, Apache-2.0)
wget -P weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
```

Check the official repositories for the current file names before downloading.
Only the two trunks are loaded; no pretrained segmentation head is used.

## Data

**LRMD is not redistributed.** The imagery comes from a commercial
very-high-resolution optical mosaic whose terms do not allow redistribution, so
this repository ships code and split definitions only. The annotations, when
released, use the MIT license; the code and weights use Apache-2.0. The four
labels are

| value | class | albedo prior |
|---|---|---|
| 0 | background (concrete roofs, asphalt, soil, vegetation, roads, water, vehicles) | — |
| 1 | high-albedo color-coated steel | 0.30–0.75 |
| 2 | low-albedo color-coated steel | 0.05–0.20 |
| 3 | tile | 0.10–0.35 |

Annotation follows a written protocol; a condensed English version is provided
in Section S10 of the supplementary material of the paper.

Expected layout (any root works):

```
dataset/roof_dataset_v2/
  images/            1024x1024 RGB PNG tiles
  masks/             single-channel PNG, values 0-3 (255 = ignore)
  split_fold0.json   written by train.py when --split_path does not exist
  ...
```

Each `split_fold*.json` holds `train`/`val`/`test` lists of absolute or relative
paths. To rebuild a five-fold, scene-disjoint split, pass a regular expression
for the scene identifier to `train.py` through `--group_pattern`; tiles sharing
a captured group are kept on the same side of every fold.

The two transfer datasets (ISPRS Potsdam at 5 cm and E-GCS at 0.3 m) are public
and are not bundled either. Potsdam is prepared in two steps:
`tools/prepare_potsdam_labels.py` converts the six-class RGB label tiles to
index masks, and `tools/prepare_potsdam_crops.py` cuts 36 anchored 1024-px tiles
per scene and writes a scene-disjoint split. The blur-degradation control of the
supplementary material is generated by `tools/make_potsdam_blur.py`.

## Training

The protocol reported in the paper, for fold *k* of LRMD:

```bash
python train.py \
  --hiera_path weights/sam2.1_hiera_large.pt \
  --dinov2_path weights/dinov2_vitl14_pretrain.pth \
  --train_image_path dataset/roof_dataset_v2/images \
  --train_mask_path  dataset/roof_dataset_v2/masks \
  --split_path       dataset/roof_dataset_v2/split_fold${k}.json \
  --save_path        runs/fold${k} \
  --epoch 100 --batch_size 2 --lr 3e-5 --weight_decay 5e-4 \
  --num_classes 4 --seed 42 \
  --warmup_epochs 10 --mosaic_close_epochs 50 --dy_edge_offset 6 \
  --num_workers 8
```

Two checkpoints are written per run: `FDS-UNeXT-best.pth` (selected on
validation mIoU, used for every number in the paper) and
`FDS-UNeXT-best-bf.pth` (selected on validation boundary F-score).

The defaults of `train.py` are those of the codebase, not those of the paper
(40 epochs, learning rate 2e-4, `warmup_epochs 0`, `mosaic_close_epochs 10`,
`dy_edge_offset 4`). Always pass the flags above when reproducing the paper.

## Evaluation

```bash
python eval.py \
  --hiera_path weights/sam2.1_hiera_large.pt \
  --dinov2_path weights/dinov2_vitl14_pretrain.pth \
  --checkpoint runs/fold${k}/FDS-UNeXT-best.pth \
  --split_path dataset/roof_dataset_v2/split_fold${k}.json --split test \
  --num_classes 4 --bf_tolerance 8 --dy_edge_offset 6 \
  --class_names background,high-albedo-steel,low-albedo-steel,tile
```

The paper reports the boundary F-score with an **8-pixel tolerance**, which is
also the default of `eval.py`; pass `--bf_tolerance 8` explicitly to keep the
protocol visible. Boundary IoU, HD95 and the penalized HD95 variant come from
`tools/b2_boundary_metrics.py` with `--group roof_fds`.

## Reproducing the tables and figures

| Paper item | How |
|---|---|
| Table 3, per-fold LRMD scores | `train.py` + `eval.py` per fold, aggregate the `test_metrics.json` files |
| Table 4, Potsdam / E-GCS | same scripts with the public splits; seeds 42/3407 (Potsdam) and 42/3407/666 (E-GCS) |
| Comparator table | `train_compare.py --model <name> --split_path ...` (it trains and evaluates in one run; boundary F-score tolerance is 8 px by default here) |
| Official SAM2-UNeXT baseline | `train_ref.py` and `eval_ref.py` (GroupNorm instead of BatchNorm, K-class head, bounded residual adapter) |
| Module ablation (Table 5) | `train.py` with `--no-use_fdm`, `--no-use_dysample`, or the three boundary-branch flags together (`--no-use_ba --no-use_boundary_head --no-use_dcn`), each over five folds; the branches were not ablated individually at five folds |
| Single-stream ablation (Table S1) | `train.py --no-use_dino --no-use_fdm` (SAM2-only) or `--no-use_sam --no-use_fdm` (DINOv2-only) |
| Boundary-supervision ablation | `train.py --warmup_epochs 100 --aux_boundary_weight 0 --weit_coeff 0` (SDF off, auxiliary head off, boundary weighting off) |
| Tanh-bound ablation | `train_notanh.py` and `eval_notanh.py`, which build `models/fds_unext_notanh.py` (unbounded adapter residual) |
| Boundary IoU / HD95 (Table 6) | `tools/b2_boundary_metrics.py` |
| Throughput spectrum (Figure 6) | `tools/a1_efficiency.py` |
| Qualitative and error maps (Figures 8, 9 and S2) | `tools/a2_qualitative.py`, `tools/a3_representative.py` |

`results/run_summary.csv` aggregates the runs behind the paper (validation and
test mIoU / boundary F-score per run, run name, and which log the numbers were
taken from).

## Pre-trained checkpoints

The ten checkpoints behind the tables are published on Hugging Face as
**trainable-only** files of about 28 MB each, rather than the 2.0 GB full
checkpoints. A trained model holds 523.1 M parameters, of which the 516.2 M
frozen-trunk parameters are exactly the public SAM 2.1 and DINOv2 weights, so
only the 6.97 M parameters that the optimizer updated are stored.

Model hub: `https://huggingface.co/Jim-myoenk/FDS-UNeXT`

| file | trained on | used for |
|---|---|---|
| `lrmd-fold0.pth` … `lrmd-fold4.pth` | LRMD, five folds | Table 3, Figures 2 and 5 |
| `potsdam-seed42.pth`, `potsdam-seed3407.pth` | Potsdam | Table 4, Figure 9a |
| `egcs-seed42.pth`, `egcs-seed3407.pth`, `egcs-seed666.pth` | E-GCS | Table 4, Figures 3 and 9b |

Rebuild a full checkpoint that `eval.py` can read:

```bash
# download one delta
hf download Jim-myoenk/FDS-UNeXT lrmd-fold0.pth --local-dir deltas

python tools/merge_trainable.py \
  --delta deltas/lrmd-fold0.pth \
  --hiera_path weights/sam2.1_hiera_large.pt \
  --dinov2_path weights/dinov2_vitl14_pretrain.pth \
  --out runs/fold0/FDS-UNeXT-best.pth
```

The round trip is exact: exporting and re-merging a checkpoint reproduces every
tensor of the original file. `tools/export_trainable.py` performs the reverse
direction and is what produced the published deltas.

## Implementation notes

* **Mixed precision.** In the version used for the paper, the autocast context
  was enabled whenever a gradient scaler object existed, so all runs executed
  the forward pass in float16 even with `--amp False`. The flag is kept for
  compatibility; see §3.5.2 of the paper for the disclosure and the measured
  effect (about 0.36 mIoU below a float32 evaluation of the same checkpoints).
* **Photometric gain.** `--photo_metric` and `--photo_gain` are implemented but
  disabled in every run reported in the paper, because the high-/low-albedo
  distinction is itself photometric.
* **MLP head.** `--use_mlp_head` is off by default and off in the paper; the
  fusion head remains in the code as an optional variant.
* **Bounded adapter.** The residual adapter of the SAM branch uses a `tanh`
  bound. The unbounded variant reported in the supplementary material is
  produced by `train_notanh.py`, which instantiates
  `models/fds_unext_notanh.py`.
* **Class absent from the ground truth.** Metrics skip a class that does not
  appear in the reference tile, following the convention of `utils/metrics.py`.

## License

Apache-2.0; see [LICENSE](LICENSE). The vendored `sam2/` package and the two
frozen trunks come from Meta AI and are distributed under the same license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{zhao2026fdsunext,
  title   = {Boundary Fidelity in Frozen Foundation-Model Adaptation: When It
             Holds and When It Fails for Fine-Grained Rooftop Segmentation},
  author  = {Zhao, Yuancheng and Wang, Jie and Li, Faguo and Liu, Erte and
             Meng, Xiangying and Chen, Yueyue and Yang, Jiapeng and
             Zhou, Yibo and Li, Qingxue},
  journal = {Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgements

The SAM 2 implementation is adapted from
[facebookresearch/sam2](https://github.com/facebookresearch/sam2); the decoder
follows SAM2-UNeXT. The computations were performed on the Supercomputing
Center of Lanzhou University.
