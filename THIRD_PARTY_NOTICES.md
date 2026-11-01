# Third-party notices

This repository is released under the Apache License 2.0 (see `LICENSE`). It
contains or depends on the following third-party components.

## Vendored source

**`sam2/` and `sam2_configs/`** — a minimal subset of the Segment Anything 2
reference implementation, © Meta Platforms, Inc. and affiliates, licensed under
the Apache License 2.0. Upstream: <https://github.com/facebookresearch/sam2>.
Only the modules needed to build and run the Hiera image encoder are included;
the original file headers are retained where present.

## Model weights (downloaded, not distributed)

Both frozen trunks are distributed by Meta AI under the Apache License 2.0 and
must be downloaded by the user:

* **SAM 2.1 Hiera-L** — <https://github.com/facebookresearch/sam2>
* **DINOv2 ViT-L/14** — <https://github.com/facebookresearch/dinov2>

## Python dependencies

`torch`, `torchvision`, `timm`, `numpy`, `scipy`, `Pillow`, `tqdm`,
`hydra-core`, `omegaconf`, `iopath`, `segmentation-models-pytorch` and
`opencv-python` are installed from PyPI and keep their own licenses. In
particular, `segmentation-models-pytorch` is used for the SegFormer-B2
comparator only.

## Data

The Lanzhou Rooftop Material Dataset is **not** included. The underlying imagery
was retrieved from a public web mapping service whose terms of use do not permit
redistribution. ISPRS Potsdam and E-GCS are public benchmarks and are obtained
from their official sources.
