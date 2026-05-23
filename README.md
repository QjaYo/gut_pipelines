# GS Preprocessing Pipelines

Data preparation pipelines for 3D Gaussian Splatting (3DGS) training,
covering both fisheye and perspective captures.

## Two pipelines

| Folder | Camera type | Examples |
|---|---|---|
| [`pipeline_fisheye/`](pipeline_fisheye/) | Circular fisheye (≈200°) | Insta360 X4 single lens, GoPro Max single lens |
| [`pipeline_perspective/`](pipeline_perspective/) | Standard perspective | Smartphones, DSLRs, extracted video frames |

## Which one to use

- **Image is a circular disc with a black border** → fisheye
- **Standard rectangular field of view** → perspective

## Common output layout (3DGS training input)

```
<dataset>/
├── images/                  ← training images (downsampled to 1/2)
├── masks/                   ← (fisheye only) grip / vignette masks
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

## Conda environments

Each step requires a specific conda environment:
- `01_resize_half.py` — PIL only (any env works)
- `02_make_masks.py` (fisheye only) — `sam3` env, with SAM3 repo as cwd
- `02_run_hloc.py` / `03_run_hloc.py` — `hloc` env
- `04_unik3d_priors.py` (fisheye, optional) — `unik3d` env

See each pipeline's README for details.

## Trainers

The output of these pipelines feeds into 3DGS trainers:
- [QjaYo/gsplat_gut](https://github.com/QjaYo/gsplat_gut) — 3DGUT trainer
  for raw distorted captures (fisheye + perspective, with optional
  depth/normal supervision).
- LichtFeld-Studio — verified RGB-only baseline (external).

## Utilities

See [`utils/`](utils/) for additional helpers (e.g. interactive sparse
outlier cleanup).
