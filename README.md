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

## Setup

The main dependencies (SAM3, hloc, UniK3D) are external repos, not pip
packages. Each is installed into its own conda environment and used for
the corresponding step. There is intentionally **no single
`requirements.txt`** — a flat `pip install` cannot reproduce these envs.

### External repos to clone and install

Follow each project's install instructions to create the conda env.

| Step | Conda env | External repo |
|---|---|---|
| `02_make_masks.py` (fisheye) | `sam3` | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) |
| `02_run_hloc.py` / `03_run_hloc.py` | `hloc` | [cvg/Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization) |
| `04_unik3d_priors.py` (fisheye, optional) | `unik3d` | [lpiccinelli-eth/UniK3D](https://github.com/lpiccinelli-eth/UniK3D) |

### Extra pip packages

Inside each env, install these as needed (most envs already include them):

```bash
pip install pillow numpy torch tqdm pycolmap plyfile opencv-python
```

### Per-script env summary

- `01_resize_half.py` — Pillow only; any env works.
- `02_make_masks.py` (fisheye) — `sam3` env, SAM3 repo as cwd.
- `02_run_hloc.py` / `03_run_hloc.py` — `hloc` env.
- `04_unik3d_priors.py` (fisheye, optional) — `unik3d` env.
- `05_normals_from_points.py` (fisheye, optional) — any env with `numpy`,
  `torch`, `pillow`.
- `utils/sparse_cleanup/*.py` — any env with `pycolmap`, `plyfile`, `numpy`.

See each pipeline's README for the exact run commands.

## Trainers

The output of these pipelines feeds into 3DGS trainers:
- [QjaYo/gsplat_gut](https://github.com/QjaYo/gsplat_gut) — 3DGUT trainer
  for raw distorted captures (fisheye + perspective, with optional
  depth/normal supervision).
- LichtFeld-Studio — verified RGB-only baseline (external).

## Utilities

See [`utils/`](utils/) for additional helpers (e.g. interactive sparse
outlier cleanup).
