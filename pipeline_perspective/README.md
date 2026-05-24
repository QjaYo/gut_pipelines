# Perspective Pipeline

Convert standard perspective captures into a 3DGS-ready sparse model.

## Starting layout

```
<dataset>/
└── raw/                     ← camera originals or extracted video frames (.jpg, .png)
```

To extract frames from a video:
```bash
ffmpeg -i video.MP4 -vf fps=2 -qscale:v 1 -qmin 1 $DATA/raw/%04d.jpg
```

## Steps

```bash
DATA=/path/to/your_dataset

# 1) Downsample 1/2 (parallel, ~seconds)
#    Target: 4K (3840×2160) → 2K (1920×1080)  (--target-longest-side 1920, default)
#    Skips automatically if images/ is already ≤ 1920 — safe to re-run.
conda activate hloc
python 01_resize_half.py --root $DATA
# → $DATA/images/  (1/2 resolution)

# 2) hloc SfM (OPENCV camera model)
#    SuperPoint + LightGlue + MegaLoc retrieval k=50
#    ~10–30 min for 100–200 images (varies with resolution and graph density)
python 02_run_hloc.py --root $DATA
# → $DATA/sparse/0/{cameras,images,points3D}.bin
# → $DATA/hloc_cache/   (rerun cache)

# 3) (optional) UniK3D depth + per-pixel 3D points — for d+n supervised training
#    Reads hloc OPENCV intrinsics from sparse/0/cameras.bin
#    155 images ~5 min (vit-l)
conda activate unik3d
python 03_unik3d_priors.py --root $DATA
# → $DATA/mono_depth/<stem>_aligned.npy
# → $DATA/unik3d_points/<stem>.npy

# 4) (optional) surface normals from UniK3D 3D points
python 04_normals_from_points.py --root $DATA
# → $DATA/normals_from_pretrain/<stem>.png

# 5) Train (example: LichtFeld-Studio, vanilla; step 3-4 not needed)
cd ~/Development/LichtFeld-Studio
build/LichtFeld-Studio \
    -d $DATA \
    -o $DATA/output \
    --strategy mcmc \
    --enable-mip \
    --train --headless

# Or train with d+n supervision (gsplat gut/ trainer, needs step 3-4 outputs):
#  python ~/Development/gsplat_gut/examples/gut/simple_trainer.py mcmc \
#     --data-dir $DATA --result-dir $DATA/output_dn \
#     --camera-model pinhole \
#     --use-mono-depth-loss --use-mono-normal-loss --disable-video
```

## Downsampling guidance

| Source resolution | Downsample | Training time (4090) | Quality |
|---|---|---|---|
| 4K (3840×2160) | 1/2 → 2K | ~40 min (30k iter) | nearly identical |
| 2K (2560×1440) | none or 1/2 | 30–60 min | ↑ |
| 1080p | none | ~25 min | OK |

Training directly on 4K is not advisable — ~4× training time for a
marginal quality difference.

## Key options

### 02_run_hloc.py
- `--num_matched 50` (default): retrieval top-50.
- `--num_matched 0`: exhaustive matching.
- `--feature superpoint_max` (default): most-tested feature.
- `--feature disk`: stronger on weak textures.

### LichtFeld training
- `--strategy mcmc`: stable.
- `--enable-mip`: anti-aliasing.
- `--gut` and `--mask-mode` are not needed for perspective.

## Sanity check (printed at the end of step 2)

For a standard perspective camera:
- registered: 90%+ (95%+ typical for video frames since inter-frame motion
  is small).
- mean reproj error: < 1.0 px.
- camera model: OPENCV.
- params: `[fx, fy, cx, cy, k1, k2, p1, p2]`
  - fx ≈ fy ≈ image width (roughly).
  - cx ≈ cy ≈ image center.
  - Distortion (k1..p2) close to 0 (typical modern lens).

## Troubleshooting

### Low registration rate (< 80%)
- Video fps too high → too little change between frames → weak matching
  graph. Try lower fps (e.g. 2 → 1).
- Strong lighting changes → weak SuperPoint matches. Try `--feature disk`.

### Reconstruction split (Reconstruction 0, 1, ...)
- COLMAP creates separate reconstructions for disconnected sub-graphs.
- Decide by registration rate:
  - 90%+ in the first reconstruction → fine to use.
  - Heavy splits → recapture the video, or increase retrieval k
    (`--num_matched 80`).
