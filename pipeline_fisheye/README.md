# Fisheye Pipeline

Convert circular fisheye captures (e.g. 200° single-lens) into a 3DGS-ready
sparse model and train via one of two paths.

- **Path A — LichtFeld-Studio**: verified baseline. Fast and stable, RGB-only.
- **Path B — gsplat GUT trainer**: fisheye + 3DGUT + DN-Splatter-style
  depth/normal supervision (experimental, in-house).

## Starting layout

```
<dataset>/
└── raw/                     ← camera originals (.insp, .jpg, .png)
```

---

## Data preparation (common to both paths)

```bash
DATA=/path/to/your_dataset

# 1) Downsample 1/2 (parallel, ~5 s)
#    Target: Insta360 X4 5984² → 2992²  (--target-longest-side 2992, default)
#    Skips automatically if images/ is already ≤ 2992 — safe to re-run.
conda activate hloc
python 01_resize_half.py --root $DATA
# → $DATA/images/  (1/2 resolution, square preserved)

# 2) SAM3 masks (fisheye disc + person/grip removal)
#    ~10 s/image (~25 min for 155 images on 4090)
conda activate sam3
cd ~/Development/sam3                      # SAM3 import path
python $DATA/02_make_masks.py --root $DATA
# → $DATA/masks/, $DATA/mask_overlays/

# 3) hloc SfM (OPENCV_FISHEYE camera model)
#    SuperPoint + LightGlue + MegaLoc retrieval k=50
#    ~30 min for 155 images
conda activate hloc
python 03_run_hloc.py --root $DATA
# → $DATA/sparse/0/{cameras,images,points3D}.bin
# → $DATA/hloc_cache/   (rerun cache)

# 4) (optional) UniK3D depth + normal — only if Path B with d+n supervision
#    ~4 min for 155 images (vit-l)
conda activate unik3d
python 04_unik3d_priors.py --root $DATA
# → $DATA/mono_depth/<stem>_aligned.npy        (metric depth)
# → $DATA/normals_from_pretrain/<stem>.png     (derived from 3D points)
# → $DATA/unik3d_points/<stem>.npy             (debug)
```

### Key options

**02_make_masks.py**
- `luma_thresh=80`, `shrink=0.93` (verified on Insta360 X4 single lens).
- Includes the fisheye vignette gradient in the mask to suppress 3DGS floaters.
- More permissive values (e.g. `luma=25`) leave the gradient as "valid" and
  produce floaters in training.

**03_run_hloc.py**
- `--num_matched 50` (default): retrieval top-50. Usually sufficient.
- `--num_matched 0`: exhaustive matching (small datasets).
- `--feature superpoint_max` (default): robust for fisheye.

### Sanity check (printed at the end of step 3)

Expected BA convergence values for an Insta360 X4 200° single-lens fisheye:
- fx ≈ fy ≈ 781
- cx ≈ cy ≈ 1493 (near center of 2992² image)
- k1 ≈ +0.08, k2 ≈ -0.03, k3 ≈ +0.01, k4 ≈ -0.003
- registered: 95%+
- mean reproj error: < 1.5 px

### Other fisheye cameras

Adjust the initial FOV with `03_run_hloc.py --fov` (default 200°).
Keep within ±20° of the real FOV to avoid BA divergence.

---

## Path A — LichtFeld-Studio (baseline)

> Pick A *or* B (both consume the same `$DATA`). You can also run both into
> different output directories and compare.

Verified, stable baseline. RGB-only (no depth/normal supervision). Step 4
not required.

```bash
cd ~/Development/LichtFeld-Studio
build/LichtFeld-Studio \
    -d $DATA \
    -o $DATA/output \
    --gut --mask-mode ignore \
    --strategy mcmc \
    --enable-mip \
    --train --headless
```

### Required LichtFeld options
- `--gut`: native OPENCV_FISHEYE rendering.
- `--mask-mode ignore`: exclude `masks/*.png` regions from the loss.
- `--strategy mcmc`: stable densification.
- `--enable-mip`: anti-aliasing.
- `--undistort` OFF (default): conflicts with `--gut`; do not enable.

---

## Path B — gsplat GUT trainer (fisheye + 3DGUT + d/n supervision)

In-house trainer at [QjaYo/gsplat_gut](https://github.com/QjaYo/gsplat_gut)
(`examples/gut/`). Upstream gsplat is not modified — `git pull` won't conflict.

> Not fisheye-only: `gut_colmap.Parser` auto-detects the COLMAP camera type
> (PINHOLE / RADIAL / OPENCV / OPENCV_FISHEYE, etc.) and passes it straight
> to the rasterizer. Same command also trains pinhole captures.

### Why a separate trainer

LichtFeld is a stable baseline but two things are blocked:

1. **Want depth + normal dense supervision on fisheye.**
   - DN-Splatter assumes perspective cameras (its mono prior nets are trained
     on perspective images).
   - UniK3D predicts metric 3D directly regardless of fisheye / perspective,
     so dense d+n GT can be generated for fisheye too.
   - Need a trainer that plugs these into the Gaussian-splatting loss.
     LichtFeld has no d/n loss slot.

2. **3DGUT only makes sense on raw distorted images.**
   - Upstream `gsplat/examples/simple_trainer.py` undistorts fisheye with
     `cv2.remap` before training. That destroys the point of 3DGUT
     (projecting gaussians through the actual non-linear camera function)
     and discards FOV beyond 180°.
   - `gut_colmap.py` skips the undistortion step and forwards
     `radial_coeffs` / `tangential_coeffs` to the rasterizer.

In short, this trainer enables **"fisheye + 3DGUT + DN-Splatter-style d/n
supervision"** on the same data/strategy as the LichtFeld baseline, for a
fair comparison.

### Basic training (RGB-only, no d+n)

```bash
conda activate nerfstudio_dev
cd ~/Development/gsplat_gut/examples/gut
python gut_trainer.py mcmc \
    --data-dir $DATA \
    --result-dir $DATA/output_gut \
    --data-factor 1 \
    --max-steps 30000 \
    --disable-video
```

### Training with d+n supervision (requires step 4)

Requires `$DATA/mono_depth/` and `$DATA/normals_from_pretrain/`.

```bash
conda activate nerfstudio_dev
cd ~/Development/gsplat_gut/examples/gut
python gut_trainer.py mcmc \
    --data-dir $DATA \
    --result-dir $DATA/output_gut_dn \
    --data-factor 1 \
    --max-steps 30000 \
    --use-mono-depth-loss --mono-depth-lambda 0.2 \
    --use-mono-normal-loss --mono-normal-lambda 0.1 \
    --disable-video
```

### gsplat GUT trainer options

**Required**
- `mcmc`: first positional arg; selects MCMC densification (only strategy
  supported by 3DGUT).
- `--data-dir`: `<dataset>/` (contains images, masks, sparse).
- `--result-dir`: output directory; auto-created with subfolders
  (`ckpts/`, `ply/`, `renders/`, `stats/`, `tb/`).
- (Note) `gut_trainer.py` does `sys.path.insert` for its own parent, so no
  `PYTHONPATH` setup is needed.

**Common options**
- `--max-steps 30000`: training step count. `save_steps` / `ply_steps`
  default to `[max_steps//3, 2*max_steps//3, max_steps]` (e.g. 10k/20k/30k).
- `--data-factor 1`: no further downsampling (already 1/2 from step 1).
- `--disable-video`: skip end-of-training video (slow).
- `--disable-viewer`: turn off viser (headless / save VRAM). On by default.
- `--eval-steps 30000` (opt-in): run evaluation + save val renders at the
  final step. Off by default — no eval unless this flag is passed.
- `--save-steps 15000 30000` / `--ply-steps ...`: manual save schedule.
  Default is automatic.

**Depth supervision**
- `--use-mono-depth-loss`: L1 between rendered depth and UniK3D metric depth.
- `--mono-depth-lambda 0.2`: depth-loss weight (recommended 0.1–0.5).
- `--mono-depth-loss-type {L1,Pearson}`: default `L1` (use with metric depth);
  use `Pearson` for relative depth (scale-shift invariant).

**Normal supervision**
- `--use-mono-normal-loss`: surface-normal loss. Takes the gaussian's
  minor axis as its normal and compares to GT.
- `--mono-normal-lambda 0.1`: normal-loss weight (recommended 0.05–0.2).
- `--mono-normal-loss-type {cosine,L1}`: default `cosine` (1 − n·g).

**Camera model**
- Default: `--camera-model fisheye` (OPENCV_FISHEYE).
- Pinhole captures: `--camera-model pinhole`; distortion still flows through
  via `radial_coeffs` / `tangential_coeffs`.
