"""
Step 4 (optional): Generate fisheye depth + per-pixel 3D points with UniK3D,
explicitly told the camera is an OPENCV_FISHEYE.

This step is only needed if you plan to train with depth/normal supervision
(DN-Splatter style). Vanilla LichtFeld GUT training does not need it.

Step 5 (separate script) derives surface normals from the 3D points produced
here. Keeping them split lets you re-derive normals without re-running the
expensive UniK3D inference, and lets you compare normal-derivation strategies.

Why pass fisheye intrinsics explicitly
--------------------------------------
UniK3D is camera-agnostic — it will run on any image without intrinsics and
produce a metric 3D point map by guessing the camera model. Its API also
accepts an optional `camera` argument and will use it when given. This
script lives in `pipeline_fisheye/` and is only used for fisheye captures,
so we **hardcode the Fisheye624 camera model** and read the per-scene
intrinsics from the hloc output `<root>/sparse/0/cameras.bin`, eliminating
the guess. Step 3 (hloc) must therefore be done before step 4.

Inputs / Outputs
----------------
Reads:
  <root>/images/*.jpg                    RGB images (from step 1)
  <root>/sparse/0/cameras.bin            hloc-refined OPENCV_FISHEYE intrinsics
                                         (from step 3; required as of this rev)

Writes:
  <root>/mono_depth/<stem>_aligned.npy   metric depth (H, W) float32
                                         (DN-Splatter convention; UniK3D is
                                         already metric so no scale alignment
                                         step is needed)
  <root>/unik3d_points/<stem>.npy        per-pixel 3D point (H, W, 3) float32
                                         in UniK3D's camera frame
                                         (input to step 5 for normals)

Environment
-----------
UniK3D requires Python 3.11+, so use the `unik3d` conda env:

  conda activate unik3d
  python 04_unik3d_priors.py --root <dataset>

Re-runs
-------
Per-image skip: if <stem>_aligned.npy and <stem>.npy already exist for an
image, that image is skipped. Pass --force to re-generate everything.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# COLMAP camera model ids (subset). 5 = OPENCV_FISHEYE.
_COLMAP_OPENCV_FISHEYE = 5


def read_opencv_fisheye_intrinsics(cameras_bin: Path):
    """Read OPENCV_FISHEYE intrinsics from a hloc cameras.bin.

    Expects exactly one camera (this pipeline runs hloc with a *shared*
    OPENCV_FISHEYE camera prior). Raises if the model is wrong or if the
    file has more than one camera with conflicting intrinsics.

    Returns:
        (width, height, fx, fy, cx, cy, k1, k2, k3, k4)
    """
    if not cameras_bin.is_file():
        raise SystemExit(
            f"[04] missing {cameras_bin}\n"
            f"     Run step 3 (03_run_hloc.py) first — UniK3D needs hloc-refined "
            f"OPENCV_FISHEYE intrinsics."
        )
    with open(cameras_bin, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        if num == 0:
            raise SystemExit(f"[04] {cameras_bin} contains 0 cameras.")
        # Read first camera; verify model.
        cam_id = struct.unpack("<I", f.read(4))[0]
        model_id = struct.unpack("<i", f.read(4))[0]
        width = struct.unpack("<Q", f.read(8))[0]
        height = struct.unpack("<Q", f.read(8))[0]
        if model_id != _COLMAP_OPENCV_FISHEYE:
            raise SystemExit(
                f"[04] {cameras_bin}: camera {cam_id} has model_id={model_id}, "
                f"expected OPENCV_FISHEYE (5). This script is fisheye-only."
            )
        # OPENCV_FISHEYE has 8 params: fx, fy, cx, cy, k1, k2, k3, k4 (doubles).
        fx, fy, cx, cy, k1, k2, k3, k4 = struct.unpack("<8d", f.read(8 * 8))
        # (If multiple cameras existed we'd just take the first; this pipeline
        # only ever has one. We don't validate the others.)
    print(f"[04] hloc OPENCV_FISHEYE intrinsics:")
    print(f"     image={width}x{height}")
    print(f"     fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"     k1={k1:.4f} k2={k2:.4f} k3={k3:.4f} k4={k4:.4f}")
    return width, height, fx, fy, cx, cy, k1, k2, k3, k4


def make_unik3d_fisheye(intrinsics, device: str = "cuda"):
    """Build a UniK3D Fisheye624 from our (8-param) OPENCV_FISHEYE.

    Fisheye624 packs 16 params: fx, fy, cx, cy, k1..k6, p1, p2, s1..s4.
    OPENCV_FISHEYE only has 4 radial coefficients and no tangential / thin-
    prism terms, so we zero-pad k5..k6, p1, p2, s1..s4.
    """
    from unik3d.utils.camera import Fisheye624
    _w, _h, fx, fy, cx, cy, k1, k2, k3, k4 = intrinsics
    params = torch.tensor(
        [fx, fy, cx, cy,
         k1, k2, k3, k4,
         0.0, 0.0,          # k5, k6
         0.0, 0.0,          # p1, p2
         0.0, 0.0, 0.0, 0.0],   # s1, s2, s3, s4
        dtype=torch.float32, device=device,
    )
    return Fisheye624(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Dataset root containing images/")
    ap.add_argument("--model", default="lpiccinelli/unik3d-vitl",
                    help="HuggingFace model id (default: vitl). Use vits for "
                         "smaller / faster, vitg for higher quality.")
    ap.add_argument("--force", action="store_true",
                    help="Re-generate even if outputs already exist.")
    ap.add_argument("--max-images", type=int, default=0,
                    help="Process only the first N images (0 = all).")
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "images"
    if not img_dir.exists():
        sys.exit(f"ERROR: {img_dir} not found. Run step 1 first.")

    depth_dir = root / "mono_depth"
    points_dir = root / "unik3d_points"
    depth_dir.mkdir(exist_ok=True, parents=True)
    points_dir.mkdir(exist_ok=True, parents=True)

    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"ERROR: no images found in {img_dir}")

    if not args.force:
        already = sum(
            1 for f in files
            if (depth_dir / f"{f.stem}_aligned.npy").exists()
            and (points_dir / f"{f.stem}.npy").exists()
        )
        if already == len(files):
            print(f"All {len(files)} images already have UniK3D depth + points. "
                  f"Pass --force to re-generate.")
            return

    # Read hloc-refined OPENCV_FISHEYE intrinsics; the Fisheye624 camera object
    # is rebuilt per image inside the loop (see note below).
    intrinsics = read_opencv_fisheye_intrinsics(root / "sparse" / "0" / "cameras.bin")
    from unik3d.models import UniK3D   # imported here so --help works without torch
    print(f"Loading UniK3D ({args.model}) ...")
    model = UniK3D.from_pretrained(args.model).cuda().eval()

    # NOTE: UniK3D's base Camera.crop / .resize mutate self.K and self.params
    # in place (and Fisheye624 / OPENCV inherit them unchanged). model.infer()
    # calls both internally, so reusing one camera object across calls causes
    # fx, fy, cx, cy to drift every iteration — fx is multiplied by
    # resize_factor (~0.26 for 2992² fisheye) each call and hits 0 within ~15
    # images, after which depths come back all-NaN. Rebuild a fresh camera
    # per image to dodge it. (Verified in UniK3D 0.x.)

    print(f"Processing {len(files)} images ...")
    for i, f in enumerate(files, 1):
        depth_path = depth_dir / f"{f.stem}_aligned.npy"
        points_path = points_dir / f"{f.stem}.npy"

        if not args.force and depth_path.exists() and points_path.exists():
            continue

        rgb = np.array(Image.open(f).convert("RGB"))                   # (H, W, 3)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda()   # (3, H, W)

        camera = make_unik3d_fisheye(intrinsics, device="cuda")

        with torch.no_grad():
            pred = model.infer(rgb_t, camera=camera)

        depth = pred["depth"].squeeze().cpu().numpy().astype(np.float32)   # (H, W)
        points = pred["points"].squeeze().cpu().numpy().astype(np.float32)  # (3, H, W) or (H, W, 3)
        if points.shape[0] == 3:
            points = points.transpose(1, 2, 0)                             # → (H, W, 3)

        np.save(depth_path, depth)
        np.save(points_path, points)

        if i % 20 == 0 or i == len(files):
            pos = depth[depth > 0]
            rng = f"{pos.min():.2f}-{pos.max():.2f}m" if len(pos) else "(empty)"
            print(f"  [{i:>3}/{len(files)}] {f.name}  depth {rng}")

    print("\nDone.")
    print(f"  depth:  {depth_dir}/")
    print(f"  points: {points_dir}/")


if __name__ == "__main__":
    main()
