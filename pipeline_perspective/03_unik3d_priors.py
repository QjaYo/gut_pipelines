"""
Step 3 (optional): perspective UniK3D depth + per-pixel 3D points.

Same purpose as `pipeline_fisheye/04_unik3d_priors.py` but for OPENCV
(perspective) cameras: reads hloc's BA-refined intrinsics from
`<root>/sparse/0/cameras.bin` and feeds them to UniK3D as an `OPENCV`
camera object so UniK3D does not have to guess the projection from the
image alone.

Reads:
  <root>/images/*.jpg                    (from step 1)
  <root>/sparse/0/cameras.bin            OPENCV camera (model 4), 8 params

Writes:
  <root>/mono_depth/<stem>_aligned.npy   metric depth (H, W) float32
  <root>/unik3d_points/<stem>.npy        per-pixel 3D point (H, W, 3) float32
                                         in UniK3D's camera frame

Environment:
  conda activate unik3d
  python 03_unik3d_priors.py --root <dataset>

Re-runs skip per image; pass --force to regenerate.
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# COLMAP camera model ids: 4 = OPENCV (perspective + 2 radial + 2 tangential).
_COLMAP_OPENCV = 4


def read_opencv_intrinsics(cameras_bin: Path):
    """Read OPENCV intrinsics from a hloc cameras.bin.

    Returns:
        (width, height, fx, fy, cx, cy, k1, k2, p1, p2)
    """
    if not cameras_bin.is_file():
        raise SystemExit(
            f"[03] missing {cameras_bin}\n"
            f"     Run step 2 (02_run_hloc.py) first."
        )
    with open(cameras_bin, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        if num == 0:
            raise SystemExit(f"[03] {cameras_bin} contains 0 cameras.")
        cam_id = struct.unpack("<I", f.read(4))[0]
        model_id = struct.unpack("<i", f.read(4))[0]
        width = struct.unpack("<Q", f.read(8))[0]
        height = struct.unpack("<Q", f.read(8))[0]
        if model_id != _COLMAP_OPENCV:
            raise SystemExit(
                f"[03] camera {cam_id} has model_id={model_id}, expected "
                f"OPENCV (4). This script is for perspective only."
            )
        # OPENCV: 8 doubles — fx, fy, cx, cy, k1, k2, p1, p2.
        fx, fy, cx, cy, k1, k2, p1, p2 = struct.unpack("<8d", f.read(8 * 8))
    print(f"[03] hloc OPENCV intrinsics:")
    print(f"     image={width}x{height}")
    print(f"     fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"     k1={k1:.4f} k2={k2:.4f} p1={p1:.4f} p2={p2:.4f}")
    return width, height, fx, fy, cx, cy, k1, k2, p1, p2


def make_unik3d_opencv(intrinsics, device: str = "cuda"):
    """Build a UniK3D OPENCV camera from the COLMAP OPENCV intrinsics.

    UniK3D's `OPENCV` packs 16 params: fx, fy, cx, cy, k1..k6, p1, p2, s1..s4.
    COLMAP's OPENCV gives only k1, k2, p1, p2 — the rest are zero-padded.
    """
    from unik3d.utils.camera import OPENCV
    _w, _h, fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
    params = torch.tensor(
        [fx, fy, cx, cy,
         k1, k2,                  # k1, k2
         0.0, 0.0, 0.0, 0.0,      # k3..k6 (poly division must be 0 per UniK3D assert)
         p1, p2,                  # p1, p2
         0.0, 0.0, 0.0, 0.0],     # s1..s4 (thin prism)
        dtype=torch.float32, device=device,
    )
    return OPENCV(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Dataset root containing images/ and sparse/0/")
    ap.add_argument("--model", default="lpiccinelli/unik3d-vitl",
                    help="HuggingFace model id (vits/vitl/vitg).")
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
    if args.max_images:
        files = files[: args.max_images]

    if not args.force:
        already = sum(
            1 for f in files
            if (depth_dir / f"{f.stem}_aligned.npy").exists()
            and (points_dir / f"{f.stem}.npy").exists()
        )
        if already == len(files):
            print(f"All {len(files)} images already have UniK3D outputs. "
                  f"Pass --force to re-generate.")
            return

    intrinsics = read_opencv_intrinsics(root / "sparse" / "0" / "cameras.bin")
    from unik3d.models import UniK3D
    print(f"Loading UniK3D ({args.model}) ...")
    model = UniK3D.from_pretrained(args.model).cuda().eval()

    # NOTE: UniK3D's base Camera.crop / .resize mutate self.K and self.params
    # in place (and OPENCV / Fisheye624 inherit them unchanged). model.infer()
    # calls both internally, so reusing one camera object across calls causes
    # fx, fy, cx, cy to drift every iteration — fx is multiplied by
    # resize_factor each call and hits 0 within ~30-110 images (exact count
    # depends on resolution / resize ratio), after which depths come back
    # all-NaN. Rebuild a fresh camera per image to dodge it. (Verified in
    # UniK3D 0.x.)

    print(f"Processing {len(files)} images ...")
    for i, f in enumerate(files, 1):
        depth_path = depth_dir / f"{f.stem}_aligned.npy"
        points_path = points_dir / f"{f.stem}.npy"

        if not args.force and depth_path.exists() and points_path.exists():
            continue

        rgb = np.array(Image.open(f).convert("RGB"))
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().cuda()

        camera = make_unik3d_opencv(intrinsics, device="cuda")

        with torch.no_grad():
            pred = model.infer(rgb_t, camera=camera)

        depth = pred["depth"].squeeze().cpu().numpy().astype(np.float32)
        points = pred["points"].squeeze().cpu().numpy().astype(np.float32)
        if points.shape[0] == 3:
            points = points.transpose(1, 2, 0)
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
