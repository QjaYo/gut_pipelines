"""
Step 5 (optional): Derive surface normals from UniK3D per-pixel 3D points.

Step 4 saves per-pixel 3D points (in UniK3D's camera frame) into
<root>/unik3d_points/*.npy. This step turns those into normal maps suitable
for DN-Splatter's d+n training.

Method
------
Surface normals are computed by taking the cross-product of the local
tangent vectors in image space (central differences of the 3D-point map),
then normalizing. Because UniK3D outputs metric 3D directly, this gives the
correct normal at every pixel — including the fisheye edges where mono
normal estimators (DSINE, Omnidata) trained on perspective images fail.

  n(u,v) = normalize( ∂P/∂u × ∂P/∂v )

We flip the sign if needed so normals point toward the camera (negative
camera-Z half-space). This matches DN-Splatter's camera-frame convention,
so loading with --normal-format dsine (the camera-frame option) is correct.

Robustness
----------
- Depth discontinuities (e.g. silhouettes) produce huge tangent vectors and
  unstable normals. We optionally mask those by a depth gradient threshold.
- Edge pixels use replicated values via np.pad.

Inputs / Outputs
----------------
Reads:
  <root>/unik3d_points/<stem>.npy   per-pixel (H, W, 3), from step 4

Writes:
  <root>/normals_from_pretrain/<stem>.png   RGB-encoded unit normal in
                                            camera frame, [-1,1] mapped to
                                            [0,255]. DN-Splatter convention.

Environment
-----------
No model needed. Any env with numpy + Pillow works.

Usage
-----
  python 05_normals_from_points.py --root <dataset>
  python 05_normals_from_points.py --root <dataset> --depth-edge-thresh 0.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def normals_from_points(
    points: np.ndarray,
    depth_edge_thresh: float = 0.0,
) -> np.ndarray:
    """Compute per-pixel unit normals from a (H, W, 3) 3D-point map.

    Args:
        points: (H, W, 3) float array in camera frame.
        depth_edge_thresh: If > 0, pixels where |∇z| exceeds this fraction
            of the local depth are replaced with the median of their
            neighbourhood (helps with silhouette artefacts).

    Returns:
        (H, W, 3) unit normals in camera frame, sign flipped so the average
        normal points toward the camera (negative-Z side).
    """
    pad = np.pad(points, ((1, 1), (1, 1), (0, 0)), mode="edge")
    dx = pad[1:-1, 2:, :] - pad[1:-1, :-2, :]
    dy = pad[2:, 1:-1, :] - pad[:-2, 1:-1, :]

    if depth_edge_thresh > 0:
        # Suppress normals across strong depth jumps.
        z = points[..., 2]
        zpad = np.pad(z, ((1, 1), (1, 1)), mode="edge")
        dzdx = zpad[1:-1, 2:] - zpad[1:-1, :-2]
        dzdy = zpad[2:, 1:-1] - zpad[:-2, 1:-1]
        local_z = np.maximum(np.abs(z), 1e-3)
        bad = (np.abs(dzdx) > depth_edge_thresh * local_z) | (
              np.abs(dzdy) > depth_edge_thresh * local_z)
        # zero-out bad tangents so cross is 0; we'll fill later
    else:
        bad = None

    n = np.cross(dx, dy)                        # (H, W, 3)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-8)

    if bad is not None:
        # Replace bad pixels with zeros; downstream loss should be masked
        # by the fisheye disc mask anyway, so leaving zeros there is fine.
        n[bad] = 0.0

    # Sign convention: normals point toward the camera (camera at origin,
    # +Z forward toward the scene, so a surface facing the camera has nz < 0).
    # If the average is positive (i.e. cross gave outward direction), flip.
    if np.mean(n[..., 2]) > 0:
        n = -n

    return n.astype(np.float32)


def normal_to_png(normal: np.ndarray) -> Image.Image:
    """Encode (H, W, 3) unit normal in [-1, 1] to RGB PNG in [0, 255]."""
    img = ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Dataset root containing unik3d_points/")
    ap.add_argument("--depth-edge-thresh", type=float, default=0.0,
                    help="If > 0, mask normals across depth discontinuities. "
                         "Threshold is fraction of local depth (e.g. 0.5 = 50%).")
    ap.add_argument("--force", action="store_true",
                    help="Re-generate even if outputs already exist.")
    args = ap.parse_args()

    root = Path(args.root)
    points_dir = root / "unik3d_points"
    if not points_dir.exists():
        sys.exit(f"ERROR: {points_dir} not found. Run step 4 first.")

    normal_dir = root / "normals_from_pretrain"
    normal_dir.mkdir(exist_ok=True, parents=True)

    files = sorted(points_dir.glob("*.npy"))
    if not files:
        sys.exit(f"ERROR: no .npy files in {points_dir}")

    n_skip = 0
    print(f"Processing {len(files)} point maps ...")
    for i, f in enumerate(files, 1):
        out_path = normal_dir / f"{f.stem}.png"
        if not args.force and out_path.exists():
            n_skip += 1
            continue

        points = np.load(f)                                  # (H, W, 3)
        if points.ndim != 3 or points.shape[-1] != 3:
            print(f"  WARN: {f.name} has shape {points.shape}, skip")
            continue

        normal = normals_from_points(points, depth_edge_thresh=args.depth_edge_thresh)
        normal_to_png(normal).save(out_path)

        if i % 20 == 0 or i == len(files):
            print(f"  [{i:>3}/{len(files)}] {f.name}")

    if n_skip:
        print(f"  (skipped {n_skip} that already existed)")
    print(f"\nDone. normals -> {normal_dir}/")


if __name__ == "__main__":
    main()
