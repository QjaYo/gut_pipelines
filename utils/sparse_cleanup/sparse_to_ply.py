"""
Export COLMAP sparse model → PLY for interactive cleanup.

Each point's `point3D_id` is preserved as a custom PLY scalar field so the
round-trip back to `points3D.bin` (via `ply_to_sparse.py`) is exact — no
nearest-neighbor matching, no precision drift.

Workflow:
    1. python sparse_to_ply.py --root <dataset>
       → writes <root>/sparse/0/points3D.ply

    2. Open the PLY in CloudCompare:
         - WASD: navigate (Camera mode = Walk in CloudCompare)
         - Right mouse: look around
         - Left drag: rectangle / polygon select (Segment tool)
         - Delete the selected points
       Save back as <root>/sparse/0/points3D_clean.ply  (keep .ply, keep
       the `point_id` scalar field — CloudCompare preserves it by default).

    3. python ply_to_sparse.py --root <dataset>
       → reads points3D_clean.ply, writes a filtered points3D.bin into
         <root>/sparse/0_clean/  (so the original sparse/0/ stays intact).

Notes:
    - Only `points3D.bin` is rewritten; `cameras.bin` and `images.bin` are
      copied as-is. The `images.bin` may keep references to deleted
      point3D_ids — that's harmless for 3DGS training (the loader only
      iterates `points3D` dict, doesn't dereference these IDs).
    - GS / DN-Splatter loaders are safe with arbitrary point removal.
      Avoid running re-triangulation / BA on the cleaned sparse though;
      that would complain about orphan references.
"""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def read_points3D_binary(path: Path):
    """Lightweight COLMAP points3D.bin reader.

    Returns three numpy arrays — ids, xyz, rgb — in dict-iteration order.
    """
    import struct
    points_ids, xyzs, rgbs = [], [], []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pid, x, y, z, r, g, b, _err = struct.unpack("<QdddBBBd", f.read(43))
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)   # skip track entries
            points_ids.append(pid)
            xyzs.append((x, y, z))
            rgbs.append((r, g, b))
    return (np.array(points_ids, dtype=np.uint64),
            np.array(xyzs, dtype=np.float64),
            np.array(rgbs, dtype=np.uint8))


def write_ply(out_path: Path, ids, xyz, rgb):
    """Write PLY with x,y,z,red,green,blue,point_id columns."""
    n = len(ids)
    vertex = np.zeros(n, dtype=[
        ("x", "f8"), ("y", "f8"), ("z", "f8"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("point_id", "u8"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vertex["point_id"] = ids

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dataset root containing sparse/0/")
    ap.add_argument("--sparse", default="sparse/0",
                    help="relative sparse path (default: sparse/0)")
    ap.add_argument("--out", default="points3D.ply",
                    help="output filename inside the sparse dir (default: points3D.ply)")
    args = ap.parse_args()

    root = Path(args.root)
    sparse = root / args.sparse
    src = sparse / "points3D.bin"
    if not src.is_file():
        raise SystemExit(f"missing {src}")

    print(f"Reading {src} ...")
    ids, xyz, rgb = read_points3D_binary(src)
    print(f"  {len(ids)} points")

    out = sparse / args.out
    print(f"Writing {out} ...")
    write_ply(out, ids, xyz, rgb)
    print("Done.\n")
    print("Next steps:")
    print(f"  1. Open {out} in CloudCompare")
    print( "     - Camera tools → switch to 'Walk' (FPS) for WASD navigation")
    print( "     - Edit → Segment (or shortcut S) to rectangle-select")
    print( "     - Delete the selected points, save as `points3D_clean.ply`")
    print(f"  2. python ply_to_sparse.py --root {args.root}")


if __name__ == "__main__":
    main()
