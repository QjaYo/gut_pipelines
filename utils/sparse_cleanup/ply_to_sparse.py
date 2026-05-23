"""
Import edited PLY back into a filtered COLMAP sparse model.

Reads `<root>/<sparse>/points3D_clean.ply` (the file you saved out of
CloudCompare/MeshLab/etc., with the `point_id` scalar field intact) and
writes a new sparse directory with `points3D.bin` filtered to just the
remaining IDs. `cameras.bin` and `images.bin` are copied verbatim.

Workflow paired with `sparse_to_ply.py` — see that file's docstring for
the round-trip steps.

Safety note: the new sparse is written to `<root>/<sparse>_clean/` by
default. Your original `<root>/<sparse>/` is never modified.
"""

import argparse
import shutil
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData


def _read_track_from_binary(path: Path):
    """Read the full points3D.bin including track lists.

    Returns dict[point_id] = (xyz_bytes, rgb_bytes, error_bytes, track_bytes)
    where each value is the *raw* bytes ready to be re-emitted. This avoids
    re-encoding floats / breaking anyone's idea of exact binary equality.
    """
    points = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            head = f.read(43)                # id(Q) + xyz(3d) + rgb(3B) + err(d)
            pid = struct.unpack("<Q", head[:8])[0]
            track_len_bytes = f.read(8)
            track_len = struct.unpack("<Q", track_len_bytes)[0]
            track_bytes = f.read(8 * track_len)  # (image_id, point2D_idx) × len
            points[pid] = head + track_len_bytes + track_bytes
    return points


def _write_points3D_bin(path: Path, points_dict):
    """Inverse of `_read_track_from_binary`. Preserves byte-for-byte payload."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points_dict)))
        for blob in points_dict.values():
            f.write(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dataset root containing sparse/0/")
    ap.add_argument("--sparse", default="sparse/0",
                    help="relative sparse path (default: sparse/0)")
    ap.add_argument("--ply", default="points3D_clean.ply",
                    help="edited PLY filename inside the sparse dir "
                         "(default: points3D_clean.ply)")
    ap.add_argument("--out", default="sparse/0_clean",
                    help="output sparse directory relative to root "
                         "(default: sparse/0_clean)")
    args = ap.parse_args()

    root = Path(args.root)
    sparse = root / args.sparse
    src_bin = sparse / "points3D.bin"
    ply_path = sparse / args.ply
    out_dir = root / args.out

    if not src_bin.is_file():
        raise SystemExit(f"missing {src_bin}")
    if not ply_path.is_file():
        raise SystemExit(f"missing {ply_path}  — did you save the cleaned "
                         f"PLY out of CloudCompare into {sparse}/?")

    # Read original points (raw bytes per id), preserving tracks / errors.
    print(f"Reading {src_bin} ...")
    points_raw = _read_track_from_binary(src_bin)
    print(f"  original: {len(points_raw)} points")

    # Read remaining point_ids from the edited PLY.
    print(f"Reading {ply_path} ...")
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    if "point_id" not in vertex.data.dtype.names:
        raise SystemExit(
            f"{ply_path} has no `point_id` scalar field. Did you save the\n"
            f"PLY without losing scalar fields? CloudCompare's default save\n"
            f"keeps them — but 'Save As' → PLY → 'Default' must NOT drop\n"
            f"the scalar field. Check the 'Scalar fields → save' option."
        )
    kept_ids = np.array(vertex.data["point_id"], dtype=np.uint64)
    kept_set = set(int(x) for x in kept_ids.tolist())
    print(f"  kept:     {len(kept_set)} points "
          f"({len(points_raw) - len(kept_set)} removed)")

    # Filter dict
    points_filtered = {pid: blob for pid, blob in points_raw.items()
                       if pid in kept_set}
    if len(points_filtered) != len(kept_set):
        missing = len(kept_set) - len(points_filtered)
        print(f"  WARNING: {missing} IDs in PLY had no match in original "
              f"points3D.bin (numeric drift?). They were dropped.")

    # Write new sparse
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy cameras + images verbatim
    for name in ("cameras.bin", "images.bin"):
        s = sparse / name
        d = out_dir / name
        if s.is_file():
            shutil.copy2(s, d)
            print(f"  copy: {name}")
    # Write filtered points3D.bin
    dst_bin = out_dir / "points3D.bin"
    _write_points3D_bin(dst_bin, points_filtered)
    print(f"  write: points3D.bin  ({len(points_filtered)} points)")
    print(f"\nNew sparse: {out_dir}")
    print("  Use this path for training:")
    print(f"    --data-dir {root}   (then have the trainer use {args.out} as sparse)")
    print("  Or rename sparse/0_clean → sparse/0 (back up the original first).")


if __name__ == "__main__":
    main()
