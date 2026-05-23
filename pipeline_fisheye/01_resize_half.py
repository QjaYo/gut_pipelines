"""
Step 1: Resize raw fisheye photos to the training resolution.

Goal
----
Take the camera's native resolution and produce a half-side downscale that is
LichtFeld-friendly for both SfM and 3DGS training:

  Insta360 X4 single-lens fisheye  5984 × 5984  →  2992 × 2992   (target)
  GoPro Max single-lens (similar)  5760 × 5760  →  2880 × 2880

The default target longest side is 2992 px. The factor is computed from the
source so a 5984² photo becomes 2992² (factor 2) without any guesswork.

If the dataset is already at (or below) the target — i.e. the camera already
exported half-res JPEGs, or someone else ran this step before — the script
exits cleanly without re-encoding. Re-running this step on already-resized
data is a no-op.

Layout
------
Reads:  <root>/raw/*.insp (or *.jpg)
Writes: <root>/images/<basename>.jpg

`.insp` files are JPEG with a custom extension; we re-encode at the smaller
resolution. Empty (0-byte) files from camera save failures are skipped.

Parallelism
-----------
PIL Lanczos resize is CPU-bound and embarrassingly parallel per image. We use
a multiprocessing pool; default workers = os.cpu_count().

Usage
-----
  python 01_resize_half.py --root /path/to/dataset
  python 01_resize_half.py --root /path/to/dataset --target-longest-side 2992
  python 01_resize_half.py --root /path/to/dataset --workers 8

Legacy fallback: if --root/raw/ does not exist but --root/images/ contains
`.insp` files (the very first version of this pipeline), the script will
rename images -> raw first.
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".insp", ".jpg", ".jpeg", ".png"}


def resize_one(args):
    src_path, out_path, factor, quality = args
    if src_path.stat().st_size == 0:
        return src_path.name, None
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    nw, nh = w // factor, h // factor
    im = im.resize((nw, nh), Image.LANCZOS)
    im.save(out_path, quality=quality, subsampling=0)
    return src_path.name, (nw, nh)


def longest_side(path: Path) -> int:
    """Return the longest side of an image at `path` (in pixels)."""
    with Image.open(path) as im:
        return max(im.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target-longest-side", type=int, default=2992,
                    help="Target longest side in px (default: 2992, i.e. 1/2 of "
                         "the 5984² Insta360 X4 native resolution).")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Parallel workers (default: all CPU cores). Use 1 for serial.")
    ap.add_argument("--force", action="store_true",
                    help="Re-resize even when images/ is already at or below target.")
    args = ap.parse_args()

    root = Path(args.root)
    raw = root / "raw"
    out = root / "images"
    target = args.target_longest_side

    # Fast-path: dataset already prepared (e.g. images/ comes pre-downscaled
    # from the camera export). If every existing image is already ≤ target,
    # nothing to do.
    if not args.force and out.exists() and any(p.suffix.lower() in IMAGE_EXTS
                                               for p in out.iterdir()):
        existing = [p for p in out.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        sample = existing[0]
        side = longest_side(sample)
        if side <= target:
            print(f"images/ already populated: {len(existing)} files, sample "
                  f"longest side {side} ≤ target {target}. Skipping resize.")
            print("(Pass --force to re-encode anyway.)")
            return
        else:
            print(f"images/ exists but sample longest side {side} > target "
                  f"{target}. Will re-process from raw/ (or these files if "
                  f"raw/ is missing).")

    # Legacy fallback: .insp files dropped directly into images/
    if not raw.exists() and out.exists():
        insp_count = sum(1 for p in out.iterdir() if p.suffix.lower() == ".insp")
        if insp_count > 0:
            print(f"Legacy layout: renaming images/ → raw/ ({insp_count} .insp files)")
            out.rename(raw)

    if not raw.exists():
        sys.exit(f"ERROR: {raw} does not exist. Put the camera files there.")

    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"ERROR: no image files in {raw}")

    # Compute factor from first image vs target. Fisheye is square, so longest
    # side = either side. We pick the smallest integer factor that brings the
    # source at or below the target.
    src_side = longest_side(files[0])
    if src_side <= target:
        factor = 1
        print(f"Source longest side ({src_side}) ≤ target ({target}). Re-saving "
              f"without downscale (factor=1).")
    else:
        factor = max(1, src_side // target)
        # Adjust: ensure the result is ≤ target.
        while src_side // factor > target:
            factor += 1
        print(f"Source longest side {src_side}, target {target} → factor {factor} "
              f"(result {src_side // factor}).")

    jobs = [(p, out / (p.stem + ".jpg"), factor, args.quality) for p in files]
    n_total = len(jobs)
    print(f"Resizing {n_total} files with {args.workers} worker(s) ...")

    skipped_empty, written = [], 0
    if args.workers <= 1:
        results = (resize_one(j) for j in jobs)
    else:
        pool = Pool(processes=args.workers)
        results = pool.imap_unordered(resize_one, jobs)

    for name, size in results:
        if size is None:
            skipped_empty.append(name)
            continue
        written += 1
        if written % 10 == 0 or written == n_total:
            print(f"[{written:>3}/{n_total}] {name} -> {size[0]}x{size[1]}")

    if args.workers > 1:
        pool.close()
        pool.join()

    print(f"\nDone. {written} images -> {out}")
    if skipped_empty:
        print(f"Skipped {len(skipped_empty)} empty/corrupt source files:")
        for n in skipped_empty:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
