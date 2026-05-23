"""
Step 1: Resize raw perspective photos / video frames to the training resolution.

Goal
----
Take native-resolution captures (e.g. 4K video frames, 6000² photos) and
produce a half-side downscale that is LichtFeld-friendly for both SfM and
3DGS training:

  4K video / DSLR  3840 × 2160   →  1920 × 1080   (target)
  4K square        4000 × 4000   →  2000 × 2000

The default target longest side is 1920 px. The factor is computed from the
source so a 3840-wide frame becomes 1920 wide (factor 2) without any guesswork.

If the dataset is already at (or below) the target — i.e. someone else ran
this step before, or the frames were exported at the smaller size — the
script exits cleanly without re-encoding. Re-running is a no-op.

Layout
------
Reads:  <root>/raw/*.jpg (or *.png, *.jpeg)
Writes: <root>/images/<basename>.jpg

For video data, extract frames into `raw/` first, e.g.
  ffmpeg -i video.MP4 -vf fps=2 -qscale:v 1 -qmin 1 <root>/raw/%04d.jpg

Parallelism
-----------
PIL Lanczos resize is CPU-bound and embarrassingly parallel per image. We use
a multiprocessing pool; default workers = os.cpu_count().

Usage
-----
  python 01_resize_half.py --root /path/to/dataset
  python 01_resize_half.py --root /path/to/dataset --target-longest-side 1920
  python 01_resize_half.py --root /path/to/dataset --workers 8
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


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
    with Image.open(path) as im:
        return max(im.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--target-longest-side", type=int, default=1920,
                    help="Target longest side in px (default: 1920, i.e. 1/2 "
                         "of 4K (3840) video frames).")
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

    # Fast-path: already prepared.
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

    if not raw.exists():
        sys.exit(f"ERROR: {raw} does not exist. Put your photos / extracted "
                 f"video frames there.")

    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in raw.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"ERROR: no image files in {raw}")

    src_side = longest_side(files[0])
    if src_side <= target:
        factor = 1
        print(f"Source longest side ({src_side}) ≤ target ({target}). Re-saving "
              f"without downscale (factor=1).")
    else:
        factor = max(1, src_side // target)
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
