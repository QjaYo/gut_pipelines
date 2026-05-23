"""
Step 3: hloc SfM for Insta360 X4 single-lens (~200deg) fisheye, with output
        written directly into the LichtFeld-Studio standard layout.

Reads:
  <root>/images/*.jpg              half-resolution RGB images (step 1)
  <root>/masks/<basename>.png      per-image masks (step 2)

Writes:
  <root>/sparse/0/cameras.bin      COLMAP sparse model (LichtFeld input)
  <root>/sparse/0/images.bin
  <root>/sparse/0/points3D.bin
  <root>/hloc_cache/               re-run cache: features.h5, matches.h5,
                                                 pairs.txt, global-feats-*.h5,
                                                 database.db, images_masked/,
                                                 sfm_work/

After this step you can train directly with:
  build/LichtFeld-Studio -d <root> -o <root>/output --gut --mask-mode ignore

Pipeline
--------
  1. Apply masks to images (write into hloc_cache/images_masked/) so
     SuperPoint does not extract features in masked regions. hloc does
     not consume a user mask channel; pre-blackening is the safe path.
  2. SuperPoint feature extraction (cached) on the *masked* images.
  3. MegaLoc global descriptors on the *original* images (not masked —
     MegaLoc was trained on natural images; large black regions degrade
     the descriptor).
  4. Retrieval pair list: top-k most similar per image (default k=50).
     Fall back to exhaustive if --num_matched <= 0.
  5. LightGlue matching (cached).
  6. COLMAP SfM with shared OPENCV_FISHEYE camera prior, BA refines all
     intrinsics + distortion.
  7. Copy the 3 result .bin files into <root>/sparse/0/.

Initial intrinsic prior (OPENCV_FISHEYE: fx,fy,cx,cy,k1,k2,k3,k4)
-----------------------------------------------------------------
  Defaults assume a 2992x2992 single-lens X4 image (1/2 of 5984).
    cx, cy: fisheye disk center from black-border detection (~image center)
    fx=fy: from equidistant model with assumed FOV=200deg
           f = (image_size/2) / (FOV_rad/2)  ~  844 px
    k1..k4: 0 (BA estimates)
  Pass --fov / --image_size for other configurations.
"""

import argparse
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

import pycolmap
from hloc import (extract_features, match_features, pairs_from_exhaustive,
                  pairs_from_retrieval, reconstruction)


def apply_masks(images_dir: Path, masks_dir: Path, out_dir: Path):
    """Black out masked regions in each image (where mask <= 127). Output to
    out_dir as JPEG. Re-run safe: skips files whose target already exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    todo = [p for p in files if not (out_dir / p.name).exists()]
    print(f"Masking {len(todo)} new images (total {len(files)}) -> {out_dir}")
    for p in todo:
        mask_path = masks_dir / (p.stem + ".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"missing mask: {mask_path}")
        img = np.array(Image.open(p).convert("RGB"))
        msk = np.array(Image.open(mask_path).convert("L"))
        if msk.shape != img.shape[:2]:
            raise ValueError(
                f"size mismatch: img={img.shape[:2]} mask={msk.shape} ({p.name})")
        img[msk <= 127] = 0
        Image.fromarray(img).save(out_dir / p.name, quality=95, subsampling=0)


def derive_intrinsic_prior(images_dir: Path, fov_deg: float):
    """Return (image_size, cx, cy, focal). cx/cy from any sample image; focal
    from equidistant fisheye model: f = (img/2) / (FOV_rad / 2)."""
    sample = next(p for p in images_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    w, h = Image.open(sample).size
    if w != h:
        raise ValueError(f"Expected square image, got {w}x{h}. The "
                         "OPENCV_FISHEYE prior here assumes a square fisheye.")
    img_sz = w
    cx, cy = img_sz / 2.0, img_sz / 2.0
    fov_rad = fov_deg * math.pi / 180.0
    focal = (img_sz / 2.0) / (fov_rad / 2.0)
    return img_sz, cx, cy, focal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Dataset root containing images/ and masks/")
    ap.add_argument("--fov", type=float, default=200.0,
                    help="Initial FOV guess in degrees for equidistant prior")
    ap.add_argument("--feature", default="superpoint_max",
                    choices=["superpoint_aachen", "superpoint_max", "disk",
                             "aliked-n16"])
    ap.add_argument("--matcher", default="superpoint+lightglue",
                    choices=["superpoint+lightglue", "disk+lightglue",
                             "aliked+lightglue", "superglue"])
    ap.add_argument("--retrieval", default="megaloc",
                    choices=["megaloc", "netvlad", "openibl", "eigenplaces"],
                    help="Global descriptor for retrieval (default: megaloc)")
    ap.add_argument("--num_matched", type=int, default=50,
                    help="Top-k retrieval pairs per image (default 50). "
                         "Set <= 0 to fall back to exhaustive matching.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    cache = root / "hloc_cache"
    cache.mkdir(parents=True, exist_ok=True)
    masked_dir = cache / "images_masked"
    sfm_work = cache / "sfm_work"
    sparse_out = root / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise SystemExit(f"missing {images_dir}. Run step 1 (resize) first.")
    if not masks_dir.exists():
        raise SystemExit(f"missing {masks_dir}. Run step 2 (masks) first.")

    use_retrieval = args.num_matched > 0
    n_steps = 5 if use_retrieval else 4

    # 1. mask -> images_masked/
    t = time.time()
    apply_masks(images_dir, masks_dir, masked_dir)
    print(f"  masked in {time.time() - t:.1f}s\n")

    # 2. Local features (SuperPoint) on masked images
    feat_conf = extract_features.confs[args.feature]
    print(f"[1/{n_steps}] Extracting local features ({args.feature}) ...")
    t = time.time()
    feature_path = extract_features.main(
        feat_conf, masked_dir, cache, feature_path=cache / "features.h5")
    print(f"  -> {feature_path}  ({time.time() - t:.1f}s)\n")

    # 3. Pair list: retrieval (default) or exhaustive
    pairs_path = cache / "pairs.txt"
    if use_retrieval:
        # Global descriptors on ORIGINAL images (MegaLoc was trained on natural
        # images; large black regions degrade the descriptor).
        retrieval_conf = extract_features.confs[args.retrieval]
        print(f"[2/{n_steps}] Extracting global descriptors ({args.retrieval}) ...")
        t = time.time()
        retrieval_path = extract_features.main(
            retrieval_conf, images_dir, cache)
        print(f"  -> {retrieval_path}  ({time.time() - t:.1f}s)\n")

        print(f"[3/{n_steps}] Retrieval pairs (top-{args.num_matched}) ...")
        pairs_from_retrieval.main(
            retrieval_path, pairs_path, num_matched=args.num_matched)
    else:
        print(f"[2/{n_steps}] Building exhaustive pair list ...")
        image_list = sorted([p.name for p in masked_dir.iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    n_pairs = sum(1 for _ in open(pairs_path))
    n_imgs = sum(1 for p in masked_dir.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"  -> {n_pairs} pairs over {n_imgs} images\n")

    # 4. LightGlue matching
    match_conf = match_features.confs[args.matcher]
    step_no = 4 if use_retrieval else 3
    print(f"[{step_no}/{n_steps}] Matching ({args.matcher}) ...")
    t = time.time()
    match_path = match_features.main(
        match_conf, pairs_path, features=feature_path,
        matches=cache / "matches.h5")
    print(f"  -> {match_path}  ({time.time() - t:.1f}s)\n")

    # 5. Reconstruction with OPENCV_FISHEYE prior
    img_sz, cx, cy, focal = derive_intrinsic_prior(images_dir, args.fov)
    cam_params = f"{focal:.3f},{focal:.3f},{cx:.3f},{cy:.3f},0,0,0,0"
    step_no = 5 if use_retrieval else 4
    print(f"[{step_no}/{n_steps}] Reconstruction (OPENCV_FISHEYE, shared camera)")
    print(f"      image={img_sz}x{img_sz}, FOV={args.fov} deg => "
          f"f={focal:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # If a previous reconstruction is in the cache, wipe it for a clean run.
    if sfm_work.exists():
        shutil.rmtree(sfm_work)
    sfm_work.mkdir(parents=True)

    t = time.time()
    rec = reconstruction.main(
        sfm_dir=sfm_work,
        image_dir=masked_dir,
        pairs=pairs_path,
        features=feature_path,
        matches=match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options={
            "camera_model": "OPENCV_FISHEYE",
            "camera_params": cam_params,
        },
        mapper_options={
            "ba_refine_focal_length": True,
            "ba_refine_principal_point": True,
            "ba_refine_extra_params": True,
        },
        verbose=args.verbose,
    )
    print(f"\nSfM done in {time.time() - t:.1f}s")
    if rec is None:
        raise SystemExit("Reconstruction FAILED — no model produced.")

    print(f"  registered: {rec.num_reg_images()}/{n_imgs}")
    print(f"  points3D : {rec.num_points3D()}")
    cam = next(iter(rec.cameras.values()))
    print(f"  camera   : {cam.model.name}  params={list(cam.params)}")

    # 6. Copy the 3 model files into LichtFeld layout.
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        src = sfm_work / name
        if not src.exists():
            raise SystemExit(f"expected {src} missing — hloc output layout changed?")
        shutil.copy2(src, sparse_out / name)
    print(f"\n=> {sparse_out}/  ready for LichtFeld training.")


if __name__ == "__main__":
    main()
