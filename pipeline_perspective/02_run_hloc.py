"""hloc SfM for perspective cameras (pinhole + radial/tangential distortion).

Reads:
  <root>/images/*.jpg     RGB images (e.g. from 01_resize_half.py)

Writes:
  <root>/sparse/0/cameras.bin     COLMAP sparse model (LichtFeld input)
  <root>/sparse/0/images.bin
  <root>/sparse/0/points3D.bin
  <root>/hloc_cache/              re-run cache: features.h5, matches.h5,
                                                 pairs.txt, global-feats-*.h5,
                                                 database.db, sfm_work/

After this step you can train directly with LichtFeld:
  build/LichtFeld-Studio -d <root> -o <root>/output --strategy mcmc

Pipeline
--------
  1. SuperPoint feature extraction (cached).
  2. MegaLoc global descriptors for retrieval.
  3. Retrieval pair list: top-k most similar per image (default k=50).
     Fall back to exhaustive if --num_matched <= 0.
  4. LightGlue matching (cached).
  5. COLMAP SfM with shared OPENCV camera (perspective + 2 radial + 2 tangential),
     BA refines all intrinsics + distortion.
  6. Copy the 3 result .bin files into <root>/sparse/0/.

This is the perspective counterpart of `pipeline_fisheye/03_run_hloc.py`. Key
differences from the fisheye version:
  - Camera model: OPENCV (not OPENCV_FISHEYE)
  - No SAM3 mask step (perspective edges don't need masking)
  - No FOV prior (COLMAP estimates intrinsics from EXIF / defaults)
"""

import argparse
import shutil
import time
from pathlib import Path

import pycolmap
from hloc import (extract_features, match_features, pairs_from_exhaustive,
                  pairs_from_retrieval, reconstruction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Dataset root containing images/")
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
    cache = root / "hloc_cache"
    cache.mkdir(parents=True, exist_ok=True)
    sfm_work = cache / "sfm_work"
    sparse_out = root / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise SystemExit(f"missing {images_dir}. Put images there (or run step 1 to resize).")

    use_retrieval = args.num_matched > 0
    n_steps = 5 if use_retrieval else 4

    # 1. Local features (SuperPoint by default)
    feat_conf = extract_features.confs[args.feature]
    print(f"[1/{n_steps}] Extracting local features ({args.feature}) ...")
    t = time.time()
    feature_path = extract_features.main(
        feat_conf, images_dir, cache, feature_path=cache / "features.h5")
    print(f"  -> {feature_path}  ({time.time() - t:.1f}s)\n")

    # 2. Pair list: retrieval (default) or exhaustive
    pairs_path = cache / "pairs.txt"
    if use_retrieval:
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
        image_list = sorted([p.name for p in images_dir.iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    n_pairs = sum(1 for _ in open(pairs_path))
    n_imgs = sum(1 for p in images_dir.iterdir()
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

    # 5. Reconstruction with OPENCV (perspective) camera, shared across images
    step_no = 5 if use_retrieval else 4
    print(f"[{step_no}/{n_steps}] Reconstruction (OPENCV, shared camera)")

    if sfm_work.exists():
        shutil.rmtree(sfm_work)
    sfm_work.mkdir(parents=True)

    t = time.time()
    rec = reconstruction.main(
        sfm_dir=sfm_work,
        image_dir=images_dir,
        pairs=pairs_path,
        features=feature_path,
        matches=match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options={
            "camera_model": "OPENCV",
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
