"""
Step 2: Per-image mask generation for Insta360 X4 single-lens ~200deg fisheye.

Final mask  = (fisheye valid circle) AND NOT (SAM3 'person' mask)
              255 = use this pixel for SfM / 3DGS photometric loss
                0 = ignore (black border, photographer's hand/foot/grip)

Reads:  <root>/images/*.jpg
Writes: <root>/masks/<basename>.png        (0/255, LichtFeld-compatible)
        <root>/mask_overlays/<basename>.jpg (small, for visual QA)

Usage:
  conda activate sam3
  cd ~/Development/sam3
  python /path/to/02_make_masks.py --root /path/to/dataset
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# SAM3 ships its weights in bfloat16; enable autocast globally for the script.
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def detect_fisheye_circle(img_rgb, luma_thresh=80):
    """Find the circular valid region from the black border.
    Returns (cx, cy, r). The circle is shrunk by 7% to skip the vignette
    gradient at the fisheye edge — empirically tuned to cover the soft
    falloff that otherwise gets learned as floaters by 3DGS."""
    gray = (0.299 * img_rgb[..., 0] +
            0.587 * img_rgb[..., 1] +
            0.114 * img_rgb[..., 2]).astype(np.uint8)
    ys, xs = np.where(gray > luma_thresh)
    if len(xs) == 0:
        h, w = gray.shape
        return w / 2, h / 2, min(h, w) / 2
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    r = max((xs.max() - xs.min()) / 2, (ys.max() - ys.min()) / 2) * 0.93
    return cx, cy, r


def make_circle_mask(h, w, cx, cy, r):
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8) * 255


def dilate_max(mask_u8, px):
    if px <= 0:
        return mask_u8
    k = 2 * px + 1
    t = torch.from_numpy(mask_u8.astype(np.float32))[None, None]
    t = torch.nn.functional.max_pool2d(t, kernel_size=k, stride=1, padding=px)
    return (t[0, 0].numpy() > 127).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--score_thresh", type=float, default=0.30)
    ap.add_argument("--dilate_px", type=int, default=15)
    args = ap.parse_args()

    root = Path(args.root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    overlays_dir = root / "mask_overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in images_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not img_paths:
        raise SystemExit(f"No images in {images_dir}. Run 01_resize_half.py first.")
    print(f"Found {len(img_paths)} images in {images_dir}")

    first = np.array(Image.open(img_paths[0]).convert("RGB"))
    H, W = first.shape[:2]
    cx, cy, r = detect_fisheye_circle(first)
    print(f"Fisheye circle: center=({cx:.1f}, {cy:.1f}) r={r:.1f}  image={W}x{H}")
    circle_mask = make_circle_mask(H, W, cx, cy, r)

    print("Loading SAM3 ...")
    t0 = time.time()
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print(f"  loaded in {time.time() - t0:.1f}s")

    for i, p in enumerate(img_paths):
        t = time.time()
        img = Image.open(p).convert("RGB")
        arr = np.array(img)

        state = processor.set_image(img)
        out = processor.set_text_prompt(state=state, prompt=args.prompt)
        masks = out["masks"]
        scores = out["scores"]

        person = np.zeros((H, W), dtype=np.uint8)
        if masks is not None and len(masks) > 0:
            m_np = (masks.detach().float().cpu().numpy()
                    if torch.is_tensor(masks) else np.asarray(masks))
            s_np = (scores.detach().float().cpu().numpy()
                    if torch.is_tensor(scores) else np.asarray(scores))
            for k in range(m_np.shape[0]):
                if float(s_np[k]) < args.score_thresh:
                    continue
                m = m_np[k]
                while m.ndim > 2:
                    m = m.squeeze(0)
                person |= (m > 0.5).astype(np.uint8) * 255

        person = dilate_max(person, args.dilate_px)
        final = circle_mask.copy()
        final[person > 0] = 0
        Image.fromarray(final, mode="L").save(masks_dir / (p.stem + ".png"))

        # Small overlay for visual QA (red = invalid).
        overlay = arr.copy()
        invalid = final == 0
        red = np.zeros_like(arr)
        red[..., 0] = 255
        overlay[invalid] = (overlay[invalid] * 0.55 + red[invalid] * 0.45).astype(np.uint8)
        ov = Image.fromarray(overlay)
        ov.thumbnail((1024, 1024))
        ov.save(overlays_dir / (p.stem + ".jpg"), quality=80)

        print(f"[{i+1:>3}/{len(img_paths)}] {p.name}  "
              f"person_px={int((person>0).sum()):>10}  "
              f"valid_frac={(final > 0).mean():.3f}  "
              f"({time.time() - t:.1f}s)")

    print(f"\nDone. Masks -> {masks_dir}\nOverlays -> {overlays_dir}")


if __name__ == "__main__":
    main()
