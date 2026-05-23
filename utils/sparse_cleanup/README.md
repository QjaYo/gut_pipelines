# Sparse outlier cleanup

Interactively remove outlier points from a COLMAP `sparse/0` model using
an external GUI (CloudCompare), then write a cleaned sparse back. Useful
before restarting training.

```bash
DATA=/path/to/dataset

# 1) sparse → PLY (preserves point_id)
python sparse_to_ply.py --root $DATA
# → $DATA/sparse/0/points3D.ply

# 2) Open in CloudCompare
cloudcompare $DATA/sparse/0/points3D.ply
#   - Camera tools → Walk mode (WASD navigation)
#   - Right-drag: look around
#   - S (Segment tool): left-drag / polygon selection
#   - Delete, then "Save As" → PLY → file name `points3D_clean.ply`
#   - Important: keep "Scalar fields → save" enabled (preserves point_id)

# 3) Edited PLY → filtered sparse directory
python ply_to_sparse.py --root $DATA
# → $DATA/sparse/0_clean/   (original sparse/0/ untouched)

# 4) Train on the cleaned sparse
#    Either point the trainer at sparse/0_clean, or rename it to sparse/0
#    after backing up the original.
```

## Safety notes

- **3DGS / DN-Splatter training**: arbitrary point removal is safe — the
  loader iterates over a dict and ignores missing ids.
- **Re-triangulation / re-running BA**: not recommended; `images.bin`
  carries orphan references that will warn.
- **Original preservation**: always written to `sparse/0_clean/`; the
  original `sparse/0/` is never modified.
