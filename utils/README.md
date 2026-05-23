# Pipeline Utilities

Helper tools that sit alongside the main pipelines. Sorted by maturity into
three tiers; only the verified tier is published.

## Folder convention

- **`utils/` root** — verified utilities. Safe to use; results trustworthy.
  **Published.**
- **`utils/research/`** — unverified / work-in-progress code. Result quality
  not guaranteed. **Local-only (gitignored).** Promoted to `utils/` root
  once verified.
- **`utils/tmp/`** — one-off diagnostic / visualization scripts. May
  disappear in the next commit. **Local-only (gitignored).**

Promotion path: `tmp/` → `research/` → `utils/` root → (if production)
the relevant `pipeline_*/`.

## Verified utilities

- [`sparse_cleanup/`](sparse_cleanup/) — interactively prune outliers from
  COLMAP `sparse/0` using an external GUI (CloudCompare). A script pair
  (`sparse_to_ply.py` ↔ `ply_to_sparse.py`) round-trips between COLMAP
  binaries and PLY.
