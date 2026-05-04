# CSE427 SST Project Structure Checklist

## Kept in Git

These files/folders are safe and should be pushed:

- `README.md`
- `.gitignore`
- `cleanup_manifest.csv`
- `src/`
- `src/pipelines/`
- `notebooks/`
- `data/processed/`
- `outputs/eda/`
- `outputs/figures/`
- `outputs/results/`
- `paper/`

## Kept Locally Only

These files/folders are required for local reproduction or raw data access, but should not be pushed:

- `data/gqa_raw/questions/val_balanced_questions.json`
- `data/gqa_raw/val_sceneGraphs.json`
- `data/gqa_images_subset_200/`

## Archived Locally Only

These folders should stay out of Git:

- `archive_cleanup/`
- `archive/`

## Ignored Junk / Temporary Files

These should never be committed:

- `.DS_Store`
- `__pycache__/`
- `.ipynb_checkpoints/`
- `logs/`
- `tmp/`
- `temp/`
- model checkpoints such as `.pt`, `.pth`, `.ckpt`
- large arrays such as `.npy`, `.npz`
- compressed archives such as `.zip`, `.tar.gz`

## Final Submission Checklist

Before pushing or submitting:

- [ ] `git status` shows only intended changes or is clean.
- [ ] No files from `data/gqa_raw/` are staged.
- [ ] No files from `data/gqa_images_subset_200/` are staged.
- [ ] No files from `archive_cleanup/` are staged.
- [ ] Final figures are present in `outputs/figures/`.
- [ ] Final CSV/Markdown results are present in `outputs/results/`.
- [ ] Final notebooks are present in `notebooks/`.
- [ ] Source code is present in `src/`.
- [ ] Paper/report assets are present in `paper/`.
