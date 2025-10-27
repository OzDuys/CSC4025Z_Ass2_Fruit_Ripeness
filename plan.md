# Project Plan / TODOs (Assignment 2)

This file tracks remaining work to finish the report, tuning, and submission. The report skeleton is in `report.tex` and includes current results and figures.

## Status Snapshot
- Dataset chosen and described (9 classes over 3 fruits × 3 ripeness).
- Baselines implemented: kNN (raw pixels), HOG + Linear (expected range).
- CNN baseline trained (5 and 20 epoch runs); best test accuracy 84.5%.
- Hyperparameter tuning notebook and methodology prepared; runs in progress.
- Report skeleton created with figures and rubric-aligned sections.

## High‑Priority TODOs
- [ ] Run W&B sweep and pick best non‑pretrained CNN config (per brief).
- [ ] Train final CNN with selected config; save best checkpoint and seed.
- [ ] Compute CNN confusion matrix on test split.
- [ ] Export CNN classification report (per‑class precision/recall/F1, macro‑F1, balanced accuracy).
- [ ] Plot ROC/PR curves (one‑vs‑rest) for CNN.
- [ ] Document augmentation/regularisation ablations (on/off comparisons).
- [ ] Produce final predictions file for test set (CSV or JSON) and include in repo.

## Baselines (verify and finalise)
- [ ] Confirm HOG + Logistic/SVM final numbers on test; capture confusion matrices (already generated image available).
- [ ] Ensure kNN notebook saves confusion matrix and K‑sweep plots (images present).
- [ ] Summarise baseline configuration in report table (done) and add any missing hyperparameters.

## Data & Splits
- [ ] Verify train/val/test split reproducibility (seeded 15% val from train).
- [ ] Record class counts and any imbalance notes; include small table in report.

## Hyperparameter Tuning
- [ ] Sweep space: optimiser, LR, weight decay, dropout, batch size, aug strength.
- [ ] Early stopping on validation macro‑F1; log best model.
- [ ] Insert tuned results table into `report.tex` (currently placeholder).

## Analysis & Visuals
- [ ] Misclassified examples montage (per confusing pairs like ripe vs unripe).
- [ ] Per‑class support and error breakdown.
- [ ] Add CNN training curves from best run (currently baseline curves shown).

## Reproducibility
- [ ] Record exact environment (PyTorch/torchvision versions; CUDA if used).
- [ ] Write a short run script or notebook cell order to reproduce final model.
- [ ] Pin seeds and document deterministic flags where feasible.

## Report
- [ ] Fill in hyperparameter tuning results and discussion sections.
- [ ] Add CNN confusion matrix and classification report figures.
- [ ] Add limitations/future work and ethics notes (safety, oversight).
- [ ] Final polish pass for clarity; keep ≈5 pages guideline.

## Presentation
- [ ] 5‑minute deck: problem, data, baselines, CNN, results, analysis.
- [ ] Figure selection: kNN confusion matrix, HOG vs CNN comparison, CNN curves.
- [ ] Rehearse to stay under time; prepare Q&A on choices.

## Submission Packaging
- [ ] Include report (PDF), code, data sample or download helper, final predictions, and slides.
- [ ] Follow naming convention with student numbers.
- [ ] Sanity‑check that a fresh environment can reproduce results.

## Nice‑to‑Haves (time permitting)
- [ ] Grad‑CAM or saliency maps for insight into model focus regions.
- [ ] Lightweight experiment tracker export (CSV of runs/metrics) for appendix.
