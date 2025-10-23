# Project Plan

- Visualise results: ROC curve etc

- Weights and biases

- Implement baseline without NN


---

# Chatgpt Project Plan

## Problem Formulation
- Select evaluation metrics (accuracy, macro F1, per-class recall) and justify them.

## Baseline Implementation
- Implement a simple non-neural baseline (e.g., k-NN on color histograms or Naïve Bayes).
- Evaluate the baseline on validation/test data and store the metrics for comparison.

## Neural Model Development
- Start from the PyTorch baseline notebook; ensure scripts/notebooks support clean training/evaluation.
- Systematically explore architectures, augmentations, and optimization hyperparameters.
- Log training curves, checkpoints, and experiment settings for reproducibility.

## Evaluation & Analysis
- Report final metrics for baseline and best neural model on validation/test splits.
- Produce diagnostics (confusion matrix, class-wise performance, misclassified samples).
- Summarize insights: architecture choices, optimization behavior, limitations.

## Reporting
- Draft the ≈5-page report covering problem, methodology, experiments, results, and analysis.
- Add reproducibility instructions (environment setup, dataset download, command order).
- Acknowledge any external code or resources used.

## Presentation Preparation
- Build a 5-minute slide deck highlighting problem formulation, data, model, results, and conclusions.
- Rehearse delivery; prepare to answer questions on ethics and future work.

## Submission Packaging
- Ensure the repository holds final notebooks/scripts, dataset download helper, and requirements.
- Generate final test predictions and include them with submission artefacts.
- Zip report, code, data samples/checkpoints, and slides following the naming convention.
